"""MUSA FlashAttention adapter for Qwen3.5 training.

Transformers 5.2 detects upstream FlashAttention through CUDA/HIP-specific
availability checks.  The MUSA package in this environment is therefore not
selected by the stock ``flash_attention_2`` integration.  This module registers
a project-local implementation that always uses the varlen API, including for
fully valid batches, because the installed dense MUSA backward kernel is not
numerically safe for Qwen3.5's head dimension 256.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


logger = logging.getLogger(__name__)

FLASH_CONFIG_ALIAS = "flash"
MUSA_VARLEN_ATTENTION = "musa_flash_varlen"
_VISION_ORIGINAL_ATTENTION_ATTR = "_starvla_qwen35_original_vision_attention"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _config_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


def _musa_is_available() -> bool:
    return bool(hasattr(torch, "musa") and torch.musa.is_available())


def _register_musa_varlen_attention(*, require_mate: bool) -> None:
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn.backends import musa as flash_attn_musa
    except ImportError as error:
        raise ImportError(
            "The Qwen3.5 MUSA varlen attention path requires the MUSA "
            "flash-attn package."
        ) from error
    if require_mate and not flash_attn_musa.is_mate_available():
        raise RuntimeError(
            "Qwen3.5 text FlashAttention requires Mate/TileLang for "
            "head_dim=256 backward."
        )

    ALL_ATTENTION_FUNCTIONS.register(
        MUSA_VARLEN_ATTENTION, musa_varlen_flash_attention_forward
    )
    ALL_MASK_ATTENTION_FUNCTIONS.register(MUSA_VARLEN_ATTENTION, _musa_varlen_mask)


def _musa_varlen_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """Return only the 2D padding mask; causal masking is done by FlashAttention."""
    del batch_size, cache_position, kwargs
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError(
            f"MUSA varlen FlashAttention expects a 2D padding mask, got {attention_mask.ndim}D."
        )
    return attention_mask[:, -kv_length:].bool()


def musa_varlen_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Run Qwen attention through the installed MUSA Mate/TileLang varlen kernel."""
    del module
    if kwargs.get("output_attentions", False):
        logger.warning("MUSA varlen FlashAttention does not return attention weights.")
    if query.device.type != "musa" or key.device.type != "musa" or value.device.type != "musa":
        raise RuntimeError("MUSA varlen FlashAttention requires query, key, and value on a MUSA device.")
    head_dims = (query.shape[-1], key.shape[-1], value.shape[-1])
    if len(set(head_dims)) != 1 or head_dims[0] not in (64, 256):
        raise RuntimeError(
            "The validated Qwen3.5 MUSA varlen FlashAttention path requires "
            f"q/k/v head_dim 64 or 256, got {head_dims}."
        )
    if dropout != 0.0:
        raise RuntimeError(f"The validated Mate/TileLang path requires dropout=0.0, got {dropout}.")
    if sliding_window is not None:
        raise RuntimeError("The validated MUSA varlen FlashAttention path does not support sliding-window attention.")
    if softcap not in (None, 0.0):
        raise RuntimeError(f"The validated MUSA varlen FlashAttention path does not support softcap={softcap}.")

    try:
        from flash_attn import flash_attn_varlen_func
        from flash_attn.backends import musa as flash_attn_musa
        from flash_attn.bert_padding import pad_input, unpad_input
    except ImportError as error:
        raise ImportError(
            "Qwen3.5 MUSA attention requires the MUSA flash-attn package; "
            "head_dim=256 additionally requires Mate/TileLang."
        ) from error

    if head_dims[0] == 256 and not flash_attn_musa.is_mate_available():
        raise RuntimeError(
            "attn_implementation=flash requires Mate/TileLang for Qwen3.5 head_dim=256."
        )

    # Qwen gives [batch, heads, sequence, dim]; FlashAttention uses
    # [batch, sequence, heads, dim].
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    batch_size, query_length = query.shape[:2]
    key_length = key.shape[1]

    # Qwen3.5 vision already supplies packed varlen metadata.  Use it directly
    # so all images/windows share one kernel call instead of launching once per
    # split.  The vision module represents packed tokens as a synthetic batch 1.
    cu_seqlens_q = kwargs.get("cu_seq_lens_q")
    cu_seqlens_k = kwargs.get("cu_seq_lens_k")
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("Both cu_seq_lens_q and cu_seq_lens_k are required for packed attention.")
        if batch_size != 1:
            raise ValueError(f"Packed Qwen3.5 vision attention expects batch=1, got {batch_size}.")
        max_seqlen_q = kwargs.get("max_length_q")
        max_seqlen_k = kwargs.get("max_length_k")
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("Packed attention requires max_length_q and max_length_k.")
        output = flash_attn_varlen_func(
            query.squeeze(0),
            key.squeeze(0),
            value.squeeze(0),
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True if is_causal is None else is_causal,
            window_size=(-1, -1),
            softcap=0.0,
            deterministic=False,
        )
        return output.unsqueeze(0), None

    # Text attention reaches this path. Always unpad, even when every token is
    # valid, so backward cannot fall through to the broken dense d=256 kernel.
    if attention_mask is None:
        key_mask = torch.ones(
            batch_size,
            key_length,
            device=query.device,
            dtype=torch.bool,
        )
    else:
        if attention_mask.ndim != 2:
            raise ValueError(
                f"MUSA varlen FlashAttention expects a 2D padding mask, got {attention_mask.ndim}D."
            )
        key_mask = attention_mask[:, -key_length:].bool()

    query_mask = key_mask if query_length == key_length else key_mask[:, -query_length:]
    query_unpadded, query_indices, cu_seqlens_q, max_seqlen_q = unpad_input(query, query_mask)
    key_unpadded, _, cu_seqlens_k, max_seqlen_k = unpad_input(key, key_mask)
    value_unpadded, _, value_cu_seqlens_k, value_max_seqlen_k = unpad_input(value, key_mask)
    if not torch.equal(cu_seqlens_k, value_cu_seqlens_k) or max_seqlen_k != value_max_seqlen_k:
        raise RuntimeError("K/V unpadding produced inconsistent sequence metadata.")

    output_unpadded = flash_attn_varlen_func(
        query_unpadded,
        key_unpadded,
        value_unpadded,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True if is_causal is None else is_causal,
        window_size=(-1, -1),
        softcap=0.0,
        deterministic=False,
    )
    output = pad_input(output_unpadded, query_indices, batch_size, query_length)
    return output, None


def resolve_qwen35_attention_implementation(requested: str) -> str:
    """Register and resolve the user-facing ``flash`` alias for MUSA training."""
    if requested != FLASH_CONFIG_ALIAS:
        return requested

    _register_musa_varlen_attention(require_mate=True)

    # The custom key intentionally contains "flash" so Qwen3.5 vision forwards
    # its native cu_seqlens to the adapter. Transformers would otherwise try to
    # preload an upstream CUDA/HIP implementation while constructing the model.
    # Suppress only that preload check; Qwen's already-imported runtime helper
    # still recognizes this key as FlashAttention.
    from transformers import modeling_utils

    current_check = modeling_utils.is_flash_attention_requested
    if not getattr(current_check, "_starvla_musa_flash_wrapper", False):
        original_check = current_check

        def check_without_custom_preload(config=None, requested_attention_implementation=None):
            implementation = (
                config._attn_implementation
                if config is not None
                else requested_attention_implementation
            )
            if implementation == MUSA_VARLEN_ATTENTION:
                return False
            return original_check(
                config=config,
                requested_attention_implementation=requested_attention_implementation,
            )

        check_without_custom_preload._starvla_musa_flash_wrapper = True
        modeling_utils.is_flash_attention_requested = check_without_custom_preload
    logger.info(
        "Resolved attn_implementation=flash to %s (MUSA Mate/TileLang varlen path).",
        MUSA_VARLEN_ATTENTION,
    )
    return MUSA_VARLEN_ATTENTION


def install_qwen35_musa_vision_flash_attention(model: torch.nn.Module) -> int:
    """Route only Qwen3.5 vision attention through packed MUSA FlashAttention."""
    _register_musa_varlen_attention(require_mate=False)

    found = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3_5VisionAttention":
            continue
        found += 1
        if not hasattr(module, _VISION_ORIGINAL_ATTENTION_ATTR):
            setattr(
                module,
                _VISION_ORIGINAL_ATTENTION_ATTR,
                module.config._attn_implementation,
            )
        module.config._attn_implementation = MUSA_VARLEN_ATTENTION

    if found == 0:
        raise RuntimeError(
            "Vision FlashAttention was requested, but no "
            "Qwen3_5VisionAttention module was found."
        )
    logger.warning(
        "Enabled packed MUSA FlashAttention for %d Qwen3.5 vision attention "
        "module(s); text attention remains unchanged.",
        found,
    )
    return found


def disable_qwen35_musa_vision_flash_attention(model: torch.nn.Module) -> int:
    """Restore the attention implementation saved on each vision module."""
    restored = 0
    restored_configs: set[int] = set()
    for module in model.modules():
        original = getattr(module, _VISION_ORIGINAL_ATTENTION_ATTR, None)
        if original is None:
            continue
        config_id = id(module.config)
        if config_id not in restored_configs:
            module.config._attn_implementation = original
            restored_configs.add(config_id)
        delattr(module, _VISION_ORIGINAL_ATTENTION_ATTR)
        restored += 1
    return restored


def configure_qwen35_musa_vision_flash_attention(
    model: torch.nn.Module, qwenvl_config: Any
) -> int:
    """Apply the YAML-selected vision-only FlashAttention policy."""
    enabled = _config_bool(
        qwenvl_config.get("musa_vision_flash_attention", False),
        name="framework.qwenvl.musa_vision_flash_attention",
    )
    if not enabled:
        disable_qwen35_musa_vision_flash_attention(model)
        return 0
    if qwenvl_config.get("attn_implementation") != "eager":
        raise RuntimeError(
            "Vision-only MUSA FlashAttention requires the model-level "
            "framework.qwenvl.attn_implementation=eager baseline."
        )
    if not _musa_is_available():
        raise RuntimeError(
            "framework.qwenvl.musa_vision_flash_attention=true was requested, "
            "but MUSA is unavailable."
        )
    return install_qwen35_musa_vision_flash_attention(model)


__all__ = [
    "FLASH_CONFIG_ALIAS",
    "MUSA_VARLEN_ATTENTION",
    "configure_qwen35_musa_vision_flash_attention",
    "disable_qwen35_musa_vision_flash_attention",
    "install_qwen35_musa_vision_flash_attention",
    "musa_varlen_flash_attention_forward",
    "resolve_qwen35_attention_implementation",
]
