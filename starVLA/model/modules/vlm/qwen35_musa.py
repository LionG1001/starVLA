"""Narrow MUSA compatibility and optional fast-path hooks for Qwen3.5."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _setting(value: Any, *, name: str) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int) and value in (0, 1):
        return "on" if value else "off"
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be one of auto/1/0/true/false/on/off, but got {value!r}."
        )
    value = value.strip().lower()
    if value in _TRUE_VALUES:
        return "on"
    if value in _FALSE_VALUES:
        return "off"
    if value == "auto":
        return value
    raise ValueError(
        f"{name} must be one of auto/1/0/true/false/on/off, but got {value!r}."
    )


def _musa_is_available() -> bool:
    return bool(
        hasattr(torch, "musa")
        and torch.musa.is_available()
    )


def _fla_causal_conv_adapter(fla_causal_conv1d):
    """Adapt Transformers' [B, D, T] causal-conv API to FLA's [B, T, D]."""

    def causal_conv1d_fn(
        x,
        weight,
        bias=None,
        activation=None,
        seq_idx=None,
        **kwargs,
    ):
        if seq_idx is not None:
            raise NotImplementedError(
                "The StarVLA FLA training adapter does not support seq_idx."
            )
        output, _ = fla_causal_conv1d(
            x=x.transpose(1, 2).contiguous(),
            weight=weight,
            bias=bias,
            activation=activation,
            output_final_state=False,
            backend="triton",
            **kwargs,
        )
        # Qwen3_5GatedDeltaNet immediately transposes this [B, D, T] result
        # back to [B, T, D] before splitting Q/K/V. Keep the intermediate as
        # a view: materializing it here copies roughly 19 MiB per layer for
        # the bs=4, T=291, D=8192 training shape, only to transpose it back.
        return output.transpose(1, 2)

    return causal_conv1d_fn


def install_qwen35_musa_fla_training_fastpath(model: nn.Module) -> int:
    """Enable FLA kernels for Qwen3.5 linear-attention training blocks.

    The hook replaces the sequence-mode causal convolution, gated delta rule,
    and gated RMSNorm. It deliberately leaves the recurrent cache/update path
    unchanged, so generation remains on the Transformers implementation.
    """
    try:
        from fla.modules import FusedRMSNormGated
        from fla.modules.convolution import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_recurrent_gated_delta_rule,
        )
    except ImportError as exc:
        raise RuntimeError(
            "framework.qwenvl.musa_fla_fastpath=true requires fla-core and "
            "flash-linear-attention. Install the validated wheels without "
            "replacing the MUSA torch/Triton stack."
        ) from exc

    causal_conv_adapter = _fla_causal_conv_adapter(causal_conv1d)
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3_5GatedDeltaNet":
            continue
        if getattr(module, "_starvla_musa_fla_fastpath", False):
            patched += 1
            continue

        old_norm = module.norm
        fused_norm = FusedRMSNormGated(
            module.head_v_dim,
            eps=module.layer_norm_epsilon,
            activation=module.activation,
            device=old_norm.weight.device,
            dtype=old_norm.weight.dtype,
        )
        with torch.no_grad():
            fused_norm.weight.copy_(old_norm.weight)

        module.causal_conv1d_fn = causal_conv_adapter
        module.chunk_gated_delta_rule = chunk_gated_delta_rule
        # StarVLA only accelerates full-sequence training here. Keep generation
        # on the Transformers reference recurrent implementation.
        module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        module.norm = fused_norm
        module._starvla_musa_fla_fastpath = True
        patched += 1

    if patched == 0:
        raise RuntimeError(
            "FLA was requested, but no Qwen3_5GatedDeltaNet modules were found."
        )

    logger.warning(
        "Enabled the experimental StarVLA MUSA FLA training fast path for "
        "%d Qwen3.5 linear-attention layers. Generation keeps the "
        "Transformers recurrent path.",
        patched,
    )
    return patched


def disable_qwen35_musa_fla_fastpath(model: nn.Module) -> int:
    """Force the Transformers reference path for a trustworthy MUSA baseline.

    Transformers 5.2.0 automatically binds FLA functions when the ``fla``
    package is importable. Merely skipping StarVLA's installer therefore does
    not disable FLA. Rebinding every GatedDeltaNet block makes the environment
    switch effective and keeps baseline/fast-path A/B runs comparable.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5RMSNormGated,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    rebound = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3_5GatedDeltaNet":
            continue

        old_norm = module.norm
        reference_norm = Qwen3_5RMSNormGated(
            module.head_v_dim,
            eps=module.layer_norm_epsilon,
        ).to(device=old_norm.weight.device, dtype=old_norm.weight.dtype)
        with torch.no_grad():
            reference_norm.weight.copy_(old_norm.weight)

        module.causal_conv1d_fn = None
        module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        module.norm = reference_norm
        module._starvla_musa_fla_fastpath = False
        rebound += 1

    if rebound:
        logger.info(
            "Forced the Transformers reference GatedDeltaNet path for %d "
            "Qwen3.5 linear-attention layers.",
            rebound,
        )
    return rebound


def configure_qwen35_musa_fla_path(model: nn.Module, qwenvl_config: Any) -> int:
    """Apply the requested Qwen3.5 FLA policy on MUSA.

    ``0`` forces the reference path, ``1`` requires StarVLA's validated FLA
    adapter, and ``auto`` leaves Transformers' import-time selection intact.
    CUDA and CPU behavior is intentionally left to Transformers.
    """
    setting = _setting(
        qwenvl_config.get("musa_fla_fastpath", False),
        name="framework.qwenvl.musa_fla_fastpath",
    )
    if not _musa_is_available():
        if setting == "on":
            raise RuntimeError(
                "framework.qwenvl.musa_fla_fastpath=true was requested, "
                "but MUSA is unavailable."
            )
        return 0
    if setting == "on":
        return install_qwen35_musa_fla_training_fastpath(model)
    if setting == "off":
        disable_qwen35_musa_fla_fastpath(model)
        return 0
    return 0
