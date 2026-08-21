"""Optional torch-musa fused operator patches for Transformers Qwen3.5.

The patches are deliberately installed after ``from_pretrained`` so the
installed Transformers package and pretrained checkpoint schema stay intact.
Every operator has an eager fallback, and a runtime kernel failure permanently
disables only that fused path for the current process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MethodType
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

logger = logging.getLogger(__name__)

_ROPE_PHASE_ATTR = "_starvla_qwen35_musa_rope_phase"
_SWIGLU_COMBINED_PROJ_ATTR = "gate_up_proj"
_SWIGLU_STATE_HOOKS_ATTR = "_starvla_musa_swiglu_state_hooks"

_RMSNORM_DISABLED = False
_RMSNORM_LOGGED = False
_GATED_RMSNORM_DISABLED = False
_GATED_RMSNORM_LOGGED = False
_ROPE_DISABLED = False
_ROPE_LOGGED = False
_SWIGLU_DISABLED = False
_SWIGLU_LOGGED = False


@dataclass(frozen=True)
class Qwen35MusaFusedOpsStatus:
    rmsnorm_modules: int = 0
    gated_rmsnorm_modules: int = 0
    rotary_embedding_modules: int = 0
    rope_function_patched: bool = False
    swiglu_modules: int = 0


def _reset_runtime_state_for_tests() -> None:
    """Reset process-wide circuit breakers between isolated unit tests."""
    global _RMSNORM_DISABLED, _RMSNORM_LOGGED
    global _GATED_RMSNORM_DISABLED, _GATED_RMSNORM_LOGGED
    global _ROPE_DISABLED, _ROPE_LOGGED
    global _SWIGLU_DISABLED, _SWIGLU_LOGGED

    _RMSNORM_DISABLED = False
    _RMSNORM_LOGGED = False
    _GATED_RMSNORM_DISABLED = False
    _GATED_RMSNORM_LOGGED = False
    _ROPE_DISABLED = False
    _ROPE_LOGGED = False
    _SWIGLU_DISABLED = False
    _SWIGLU_LOGGED = False


def _config_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value, but got {value!r}.")


def _musa_is_available() -> bool:
    return bool(hasattr(torch, "musa") and torch.musa.is_available())


def _is_supported_transformers_version() -> bool:
    import transformers

    return Version(transformers.__version__).release[:2] == (5, 2)


def _raise_if_oom(error: RuntimeError) -> None:
    message = str(error).lower()
    if "out of memory" in message or ("memory" in message and "alloc" in message):
        raise error


def qwen35_rms_norm_eager(module, hidden_states: torch.Tensor) -> torch.Tensor:
    output = hidden_states.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + module.eps)
    output = output * (1.0 + module.weight.float())
    return output.type_as(hidden_states)


def qwen35_rms_norm_musa(module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Use fused RMS reduction while retaining Qwen3.5's ``1 + weight`` rule."""
    global _RMSNORM_DISABLED, _RMSNORM_LOGGED

    fast_path = (
        not _RMSNORM_DISABLED
        and hidden_states.device.type == "musa"
        and hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and hidden_states.shape[-1] == module.weight.numel()
        and hasattr(F, "rms_norm")
    )
    if not fast_path:
        return qwen35_rms_norm_eager(module, hidden_states)

    try:
        # torch-musa requires input and weight to have the same dtype. Passing
        # no weight lets muDNN fuse the expensive reduction while preserving
        # Qwen3.5's FP32 (1 + weight) scale outside the kernel.
        output = F.rms_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            weight=None,
            eps=module.eps,
        )
        output = output.float() * (1.0 + module.weight.float())
        output = output.type_as(hidden_states)
        if not _RMSNORM_LOGGED:
            logger.warning("Qwen3.5 MUSA fused RMSNorm fast path is active.")
            _RMSNORM_LOGGED = True
        return output
    except RuntimeError as error:
        _raise_if_oom(error)
        _RMSNORM_DISABLED = True
        logger.warning(
            "Qwen3.5 MUSA fused RMSNorm failed once and is now disabled; using eager fallback: %s",
            error,
        )
        return qwen35_rms_norm_eager(module, hidden_states)


def qwen35_gated_rms_norm_eager(
    module,
    hidden_states: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    output = hidden_states.float()
    variance = output.pow(2).mean(-1, keepdim=True)
    output = output * torch.rsqrt(variance + module.variance_epsilon)
    output = module.weight * output.to(input_dtype)
    output = output * F.silu(gate.float())
    return output.to(input_dtype)


def qwen35_gated_rms_norm_musa(
    module,
    hidden_states: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse the RMS reduction in Qwen3.5's gated linear-attention norm."""
    global _GATED_RMSNORM_DISABLED, _GATED_RMSNORM_LOGGED

    fast_path = (
        not _GATED_RMSNORM_DISABLED
        and gate is not None
        and hidden_states.device.type == "musa"
        and gate.device == hidden_states.device
        and hidden_states.dtype == module.weight.dtype
        and hidden_states.shape == gate.shape
        and hidden_states.shape[-1] == module.weight.numel()
        and hasattr(F, "rms_norm")
    )
    if not fast_path:
        return qwen35_gated_rms_norm_eager(module, hidden_states, gate)

    try:
        output = F.rms_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            weight=module.weight,
            eps=module.variance_epsilon,
        )
        output = (output * F.silu(gate.float())).to(hidden_states.dtype)
        if not _GATED_RMSNORM_LOGGED:
            logger.warning("Qwen3.5 MUSA fused gated RMSNorm fast path is active.")
            _GATED_RMSNORM_LOGGED = True
        return output
    except RuntimeError as error:
        _raise_if_oom(error)
        _GATED_RMSNORM_DISABLED = True
        logger.warning(
            "Qwen3.5 MUSA fused gated RMSNorm failed once and is now disabled; using eager fallback: %s",
            error,
        )
        return qwen35_gated_rms_norm_eager(module, hidden_states, gate)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def qwen35_rotary_pos_emb_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_broadcast = cos.unsqueeze(unsqueeze_dim)
    sin_broadcast = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_embed = (query_rotary * cos_broadcast) + (_rotate_half(query_rotary) * sin_broadcast)
    key_embed = (key_rotary * cos_broadcast) + (_rotate_half(key_rotary) * sin_broadcast)
    return (
        torch.cat((query_embed, query_pass), dim=-1),
        torch.cat((key_embed, key_pass), dim=-1),
    )


def _qwen35_rope_one_musa(
    hidden_states: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, sequence_length, _ = hidden_states.shape
    rotary_dim = phase.shape[-1]
    rotary = hidden_states[..., :rotary_dim]
    passthrough = hidden_states[..., rotary_dim:]

    # muDNN RoPE accepts one [sequence, rotary_dim] phase shared over its
    # batch. Flattening B x S into the sequence axis preserves each sample's
    # MRoPE phase without launching one kernel per sample.
    rope_input = rotary.transpose(1, 2).reshape(
        batch_size * sequence_length,
        1,
        num_heads,
        rotary_dim,
    )
    rope_output = torch.rope(
        rope_input,
        phase.reshape(batch_size * sequence_length, rotary_dim),
        rotary_interleaved=False,
        batch_first=False,
        multi_latent_attention=False,
    )
    rope_output = rope_output.reshape(
        batch_size,
        sequence_length,
        num_heads,
        rotary_dim,
    ).transpose(1, 2)
    return torch.cat((rope_output, passthrough), dim=-1)


def qwen35_rotary_pos_emb_musa(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fused partial MRoPE to batch-specific Qwen3.5 phases."""
    global _ROPE_DISABLED, _ROPE_LOGGED

    phase = getattr(cos, _ROPE_PHASE_ATTR, None)
    fast_path = (
        not _ROPE_DISABLED
        and hasattr(torch, "rope")
        and query.device.type == "musa"
        and key.device == query.device
        and cos.device == query.device
        and sin.device == query.device
        and query.ndim == key.ndim == 4
        and phase is not None
        and phase.device == query.device
        and phase.ndim == 3
        and phase.dtype == torch.float32
        and unsqueeze_dim == 1
        and query.shape[0] == key.shape[0] == phase.shape[0]
        and query.shape[2] == key.shape[2] == phase.shape[1]
        and phase.shape[-1] == cos.shape[-1] == sin.shape[-1]
        and phase.shape[-1] <= query.shape[-1]
        and query.shape[-1] == key.shape[-1]
        and phase.shape[-1] % 2 == 0
    )
    if not fast_path:
        return qwen35_rotary_pos_emb_eager(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)

    try:
        query_embed = _qwen35_rope_one_musa(query, phase)
        key_embed = _qwen35_rope_one_musa(key, phase)
        if not _ROPE_LOGGED:
            logger.warning("Qwen3.5 MUSA fused partial MRoPE fast path is active.")
            _ROPE_LOGGED = True
        return query_embed.to(query.dtype), key_embed.to(key.dtype)
    except RuntimeError as error:
        _raise_if_oom(error)
        _ROPE_DISABLED = True
        logger.warning(
            "Qwen3.5 MUSA fused partial MRoPE failed once and is now disabled; using eager fallback: %s",
            error,
        )
        return qwen35_rotary_pos_emb_eager(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)


def _qwen35_rotary_embedding_forward(module, hidden_states, position_ids):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    inv_freq_expanded = module.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()

    device_type = (
        hidden_states.device.type
        if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
        else "cpu"
    )
    from transformers.models.qwen3_5 import modeling_qwen3_5

    with modeling_qwen3_5.maybe_autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = module.apply_interleaved_mrope(freqs, module.mrope_section)
        phase = torch.cat((freqs, freqs), dim=-1)
        cos = phase.cos() * module.attention_scaling
        sin = phase.sin() * module.attention_scaling

    cos = cos.to(dtype=hidden_states.dtype)
    sin = sin.to(dtype=hidden_states.dtype)
    if (
        hidden_states.device.type == "musa"
        and isinstance(module.attention_scaling, (int, float))
        and float(module.attention_scaling) == 1.0
    ):
        setattr(cos, _ROPE_PHASE_ATTR, phase.contiguous())
    return cos, sin


def qwen35_swiglu_eager(module, hidden_states: torch.Tensor) -> torch.Tensor:
    combined_projection = getattr(module, _SWIGLU_COMBINED_PROJ_ATTR, None)
    if combined_projection is not None:
        gate, up = combined_projection(hidden_states).chunk(2, dim=-1)
        return module.down_proj(module.act_fn(gate) * up)
    return module.down_proj(module.act_fn(module.gate_proj(hidden_states)) * module.up_proj(hidden_states))


class _Qwen35CombinedGateUpProjection(nn.Module):
    """One trainable projection whose output layout is ``[gate | up]``."""

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear) -> None:
        super().__init__()
        if gate_proj.bias is not None or up_proj.bias is not None:
            raise RuntimeError("Qwen3.5 combined gate/up projection requires bias=False.")
        if gate_proj.weight.shape != up_proj.weight.shape:
            raise RuntimeError("Qwen3.5 gate_proj and up_proj weights must have identical shapes.")
        if gate_proj.weight.device != up_proj.weight.device:
            raise RuntimeError("Qwen3.5 gate_proj and up_proj weights must be on the same device.")
        if gate_proj.weight.dtype != up_proj.weight.dtype:
            raise RuntimeError("Qwen3.5 gate_proj and up_proj weights must have the same dtype.")
        if gate_proj.weight.requires_grad != up_proj.weight.requires_grad:
            raise RuntimeError("Qwen3.5 gate_proj and up_proj must share requires_grad state.")

        self.in_features = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features
        combined_weight = torch.cat((gate_proj.weight.detach(), up_proj.weight.detach()), dim=0)
        self.weight = nn.Parameter(
            combined_weight,
            requires_grad=gate_proj.weight.requires_grad,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, bias=None)


def _qwen35_swiglu_state_dict_post_hook(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: Mapping[str, Any],
) -> None:
    """Expose the original Transformers gate/up keys in model checkpoints."""
    del local_metadata
    combined_key = f"{prefix}{_SWIGLU_COMBINED_PROJ_ATTR}.weight"
    combined_weight = state_dict.pop(combined_key, None)
    if combined_weight is None:
        return
    intermediate_size = getattr(module, _SWIGLU_COMBINED_PROJ_ATTR).intermediate_size
    # Independent storage keeps both torch.save and safetensors compatible;
    # exporting a pair of views into one combined parameter is rejected by
    # safetensors' shared-storage check.
    state_dict[f"{prefix}gate_proj.weight"] = combined_weight[:intermediate_size].clone()
    state_dict[f"{prefix}up_proj.weight"] = combined_weight[intermediate_size:].clone()


def _qwen35_swiglu_load_state_dict_pre_hook(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: Mapping[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Pack original Transformers gate/up checkpoint tensors exactly once."""
    del local_metadata, strict, missing_keys, unexpected_keys
    combined_key = f"{prefix}{_SWIGLU_COMBINED_PROJ_ATTR}.weight"
    gate_key = f"{prefix}gate_proj.weight"
    up_key = f"{prefix}up_proj.weight"
    gate_weight = state_dict.get(gate_key)
    up_weight = state_dict.get(up_key)
    if gate_weight is None and up_weight is None:
        return
    if gate_weight is None or up_weight is None:
        error_msgs.append(f"Both {gate_key!r} and {up_key!r} are required by the combined Qwen3.5 SwiGLU projection.")
        return
    if combined_key in state_dict:
        error_msgs.append(f"Checkpoint contains both original gate/up weights and {combined_key!r}.")
        return
    if gate_weight.shape != up_weight.shape:
        error_msgs.append(f"Checkpoint gate/up shapes differ: {tuple(gate_weight.shape)} vs {tuple(up_weight.shape)}.")
        return
    state_dict[combined_key] = torch.cat((gate_weight, up_weight), dim=0)
    state_dict.pop(gate_key)
    state_dict.pop(up_key)


def _install_qwen35_combined_swiglu_projection(module: nn.Module) -> None:
    """Replace two Qwen3.5 MLP projections with one producer-side projection.

    This must run before the optimizer and distributed wrappers are created.
    Model state_dicts remain compatible with the original Transformers keys,
    while optimizer checkpoints intentionally follow the new one-parameter
    layout.
    """
    if getattr(module, _SWIGLU_COMBINED_PROJ_ATTR, None) is not None:
        return
    gate_proj = getattr(module, "gate_proj", None)
    up_proj = getattr(module, "up_proj", None)
    if not isinstance(gate_proj, nn.Linear) or not isinstance(up_proj, nn.Linear):
        raise RuntimeError("Qwen3.5 fused SwiGLU requires nn.Linear gate_proj and up_proj modules.")

    combined_projection = _Qwen35CombinedGateUpProjection(gate_proj, up_proj)
    del module.gate_proj
    del module.up_proj
    module.add_module(_SWIGLU_COMBINED_PROJ_ATTR, combined_projection)
    state_hook = module.register_state_dict_post_hook(_qwen35_swiglu_state_dict_post_hook)
    load_hook = module.register_load_state_dict_pre_hook(_qwen35_swiglu_load_state_dict_pre_hook)
    setattr(module, _SWIGLU_STATE_HOOKS_ATTR, (state_hook, load_hook))


def qwen35_swiglu_musa(module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run one combined projection followed by torch-musa fused SwiGLU."""
    global _SWIGLU_DISABLED, _SWIGLU_LOGGED

    combined_projection = getattr(module, _SWIGLU_COMBINED_PROJ_ATTR, None)
    if combined_projection is None:
        return qwen35_swiglu_eager(module, hidden_states)

    gate_up = combined_projection(hidden_states)
    fast_path = (
        not _SWIGLU_DISABLED
        and hidden_states.device.type == "musa"
        and getattr(module.config, "hidden_act", None) in {"silu", "swish"}
        and hasattr(F, "swish_glu")
        and gate_up.is_contiguous()
    )
    if not fast_path:
        gate, up = gate_up.chunk(2, dim=-1)
        return module.down_proj(module.act_fn(gate) * up)

    try:
        output = module.down_proj(F.swish_glu(gate_up))
        if not _SWIGLU_LOGGED:
            logger.warning("Qwen3.5 MUSA combined-projection fused SwiGLU path is active.")
            _SWIGLU_LOGGED = True
        return output
    except RuntimeError as error:
        _raise_if_oom(error)
        _SWIGLU_DISABLED = True
        logger.warning(
            "Qwen3.5 MUSA fused SwiGLU failed once and is now disabled; "
            "using combined-projection eager activation fallback: %s",
            error,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        return module.down_proj(module.act_fn(gate) * up)


def configure_qwen35_musa_fused_ops(
    model: torch.nn.Module,
    qwenvl_config: Any,
) -> Qwen35MusaFusedOpsStatus:
    """Install enabled Qwen3.5 fused-op candidates and return patch counts."""
    rmsnorm_enabled = _config_bool(
        qwenvl_config.get("musa_fused_rmsnorm", False),
        name="framework.qwenvl.musa_fused_rmsnorm",
    )
    gated_rmsnorm_enabled = _config_bool(
        qwenvl_config.get("musa_fused_gated_rmsnorm", False),
        name="framework.qwenvl.musa_fused_gated_rmsnorm",
    )
    rope_enabled = _config_bool(
        qwenvl_config.get("musa_fused_rope", False),
        name="framework.qwenvl.musa_fused_rope",
    )
    swiglu_enabled = _config_bool(
        qwenvl_config.get("musa_fused_swiglu", False),
        name="framework.qwenvl.musa_fused_swiglu",
    )
    if not any((rmsnorm_enabled, gated_rmsnorm_enabled, rope_enabled, swiglu_enabled)):
        return Qwen35MusaFusedOpsStatus()

    if not _musa_is_available():
        raise RuntimeError("Qwen3.5 MUSA fused ops were enabled, but MUSA is unavailable.")
    if not _is_supported_transformers_version():
        raise RuntimeError("Qwen3.5 MUSA fused-op patches currently support Transformers 5.2.x only.")

    from transformers.models.qwen3_5 import modeling_qwen3_5

    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3_5":
        raise RuntimeError("Qwen3.5 fused ops were enabled for a non-Qwen3.5 model.")

    rmsnorm_modules = 0
    gated_rmsnorm_modules = 0
    rotary_embedding_modules = 0
    swiglu_modules = 0

    for module in model.modules():
        if rmsnorm_enabled and type(module) is modeling_qwen3_5.Qwen3_5RMSNorm:
            module.forward = MethodType(qwen35_rms_norm_musa, module)
            rmsnorm_modules += 1
        elif gated_rmsnorm_enabled and type(module) is modeling_qwen3_5.Qwen3_5RMSNormGated:
            module.forward = MethodType(qwen35_gated_rms_norm_musa, module)
            gated_rmsnorm_modules += 1

        if rope_enabled and type(module) is modeling_qwen3_5.Qwen3_5TextRotaryEmbedding:
            forward = torch.no_grad()(modeling_qwen3_5.dynamic_rope_update(_qwen35_rotary_embedding_forward))
            module.forward = MethodType(forward, module)
            rotary_embedding_modules += 1

        if swiglu_enabled and type(module) is modeling_qwen3_5.Qwen3_5MLP:
            _install_qwen35_combined_swiglu_projection(module)
            module.forward = MethodType(qwen35_swiglu_musa, module)
            swiglu_modules += 1

    rope_function_patched = False
    if rope_enabled and rotary_embedding_modules:
        modeling_qwen3_5.apply_rotary_pos_emb = qwen35_rotary_pos_emb_musa
        rope_function_patched = True

    missing_patches = []
    if rmsnorm_enabled and not rmsnorm_modules:
        missing_patches.append("RMSNorm")
    if rope_enabled and not rotary_embedding_modules:
        missing_patches.append("RoPE")
    if swiglu_enabled and not swiglu_modules:
        missing_patches.append("SwiGLU")
    if missing_patches:
        raise RuntimeError("Enabled Qwen3.5 fused ops did not match any target modules: " + ", ".join(missing_patches))
    if gated_rmsnorm_enabled and not gated_rmsnorm_modules:
        logger.warning(
            "Qwen3.5 fused gated RMSNorm was enabled but no exact modules remained; "
            "an earlier FLA patch may already own this path."
        )

    status = Qwen35MusaFusedOpsStatus(
        rmsnorm_modules=rmsnorm_modules,
        gated_rmsnorm_modules=gated_rmsnorm_modules,
        rotary_embedding_modules=rotary_embedding_modules,
        rope_function_patched=rope_function_patched,
        swiglu_modules=swiglu_modules,
    )
    logger.warning("Installed Qwen3.5 MUSA fused-op candidates: %s", status)
    return status


__all__ = [
    "Qwen35MusaFusedOpsStatus",
    "configure_qwen35_musa_fused_ops",
    "qwen35_gated_rms_norm_eager",
    "qwen35_gated_rms_norm_musa",
    "qwen35_rms_norm_eager",
    "qwen35_rms_norm_musa",
    "qwen35_rotary_pos_emb_eager",
    "qwen35_rotary_pos_emb_musa",
    "qwen35_swiglu_eager",
    "qwen35_swiglu_musa",
]
