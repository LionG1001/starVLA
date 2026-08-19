"""MUSA fast path for Qwen3.5's non-overlapping vision patch projection."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_ORIGINAL_FORWARD_ATTR = "_starvla_qwen35_patch_embed_conv3d_forward"


def _setting(value: Any, *, name: str) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int) and value in (0, 1):
        return "on" if value else "off"
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be one of auto/1/0/true/false/on/off, but got {value!r}."
        )
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return "on"
    if normalized in _FALSE_VALUES:
        return "off"
    if normalized == "auto":
        return normalized
    raise ValueError(
        f"{name} must be one of auto/1/0/true/false/on/off, but got {value!r}."
    )


def _musa_is_available() -> bool:
    return bool(hasattr(torch, "musa") and torch.musa.is_available())


def _patch_geometry(module: nn.Module) -> tuple[int, int, int, int]:
    temporal_patch_size = int(module.temporal_patch_size)
    patch_size = int(module.patch_size)
    in_channels = int(module.in_channels)
    embed_dim = int(module.embed_dim)
    return temporal_patch_size, patch_size, in_channels, embed_dim


def _supports_linear_projection(module: nn.Module) -> bool:
    proj = getattr(module, "proj", None)
    if not isinstance(proj, nn.Conv3d):
        return False

    temporal_patch_size, patch_size, in_channels, embed_dim = _patch_geometry(
        module
    )
    kernel = (temporal_patch_size, patch_size, patch_size)
    return bool(
        proj.in_channels == in_channels
        and proj.out_channels == embed_dim
        and tuple(proj.kernel_size) == kernel
        and tuple(proj.stride) == kernel
        and tuple(proj.padding) == (0, 0, 0)
        and tuple(proj.dilation) == (1, 1, 1)
        and proj.groups == 1
        and tuple(proj.weight.shape) == (embed_dim, in_channels, *kernel)
    )


def qwen35_vision_patch_linear_forward(
    module: nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Evaluate the one-patch Conv3D as its exactly equivalent linear map.

    Transformers reshapes every input row into one complete spatiotemporal
    patch. Because Conv3D's kernel and stride cover that entire patch, each
    output has spatial size 1x1x1 and no windows overlap. Flattening the patch
    and Conv3D weight therefore gives the same affine projection while letting
    MUSA use its substantially faster GEMM kernels in forward and backward.
    """
    if not _supports_linear_projection(module):
        original_forward = getattr(module, _ORIGINAL_FORWARD_ATTR, None)
        if original_forward is None:
            raise RuntimeError(
                "Qwen3.5 vision patch linear fast path has no Conv3D fallback."
            )
        return original_forward(hidden_states)

    temporal_patch_size, patch_size, in_channels, embed_dim = _patch_geometry(
        module
    )
    patch_volume = in_channels * temporal_patch_size * patch_size * patch_size
    target_dtype = module.proj.weight.dtype
    flat_patches = hidden_states.view(-1, patch_volume).to(dtype=target_dtype)
    flat_weight = module.proj.weight.view(embed_dim, patch_volume)
    return F.linear(flat_patches, flat_weight, module.proj.bias)


def _installed_forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return qwen35_vision_patch_linear_forward(module, hidden_states)


def install_qwen35_musa_vision_patch_fastpath(model: nn.Module) -> int:
    """Replace eligible Qwen3.5 PatchEmbed forwards without replacing weights."""
    patched = 0
    found = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3_5VisionPatchEmbed":
            continue
        found += 1
        if hasattr(module, _ORIGINAL_FORWARD_ATTR):
            patched += 1
            continue
        if not _supports_linear_projection(module):
            logger.warning(
                "Keeping Qwen3.5 vision PatchEmbed on Conv3D because its "
                "geometry is not an eligible one-patch projection."
            )
            continue

        setattr(module, _ORIGINAL_FORWARD_ATTR, module.forward)
        module.forward = MethodType(_installed_forward, module)
        patched += 1

    if found == 0:
        raise RuntimeError(
            "Vision patch fast path was requested, but no "
            "Qwen3_5VisionPatchEmbed module was found."
        )
    if patched == 0:
        raise RuntimeError(
            "Vision patch fast path was requested, but every Qwen3.5 "
            "PatchEmbed module had unsupported Conv3D geometry."
        )

    logger.warning(
        "Enabled the experimental StarVLA MUSA linear projection fast path "
        "for %d Qwen3.5 vision PatchEmbed module(s).",
        patched,
    )
    return patched


def disable_qwen35_musa_vision_patch_fastpath(model: nn.Module) -> int:
    """Restore every PatchEmbed instance to its original Conv3D forward."""
    restored = 0
    for module in model.modules():
        original_forward = getattr(module, _ORIGINAL_FORWARD_ATTR, None)
        if original_forward is None:
            continue
        module.forward = original_forward
        delattr(module, _ORIGINAL_FORWARD_ATTR)
        restored += 1
    return restored


def configure_qwen35_musa_vision_patch_path(
    model: nn.Module, qwenvl_config: Any
) -> int:
    """Apply the YAML-selected Qwen3.5 vision PatchEmbed policy on MUSA."""
    setting = _setting(
        qwenvl_config.get("musa_vision_patch_linear_fastpath", False),
        name="framework.qwenvl.musa_vision_patch_linear_fastpath",
    )
    if not _musa_is_available():
        if setting == "on":
            raise RuntimeError(
                "framework.qwenvl.musa_vision_patch_linear_fastpath=true was "
                "requested, but MUSA is unavailable."
            )
        return 0
    if setting == "on":
        return install_qwen35_musa_vision_patch_fastpath(model)
    if setting == "off":
        disable_qwen35_musa_vision_patch_fastpath(model)
    return 0


__all__ = [
    "configure_qwen35_musa_vision_patch_path",
    "disable_qwen35_musa_vision_patch_fastpath",
    "install_qwen35_musa_vision_patch_fastpath",
    "qwen35_vision_patch_linear_forward",
]
