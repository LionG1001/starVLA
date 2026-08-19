"""Model FLOPs helpers used by StarVLA's per-device MFU reporting.

The estimates in this module count useful model matrix FLOPs. They exclude
optimizer work, communication, elementwise operations, and activation
recomputation introduced by gradient checkpointing. That is the conventional
MFU numerator; executed/recomputed FLOPs would instead be an HFU-style metric.
"""

from __future__ import annotations

from dataclasses import dataclass

QWEN35_MFU_FORMULA_VERSION = "qwen3_5_v2"


@dataclass(frozen=True)
class Qwen35ModelFlopConfig:
    """Static Qwen3.5 component sizes needed by the FLOPs estimator."""

    text_layer_parameters: int
    full_attention_layers: int
    linear_attention_layers: int
    text_attention_heads: int
    text_head_dim: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    vision_block_parameters: int
    vision_patch_parameters: int
    vision_merger_parameters: int
    vision_layers: int
    vision_attention_heads: int
    vision_head_dim: int
    vision_spatial_merge_size: int
    action_parameters: int
    lm_head_parameters: int


@dataclass(frozen=True)
class Qwen35BatchFlopShape:
    """Dynamic per-device shapes for one Qwen3.5 training micro-batch."""

    batch_size: int
    padded_language_sequence_length: int
    vision_patch_tokens: int
    num_images: int
    action_tokens_per_sample: int
    logits_tokens_per_sample: int = 1


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def calculate_per_device_mfu(
    estimated_tflops_per_device_step: float,
    model_time_seconds: float,
    peak_tflops_per_device: float,
) -> dict[str, float]:
    """Convert useful per-device step FLOPs and time into throughput and MFU."""

    if estimated_tflops_per_device_step < 0:
        raise ValueError("estimated_tflops_per_device_step must be non-negative")
    if model_time_seconds <= 0:
        raise ValueError("model_time_seconds must be positive")
    if peak_tflops_per_device <= 0:
        raise ValueError("peak_tflops_per_device must be positive")

    achieved_tflops = estimated_tflops_per_device_step / model_time_seconds
    return {
        "achieved_tflops_per_device": achieved_tflops,
        "mfu_percent": achieved_tflops / peak_tflops_per_device * 100,
    }


def estimate_qwen35_training_flops(
    model: Qwen35ModelFlopConfig,
    batch: Qwen35BatchFlopShape,
) -> dict[str, float | int | str]:
    """Estimate useful per-device training FLOPs for Qwen3.5 QwenOFT.

    A trainable matrix parameter contributes approximately six FLOPs per
    token: two in the forward GEMM and four across activation/weight backward.
    Dense attention matrix products and the parameter-free Gated DeltaNet
    state updates are added separately. QwenOFT requests one LM-head logit
    position and does not use those logits in its action loss, so that branch
    contributes forward FLOPs only.

    The vision attention term assumes images in a micro-batch use the same
    patch grid. StarVLA currently resizes every RoboTwin observation to the
    configured fixed image size, so this is exact for the supported run.
    """

    for name, value in vars(model).items():
        _require_non_negative(name, value)
    for name, value in vars(batch).items():
        _require_non_negative(name, value)
    if batch.batch_size == 0:
        raise ValueError("batch_size must be positive")
    if batch.padded_language_sequence_length == 0:
        raise ValueError("padded_language_sequence_length must be positive")
    if model.vision_spatial_merge_size == 0:
        raise ValueError("vision_spatial_merge_size must be positive")
    if batch.vision_patch_tokens and batch.num_images == 0:
        raise ValueError("num_images must be positive when vision patches are present")

    language_tokens = batch.batch_size * batch.padded_language_sequence_length
    action_tokens = batch.batch_size * batch.action_tokens_per_sample
    logits_tokens = batch.batch_size * batch.logits_tokens_per_sample

    merge_unit = model.vision_spatial_merge_size**2
    if batch.vision_patch_tokens % merge_unit:
        raise ValueError(
            "vision_patch_tokens must be divisible by vision_spatial_merge_size squared"
        )
    vision_merged_tokens = batch.vision_patch_tokens // merge_unit

    # Forward + activation backward + weight backward for trainable matrices.
    text_weight_flops = 6 * model.text_layer_parameters * language_tokens
    action_weight_flops = 6 * model.action_parameters * action_tokens
    vision_block_weight_flops = 6 * model.vision_block_parameters * batch.vision_patch_tokens
    vision_patch_weight_flops = 6 * model.vision_patch_parameters * batch.vision_patch_tokens
    vision_merger_weight_flops = 6 * model.vision_merger_parameters * vision_merged_tokens

    # QK^T and attention-value products, including their backward passes.
    text_attention_flops = (
        12
        * model.full_attention_layers
        * batch.batch_size
        * model.text_attention_heads
        * model.text_head_dim
        * batch.padded_language_sequence_length**2
    )
    # Gated DeltaNet keeps a [key_dim, value_dim] state for every value head.
    # A useful forward token performs roughly one state decay, two state reads,
    # and one outer-product update: 7 * K * V FLOPs per head. Multiplying by
    # three approximates forward + activation backward + weight/input backward,
    # matching the conventional 6 * parameters * tokens training estimate.
    linear_attention_core_flops = (
        21
        * model.linear_attention_layers
        * language_tokens
        * model.linear_num_value_heads
        * model.linear_key_head_dim
        * model.linear_value_head_dim
    )
    if batch.num_images:
        patches_per_image = batch.vision_patch_tokens / batch.num_images
        vision_attention_token_squares = batch.num_images * patches_per_image**2
    else:
        vision_attention_token_squares = 0.0
    vision_attention_flops = (
        12
        * model.vision_layers
        * model.vision_attention_heads
        * model.vision_head_dim
        * vision_attention_token_squares
    )

    # The logits branch is executed for logits_to_keep positions but is not an
    # ancestor of action_loss, so no LM-head backward GEMM runs.
    lm_head_forward_flops = 2 * model.lm_head_parameters * logits_tokens

    breakdown = {
        "text_weight_flops": text_weight_flops,
        "text_attention_flops": text_attention_flops,
        "linear_attention_core_flops": linear_attention_core_flops,
        "vision_block_weight_flops": vision_block_weight_flops,
        "vision_attention_flops": vision_attention_flops,
        "vision_patch_weight_flops": vision_patch_weight_flops,
        "vision_merger_weight_flops": vision_merger_weight_flops,
        "action_weight_flops": action_weight_flops,
        "lm_head_forward_flops": lm_head_forward_flops,
    }
    text_flops = (
        text_weight_flops
        + text_attention_flops
        + linear_attention_core_flops
        + lm_head_forward_flops
    )
    vision_flops = (
        vision_block_weight_flops
        + vision_attention_flops
        + vision_patch_weight_flops
        + vision_merger_weight_flops
    )
    action_flops = action_weight_flops
    total_flops = float(sum(breakdown.values()))
    return {
        "formula_version": QWEN35_MFU_FORMULA_VERSION,
        "total_flops": total_flops,
        "estimated_tflops_per_device_step": total_flops / 1e12,
        "estimated_text_tflops_per_device_step": text_flops / 1e12,
        "estimated_vision_tflops_per_device_step": vision_flops / 1e12,
        "estimated_action_tflops_per_device_step": action_flops / 1e12,
        "language_tokens_per_device": language_tokens,
        "vision_patch_tokens_per_device": batch.vision_patch_tokens,
        "vision_merged_tokens_per_device": int(vision_merged_tokens),
        "action_tokens_per_device": action_tokens,
        **breakdown,
    }
