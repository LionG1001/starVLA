"""Numerical regression test for Qwen3.5 FlashAttention backward on MUSA.

The production Qwen3.5-4B attention boundary uses GQA with 16 query heads,
4 key/value heads, head dimension 256, and left-padded batches.  FlashAttention
represents that padding by unpadding the valid tokens and calling its varlen API.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention

from starVLA.model.modules.vlm.qwen35_musa_flash_attention import (
    MUSA_VARLEN_ATTENTION,
    musa_varlen_flash_attention_forward,
    resolve_qwen35_attention_implementation,
)

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.backends import musa as flash_attn_musa
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_musa = None


QUERY_HEADS = 16
KEY_VALUE_HEADS = 4
HEAD_DIM = 256
VALID_LENGTHS = (291, 286, 290, 290)
VISION_LENGTHS = (196, 196)
DTYPE = torch.bfloat16
SEED = 20260818


def _musa_flash_attn_is_available() -> bool:
    return (
        torch_musa is not None
        and hasattr(torch, "musa")
        and torch.musa.is_available()
        and flash_attn_varlen_func is not None
        and flash_attn_musa is not None
        and flash_attn_musa.is_mate_available()
    )


@dataclass(frozen=True)
class ErrorStats:
    finite: bool
    max_abs: float
    mean_abs: float
    relative_l2: float
    actual_abs_max: float
    reference_abs_max: float


def _error_stats(actual: torch.Tensor, reference: torch.Tensor) -> ErrorStats:
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    finite = bool(torch.isfinite(actual).all().item())
    if not finite:
        return ErrorStats(False, math.inf, math.inf, math.inf, math.inf, reference.abs().max().item())
    difference = actual - reference
    return ErrorStats(
        finite=True,
        max_abs=difference.abs().max().item(),
        mean_abs=difference.abs().mean().item(),
        relative_l2=(difference.norm() / reference.norm().clamp_min(1e-12)).item(),
        actual_abs_max=actual.abs().max().item(),
        reference_abs_max=reference.abs().max().item(),
    )


def _eager_varlen_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_gradient: torch.Tensor,
    valid_lengths: tuple[int, ...],
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run an explicit FP32 GQA reference, one unpadded sample at a time."""
    query = query.detach().float().requires_grad_(True)
    key = key.detach().float().requires_grad_(True)
    value = value.detach().float().requires_grad_(True)

    outputs = []
    offset = 0
    groups = query.shape[1] // key.shape[1]
    scale = query.shape[-1] ** -0.5
    for length in valid_lengths:
        sample_query = query[offset : offset + length].transpose(0, 1)
        sample_key = key[offset : offset + length].repeat_interleave(groups, dim=1).transpose(0, 1)
        sample_value = value[offset : offset + length].repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = torch.matmul(sample_query, sample_key.transpose(-2, -1)) * scale
        if causal:
            causal_mask = torch.ones(length, length, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probabilities, sample_value).transpose(0, 1))
        offset += length

    output = torch.cat(outputs, dim=0)
    output.backward(output_gradient.detach().float())
    return output.detach(), query.grad.detach(), key.grad.detach(), value.grad.detach()


@unittest.skipUnless(_musa_flash_attn_is_available(), "requires MUSA FlashAttention and Mate")
class Qwen35FlashAttentionMusaTest(unittest.TestCase):
    def test_real_shape_varlen_backward_matches_fp32_eager(self):
        total_tokens = sum(VALID_LENGTHS)
        generator = torch.Generator().manual_seed(SEED)

        # Generate on CPU so the reference and MUSA path consume bit-identical BF16 inputs.
        query_cpu = torch.randn(total_tokens, QUERY_HEADS, HEAD_DIM, generator=generator).to(DTYPE)
        key_cpu = torch.randn(total_tokens, KEY_VALUE_HEADS, HEAD_DIM, generator=generator).to(DTYPE)
        value_cpu = torch.randn(total_tokens, KEY_VALUE_HEADS, HEAD_DIM, generator=generator).to(DTYPE)
        output_gradient_cpu = (
            torch.randn(total_tokens, QUERY_HEADS, HEAD_DIM, generator=generator) * 0.01
        ).to(DTYPE)

        reference = _eager_varlen_reference(
            query_cpu,
            key_cpu,
            value_cpu,
            output_gradient_cpu,
            VALID_LENGTHS,
            True,
        )

        device = torch.device("musa:0")
        query = query_cpu.to(device).requires_grad_(True)
        key = key_cpu.to(device).requires_grad_(True)
        value = value_cpu.to(device).requires_grad_(True)
        output_gradient = output_gradient_cpu.to(device)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(VALID_LENGTHS).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        )

        mate_calls = []
        original_varlen_mate = flash_attn_musa.varlen_mate

        def observed_varlen_mate(*args, **kwargs):
            mate_calls.append(
                {
                    "q_shape": tuple(args[0].shape),
                    "k_shape": tuple(args[1].shape),
                    "causal": kwargs.get("causal", args[8] if len(args) > 8 else None),
                }
            )
            return original_varlen_mate(*args, **kwargs)

        flash_attn_musa.varlen_mate = observed_varlen_mate
        try:
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,
                max(VALID_LENGTHS),
                max(VALID_LENGTHS),
                dropout_p=0.0,
                softmax_scale=HEAD_DIM**-0.5,
                causal=True,
                deterministic=True,
            )
            output.backward(output_gradient)
            torch.musa.synchronize()
        finally:
            flash_attn_musa.varlen_mate = original_varlen_mate

        stats = {
            "output": _error_stats(output, reference[0]),
            "dq": _error_stats(query.grad, reference[1]),
            "dk": _error_stats(key.grad, reference[2]),
            "dv": _error_stats(value.grad, reference[3]),
        }
        print(
            {
                "backend": "mate_tilelang_varlen",
                "dtype": str(DTYPE),
                "valid_lengths": VALID_LENGTHS,
                "q_shape": tuple(query.shape),
                "k_shape": tuple(key.shape),
                "mate_calls": mate_calls,
                "stats": stats,
            }
        )

        self.assertEqual(len(mate_calls), 1, "the call did not dispatch to the Mate varlen backend")
        for name, tensor_stats in stats.items():
            self.assertTrue(tensor_stats.finite, f"{name} contains NaN or Inf")
            self.assertLessEqual(
                tensor_stats.relative_l2,
                0.05,
                f"{name} differs excessively from the FP32 eager reference",
            )

    def test_qwen35_vision_shape_varlen_backward_matches_fp32_eager(self):
        total_tokens = sum(VISION_LENGTHS)
        generator = torch.Generator().manual_seed(SEED + 1)
        tensors = [
            torch.randn(total_tokens, QUERY_HEADS, 64, generator=generator).to(DTYPE)
            for _ in range(3)
        ]
        output_gradient_cpu = (
            torch.randn(total_tokens, QUERY_HEADS, 64, generator=generator) * 0.01
        ).to(DTYPE)
        reference = _eager_varlen_reference(
            tensors[0], tensors[1], tensors[2], output_gradient_cpu, VISION_LENGTHS, False
        )

        query, key, value = [tensor.to("musa:0").requires_grad_(True) for tensor in tensors]
        output_gradient = output_gradient_cpu.to("musa:0")
        cu_seqlens = torch.tensor([0, 196, 392], device="musa:0", dtype=torch.int32)
        mate_calls = []
        original_varlen_mate = flash_attn_musa.varlen_mate

        def observed_varlen_mate(*args, **kwargs):
            mate_calls.append(tuple(args[0].shape))
            return original_varlen_mate(*args, **kwargs)

        flash_attn_musa.varlen_mate = observed_varlen_mate
        try:
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,
                196,
                196,
                dropout_p=0.0,
                softmax_scale=64**-0.5,
                causal=False,
                deterministic=True,
            )
            output.backward(output_gradient)
            torch.musa.synchronize()
        finally:
            flash_attn_musa.varlen_mate = original_varlen_mate

        stats = {
            "output": _error_stats(output, reference[0]),
            "dq": _error_stats(query.grad, reference[1]),
            "dk": _error_stats(key.grad, reference[2]),
            "dv": _error_stats(value.grad, reference[3]),
        }
        print({"backend": "musa_extension_varlen", "head_dim": 64, "stats": stats})
        self.assertEqual(mate_calls, [], "head_dim=64 should use the MUSA extension, not Mate")
        for name, tensor_stats in stats.items():
            self.assertTrue(tensor_stats.finite, f"vision {name} contains NaN or Inf")
            self.assertLessEqual(tensor_stats.relative_l2, 0.05, f"vision {name} error is excessive")

    def test_qwen35_vision_dispatch_preserves_packed_cu_seqlens(self):
        implementation = resolve_qwen35_attention_implementation("flash")
        config = Qwen3_5VisionConfig(hidden_size=1024, num_heads=16, depth=1)
        config._attn_implementation = implementation
        attention = Qwen3_5VisionAttention(config).to(device="musa:0", dtype=DTYPE).train()
        hidden_states = torch.randn(392, 1024, device="musa:0", dtype=DTYPE, requires_grad=True)
        cu_seqlens = torch.tensor([0, 196, 392], device="musa:0", dtype=torch.int32)
        position_embeddings = (
            torch.ones(392, 64, device="musa:0", dtype=DTYPE),
            torch.zeros(392, 64, device="musa:0", dtype=DTYPE),
        )
        calls = []
        original_interface = ALL_ATTENTION_FUNCTIONS[MUSA_VARLEN_ATTENTION]

        def observed_interface(*args, **kwargs):
            calls.append(
                {
                    "cu_q": kwargs.get("cu_seq_lens_q"),
                    "q_shape": tuple(args[1].shape),
                }
            )
            return musa_varlen_flash_attention_forward(*args, **kwargs)

        ALL_ATTENTION_FUNCTIONS.register(MUSA_VARLEN_ATTENTION, observed_interface)
        try:
            output = attention(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            output.float().square().mean().backward()
            torch.musa.synchronize()
        finally:
            ALL_ATTENTION_FUNCTIONS.register(MUSA_VARLEN_ATTENTION, original_interface)

        print(
            {
                "vision_resolved_attention": config._attn_implementation,
                "interface_calls": [
                    {"q_shape": call["q_shape"], "cu_q": call["cu_q"].tolist()}
                    for call in calls
                ],
                "output_finite": bool(torch.isfinite(output).all().item()),
            }
        )
        self.assertEqual(len(calls), 1, "vision attention should make one packed varlen call")
        self.assertTrue(torch.equal(calls[0]["cu_q"], cu_seqlens))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.isfinite(hidden_states.grad).all().item())

    def test_qwen35_dispatches_flash_alias_to_mate_varlen_backward(self):
        implementation = resolve_qwen35_attention_implementation("flash")
        self.assertEqual(implementation, MUSA_VARLEN_ATTENTION)

        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=QUERY_HEADS,
            num_key_value_heads=KEY_VALUE_HEADS,
            head_dim=HEAD_DIM,
            layer_types=["full_attention"],
            attention_dropout=0.0,
            use_cache=False,
        )
        config._attn_implementation = implementation
        model = Qwen3_5TextModel(config).to(device="musa:0", dtype=DTYPE).train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device="musa:0")
        attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]],
            device="musa:0",
            dtype=torch.long,
        )

        mate_calls = []
        original_varlen_mate = flash_attn_musa.varlen_mate

        def observed_varlen_mate(*args, **kwargs):
            mate_calls.append((tuple(args[0].shape), tuple(args[1].shape)))
            return original_varlen_mate(*args, **kwargs)

        flash_attn_musa.varlen_mate = observed_varlen_mate
        try:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state
            output.float().square().mean().backward()
            torch.musa.synchronize()
        finally:
            flash_attn_musa.varlen_mate = original_varlen_mate

        print(
            {
                "requested_attention": "flash",
                "resolved_attention": model.config._attn_implementation,
                "mate_calls": mate_calls,
                "output_finite": bool(torch.isfinite(output).all().item()),
            }
        )
        self.assertEqual(model.config._attn_implementation, MUSA_VARLEN_ATTENTION)
        self.assertEqual(len(mate_calls), 1)
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.isfinite(model.embed_tokens.weight.grad).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
