"""Real packed-shape benchmark for Qwen3.5 vision-only MUSA FlashAttention."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention

from starVLA.model.modules.vlm.qwen35_musa_flash_attention import (
    MUSA_VARLEN_ATTENTION,
    install_qwen35_musa_vision_flash_attention,
)


def _sync() -> None:
    torch.musa.synchronize()


def _stats(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    actual = actual.detach().float()
    reference = reference.detach().float()
    difference = actual - reference
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "relative_l2": float(
            (difference.norm() / reference.norm().clamp_min(1e-12)).item()
        ),
    }


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _measure_pair(eager, flash, *, warmup: int, repeats: int):
    for _ in range(warmup):
        eager()
        flash()
    _sync()
    samples = {"eager": [], "flash": []}
    for repeat in range(repeats):
        order = (("eager", eager), ("flash", flash))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, function in order:
            start = time.perf_counter()
            function()
            _sync()
            samples[name].append((time.perf_counter() - start) * 1_000)
    eager_stats = _summary(samples["eager"])
    flash_stats = _summary(samples["flash"])
    return {
        "eager": eager_stats,
        "flash": flash_stats,
        "speedup_ratio": eager_stats["median_ms"] / flash_stats["median_ms"],
        "improvement_percent": (
            1.0 - flash_stats["median_ms"] / eager_stats["median_ms"]
        )
        * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-sequences", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("This benchmark requires MUSA.")

    torch.manual_seed(args.seed)
    torch.musa.set_device(0)
    device = torch.device("musa:0")
    dtype = torch.bfloat16
    total_tokens = args.packed_sequences * args.sequence_length

    eager_config = Qwen3_5VisionConfig(hidden_size=1024, num_heads=16, depth=1)
    eager_config._attn_implementation = "eager"
    flash_config = Qwen3_5VisionConfig(hidden_size=1024, num_heads=16, depth=1)
    flash_config._attn_implementation = "eager"
    eager_module = (
        Qwen3_5VisionAttention(eager_config).to(device=device, dtype=dtype).train()
    )
    flash_module = (
        Qwen3_5VisionAttention(flash_config).to(device=device, dtype=dtype).train()
    )
    flash_module.load_state_dict(eager_module.state_dict())
    install_qwen35_musa_vision_flash_attention(flash_module)

    hidden_cpu = torch.randn(total_tokens, 1024).to(dtype)
    phase_cpu = torch.randn(total_tokens, 64)
    cos = phase_cpu.cos().to(dtype).to(device)
    sin = phase_cpu.sin().to(dtype).to(device)
    position_embeddings = (cos, sin)
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        args.sequence_length,
        device=device,
        dtype=torch.int32,
    )
    grad_output = (torch.randn(total_tokens, 1024) * 0.01).to(dtype).to(device)

    def run_backward(module, hidden):
        module.zero_grad(set_to_none=True)
        hidden.grad = None
        output = module(
            hidden,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        output.backward(grad_output)
        return output

    eager_correctness_input = hidden_cpu.to(device).requires_grad_(True)
    flash_correctness_input = hidden_cpu.to(device).requires_grad_(True)
    eager_output = run_backward(eager_module, eager_correctness_input)

    from flash_attn.backends import musa as flash_attn_musa

    mate_calls = []
    original_varlen_mate = flash_attn_musa.varlen_mate

    def observed_varlen_mate(*call_args, **call_kwargs):
        mate_calls.append(tuple(call_args[0].shape))
        return original_varlen_mate(*call_args, **call_kwargs)

    flash_attn_musa.varlen_mate = observed_varlen_mate
    try:
        flash_output = run_backward(flash_module, flash_correctness_input)
        _sync()
    finally:
        flash_attn_musa.varlen_mate = original_varlen_mate

    correctness = {
        "output": _stats(flash_output, eager_output),
        "input_grad": _stats(
            flash_correctness_input.grad, eager_correctness_input.grad
        ),
        "qkv_weight_grad": _stats(
            flash_module.qkv.weight.grad, eager_module.qkv.weight.grad
        ),
        "proj_weight_grad": _stats(
            flash_module.proj.weight.grad, eager_module.proj.weight.grad
        ),
    }
    for name, values in correctness.items():
        if not values["finite"] or values["relative_l2"] > 0.05:
            raise RuntimeError(f"{name} failed the numerical gate: {values}")
    if mate_calls:
        raise RuntimeError(f"Vision head_dim=64 unexpectedly dispatched to Mate: {mate_calls}")

    eager_forward_input = hidden_cpu.to(device)
    flash_forward_input = hidden_cpu.to(device)

    def eager_forward():
        with torch.no_grad():
            eager_module(
                eager_forward_input,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

    def flash_forward():
        with torch.no_grad():
            flash_module(
                flash_forward_input,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

    eager_backward_input = hidden_cpu.to(device).requires_grad_(True)
    flash_backward_input = hidden_cpu.to(device).requires_grad_(True)
    timing = {
        "forward": _measure_pair(
            eager_forward,
            flash_forward,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "forward_backward": _measure_pair(
            lambda: run_backward(eager_module, eager_backward_input),
            lambda: run_backward(flash_module, flash_backward_input),
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }
    print(
        json.dumps(
            {
                "device": torch.musa.get_device_name(0),
                "torch": torch.__version__,
                "dtype": str(dtype),
                "packed_sequences": args.packed_sequences,
                "sequence_length": args.sequence_length,
                "total_tokens": total_tokens,
                "heads": 16,
                "head_dim": 64,
                "candidate_implementation": flash_module.config._attn_implementation,
                "expected_implementation": MUSA_VARLEN_ATTENTION,
                "mate_calls": mate_calls,
                "correctness": correctness,
                "timing": timing,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
