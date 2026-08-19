"""Real-shape numerical and timing A/B for Qwen3.5 FLA on MUSA.

This benchmark isolates the 24 linear-attention layers' gated-delta-rule.
It compares Transformers' reference implementation with FLA using the shape
observed in the bs=4 StarVLA training trace.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch_musa  # noqa: F401
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
)


def _leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = actual_float - expected_float
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
        ),
        "all_finite": bool(torch.isfinite(actual_float).all()),
    }


def _run(
    implementation,
    base_inputs: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    inputs = [_leaf(value) for value in base_inputs]
    output, _ = implementation(
        *inputs,
        use_qk_l2norm_in_kernel=True,
    )
    output.backward(grad_output)
    return output.detach(), [value.grad.detach() for value in inputs]


def _measure(
    implementation,
    base_inputs: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        _run(implementation, base_inputs, grad_output)
    torch.musa.synchronize()

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        _run(implementation, base_inputs, grad_output)
        torch.musa.synchronize()
        samples.append((time.perf_counter() - started) * 1_000)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=291)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    if min(
        args.batch_size,
        args.sequence_length,
        args.heads,
        args.head_dim,
        args.warmup,
        args.repeats,
    ) < 1:
        raise ValueError("shape, warmup, and repeats must all be positive")
    if not torch.musa.is_available():
        raise RuntimeError("This benchmark requires a MUSA device")

    torch.manual_seed(args.seed)
    torch.musa.set_device(0)
    shape = (
        args.batch_size,
        args.sequence_length,
        args.heads,
        args.head_dim,
    )
    state_shape = shape[:-1]
    query = torch.randn(shape, device="musa", dtype=torch.bfloat16) * 0.1
    key = torch.randn(shape, device="musa", dtype=torch.bfloat16) * 0.1
    value = torch.randn(shape, device="musa", dtype=torch.bfloat16) * 0.1
    decay = -torch.rand(state_shape, device="musa", dtype=torch.float32) * 0.1
    beta = torch.rand(state_shape, device="musa", dtype=torch.bfloat16)
    grad_output = torch.randn_like(value) * 0.01
    base_inputs = (query, key, value, decay, beta)

    reference_output, reference_grads = _run(
        torch_chunk_gated_delta_rule,
        base_inputs,
        grad_output,
    )
    fla_output, fla_grads = _run(
        chunk_gated_delta_rule,
        base_inputs,
        grad_output,
    )
    torch.musa.synchronize()

    reference_timing = _measure(
        torch_chunk_gated_delta_rule,
        base_inputs,
        grad_output,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    fla_timing = _measure(
        chunk_gated_delta_rule,
        base_inputs,
        grad_output,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    result = {
        "shape": list(shape),
        "dtype": str(query.dtype),
        "output": _error_stats(fla_output, reference_output),
        "gradients": {
            name: _error_stats(actual, expected)
            for name, actual, expected in zip(
                ("query", "key", "value", "decay", "beta"),
                fla_grads,
                reference_grads,
                strict=True,
            )
        },
        "reference_fwd_bwd": reference_timing,
        "fla_fwd_bwd": fla_timing,
        "speedup_ratio": (
            reference_timing["median_ms"] / fla_timing["median_ms"]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
