"""Real-shape MUSA A/B for Qwen3.5 FLA causal-conv layout copies."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch_musa  # noqa: F401
from fla.modules.convolution import causal_conv1d


def _leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    difference = actual.float() - expected.float()
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "all_finite": bool(torch.isfinite(actual).all()),
    }


def _run(
    base_input: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    materialize_adapter_output: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    qwen_input = _leaf(base_input)
    weight = _leaf(base_weight)
    bias = _leaf(base_bias)
    output, _ = causal_conv1d(
        x=qwen_input.transpose(1, 2).contiguous(),
        weight=weight,
        bias=bias,
        activation="silu",
        output_final_state=False,
        backend="triton",
    )
    adapter_output = output.transpose(1, 2)
    if materialize_adapter_output:
        adapter_output = adapter_output.contiguous()

    # Qwen3_5GatedDeltaNet performs this transpose immediately after the
    # adapter, before splitting the last dimension into Q/K/V.
    consumed_output = adapter_output.transpose(1, 2)
    consumed_output.backward(grad_output)
    return consumed_output.detach(), (
        qwen_input.grad.detach(),
        weight.grad.detach(),
        bias.grad.detach(),
    )


def _measure(base_inputs, grad_output, *, materialize, warmup, repeats):
    for _ in range(warmup):
        _run(*base_inputs, grad_output, materialize_adapter_output=materialize)
    torch.musa.synchronize()

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        _run(*base_inputs, grad_output, materialize_adapter_output=materialize)
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
    parser.add_argument("--channels", type=int, default=8192)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if min(vars(args).values()) < 1:
        raise ValueError("shape, warmup, repeats, and seed must be positive")
    if not torch.musa.is_available():
        raise RuntimeError("This benchmark requires a MUSA device")

    torch.manual_seed(args.seed)
    torch.musa.set_device(0)
    dtype = torch.bfloat16
    source = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.channels,
        device="musa",
        dtype=dtype,
    )
    qwen_input = source.transpose(1, 2)
    weight = torch.randn(
        args.channels,
        args.kernel_size,
        device="musa",
        dtype=dtype,
    ) * 0.02
    bias = torch.randn(args.channels, device="musa", dtype=dtype) * 0.02
    grad_output = torch.randn_like(source) * 0.01
    base_inputs = (qwen_input, weight, bias)

    reference_output, reference_grads = _run(
        *base_inputs,
        grad_output,
        materialize_adapter_output=True,
    )
    candidate_output, candidate_grads = _run(
        *base_inputs,
        grad_output,
        materialize_adapter_output=False,
    )
    torch.musa.synchronize()

    reference_timing = _measure(
        base_inputs,
        grad_output,
        materialize=True,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    candidate_timing = _measure(
        base_inputs,
        grad_output,
        materialize=False,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    result = {
        "qwen_input_shape": list(qwen_input.shape),
        "fla_input_shape": list(source.shape),
        "dtype": str(dtype),
        "copied_bytes_per_layer": source.numel() * source.element_size(),
        "output": _error_stats(candidate_output, reference_output),
        "gradients": {
            name: _error_stats(actual, expected)
            for name, actual, expected in zip(
                ("input", "weight", "bias"),
                candidate_grads,
                reference_grads,
                strict=True,
            )
        },
        "materialized_fwd_bwd": reference_timing,
        "view_fwd_bwd": candidate_timing,
        "speedup_ratio": (
            reference_timing["median_ms"] / candidate_timing["median_ms"]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
