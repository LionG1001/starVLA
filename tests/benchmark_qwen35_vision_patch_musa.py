"""Numerical and timing gate for the real Qwen3.5 PatchEmbed MUSA shape."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchEmbed

from starVLA.model.modules.vlm.qwen35_musa_vision_patch import (
    install_qwen35_musa_vision_patch_fastpath,
)


def _sync() -> None:
    torch.musa.synchronize()


def _step(module, hidden_states, grad_output):
    module.zero_grad(set_to_none=True)
    if hidden_states.grad is not None:
        hidden_states.grad = None
    output = module(hidden_states)
    output.backward(grad_output)
    return output


def _measure(module, hidden_states, grad_output, warmup: int, iterations: int):
    for _ in range(warmup):
        _step(module, hidden_states, grad_output)
    _sync()

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        _step(module, hidden_states, grad_output)
        _sync()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-relative-l2", type=float, default=1.0e-3)
    parser.add_argument("--max-normalized-abs", type=float, default=1.0e-2)
    parser.add_argument("--min-speedup", type=float, default=1.1)
    args = parser.parse_args()

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("This benchmark requires an available MUSA device.")

    torch.manual_seed(23)
    device = torch.device("musa")
    vision_config = Qwen3_5VisionConfig(
        hidden_size=1024,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
    )
    reference = Qwen3_5VisionPatchEmbed(vision_config).to(
        device=device, dtype=torch.bfloat16
    )
    fast = copy.deepcopy(reference)
    install_qwen35_musa_vision_patch_fastpath(fast)
    patch_volume = 3 * 2 * 16 * 16
    base_input = torch.randn(
        args.patches, patch_volume, device=device, dtype=torch.bfloat16
    )
    grad_output = torch.randn(
        args.patches, 1024, device=device, dtype=torch.bfloat16
    )

    reference_input = base_input.detach().clone().requires_grad_(True)
    fast_input = base_input.detach().clone().requires_grad_(True)
    reference_output = _step(reference, reference_input, grad_output)
    fast_output = _step(fast, fast_input, grad_output)
    _sync()

    comparisons = {}
    tensors = {
        "output": (fast_output, reference_output),
        "input_grad": (fast_input.grad, reference_input.grad),
        "weight_grad": (fast.proj.weight.grad, reference.proj.weight.grad),
        "bias_grad": (fast.proj.bias.grad, reference.proj.bias.grad),
    }
    for name, (actual, expected) in tensors.items():
        diff = (actual.float() - expected.float()).abs()
        expected_float = expected.float()
        reference_abs_max = expected_float.abs().max().item()
        reference_l2 = torch.linalg.vector_norm(expected_float).item()
        comparisons[name] = {
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
            "max_abs_over_reference_max": diff.max().item()
            / max(reference_abs_max, 1.0e-12),
            "relative_l2": torch.linalg.vector_norm(diff).item()
            / max(reference_l2, 1.0e-12),
            "reference_abs_max": reference_abs_max,
            "finite": bool(torch.isfinite(actual).all().item()),
        }

    # Real training does not differentiate through input images. Keep the
    # input-gradient comparison above as a stronger semantic gate, but time
    # the actual weight/bias-gradient workload seen by PatchEmbed.
    reference_timing_input = base_input.detach().clone()
    fast_timing_input = base_input.detach().clone()
    reference_timing = _measure(
        reference,
        reference_timing_input,
        grad_output,
        args.warmup,
        args.iterations,
    )
    fast_timing = _measure(
        fast,
        fast_timing_input,
        grad_output,
        args.warmup,
        args.iterations,
    )
    speedup = reference_timing["median_ms"] / fast_timing["median_ms"]
    failures = []
    for name, comparison in comparisons.items():
        if not comparison["finite"]:
            failures.append(f"{name} contains non-finite values")
        if comparison["relative_l2"] > args.max_relative_l2:
            failures.append(
                f"{name} relative_l2={comparison['relative_l2']:.6g} exceeds "
                f"{args.max_relative_l2:.6g}"
            )
        if (
            comparison["max_abs_over_reference_max"]
            > args.max_normalized_abs
        ):
            failures.append(
                f"{name} normalized max error="
                f"{comparison['max_abs_over_reference_max']:.6g} exceeds "
                f"{args.max_normalized_abs:.6g}"
            )
    if speedup < args.min_speedup:
        failures.append(
            f"speedup={speedup:.3f} is below {args.min_speedup:.3f}"
        )

    result = {
        "shape": {
            "patches": args.patches,
            "input": [args.patches, patch_volume],
            "weight": [1024, 3, 2, 16, 16],
            "dtype": "bfloat16",
            "timed_input_requires_grad": False,
        },
        "comparisons": comparisons,
        "conv3d": reference_timing,
        "linear": fast_timing,
        "speedup": speedup,
        "thresholds": {
            "max_relative_l2": args.max_relative_l2,
            "max_normalized_abs": args.max_normalized_abs,
            "min_speedup": args.min_speedup,
        },
        "passed": not failures,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
