"""Real-shape microbenchmark for StarVLA Qwen3.5 MUSA fused-op candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.vlm import qwen35_musa_fused_ops as fused_ops


class RMSNormState:
    def __init__(self, hidden_size: int, *, gated: bool = False):
        initial = torch.ones if gated else torch.zeros
        self.weight = nn.Parameter(initial(hidden_size, device="musa", dtype=torch.bfloat16))
        if gated:
            self.variance_epsilon = 1e-6
        else:
            self.eps = 1e-6


class MLPState(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu")
        factory_kwargs = {"device": "musa", "dtype": torch.bfloat16}
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, **factory_kwargs)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, **factory_kwargs)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, **factory_kwargs)
        self.act_fn = F.silu


def _sync() -> None:
    torch.musa.synchronize()


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _measure_pair(
    eager_callable,
    candidate_callable,
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, float], dict[str, float]]:
    for _ in range(warmup):
        eager_callable()
        candidate_callable()
    _sync()
    samples = {"eager": [], "candidate": []}
    for repeat in range(repeats):
        order = (
            (("eager", eager_callable), ("candidate", candidate_callable))
            if repeat % 2 == 0
            else (("candidate", candidate_callable), ("eager", eager_callable))
        )
        for name, callable_ in order:
            start = time.perf_counter()
            callable_()
            _sync()
            samples[name].append((time.perf_counter() - start) * 1_000)
    return _summarize(samples["eager"]), _summarize(samples["candidate"])


def _compare(eager, candidate) -> dict[str, float | bool]:
    difference = (candidate.float() - eager.float()).abs()
    denominator = eager.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(eager.float()).clamp_min(1e-6)
        ),
        "all_finite": bool(torch.isfinite(candidate).all()),
        "max_relative": float((difference / denominator).max()),
    }


def _speedup(eager: dict[str, float], candidate: dict[str, float]) -> float:
    return (eager["median_ms"] / candidate["median_ms"] - 1.0) * 100.0


def _timing_result(eager: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    return {
        "eager": eager,
        "candidate": candidate,
        "speedup_percent": _speedup(eager, candidate),
    }


def benchmark_standard_rmsnorm(warmup: int, repeats: int) -> dict:
    module = RMSNormState(2560)
    hidden_states = torch.randn(4, 291, 2560, device="musa", dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn_like(hidden_states)

    with torch.no_grad():
        eager_output = fused_ops.qwen35_rms_norm_eager(module, hidden_states)
        candidate_output = fused_ops.qwen35_rms_norm_musa(module, hidden_states)

    eager_forward, candidate_forward = _measure_pair(
        lambda: fused_ops.qwen35_rms_norm_eager(module, hidden_states),
        lambda: fused_ops.qwen35_rms_norm_musa(module, hidden_states),
        warmup=warmup,
        repeats=repeats,
    )

    def eager_step():
        hidden_states.grad = None
        module.weight.grad = None
        fused_ops.qwen35_rms_norm_eager(module, hidden_states).backward(grad_output)

    def candidate_step():
        hidden_states.grad = None
        module.weight.grad = None
        fused_ops.qwen35_rms_norm_musa(module, hidden_states).backward(grad_output)

    eager_fwd_bwd, candidate_fwd_bwd = _measure_pair(eager_step, candidate_step, warmup=warmup, repeats=repeats)
    return {
        "shape": list(hidden_states.shape),
        "correctness": _compare(eager_output, candidate_output),
        "fast_path_observed": fused_ops._RMSNORM_LOGGED,
        "forward": _timing_result(eager_forward, candidate_forward),
        "forward_backward": _timing_result(eager_fwd_bwd, candidate_fwd_bwd),
    }


def benchmark_gated_rmsnorm(warmup: int, repeats: int) -> dict:
    module = RMSNormState(128, gated=True)
    hidden_states = torch.randn(
        4 * 291 * 32,
        128,
        device="musa",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate = torch.randn_like(hidden_states, requires_grad=True)
    grad_output = torch.randn_like(hidden_states)
    with torch.no_grad():
        eager_output = fused_ops.qwen35_gated_rms_norm_eager(module, hidden_states, gate)
        candidate_output = fused_ops.qwen35_gated_rms_norm_musa(module, hidden_states, gate)

    eager_forward, candidate_forward = _measure_pair(
        lambda: fused_ops.qwen35_gated_rms_norm_eager(module, hidden_states, gate),
        lambda: fused_ops.qwen35_gated_rms_norm_musa(module, hidden_states, gate),
        warmup=warmup,
        repeats=repeats,
    )

    def eager_step():
        hidden_states.grad = gate.grad = module.weight.grad = None
        fused_ops.qwen35_gated_rms_norm_eager(module, hidden_states, gate).backward(grad_output)

    def candidate_step():
        hidden_states.grad = gate.grad = module.weight.grad = None
        fused_ops.qwen35_gated_rms_norm_musa(module, hidden_states, gate).backward(grad_output)

    eager_fwd_bwd, candidate_fwd_bwd = _measure_pair(eager_step, candidate_step, warmup=warmup, repeats=repeats)
    return {
        "shape": list(hidden_states.shape),
        "correctness": _compare(eager_output, candidate_output),
        "fast_path_observed": fused_ops._GATED_RMSNORM_LOGGED,
        "forward": _timing_result(eager_forward, candidate_forward),
        "forward_backward": _timing_result(eager_fwd_bwd, candidate_fwd_bwd),
    }


def benchmark_rope(warmup: int, repeats: int) -> dict:
    batch_size, sequence_length, head_dim, rotary_dim = 4, 291, 256, 64
    query_base = torch.randn(
        batch_size,
        sequence_length,
        16,
        head_dim,
        device="musa",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key_base = torch.randn(
        batch_size,
        sequence_length,
        4,
        head_dim,
        device="musa",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    query = query_base.transpose(1, 2)
    key = key_base.transpose(1, 2)
    phase = torch.randn(batch_size, sequence_length, rotary_dim, device="musa", dtype=torch.float32)
    cos = phase.cos().to(torch.bfloat16)
    sin = phase.sin().to(torch.bfloat16)
    setattr(cos, fused_ops._ROPE_PHASE_ATTR, phase)
    query_grad = torch.randn_like(query)
    key_grad = torch.randn_like(key)
    with torch.no_grad():
        eager_output = fused_ops.qwen35_rotary_pos_emb_eager(query, key, cos, sin)
        candidate_output = fused_ops.qwen35_rotary_pos_emb_musa(query, key, cos, sin)

    eager_forward, candidate_forward = _measure_pair(
        lambda: fused_ops.qwen35_rotary_pos_emb_eager(query, key, cos, sin),
        lambda: fused_ops.qwen35_rotary_pos_emb_musa(query, key, cos, sin),
        warmup=warmup,
        repeats=repeats,
    )

    def eager_step():
        query_base.grad = key_base.grad = None
        outputs = fused_ops.qwen35_rotary_pos_emb_eager(query, key, cos, sin)
        torch.autograd.backward(outputs, (query_grad, key_grad))

    def candidate_step():
        query_base.grad = key_base.grad = None
        outputs = fused_ops.qwen35_rotary_pos_emb_musa(query, key, cos, sin)
        torch.autograd.backward(outputs, (query_grad, key_grad))

    eager_fwd_bwd, candidate_fwd_bwd = _measure_pair(eager_step, candidate_step, warmup=warmup, repeats=repeats)
    return {
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "query_stride": list(query.stride()),
        "key_stride": list(key.stride()),
        "inputs_contiguous": query.is_contiguous() or key.is_contiguous(),
        "rotary_dim": rotary_dim,
        "query_correctness": _compare(eager_output[0], candidate_output[0]),
        "key_correctness": _compare(eager_output[1], candidate_output[1]),
        "fast_path_observed": fused_ops._ROPE_LOGGED,
        "forward": _timing_result(eager_forward, candidate_forward),
        "forward_backward": _timing_result(eager_fwd_bwd, candidate_fwd_bwd),
    }


def benchmark_swiglu_mlp(warmup: int, repeats: int) -> dict:
    shape = (4, 291, 2560)
    hidden_size, intermediate_size = 2560, 9216
    eager_module = MLPState(hidden_size, intermediate_size)
    candidate_module = MLPState(hidden_size, intermediate_size)
    candidate_module.load_state_dict(eager_module.state_dict())
    fused_ops._install_qwen35_combined_swiglu_projection(candidate_module)

    eager_input = torch.randn(shape, device="musa", dtype=torch.bfloat16, requires_grad=True)
    candidate_input = eager_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(eager_input)

    def eager_function():
        return fused_ops.qwen35_swiglu_eager(eager_module, eager_input)

    def candidate_function():
        return fused_ops.qwen35_swiglu_musa(candidate_module, candidate_input)

    with torch.no_grad():
        eager_output = eager_function()
        candidate_output = candidate_function()

    eager_forward, candidate_forward = _measure_pair(
        eager_function,
        candidate_function,
        warmup=warmup,
        repeats=repeats,
    )

    def eager_step():
        eager_input.grad = None
        eager_module.zero_grad(set_to_none=True)
        eager_function().backward(grad_output)

    def candidate_step():
        candidate_input.grad = None
        candidate_module.zero_grad(set_to_none=True)
        candidate_function().backward(grad_output)

    eager_fwd_bwd, candidate_fwd_bwd = _measure_pair(eager_step, candidate_step, warmup=warmup, repeats=repeats)

    eager_step()
    candidate_step()
    combined_grad = candidate_module.gate_up_proj.weight.grad
    return {
        "input_shape": list(shape),
        "combined_projection_shape": [2 * intermediate_size, hidden_size],
        "correctness": _compare(eager_output, candidate_output),
        "input_gradient_correctness": _compare(eager_input.grad, candidate_input.grad),
        "gate_gradient_correctness": _compare(
            eager_module.gate_proj.weight.grad,
            combined_grad[:intermediate_size],
        ),
        "up_gradient_correctness": _compare(
            eager_module.up_proj.weight.grad,
            combined_grad[intermediate_size:],
        ),
        "down_gradient_correctness": _compare(
            eager_module.down_proj.weight.grad,
            candidate_module.down_proj.weight.grad,
        ),
        "fast_path_observed": fused_ops._SWIGLU_LOGGED,
        "forward": _timing_result(eager_forward, candidate_forward),
        "forward_backward": _timing_result(eager_fwd_bwd, candidate_fwd_bwd),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("This benchmark requires a MUSA device.")

    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("warmup must be >= 1 and repeats must be >= 2")

    torch.manual_seed(args.seed)
    torch.musa.set_device(0)
    fused_ops._reset_runtime_state_for_tests()
    results = {
        "device": torch.musa.get_device_name(0),
        "torch": torch.__version__,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "measurement_order": "alternating eager/candidate then candidate/eager",
        "standard_rmsnorm": benchmark_standard_rmsnorm(args.warmup, args.repeats),
        "gated_rmsnorm": benchmark_gated_rmsnorm(args.warmup, args.repeats),
        "partial_mrope": benchmark_rope(args.warmup, args.repeats),
        "swiglu_mlp": benchmark_swiglu_mlp(args.warmup, args.repeats),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
