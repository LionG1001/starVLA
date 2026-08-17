"""Manual numerical A/B for Qwen3.5's FLA gated-delta-rule on MUSA."""

import torch
import torch_musa  # noqa: F401
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
)


def make_leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


def error_stats(actual, expected):
    actual = actual.float()
    expected = expected.float()
    difference = actual - expected
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (
            difference.norm() / expected.norm().clamp_min(1e-12)
        ).item(),
    }


def run_once(device="musa:0"):
    torch.manual_seed(1234)
    shape = (1, 64, 8, 64)
    q_base = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    k_base = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    v_base = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    g_base = -torch.rand(shape[:-1], device=device, dtype=torch.float32) * 0.1
    beta_base = torch.rand(shape[:-1], device=device, dtype=torch.bfloat16)

    reference_inputs = [
        make_leaf(value)
        for value in (q_base, k_base, v_base, g_base, beta_base)
    ]
    fla_inputs = [make_leaf(value) for value in (q_base, k_base, v_base, g_base, beta_base)]

    reference, _ = torch_chunk_gated_delta_rule(
        *reference_inputs,
        use_qk_l2norm_in_kernel=True,
    )
    reference.float().square().mean().backward()

    actual, _ = chunk_gated_delta_rule(
        *fla_inputs,
        use_qk_l2norm_in_kernel=True,
    )
    actual.float().square().mean().backward()
    torch.musa.synchronize()

    print("forward", error_stats(actual, reference))
    for name, actual_input, reference_input in zip(
        ("q_grad", "k_grad", "v_grad", "g_grad", "beta_grad"),
        fla_inputs,
        reference_inputs,
        strict=True,
    ):
        print(name, error_stats(actual_input.grad, reference_input.grad))


if __name__ == "__main__":
    run_once()
