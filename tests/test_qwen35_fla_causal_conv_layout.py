"""Layout contract tests for the Qwen3.5 FLA causal-conv adapter."""

from __future__ import annotations

import torch

from starVLA.model.modules.vlm.qwen35_musa import _fla_causal_conv_adapter


def _materialized_adapter(fake_causal_conv1d):
    def causal_conv1d_fn(x, weight, bias=None, activation=None, **kwargs):
        output, _ = fake_causal_conv1d(
            x=x.transpose(1, 2).contiguous(),
            weight=weight,
            bias=bias,
            activation=activation,
            output_final_state=False,
            backend="triton",
            **kwargs,
        )
        return output.transpose(1, 2).contiguous()

    return causal_conv1d_fn


def _run_adapter(adapter_factory, source, grad_output):
    source = source.detach().clone().requires_grad_(True)
    qwen_input = source.transpose(1, 2)
    weight = torch.ones(source.shape[-1], 1)
    seen = {}

    def fake_causal_conv1d(*, x, weight, bias, activation, **kwargs):
        seen["input_contiguous"] = x.is_contiguous()
        seen["kwargs"] = kwargs
        return x * weight[:, 0], None

    adapter_output = adapter_factory(fake_causal_conv1d)(
        qwen_input,
        weight,
        activation="silu",
    )
    consumed_output = adapter_output.transpose(1, 2)
    consumed_output.backward(grad_output)
    return adapter_output.detach(), consumed_output.detach(), source.grad, seen


def test_copy_elision_preserves_output_gradient_and_consumer_layout():
    torch.manual_seed(20260819)
    source = torch.randn(2, 7, 11)
    grad_output = torch.randn_like(source)

    reference = _run_adapter(_materialized_adapter, source, grad_output)
    candidate = _run_adapter(_fla_causal_conv_adapter, source, grad_output)

    reference_adapter, reference_consumed, reference_grad, _ = reference
    candidate_adapter, candidate_consumed, candidate_grad, seen = candidate

    torch.testing.assert_close(candidate_adapter, reference_adapter)
    torch.testing.assert_close(candidate_consumed, reference_consumed)
    torch.testing.assert_close(candidate_grad, reference_grad)
    assert seen["input_contiguous"] is True
    assert seen["kwargs"]["output_final_state"] is False
    assert seen["kwargs"]["backend"] == "triton"
    assert candidate_adapter.is_contiguous() is False
    assert candidate_consumed.is_contiguous() is True
    assert reference_adapter.is_contiguous() is True
    assert reference_consumed.is_contiguous() is False
