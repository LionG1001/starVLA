from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.vlm import qwen35_musa_fused_ops as fused_ops

try:
    import torch_musa  # noqa: F401

    _MUSA_AVAILABLE = torch.musa.is_available()
except ImportError:
    _MUSA_AVAILABLE = False


class _FakeRMSNorm:
    def __init__(self, hidden_size: int, *, gated: bool = False):
        self.weight = nn.Parameter(torch.randn(hidden_size))
        if gated:
            self.variance_epsilon = 1e-6
        else:
            self.eps = 1e-6


class _FakeMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu")
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu


@pytest.fixture(autouse=True)
def _reset_fused_runtime_state():
    fused_ops._reset_runtime_state_for_tests()
    yield
    fused_ops._reset_runtime_state_for_tests()


def test_qwen35_fused_ops_configure_exact_transformers_classes(monkeypatch) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5")
            self.rms = modeling_qwen3_5.Qwen3_5RMSNorm(32)
            self.gated = modeling_qwen3_5.Qwen3_5RMSNormGated(32)
            mlp_config = SimpleNamespace(hidden_size=32, hidden_act="silu")
            self.mlp = modeling_qwen3_5.Qwen3_5MLP(mlp_config, 64)
            self.rotary = modeling_qwen3_5.Qwen3_5TextRotaryEmbedding.__new__(
                modeling_qwen3_5.Qwen3_5TextRotaryEmbedding
            )
            nn.Module.__init__(self.rotary)

    monkeypatch.setattr(fused_ops, "_musa_is_available", lambda: True)
    monkeypatch.setattr(fused_ops, "_is_supported_transformers_version", lambda: True)
    original_apply_rope = modeling_qwen3_5.apply_rotary_pos_emb
    try:
        model = _Root()
        original_mlp_state = {name: tensor.detach().clone() for name, tensor in model.mlp.state_dict().items()}
        original_mlp_numel = sum(parameter.numel() for parameter in model.mlp.parameters())
        status = fused_ops.configure_qwen35_musa_fused_ops(
            model,
            {
                "musa_fused_rmsnorm": True,
                "musa_fused_gated_rmsnorm": True,
                "musa_fused_rope": True,
                "musa_fused_swiglu": True,
            },
        )
        assert status == fused_ops.Qwen35MusaFusedOpsStatus(
            rmsnorm_modules=1,
            gated_rmsnorm_modules=1,
            rotary_embedding_modules=1,
            rope_function_patched=True,
            swiglu_modules=1,
        )
        assert model.rms.forward.__func__ is fused_ops.qwen35_rms_norm_musa
        assert model.gated.forward.__func__ is fused_ops.qwen35_gated_rms_norm_musa
        assert model.mlp.forward.__func__ is fused_ops.qwen35_swiglu_musa
        assert not hasattr(model.mlp, "gate_proj")
        assert not hasattr(model.mlp, "up_proj")
        assert model.mlp.gate_up_proj.weight.shape == (128, 32)
        assert sum(parameter.numel() for parameter in model.mlp.parameters()) == original_mlp_numel
        actual_mlp_state = model.mlp.state_dict()
        assert actual_mlp_state.keys() == original_mlp_state.keys()
        assert (
            actual_mlp_state["gate_proj.weight"].untyped_storage().data_ptr()
            != actual_mlp_state["up_proj.weight"].untyped_storage().data_ptr()
        )
        for name, expected in original_mlp_state.items():
            torch.testing.assert_close(actual_mlp_state[name], expected)

        with torch.no_grad():
            model.mlp.gate_up_proj.weight.zero_()
        model.mlp.load_state_dict(original_mlp_state, strict=True)
        torch.testing.assert_close(
            model.mlp.gate_up_proj.weight[:64],
            original_mlp_state["gate_proj.weight"],
        )
        torch.testing.assert_close(
            model.mlp.gate_up_proj.weight[64:],
            original_mlp_state["up_proj.weight"],
        )
        assert modeling_qwen3_5.apply_rotary_pos_emb is fused_ops.qwen35_rotary_pos_emb_musa
    finally:
        modeling_qwen3_5.apply_rotary_pos_emb = original_apply_rope


def test_qwen35_fused_ops_reject_enabled_path_without_matching_modules(
    monkeypatch,
) -> None:
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_5")
    monkeypatch.setattr(fused_ops, "_musa_is_available", lambda: True)
    monkeypatch.setattr(fused_ops, "_is_supported_transformers_version", lambda: True)

    with pytest.raises(RuntimeError, match="RMSNorm"):
        fused_ops.configure_qwen35_musa_fused_ops(model, {"musa_fused_rmsnorm": True})


def test_qwen35_fused_ops_cpu_fallbacks_match_eager() -> None:
    hidden_states = torch.randn(2, 17, 32)

    norm = _FakeRMSNorm(32)
    torch.testing.assert_close(
        fused_ops.qwen35_rms_norm_musa(norm, hidden_states),
        fused_ops.qwen35_rms_norm_eager(norm, hidden_states),
    )

    gated_norm = _FakeRMSNorm(32, gated=True)
    gate = torch.randn_like(hidden_states)
    torch.testing.assert_close(
        fused_ops.qwen35_gated_rms_norm_musa(gated_norm, hidden_states, gate),
        fused_ops.qwen35_gated_rms_norm_eager(gated_norm, hidden_states, gate),
    )

    query = torch.randn(2, 8, 17, 32)
    key = torch.randn(2, 2, 17, 32)
    cos = torch.randn(2, 17, 16)
    sin = torch.randn(2, 17, 16)
    actual_query, actual_key = fused_ops.qwen35_rotary_pos_emb_musa(query, key, cos, sin)
    expected_query, expected_key = fused_ops.qwen35_rotary_pos_emb_eager(query, key, cos, sin)
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)

    mlp = _FakeMLP(32, 64)
    torch.testing.assert_close(
        fused_ops.qwen35_swiglu_musa(mlp, hidden_states),
        fused_ops.qwen35_swiglu_eager(mlp, hidden_states),
    )


def test_qwen35_combined_swiglu_cpu_avoids_activation_cat(monkeypatch) -> None:
    torch.manual_seed(7)
    candidate = _FakeMLP(32, 64)
    reference = _FakeMLP(32, 64)
    reference.load_state_dict(candidate.state_dict())
    original_state = {name: tensor.detach().clone() for name, tensor in candidate.state_dict().items()}
    original_numel = sum(parameter.numel() for parameter in candidate.parameters())
    fused_ops._install_qwen35_combined_swiglu_projection(candidate)

    assert sum(parameter.numel() for parameter in candidate.parameters()) == original_numel
    assert candidate.state_dict().keys() == original_state.keys()
    for name, expected in original_state.items():
        torch.testing.assert_close(candidate.state_dict()[name], expected)

    hidden_states = torch.randn(2, 17, 32, requires_grad=True)
    reference_input = hidden_states.detach().clone().requires_grad_(True)
    grad_output = torch.randn(2, 17, 32)

    def fail_runtime_cat(*args, **kwargs):
        raise AssertionError("combined SwiGLU forward must not call torch.cat")

    monkeypatch.setattr(torch, "cat", fail_runtime_cat)
    actual = fused_ops.qwen35_swiglu_musa(candidate, hidden_states)
    expected = fused_ops.qwen35_swiglu_eager(reference, reference_input)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(hidden_states.grad, reference_input.grad)
    torch.testing.assert_close(candidate.gate_up_proj.weight.grad[:64], reference.gate_proj.weight.grad)
    torch.testing.assert_close(candidate.gate_up_proj.weight.grad[64:], reference.up_proj.weight.grad)
    torch.testing.assert_close(candidate.down_proj.weight.grad, reference.down_proj.weight.grad)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen35_fused_rmsnorm_real_shape_forward_backward() -> None:
    torch.manual_seed(11)
    hidden_size = 2560
    shape = (4, 291, hidden_size)
    norm = _FakeRMSNorm(hidden_size)
    norm.weight = nn.Parameter((torch.randn(hidden_size, device="musa", dtype=torch.bfloat16) * 0.02))
    reference_weight = norm.weight.detach().clone().requires_grad_(True)
    actual_input = torch.randn(shape, device="musa", dtype=torch.bfloat16, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(actual_input)

    actual = fused_ops.qwen35_rms_norm_musa(norm, actual_input)
    reference_norm = SimpleNamespace(weight=reference_weight, eps=norm.eps)
    expected = fused_ops.qwen35_rms_norm_eager(reference_norm, reference_input)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.03125)

    actual.backward(grad_output)
    expected.backward(grad_output)
    assert torch.isfinite(actual_input.grad).all()
    assert torch.isfinite(norm.weight.grad).all()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0.1, atol=0.03125)
    torch.testing.assert_close(norm.weight.grad, reference_weight.grad, rtol=0.05, atol=0.5)

    learning_rate = 1e-3
    actual_updated = norm.weight.detach() - learning_rate * norm.weight.grad
    reference_updated = reference_weight.detach() - learning_rate * reference_weight.grad
    torch.testing.assert_close(actual_updated, reference_updated, rtol=0.01, atol=0.001)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen35_fused_gated_rmsnorm_real_shape_forward_backward() -> None:
    torch.manual_seed(13)
    hidden_size = 128
    shape = (4 * 291 * 32, hidden_size)
    norm = _FakeRMSNorm(hidden_size, gated=True)
    norm.weight = nn.Parameter(torch.ones(hidden_size, device="musa", dtype=torch.bfloat16))
    reference_weight = norm.weight.detach().clone().requires_grad_(True)
    actual_input = torch.randn(shape, device="musa", dtype=torch.bfloat16, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    actual_gate = torch.randn_like(actual_input, requires_grad=True)
    reference_gate = actual_gate.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(actual_input)

    actual = fused_ops.qwen35_gated_rms_norm_musa(norm, actual_input, actual_gate)
    reference_norm = SimpleNamespace(
        weight=reference_weight,
        variance_epsilon=norm.variance_epsilon,
    )
    expected = fused_ops.qwen35_gated_rms_norm_eager(reference_norm, reference_input, reference_gate)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.015625)

    actual.backward(grad_output)
    expected.backward(grad_output)
    for gradient in (actual_input.grad, actual_gate.grad, norm.weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0.03, atol=0.015625)
    torch.testing.assert_close(actual_gate.grad, reference_gate.grad, rtol=0.03, atol=0.015625)
    torch.testing.assert_close(norm.weight.grad, reference_weight.grad, rtol=0.2, atol=1.0)

    learning_rate = 1e-3
    actual_updated = norm.weight.detach() - learning_rate * norm.weight.grad
    reference_updated = reference_weight.detach() - learning_rate * reference_weight.grad
    torch.testing.assert_close(actual_updated, reference_updated, rtol=0.01, atol=0.002)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen35_fused_rmsnorm_failure_uses_permanent_fallback(monkeypatch) -> None:
    norm = _FakeRMSNorm(32)
    norm.weight = nn.Parameter(torch.zeros(32, device="musa", dtype=torch.bfloat16))
    hidden_states = torch.randn(2, 7, 32, device="musa", dtype=torch.bfloat16)
    expected = fused_ops.qwen35_rms_norm_eager(norm, hidden_states)
    calls = 0

    def _fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic fused-kernel failure")

    monkeypatch.setattr(F, "rms_norm", _fail_once)
    first = fused_ops.qwen35_rms_norm_musa(norm, hidden_states)
    second = fused_ops.qwen35_rms_norm_musa(norm, hidden_states)

    assert calls == 1
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen35_fused_rmsnorm_does_not_swallow_oom(monkeypatch) -> None:
    norm = _FakeRMSNorm(32)
    norm.weight = nn.Parameter(torch.zeros(32, device="musa", dtype=torch.bfloat16))
    hidden_states = torch.randn(2, 7, 32, device="musa", dtype=torch.bfloat16)

    def _oom(*args, **kwargs):
        raise RuntimeError("MUSA out of memory")

    monkeypatch.setattr(F, "rms_norm", _oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        fused_ops.qwen35_rms_norm_musa(norm, hidden_states)
    assert not fused_ops._RMSNORM_DISABLED


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
@pytest.mark.parametrize("sequence_length", [1, 37, 291])
def test_qwen35_fused_partial_mrope_real_shape_forward_backward(
    sequence_length: int,
) -> None:
    torch.manual_seed(17)
    batch_size, head_dim, rotary_dim = 4, 256, 64
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
    reference_query_base = query_base.detach().clone().requires_grad_(True)
    reference_key_base = key_base.detach().clone().requires_grad_(True)
    query = query_base.transpose(1, 2)
    key = key_base.transpose(1, 2)
    reference_query = reference_query_base.transpose(1, 2)
    reference_key = reference_key_base.transpose(1, 2)
    if sequence_length > 1:
        assert not query.is_contiguous() and not key.is_contiguous()
    phase = torch.randn(batch_size, sequence_length, rotary_dim, device="musa", dtype=torch.float32)
    cos = phase.cos().to(torch.bfloat16)
    sin = phase.sin().to(torch.bfloat16)
    setattr(cos, fused_ops._ROPE_PHASE_ATTR, phase)

    actual_query, actual_key = fused_ops.qwen35_rotary_pos_emb_musa(query, key, cos, sin)
    expected_query, expected_key = fused_ops.qwen35_rotary_pos_emb_eager(reference_query, reference_key, cos, sin)

    assert torch.isfinite(actual_query).all() and torch.isfinite(actual_key).all()
    torch.testing.assert_close(actual_query, expected_query, rtol=0.02, atol=0.03125)
    torch.testing.assert_close(actual_key, expected_key, rtol=0.02, atol=0.03125)

    query_grad = torch.randn_like(actual_query)
    key_grad = torch.randn_like(actual_key)
    torch.autograd.backward((actual_query, actual_key), (query_grad, key_grad))
    torch.autograd.backward((expected_query, expected_key), (query_grad, key_grad))
    assert torch.isfinite(query_base.grad).all()
    assert torch.isfinite(key_base.grad).all()
    torch.testing.assert_close(
        query_base.grad,
        reference_query_base.grad,
        rtol=0.02,
        atol=0.03125,
    )
    torch.testing.assert_close(
        key_base.grad,
        reference_key_base.grad,
        rtol=0.02,
        atol=0.03125,
    )


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen35_combined_fused_swiglu_real_shape_forward_backward() -> None:
    torch.manual_seed(19)
    mlp = _FakeMLP(64, 128).to(device="musa", dtype=torch.bfloat16)
    reference_mlp = _FakeMLP(64, 128).to(device="musa", dtype=torch.bfloat16)
    reference_mlp.load_state_dict(mlp.state_dict())
    fused_ops._install_qwen35_combined_swiglu_projection(mlp)
    actual_input = torch.randn(4, 37, 64, device="musa", dtype=torch.bfloat16, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(actual_input)

    actual = fused_ops.qwen35_swiglu_musa(mlp, actual_input)
    expected = fused_ops.qwen35_swiglu_eager(reference_mlp, reference_input)
    assert fused_ops._SWIGLU_LOGGED
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)

    actual.backward(grad_output)
    expected.backward(grad_output)
    assert torch.isfinite(actual_input.grad).all()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0.03, atol=0.0625)
    combined_grad = mlp.gate_up_proj.weight.grad
    assert combined_grad is not None and torch.isfinite(combined_grad).all()
    torch.testing.assert_close(
        combined_grad[:128],
        reference_mlp.gate_proj.weight.grad,
        rtol=0.05,
        atol=0.0625,
    )
    torch.testing.assert_close(
        combined_grad[128:],
        reference_mlp.up_proj.weight.grad,
        rtol=0.05,
        atol=0.0625,
    )
    torch.testing.assert_close(
        mlp.down_proj.weight.grad,
        reference_mlp.down_proj.weight.grad,
        rtol=0.05,
        atol=0.0625,
    )
