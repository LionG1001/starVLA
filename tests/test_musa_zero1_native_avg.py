from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from starVLA.training import musa_zero1_native_avg as native_avg


class _FakeReduceOp:
    AVG = object()


class _FakeComm:
    ReduceOp = _FakeReduceOp

    def __init__(self):
        self.calls = []

    def all_reduce(self, tensor, *, op, group):
        self.calls.append((tensor, op, group))
        tensor.add_(1)


class _FakeZeroOptimizer:
    def __init__(self):
        self.dp_process_group = object()
        self.reduce_scatter = True
        self.sequence_parallel_size = 1
        self.partition_gradients = False
        self.overlap_comm = False
        self.original_calls = 0

        def original(instance, tensor, communication_data_type):
            instance.original_calls += 1
            return tensor

        self.gradient_reduction_w_predivide = MethodType(original, self)


def _prepared(zero_optimizer):
    return SimpleNamespace(optimizer=zero_optimizer)


def test_install_uses_direct_native_avg_and_restores(monkeypatch):
    comm = _FakeComm()
    monkeypatch.setattr(native_avg, "_get_deepspeed_comm", lambda: comm)
    zero = _FakeZeroOptimizer()
    original_function = zero.gradient_reduction_w_predivide.__func__

    installed = native_avg.install_musa_zero1_native_avg(
        SimpleNamespace(),
        _prepared(zero),
    )
    assert installed is zero
    assert zero.reduce_scatter is False

    tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    result = zero.gradient_reduction_w_predivide(tensor, torch.float32)
    assert result is tensor
    assert torch.equal(tensor, torch.tensor([2.0, 3.0], dtype=torch.bfloat16))
    assert len(comm.calls) == 1
    reduced, op, group = comm.calls[0]
    assert reduced.dtype == torch.float32
    assert op is _FakeReduceOp.AVG
    assert group is zero.dp_process_group
    assert zero._starvla_native_avg_calls == 1

    assert native_avg.disable_musa_zero1_native_avg(
        SimpleNamespace(),
        _prepared(zero),
    )
    assert zero.reduce_scatter is True
    assert zero.gradient_reduction_w_predivide.__func__ is original_function


def test_configure_disabled_leaves_optimizer_unchanged(monkeypatch):
    monkeypatch.setattr(native_avg, "_musa_is_available", lambda: False)
    zero = _FakeZeroOptimizer()
    original_function = zero.gradient_reduction_w_predivide.__func__

    assert (
        native_avg.configure_musa_zero1_native_avg(
            SimpleNamespace(),
            _prepared(zero),
            {"musa_zero1_native_avg": False},
        )
        is None
    )
    assert zero.reduce_scatter is True
    assert zero.gradient_reduction_w_predivide.__func__ is original_function


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("partition_gradients", True, "ZeRO-1 only"),
        ("overlap_comm", True, "overlap_comm=false"),
        ("sequence_parallel_size", 2, "sequence_parallel_size=1"),
    ],
)
def test_install_fails_closed_outside_validated_boundary(attribute, value, message):
    zero = _FakeZeroOptimizer()
    setattr(zero, attribute, value)
    with pytest.raises(RuntimeError, match=message):
        native_avg.install_musa_zero1_native_avg(
            SimpleNamespace(),
            _prepared(zero),
        )


def test_configure_enabled_requires_musa(monkeypatch):
    monkeypatch.setattr(native_avg, "_musa_is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MUSA is unavailable"):
        native_avg.configure_musa_zero1_native_avg(
            SimpleNamespace(),
            _prepared(_FakeZeroOptimizer()),
            {"musa_zero1_native_avg": True},
        )
