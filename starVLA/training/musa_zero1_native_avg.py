"""MUSA fast path for DeepSpeed ZeRO-1 gradient averaging.

DeepSpeed's default ZeRO-1 reduce-scatter path first divides the complete BF16
gradient bucket, repacks rank slices with ``torch.cat``, and then performs a SUM
all-reduce.  MCCL supports a native AVG reduction, so the same averaged gradient
can be produced directly from DeepSpeed's already-contiguous IPG bucket.

The patch is deliberately installed on one optimizer instance after
``Accelerator.prepare``.  It does not modify DeepSpeed globally and fails closed
outside the validated ZeRO-1, non-overlapped, sequence-parallel-size-1 setup.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)

_ORIGINAL_REDUCER_ATTR = "_starvla_original_gradient_reduction_w_predivide"
_ORIGINAL_REDUCE_SCATTER_ATTR = "_starvla_original_reduce_scatter"
_NATIVE_AVG_CALLS_ATTR = "_starvla_native_avg_calls"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _config_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


def _musa_is_available() -> bool:
    return bool(hasattr(torch, "musa") and torch.musa.is_available())


def _get_deepspeed_comm():
    import deepspeed.comm as ds_dist

    return ds_dist


def _iter_optimizer_candidates(model: Any, prepared_optimizer: Any) -> Iterable[Any]:
    seen: set[int] = set()
    queue = [getattr(model, "optimizer", None), prepared_optimizer]
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        queue.append(getattr(candidate, "optimizer", None))


def _resolve_zero_optimizer(model: Any, prepared_optimizer: Any) -> Any:
    required = (
        "gradient_reduction_w_predivide",
        "dp_process_group",
        "reduce_scatter",
        "sequence_parallel_size",
    )
    for candidate in _iter_optimizer_candidates(model, prepared_optimizer):
        if all(hasattr(candidate, name) for name in required):
            return candidate
    raise RuntimeError(
        "MUSA ZeRO-1 native AVG was requested, but the prepared DeepSpeed "
        "ZeRO optimizer could not be found."
    )


def _native_avg_gradient_reduction(
    zero_optimizer: Any,
    tensor: torch.Tensor,
    communication_data_type: torch.dtype,
) -> torch.Tensor:
    """Average one contiguous IPG bucket with a native MCCL AVG collective."""
    if tensor.numel() == 0:
        return tensor

    tensor_to_allreduce = tensor
    if communication_data_type != tensor.dtype:
        tensor_to_allreduce = tensor.to(communication_data_type)

    ds_dist = _get_deepspeed_comm()
    ds_dist.all_reduce(
        tensor_to_allreduce,
        op=ds_dist.ReduceOp.AVG,
        group=zero_optimizer.dp_process_group,
    )

    if tensor_to_allreduce is not tensor:
        tensor.copy_(tensor_to_allreduce)
    setattr(
        zero_optimizer,
        _NATIVE_AVG_CALLS_ATTR,
        getattr(zero_optimizer, _NATIVE_AVG_CALLS_ATTR, 0) + 1,
    )
    return tensor


def install_musa_zero1_native_avg(model: Any, prepared_optimizer: Any) -> Any:
    """Install the instance-local native AVG reducer and return its optimizer."""
    zero_optimizer = _resolve_zero_optimizer(model, prepared_optimizer)
    if hasattr(zero_optimizer, _ORIGINAL_REDUCER_ATTR):
        return zero_optimizer

    if bool(getattr(zero_optimizer, "partition_gradients", False)):
        raise RuntimeError("MUSA native AVG currently supports ZeRO-1 only, not ZeRO-2.")
    if bool(getattr(zero_optimizer, "overlap_comm", False)):
        raise RuntimeError(
            "MUSA ZeRO-1 native AVG requires overlap_comm=false; the overlapped "
            "bucket lifetime has not been validated."
        )
    if int(zero_optimizer.sequence_parallel_size) != 1:
        raise RuntimeError(
            "MUSA ZeRO-1 native AVG requires sequence_parallel_size=1, got "
            f"{zero_optimizer.sequence_parallel_size}."
        )

    setattr(
        zero_optimizer,
        _ORIGINAL_REDUCER_ATTR,
        zero_optimizer.gradient_reduction_w_predivide,
    )
    setattr(
        zero_optimizer,
        _ORIGINAL_REDUCE_SCATTER_ATTR,
        bool(zero_optimizer.reduce_scatter),
    )
    setattr(zero_optimizer, _NATIVE_AVG_CALLS_ATTR, 0)

    # With reduce_scatter disabled, DeepSpeed sends its existing contiguous IPG
    # buffer directly to gradient_reduction_w_predivide.  The bound replacement
    # below then uses AVG, eliminating both the full-buffer div and rank-slice cat.
    zero_optimizer.reduce_scatter = False
    zero_optimizer.gradient_reduction_w_predivide = MethodType(
        _native_avg_gradient_reduction,
        zero_optimizer,
    )
    logger.warning(
        "Enabled StarVLA MUSA ZeRO-1 native AVG: direct contiguous IPG "
        "all-reduce, no BF16 pre-divide or rank-slice flatten cat."
    )
    return zero_optimizer


def disable_musa_zero1_native_avg(model: Any, prepared_optimizer: Any) -> bool:
    """Restore the reducer state saved by :func:`install_musa_zero1_native_avg`."""
    try:
        zero_optimizer = _resolve_zero_optimizer(model, prepared_optimizer)
    except RuntimeError:
        return False
    if not hasattr(zero_optimizer, _ORIGINAL_REDUCER_ATTR):
        return False

    zero_optimizer.gradient_reduction_w_predivide = getattr(
        zero_optimizer,
        _ORIGINAL_REDUCER_ATTR,
    )
    zero_optimizer.reduce_scatter = getattr(
        zero_optimizer,
        _ORIGINAL_REDUCE_SCATTER_ATTR,
    )
    delattr(zero_optimizer, _ORIGINAL_REDUCER_ATTR)
    delattr(zero_optimizer, _ORIGINAL_REDUCE_SCATTER_ATTR)
    return True


def configure_musa_zero1_native_avg(
    model: Any,
    prepared_optimizer: Any,
    trainer_config: Any,
) -> Any | None:
    """Apply the YAML-selected ZeRO-1 native AVG policy after preparation."""
    enabled = _config_bool(
        trainer_config.get("musa_zero1_native_avg", False),
        name="trainer.musa_zero1_native_avg",
    )
    if not enabled:
        disable_musa_zero1_native_avg(model, prepared_optimizer)
        return None
    if not _musa_is_available():
        raise RuntimeError(
            "trainer.musa_zero1_native_avg=true was requested, but MUSA is unavailable."
        )
    return install_musa_zero1_native_avg(model, prepared_optimizer)


__all__ = [
    "configure_musa_zero1_native_avg",
    "disable_musa_zero1_native_avg",
    "install_musa_zero1_native_avg",
]
