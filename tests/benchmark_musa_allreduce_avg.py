"""Compare BF16 pre-divide+SUM with native AVG all-reduce on MUSA.

Run on one node, for example::

    torchrun --standalone --nproc_per_node=8 \
        tests/benchmark_musa_allreduce_avg.py --numel 134217728
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401


def _sync() -> None:
    torch.musa.synchronize()
    dist.barrier()


def _measure(source: torch.Tensor, *, op: dist.ReduceOp, predivide: bool, iters: int):
    output = torch.empty_like(source)
    timings_ms = []
    for _ in range(iters):
        output.copy_(source)
        _sync()
        start = time.perf_counter()
        if predivide:
            output.div_(dist.get_world_size())
        dist.all_reduce(output, op=op)
        _sync()
        timings_ms.append((time.perf_counter() - start) * 1_000)
    return output, timings_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.musa.set_device(local_rank)
    dist.init_process_group("mccl")

    generator = torch.Generator(device=f"musa:{local_rank}")
    generator.manual_seed(20260820 + dist.get_rank())
    source = torch.randn(
        args.numel,
        device=f"musa:{local_rank}",
        dtype=torch.bfloat16,
        generator=generator,
    )

    for _ in range(args.warmup):
        _measure(source, op=dist.ReduceOp.SUM, predivide=True, iters=1)
        _measure(source, op=dist.ReduceOp.AVG, predivide=False, iters=1)

    sum_output, sum_ms = _measure(
        source,
        op=dist.ReduceOp.SUM,
        predivide=True,
        iters=args.iters,
    )
    avg_output, avg_ms = _measure(
        source,
        op=dist.ReduceOp.AVG,
        predivide=False,
        iters=args.iters,
    )

    delta = (avg_output.float() - sum_output.float()).norm()
    reference = sum_output.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    relative_l2 = (delta / reference).item()
    max_abs = (avg_output.float() - sum_output.float()).abs().max().item()
    finite = bool(torch.isfinite(avg_output).all() and torch.isfinite(sum_output).all())

    if dist.get_rank() == 0:
        sum_median = statistics.median(sum_ms)
        avg_median = statistics.median(avg_ms)
        print(
            {
                "world_size": dist.get_world_size(),
                "numel": args.numel,
                "dtype": str(source.dtype),
                "predivide_sum_median_ms": sum_median,
                "native_avg_median_ms": avg_median,
                "speedup": sum_median / avg_median,
                "finite": finite,
                "relative_l2": relative_l2,
                "max_abs": max_abs,
            },
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
