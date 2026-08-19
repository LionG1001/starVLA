"""Stress reproduction for the Mate/TileLang FlashAttention backward hang.

This test intentionally does not initialize a distributed process group.  Run
one process per MUSA device so that every device launches the production
``flashattn_bwd_ws_kernel`` concurrently, matching the failure mode described
by SW-86152 without involving MCCL or the StarVLA model.

Single-node, eight-device reproduction::

    STARVLA_RUN_MATE_HANG_STRESS=1 \
    STARVLA_MATE_HANG_ITERS=100 \
    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
      tests/test_qwen35_mate_tilelang_hang_musa.py

Use an external timeout around ``torchrun``.  A failing kernel can block inside
``torch.musa.synchronize()`` until the MUSA launch watchdog reports
``MUSA_ERROR_LAUNCH_TIMEOUT``.
"""

from __future__ import annotations

import json
import os
import time
import unittest

import torch

try:
    import torch_musa  # noqa: F401
    from mate.flash_attention.tilelang.flash_attention_varlen_bwd import (
        flashattn_varlen_bwd_interface,
    )
    from mate.mha_interface import flash_attn_varlen_func
except ImportError:
    torch_musa = None
    flashattn_varlen_bwd_interface = None
    flash_attn_varlen_func = None


QUERY_HEADS = 16
KEY_VALUE_HEADS = 4
HEAD_DIM = 256
VALID_LENGTHS = (291, 286, 290, 290)
TOTAL_TOKENS = sum(VALID_LENGTHS)
DTYPE = torch.bfloat16
SEED = 20260818


def _stress_enabled() -> bool:
    return os.environ.get("STARVLA_RUN_MATE_HANG_STRESS", "0") == "1"


def _runtime_available() -> bool:
    return (
        torch_musa is not None
        and hasattr(torch, "musa")
        and torch.musa.is_available()
        and flashattn_varlen_bwd_interface is not None
        and flash_attn_varlen_func is not None
    )


def _emit(event: str, **fields) -> None:
    payload = {
        "event": event,
        "pid": os.getpid(),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "timestamp": time.time(),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


@unittest.skipUnless(
    _stress_enabled() and _runtime_available(),
    "set STARVLA_RUN_MATE_HANG_STRESS=1 and run on a MUSA Mate/TileLang environment",
)
class Qwen35MateTileLangHangStressTest(unittest.TestCase):
    def test_multidevice_repeated_real_shape_backward(self):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        iterations = int(os.environ.get("STARVLA_MATE_HANG_ITERS", "100"))
        start_at = float(os.environ.get("STARVLA_MATE_HANG_START_AT", "0"))
        if iterations <= 0:
            self.fail("STARVLA_MATE_HANG_ITERS must be positive")

        torch.musa.set_device(local_rank)
        device = torch.device(f"musa:{local_rank}")
        generator = torch.Generator().manual_seed(SEED + local_rank)
        query = torch.randn(
            TOTAL_TOKENS, QUERY_HEADS, HEAD_DIM, generator=generator, dtype=DTYPE
        ).to(device)
        key = torch.randn(
            TOTAL_TOKENS, KEY_VALUE_HEADS, HEAD_DIM, generator=generator, dtype=DTYPE
        ).to(device)
        value = torch.randn(
            TOTAL_TOKENS, KEY_VALUE_HEADS, HEAD_DIM, generator=generator, dtype=DTYPE
        ).to(device)
        output_gradient = (
            torch.randn(
                TOTAL_TOKENS,
                QUERY_HEADS,
                HEAD_DIM,
                generator=generator,
                dtype=DTYPE,
            )
            * 0.01
        ).to(device)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(VALID_LENGTHS).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        )

        # Generate the exact forward artifacts consumed by the TileLang
        # backward interface.  Inputs do not require gradients, so this does
        # not invoke Mate's autograd backward path.
        output, softmax_lse = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            max(VALID_LENGTHS),
            max(VALID_LENGTHS),
            softmax_scale=HEAD_DIM**-0.5,
            causal=True,
            deterministic=False,
            return_softmax_lse=True,
        )
        torch.musa.synchronize()

        if start_at > 0:
            while time.time() < start_at:
                time.sleep(min(0.05, start_at - time.time()))

        _emit(
            "stress_start",
            iterations=iterations,
            q_shape=tuple(query.shape),
            kv_shape=tuple(key.shape),
            valid_lengths=VALID_LENGTHS,
            uses_mccl=False,
        )

        for iteration in range(1, iterations + 1):
            _emit("backward_begin", iteration=iteration)
            dq, dk, dv = flashattn_varlen_bwd_interface(
                query,
                key,
                value,
                output,
                output_gradient,
                softmax_lse,
                max(VALID_LENGTHS),
                max(VALID_LENGTHS),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                is_causal=True,
                smscale=HEAD_DIM**-0.5,
                dtype=None,
                block_M=64,
                block_N=64,
                threads=640,
                is_bhsd=False,
            )
            # This is the diagnostic boundary: a TileLang launch hang or
            # MUSA_ERROR_LAUNCH_TIMEOUT surfaces here, before any collective.
            torch.musa.synchronize()

            if iteration == 1 or iteration % 10 == 0 or iteration == iterations:
                finite = all(bool(torch.isfinite(grad).all().item()) for grad in (dq, dk, dv))
                self.assertTrue(finite, f"non-finite gradient at iteration {iteration}")
            _emit("backward_end", iteration=iteration)

        _emit("stress_pass", iterations=iterations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
