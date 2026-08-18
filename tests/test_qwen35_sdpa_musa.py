"""Regression tests for Qwen3.5 SDPA numerical behavior on MUSA.

The tensor and mask shapes mirror the SDPA operator boundary observed in the
Qwen3.5-4B StarVLA batch-size-4 workload.  In particular, the left-padded
samples create fully masked query rows, which must not produce non-finite
outputs.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3_5TextConfig
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.masking_utils import create_causal_mask

try:
    import torch_musa
except ImportError:
    torch_musa = None


BATCH_SIZE = 4
QUERY_HEADS = 16
KEY_VALUE_HEADS = 4
KEY_VALUE_GROUPS = QUERY_HEADS // KEY_VALUE_HEADS
SEQUENCE_LENGTH = 291
HEAD_DIM = 256
HIDDEN_SIZE = 2560
VALID_LENGTHS = (291, 286, 290, 290)
DTYPE = torch.bfloat16
SEED = 20260817


def _musa_is_available() -> bool:
    return torch_musa is not None and hasattr(torch, "musa") and torch.musa.is_available()


@dataclass(frozen=True)
class ErrorStats:
    all_finite: bool
    valid_finite: bool
    nan_count: int
    inf_count: int
    max_abs: float
    mean_abs: float
    relative_l2: float
    bad_query_rows_per_batch: tuple[int, ...]
    bad_valid_query_rows_per_batch: tuple[int, ...]


def _make_attention_mask(device: torch.device, valid_lengths: tuple[int, ...]) -> torch.Tensor:
    attention_mask = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH, device=device, dtype=torch.long)
    for batch_index, valid_length in enumerate(valid_lengths):
        attention_mask[batch_index, -valid_length:] = 1
    return attention_mask


def _make_causal_mask(
    device: torch.device,
    attention_mask: torch.Tensor,
    implementation: str,
) -> torch.Tensor:
    config = Qwen3_5TextConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=QUERY_HEADS,
        num_key_value_heads=KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
    )
    config._attn_implementation = implementation
    inputs_embeds = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH, 1, device=device, dtype=DTYPE)
    cache_position = torch.arange(SEQUENCE_LENGTH, device=device)
    return create_causal_mask(config, inputs_embeds, attention_mask, cache_position, None)


def _eager_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    additive_mask: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) * (HEAD_DIM**-0.5)
    probabilities = torch.softmax(scores + additive_mask, dim=-1, dtype=torch.float32).to(DTYPE)
    return torch.matmul(probabilities, value)


def _error_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    valid_queries: torch.Tensor,
) -> ErrorStats:
    finite = torch.isfinite(actual)
    finite_query_rows = finite.all(dim=(1, 3))
    valid = valid_queries[:, None, :, None].expand_as(actual)
    actual_valid = actual.masked_select(valid).float()
    expected_valid = expected.masked_select(valid).float()
    valid_finite = bool(torch.isfinite(actual_valid).all().item())

    if valid_finite:
        difference = actual_valid - expected_valid
        max_abs = difference.abs().max().item()
        mean_abs = difference.abs().mean().item()
        relative_l2 = (difference.norm() / expected_valid.norm().clamp_min(1e-12)).item()
    else:
        max_abs = math.inf
        mean_abs = math.inf
        relative_l2 = math.inf

    return ErrorStats(
        all_finite=bool(finite.all().item()),
        valid_finite=valid_finite,
        nan_count=int(torch.isnan(actual).sum().item()),
        inf_count=int(torch.isinf(actual).sum().item()),
        max_abs=max_abs,
        mean_abs=mean_abs,
        relative_l2=relative_l2,
        bad_query_rows_per_batch=tuple((~finite_query_rows).sum(dim=-1).cpu().tolist()),
        bad_valid_query_rows_per_batch=tuple(
            ((~finite_query_rows) & valid_queries).sum(dim=-1).cpu().tolist()
        ),
    )


@unittest.skipUnless(_musa_is_available(), "requires a MUSA device and torch_musa")
class Qwen35SdpaMusaTest(unittest.TestCase):
    """Check forced SDPA backends against the production-shaped eager path."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("musa:0")
        torch.manual_seed(SEED)

        # Qwen attention projections are [batch, sequence, heads, head_dim]
        # before transpose. Preserve the resulting non-contiguous query stride.
        cls.query = torch.randn(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            QUERY_HEADS,
            HEAD_DIM,
            device=cls.device,
            dtype=DTYPE,
        ).transpose(1, 2)
        compact_key = torch.randn(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            KEY_VALUE_HEADS,
            HEAD_DIM,
            device=cls.device,
            dtype=DTYPE,
        ).transpose(1, 2)
        compact_value = torch.randn_like(compact_key)

        # With a 4D padding mask Transformers does not use SDPA's GQA option;
        # it repeats K/V to 16 heads before calling the operator.
        cls.key = repeat_kv(compact_key, KEY_VALUE_GROUPS)
        cls.value = repeat_kv(compact_value, KEY_VALUE_GROUPS)

        cls.padding_mask = _make_attention_mask(cls.device, VALID_LENGTHS)
        cls.no_padding_mask = _make_attention_mask(cls.device, (SEQUENCE_LENGTH,) * BATCH_SIZE)
        cls.sdpa_padding_mask = _make_causal_mask(cls.device, cls.padding_mask, "sdpa")
        cls.sdpa_no_padding_mask = _make_causal_mask(cls.device, cls.no_padding_mask, "sdpa")
        cls.eager_padding_mask = _make_causal_mask(cls.device, cls.padding_mask, "eager")
        cls.eager_no_padding_mask = _make_causal_mask(cls.device, cls.no_padding_mask, "eager")

        with torch.no_grad():
            cls.eager_padding_output = _eager_reference(
                cls.query,
                cls.key,
                cls.value,
                cls.eager_padding_mask,
            )
            cls.eager_no_padding_output = _eager_reference(
                cls.query,
                cls.key,
                cls.value,
                cls.eager_no_padding_mask,
            )
        torch.musa.synchronize()

    @classmethod
    def tearDownClass(cls):
        for name in (
            "query",
            "key",
            "value",
            "padding_mask",
            "no_padding_mask",
            "sdpa_padding_mask",
            "sdpa_no_padding_mask",
            "eager_padding_mask",
            "eager_no_padding_mask",
            "eager_padding_output",
            "eager_no_padding_output",
        ):
            if hasattr(cls, name):
                delattr(cls, name)
        torch.musa.empty_cache()

    def _run_backend(self, backend: SDPBackend, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), sdpa_kernel(backend):
            output = F.scaled_dot_product_attention(
                self.query,
                self.key,
                self.value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=HEAD_DIM**-0.5,
            )
        torch.musa.synchronize()
        return output

    def _assert_backend_is_numerically_safe(
        self,
        backend: SDPBackend,
        backend_name: str,
        *,
        additive_mask: bool = False,
    ):
        no_padding_mask = self.eager_no_padding_mask if additive_mask else self.sdpa_no_padding_mask
        padding_mask = self.eager_padding_mask if additive_mask else self.sdpa_padding_mask
        mask_representation = "additive_bfloat16" if additive_mask else "boolean"
        try:
            no_padding_output = self._run_backend(backend, no_padding_mask)
            padding_output = self._run_backend(backend, padding_mask)
        except RuntimeError as error:
            if backend == SDPBackend.FLASH_ATTENTION:
                self.skipTest(f"{backend_name} is unavailable for the production shape: {error}")
            raise

        no_padding_stats = _error_stats(no_padding_output, self.eager_no_padding_output, self.no_padding_mask.bool())
        padding_stats = _error_stats(padding_output, self.eager_padding_output, self.padding_mask.bool())
        fully_masked_rows = (~self.sdpa_padding_mask).all(dim=-1).sum(dim=(1, 2)).cpu().tolist()

        print(
            {
                "backend": backend_name,
                "mask_representation": mask_representation,
                "query_shape": tuple(self.query.shape),
                "key_shape": tuple(self.key.shape),
                "value_shape": tuple(self.value.shape),
                "query_stride": self.query.stride(),
                "mask_shape": tuple(self.sdpa_padding_mask.shape),
                "mask_dtype": str(padding_mask.dtype),
                "valid_lengths": VALID_LENGTHS,
                "fully_masked_rows": fully_masked_rows,
                "no_padding": no_padding_stats,
                "left_padding": padding_stats,
            }
        )

        label = f"{backend_name}/{mask_representation}"
        self.assertTrue(no_padding_stats.all_finite, f"{label} produced non-finite values without padding")
        self.assertLessEqual(no_padding_stats.relative_l2, 0.05, f"{label} has excessive no-padding error")
        self.assertTrue(padding_stats.valid_finite, f"{label} corrupted valid query rows")
        self.assertLessEqual(padding_stats.relative_l2, 0.05, f"{label} has excessive valid-row error")
        self.assertTrue(
            padding_stats.all_finite,
            f"{label} produced {padding_stats.nan_count} NaNs and {padding_stats.inf_count} Infs "
            "for Qwen3.5 left padding",
        )

    def test_math_backend_with_boolean_mask_is_finite_and_matches_eager(self):
        self._assert_backend_is_numerically_safe(SDPBackend.MATH, "math", additive_mask=False)

    def test_math_backend_with_additive_mask_is_finite_and_matches_eager(self):
        self._assert_backend_is_numerically_safe(SDPBackend.MATH, "math", additive_mask=True)

    def test_flash_backend_with_boolean_mask_is_finite_and_matches_eager(self):
        self._assert_backend_is_numerically_safe(SDPBackend.FLASH_ATTENTION, "flash", additive_mask=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
