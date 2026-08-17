import os
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from starVLA.model.modules.vlm import qwen35_musa
from starVLA.model.modules.vlm.qwen35_musa import _fla_causal_conv_adapter


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_v_dim = 4
        self.layer_norm_epsilon = 1e-6
        self.norm = nn.LayerNorm(4)
        self.causal_conv1d_fn = object()
        self.chunk_gated_delta_rule = object()
        self.recurrent_gated_delta_rule = object()


class FakeQwen35Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attention = Qwen3_5GatedDeltaNet()


class Qwen35MusaCompatibilityTest(unittest.TestCase):
    def test_fla_causal_conv_adapter_transposes_layout(self):
        observed = {}

        def fake_fla_conv(**kwargs):
            observed.update(kwargs)
            return kwargs["x"] + 1, None

        adapter = _fla_causal_conv_adapter(fake_fla_conv)
        input_tensor = torch.zeros(2, 5, 7)
        output = adapter(
            input_tensor,
            weight=torch.ones(5, 4),
            activation="silu",
        )

        self.assertEqual(observed["x"].shape, (2, 7, 5))
        self.assertEqual(output.shape, input_tensor.shape)
        torch.testing.assert_close(output, torch.ones_like(output))

    def test_off_policy_rebinds_transformers_auto_fla_path(self):
        class ReferenceNorm(nn.Module):
            def __init__(self, hidden_size, eps):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.eps = eps

        reference_module = types.ModuleType(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        reference_module.Qwen3_5RMSNormGated = ReferenceNorm
        reference_module.torch_chunk_gated_delta_rule = object()
        reference_module.torch_recurrent_gated_delta_rule = object()
        model = FakeQwen35Model()

        with mock.patch.dict(
            sys.modules,
            {
                "transformers.models.qwen3_5.modeling_qwen3_5": reference_module,
            },
        ), mock.patch.object(
            qwen35_musa,
            "_musa_is_available",
            return_value=True,
        ), mock.patch.dict(
            os.environ,
            {"STARVLA_QWEN35_FLA_FASTPATH": "0"},
        ):
            patched = qwen35_musa.configure_qwen35_musa_fla_path(model)

        layer = model.linear_attention
        self.assertEqual(patched, 0)
        self.assertIsNone(layer.causal_conv1d_fn)
        self.assertIs(
            layer.chunk_gated_delta_rule,
            reference_module.torch_chunk_gated_delta_rule,
        )
        self.assertIsInstance(layer.norm, ReferenceNorm)


if __name__ == "__main__":
    unittest.main()
