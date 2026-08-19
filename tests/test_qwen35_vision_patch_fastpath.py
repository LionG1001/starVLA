import unittest

import torch
import torch.nn as nn

from starVLA.model.modules.vlm.qwen35_musa_vision_patch import (
    disable_qwen35_musa_vision_patch_fastpath,
    install_qwen35_musa_vision_patch_fastpath,
    qwen35_vision_patch_linear_forward,
)


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        temporal_patch_size: int = 2,
        patch_size: int = 4,
        embed_dim: int = 7,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )


def _run_forward_backward(module, hidden_states, grad_output):
    output = module(hidden_states)
    output.backward(grad_output)
    return (
        output.detach().clone(),
        hidden_states.grad.detach().clone(),
        module.proj.weight.grad.detach().clone(),
        module.proj.bias.grad.detach().clone(),
    )


class Qwen35VisionPatchFastpathTest(unittest.TestCase):
    def test_linear_projection_matches_conv3d_forward_and_backward(self):
        torch.manual_seed(7)
        reference = Qwen3_5VisionPatchEmbed()
        fast = Qwen3_5VisionPatchEmbed()
        fast.load_state_dict(reference.state_dict())
        patch_volume = 3 * 2 * 4 * 4
        reference_input = torch.randn(11, patch_volume, requires_grad=True)
        fast_input = reference_input.detach().clone().requires_grad_(True)
        grad_output = torch.randn(11, 7)

        reference_values = _run_forward_backward(
            reference, reference_input, grad_output
        )
        fast.forward = lambda hidden_states: qwen35_vision_patch_linear_forward(
            fast, hidden_states
        )
        fast_values = _run_forward_backward(fast, fast_input, grad_output)

        for actual, expected in zip(fast_values, reference_values):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_install_is_idempotent_and_disable_restores_conv3d(self):
        module = Qwen3_5VisionPatchEmbed()
        original_function = module.forward.__func__

        self.assertEqual(install_qwen35_musa_vision_patch_fastpath(module), 1)
        self.assertEqual(install_qwen35_musa_vision_patch_fastpath(module), 1)
        self.assertNotEqual(module.forward.__func__, original_function)
        self.assertEqual(disable_qwen35_musa_vision_patch_fastpath(module), 1)
        self.assertEqual(module.forward.__func__, original_function)
        self.assertEqual(disable_qwen35_musa_vision_patch_fastpath(module), 0)

    def test_unsupported_geometry_keeps_conv3d_and_fails_closed(self):
        module = Qwen3_5VisionPatchEmbed()
        module.proj.stride = (1, 4, 4)

        with self.assertRaisesRegex(RuntimeError, "unsupported Conv3D geometry"):
            install_qwen35_musa_vision_patch_fastpath(module)
        self.assertFalse(hasattr(module, "_starvla_qwen35_patch_embed_conv3d_forward"))


if __name__ == "__main__":
    unittest.main()
