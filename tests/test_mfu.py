import unittest

from starVLA.training.mfu import (
    QWEN35_MFU_FORMULA_VERSION,
    Qwen35BatchFlopShape,
    Qwen35ModelFlopConfig,
    calculate_per_device_mfu,
    estimate_qwen35_training_flops,
)


class Qwen35MFUTest(unittest.TestCase):
    def setUp(self):
        # Values read from the Qwen3.5-4B checkpoint used by the K8s job.
        self.model = Qwen35ModelFlopConfig(
            text_layer_parameters=3_570_049_536,
            full_attention_layers=8,
            linear_attention_layers=24,
            text_attention_heads=16,
            text_head_dim=256,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            vision_block_parameters=302_309_376,
            vision_patch_parameters=1_573_888,
            vision_merger_parameters=27_271_680,
            vision_layers=24,
            vision_attention_heads=16,
            vision_head_dim=64,
            vision_spatial_merge_size=2,
            action_parameters=65_658_894,
            lm_head_parameters=635_699_200,
        )

    def test_robotwin_qwen35_reference_shape(self):
        result = estimate_qwen35_training_flops(
            self.model,
            Qwen35BatchFlopShape(
                batch_size=1,
                padded_language_sequence_length=291,
                vision_patch_tokens=768,
                num_images=3,
                action_tokens_per_sample=50,
            ),
        )

        self.assertEqual(result["formula_version"], QWEN35_MFU_FORMULA_VERSION)
        self.assertEqual(result["language_tokens_per_device"], 291)
        self.assertEqual(result["vision_merged_tokens_per_device"], 192)
        self.assertEqual(result["action_tokens_per_device"], 50)
        self.assertAlmostEqual(result["estimated_tflops_per_device_step"], 7.854160770152)
        self.assertEqual(result["linear_attention_core_flops"], 76_894_175_232)
        self.assertAlmostEqual(
            result["estimated_tflops_per_device_step"],
            result["estimated_text_tflops_per_device_step"]
            + result["estimated_vision_tflops_per_device_step"]
            + result["estimated_action_tflops_per_device_step"],
        )

    def test_reference_shape_mfu_hand_calculation(self):
        result = calculate_per_device_mfu(
            estimated_tflops_per_device_step=7.854160770152,
            model_time_seconds=0.5,
            peak_tflops_per_device=460.0,
        )

        self.assertAlmostEqual(result["achieved_tflops_per_device"], 15.708321540304)
        self.assertAlmostEqual(result["mfu_percent"], 3.414852508761739)

    def test_mfu_rejects_non_positive_time_and_peak(self):
        with self.assertRaisesRegex(ValueError, "model_time_seconds"):
            calculate_per_device_mfu(7.85, 0.0, 460.0)
        with self.assertRaisesRegex(ValueError, "peak_tflops_per_device"):
            calculate_per_device_mfu(7.85, 0.5, 0.0)

    def test_batch_scaling_uses_local_device_shapes(self):
        one = estimate_qwen35_training_flops(
            self.model,
            Qwen35BatchFlopShape(1, 291, 768, 3, 50),
        )
        two = estimate_qwen35_training_flops(
            self.model,
            Qwen35BatchFlopShape(2, 291, 1536, 6, 50),
        )

        self.assertAlmostEqual(two["total_flops"], 2 * one["total_flops"])

    def test_rejects_patches_without_images(self):
        with self.assertRaisesRegex(ValueError, "num_images"):
            estimate_qwen35_training_flops(
                self.model,
                Qwen35BatchFlopShape(1, 291, 768, 0, 50),
            )


if __name__ == "__main__":
    unittest.main()
