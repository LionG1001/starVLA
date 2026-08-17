import unittest

import numpy as np
import torch

from starVLA.model.framework.QwenOFT import Qwenvl_OFT


class QwenOFTPredictionDtypeTest(unittest.TestCase):
    def test_bfloat16_action_can_be_exported_as_float32_numpy(self):
        predicted_actions = torch.tensor([1.25, -0.5], dtype=torch.bfloat16)

        normalized_actions = predicted_actions.detach().float().cpu().numpy()

        self.assertEqual(normalized_actions.dtype, np.float32)
        np.testing.assert_allclose(normalized_actions, [1.25, -0.5])

    def test_action_token_validation_checks_repeated_bpe_encoding(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                if text == "x":
                    return {"input_ids": [7]}
                return {"input_ids": [8]}

        with self.assertRaisesRegex(ValueError, "Repeated action placeholder"):
            Qwenvl_OFT._validate_action_token(FakeTokenizer(), "x", 2)

    def test_action_token_validation_accepts_stable_repetition(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                return {"input_ids": [7] * len(text)}

        token_id = Qwenvl_OFT._validate_action_token(FakeTokenizer(), "x", 3)

        self.assertEqual(token_id, 7)


if __name__ == "__main__":
    unittest.main()
