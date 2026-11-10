import unittest

import torch

from src.improved_signer_cv import _landmark_feature_variants
from src.landmark_extractor import (
    HAND_FEATURE_DIM,
    LANDMARK_FEATURE_DIM,
    SHAPE_FEATURE_DIM,
)
from src.landmark_image_ensemble import landmark_model, landmark_model_logits
from src.signer_cross_validation import fold_indices


class SignerFoldTests(unittest.TestCase):
    def test_feature_ablation_inputs_keep_expected_blocks_and_masks(self):
        inputs = torch.arange(2 * (LANDMARK_FEATURE_DIM + 12), dtype=torch.float32).reshape(2, -1)

        variants = _landmark_feature_variants(inputs)

        self.assertEqual(variants["hand_only"].shape, (2, HAND_FEATURE_DIM + 4))
        self.assertEqual(variants["hand_only_position"].shape, (2, HAND_FEATURE_DIM + 12))
        self.assertEqual(variants["hand_plus_shape"].shape, (2, HAND_FEATURE_DIM + SHAPE_FEATURE_DIM + 4))
        self.assertEqual(variants["hand_plus_shape_position"].shape, (2, HAND_FEATURE_DIM + SHAPE_FEATURE_DIM + 12))
        self.assertTrue(torch.equal(variants["hand_only"][:, :HAND_FEATURE_DIM], inputs[:, :HAND_FEATURE_DIM]))
        self.assertTrue(torch.equal(variants["hand_only"][:, -4:], inputs[:, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 4]))
        self.assertTrue(torch.equal(
            variants["hand_plus_shape"][:, HAND_FEATURE_DIM:HAND_FEATURE_DIM + SHAPE_FEATURE_DIM],
            inputs[:, HAND_FEATURE_DIM:HAND_FEATURE_DIM + SHAPE_FEATURE_DIM],
        ))
        self.assertTrue(torch.equal(variants["hand_shape_pose"], inputs[:, :LANDMARK_FEATURE_DIM + 4]))
        self.assertTrue(torch.equal(variants["hand_shape_pose_position"], inputs))

    def test_feature_ablation_rejects_wrong_input_width(self):
        with self.assertRaises(ValueError):
            _landmark_feature_variants(torch.zeros((2, LANDMARK_FEATURE_DIM + 4)))

    def test_feature_variants_route_to_matching_model_dimensions(self):
        inputs = torch.randn(3, LANDMARK_FEATURE_DIM + 12)

        for values in _landmark_feature_variants(inputs).values():
            model = landmark_model(4, input_dim=values.shape[1])
            self.assertEqual(landmark_model_logits(model, inputs).shape, (3, 4))

    def test_excludes_held_signer_and_duplicate_groups(self):
        rows = [
            {"signer_id": "a", "duplicate_group": "shared"},
            {"signer_id": "b", "duplicate_group": "shared"},
            {"signer_id": "b", "duplicate_group": "independent"},
            {"signer_id": "c", "duplicate_group": "other"},
        ]
        train, validation = fold_indices(rows, "a")
        self.assertEqual(validation, [0])
        self.assertEqual(train, [2, 3])

    def test_rejects_missing_signer_or_empty_training(self):
        rows = [{"signer_id": "a", "duplicate_group": "shared"}]
        with self.assertRaises(ValueError):
            fold_indices(rows, "missing")
        with self.assertRaises(ValueError):
            fold_indices(rows, "a")


if __name__ == "__main__":
    unittest.main()
