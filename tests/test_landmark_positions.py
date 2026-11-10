import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.landmark_extractor import (
    BODY_FEATURE_DIM,
    BODY_FEATURE_OFFSET,
    HAND_FEATURE_DIM,
    LANDMARK_CACHE_FORMAT_VERSION,
    LANDMARK_FEATURE_DIM,
    SHAPE_FEATURES_PER_HAND,
    HandLandmarkExtractor,
    body_relative_features,
    hand_shape_features,
)
from src.robust_landmark_ensemble import add_landmark_noise


def make_hand(x, y, scale):
    return SimpleNamespace(landmark=[
        SimpleNamespace(x=x + i * scale, y=y, z=i * scale / 10) for i in range(21)
    ])


def category(name, score):
    return SimpleNamespace(classification=[SimpleNamespace(label=name, score=score)])


class LandmarkPositionTests(unittest.TestCase):
    def test_positions_follow_handedness_and_duplicate_reassignment(self):
        result = SimpleNamespace(
            multi_hand_landmarks=[make_hand(0.1, 0.2, 0.01), make_hand(0.6, 0.7, 0.005)],
            multi_handedness=[category("Left", 0.8), category("Left", 0.9)],
        )
        extractor = HandLandmarkExtractor.__new__(HandLandmarkExtractor)
        extractor._hands = SimpleNamespace(process=lambda image: result)
        extractor._pose = SimpleNamespace(process=lambda image: SimpleNamespace(pose_landmarks=None))
        extractor.input_is_mirrored = False
        extractor.handedness_confidence_threshold = 0.7
        output = extractor.extract(np.zeros((10, 10, 3), dtype=np.uint8))
        positions = output["positions"].reshape(2, 4)
        np.testing.assert_allclose(positions[0], [0.1, 0.2, 0, 0.2], atol=1e-6)
        np.testing.assert_allclose(positions[1], [0.6, 0.7, 0, 0.1], atol=1e-6)
        np.testing.assert_array_equal(output["presence"], [1, 1])
        np.testing.assert_array_equal(output["handedness_reliable"], [0, 1])
        np.testing.assert_allclose(output["features"][:HAND_FEATURE_DIM].reshape(2, 21, 3)[:, 0], 0)
        self.assertEqual(len(output["features"]), LANDMARK_FEATURE_DIM)
        self.assertEqual(output["features"][HAND_FEATURE_DIM:BODY_FEATURE_OFFSET].size, 2 * SHAPE_FEATURES_PER_HAND)

    def test_missing_hands_remain_zero(self):
        extractor = HandLandmarkExtractor.__new__(HandLandmarkExtractor)
        extractor._hands = SimpleNamespace(process=lambda image: SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None))
        extractor._pose = SimpleNamespace(process=lambda image: SimpleNamespace(pose_landmarks=None))
        extractor.input_is_mirrored = False
        extractor.handedness_confidence_threshold = 0.7
        output = extractor.extract(np.zeros((10, 10, 3), dtype=np.uint8))
        np.testing.assert_array_equal(output["positions"], np.zeros(8))
        tensor = torch.zeros((2, LANDMARK_FEATURE_DIM + 12))
        self.assertTrue(torch.equal(add_landmark_noise(tensor, 0.1), tensor))

    def test_body_relative_features_use_shoulder_width_and_validity_mask(self):
        hand_norm = np.zeros((2, 21, 3), dtype=np.float32)
        positions = np.zeros((2, 4), dtype=np.float32)
        positions[0] = [0.5, 0.5, 0.0, 0.1]
        presence = np.array([1.0, 0.0], dtype=np.float32)
        pose_xy = np.zeros((33, 2), dtype=np.float32)
        pose_xy[0] = [0.5, 0.1]
        pose_xy[11] = [0.25, 0.25]
        pose_xy[12] = [0.75, 0.25]
        visibility = np.ones(33, dtype=np.float32)

        body = body_relative_features(
            hand_norm, positions, presence, pose_xy, visibility, width=100, height=100,
        )

        self.assertEqual(len(body), BODY_FEATURE_DIM)
        np.testing.assert_allclose(body[1], 0.8, atol=1e-6)
        np.testing.assert_allclose(body[11], 0.5, atol=1e-6)
        self.assertEqual(body[-1], 1.0)
        self.assertTrue(np.all(body[21:42] == 0))

        visibility[11] = 0.1
        invalid = body_relative_features(
            hand_norm, positions, presence, pose_xy, visibility, width=100, height=100,
        )
        self.assertTrue(np.all(invalid == 0))
        visibility[11] = 1.0
        visibility[0] = 0.1
        invalid = body_relative_features(
            hand_norm, positions, presence, pose_xy, visibility, width=100, height=100,
        )
        self.assertTrue(np.all(invalid == 0))

    def test_hand_shape_features_are_rotation_and_scale_invariant(self):
        points = np.zeros((2, 21, 3), dtype=np.float32)
        points[0, :, :2] = np.random.default_rng(42).normal(size=(21, 2))
        theta = 0.37
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        transformed = points.copy()
        transformed[0, :, :2] = points[0, :, :2] @ rotation.T * 3.5

        original = hand_shape_features(points)
        rotated_scaled = hand_shape_features(transformed)

        self.assertEqual(original.shape, (2 * SHAPE_FEATURES_PER_HAND,))
        np.testing.assert_allclose(original, rotated_scaled, atol=3e-6)
        np.testing.assert_array_equal(original[SHAPE_FEATURES_PER_HAND:], np.zeros(SHAPE_FEATURES_PER_HAND))

    def test_landmark_noise_preserves_masks_and_body_validity(self):
        torch.manual_seed(4)
        inputs = torch.zeros((2, LANDMARK_FEATURE_DIM + 12))
        inputs[0, :HAND_FEATURE_DIM] = 1
        inputs[0, HAND_FEATURE_DIM:HAND_FEATURE_DIM + SHAPE_FEATURES_PER_HAND] = 0.5
        inputs[0, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 2] = torch.tensor([1.0, 0.0])
        inputs[0, LANDMARK_FEATURE_DIM - 1] = 1.0
        inputs[0, LANDMARK_FEATURE_DIM + 2:LANDMARK_FEATURE_DIM + 4] = torch.tensor([1.0, 0.0])

        noisy = add_landmark_noise(inputs, 0.1)

        self.assertTrue(torch.equal(noisy[:, LANDMARK_FEATURE_DIM:], inputs[:, LANDMARK_FEATURE_DIM:]))
        self.assertEqual(noisy[0, LANDMARK_FEATURE_DIM - 1].item(), 1.0)
        self.assertFalse(torch.equal(noisy[0, HAND_FEATURE_DIM:HAND_FEATURE_DIM + SHAPE_FEATURES_PER_HAND], inputs[0, HAND_FEATURE_DIM:HAND_FEATURE_DIM + SHAPE_FEATURES_PER_HAND]))
        self.assertTrue(torch.equal(noisy[0, HAND_FEATURE_DIM + SHAPE_FEATURES_PER_HAND:BODY_FEATURE_OFFSET], inputs[0, HAND_FEATURE_DIM + SHAPE_FEATURES_PER_HAND:BODY_FEATURE_OFFSET]))
        self.assertFalse(torch.equal(noisy[0, :HAND_FEATURE_DIM], inputs[0, :HAND_FEATURE_DIM]))
        self.assertTrue(torch.equal(noisy[1], inputs[1]))
        with self.assertRaises(ValueError):
            add_landmark_noise(torch.zeros((2, 126)), 0.1)


if __name__ == "__main__":
    unittest.main()
