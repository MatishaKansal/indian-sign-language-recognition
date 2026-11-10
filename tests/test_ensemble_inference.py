import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from src.ensemble_inference import LandmarkPredictor, predict_frame, run_landmark_webcam
from src.landmark_extractor import HAND_FEATURE_DIM, LANDMARK_FEATURE_DIM
from src.landmark_image_ensemble import (
    MOBILENET_WEIGHTS_NAME,
    MOBILENET_LARGE_WEIGHTS_NAME,
    configure_image_finetuning,
    compact_landmark_model,
    image_backbone,
    image_feature_dim,
    image_model,
    image_train_transform,
    landmark_model,
)


def baseline_checkpoint():
    return {
        "model_type": "landmark_mlp",
        "class_to_idx": {"first": 0, "second": 1},
        "model_state_dict": landmark_model(2, input_dim=HAND_FEATURE_DIM + 4).state_dict(),
        "input_is_mirrored": False,
    }


class LandmarkInferenceTests(unittest.TestCase):
    def test_improved_model_finetunes_only_final_mobilenet_stages(self):
        backbone = image_backbone(MOBILENET_WEIGHTS_NAME)
        trainable = configure_image_finetuning(backbone)
        trainable_ids = {id(parameter) for parameter in trainable}
        self.assertTrue(trainable)
        for index, stage in enumerate(backbone.features):
            for parameter in stage.parameters():
                self.assertEqual(parameter.requires_grad, index >= len(backbone.features) - 2)
                self.assertEqual(parameter.requires_grad, id(parameter) in trainable_ids)
        for parameter in backbone.parameters():
            if not any(id(parameter) == id(item) for stage in backbone.features for item in stage.parameters()):
                self.assertFalse(parameter.requires_grad)

    def test_improved_model_augmentation_preserves_image_shape(self):
        transform = image_train_transform(MOBILENET_WEIGHTS_NAME)
        transformed = transform(Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)))
        self.assertEqual(tuple(transformed.shape), (3, 224, 224))

    def test_mobilenet_image_features_match_classifier_input(self):
        backbone = image_backbone(MOBILENET_WEIGHTS_NAME)
        backbone.eval()
        with torch.inference_mode():
            features = backbone(torch.zeros((1, 3, 224, 224)))
        self.assertEqual(features.shape, (1, 576))
        self.assertEqual(image_feature_dim(MOBILENET_WEIGHTS_NAME), 576)
        self.assertEqual(image_model(2, head_type="legacy")[0].in_features, 576)
        with torch.inference_mode():
            logits = image_model(2, head_type="gated_residual")(features)
        self.assertEqual(logits.shape, (1, 2))

    def test_mobilenet_v3_large_image_features_match_classifier_input(self):
        backbone = image_backbone(MOBILENET_LARGE_WEIGHTS_NAME)
        backbone.eval()
        with torch.inference_mode():
            features = backbone(torch.zeros((1, 3, 224, 224)))
        self.assertEqual(features.shape, (1, 960))
        self.assertEqual(image_feature_dim(MOBILENET_LARGE_WEIGHTS_NAME), 960)
        self.assertEqual(image_model(2, 960, "legacy")[0].in_features, 960)
        with torch.inference_mode():
            self.assertEqual(image_model(2, 960)(features).shape, (1, 2))

    def test_baseline_probabilities_and_frame_prediction_match(self):
        torch.manual_seed(3)
        predictor = LandmarkPredictor(baseline_checkpoint())
        self.assertEqual(predictor.models[0][0].in_features, HAND_FEATURE_DIM + 4)
        detection = {
            "features": np.zeros(LANDMARK_FEATURE_DIM, dtype=np.float32),
            "presence": np.array([1, 0], dtype=np.float32),
            "handedness_reliable": np.array([1, 0], dtype=np.float32),
            "positions": np.array([0.2, 0.3, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        inputs = torch.from_numpy(np.concatenate(list(detection.values()))).unsqueeze(0)
        with torch.inference_mode():
            expected = predictor.probabilities(inputs).numpy()[0]
        result = predict_frame(predictor, np.zeros((32, 32, 3), dtype=np.uint8), detection)
        np.testing.assert_allclose(result["probabilities"], expected)
        self.assertAlmostEqual(float(expected.sum()), 1)
        inputs[:, HAND_FEATURE_DIM:LANDMARK_FEATURE_DIM] = 1.0
        with torch.inference_mode():
            unchanged = predictor.probabilities(inputs).numpy()[0]
        np.testing.assert_allclose(unchanged, expected)

    def test_compact_baseline_checkpoint_loads_and_predicts(self):
        torch.manual_seed(7)
        checkpoint = {
            "model_type": "landmark_mlp",
            "class_to_idx": {"first": 0, "second": 1},
            "model_state_dict": compact_landmark_model(
                2, input_dim=HAND_FEATURE_DIM + 4,
            ).state_dict(),
            "input_is_mirrored": False,
        }
        predictor = LandmarkPredictor(checkpoint)
        inputs = torch.zeros((2, LANDMARK_FEATURE_DIM + 12))

        with torch.inference_mode():
            probabilities = predictor.probabilities(inputs)

        self.assertEqual(predictor.models[0][0].out_features, 64)
        self.assertEqual(tuple(probabilities.shape), (2, 2))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))

    def test_invalid_checkpoint_and_inputs_raise(self):
        checkpoint = baseline_checkpoint()
        checkpoint["class_to_idx"] = {"bad": 1}
        with self.assertRaises(ValueError):
            LandmarkPredictor(checkpoint)
        predictor = LandmarkPredictor(baseline_checkpoint())
        with self.assertRaises(ValueError):
            predictor.probabilities(torch.zeros((1, 3)))
        with self.assertRaises(ValueError):
            predict_frame(predictor, np.zeros((2, 2), dtype=np.uint8), {})

    def test_webcam_requires_valid_threshold(self):
        with self.assertRaises(ValueError):
            run_landmark_webcam("unused.pt", confidence_threshold=1.5)

    @patch("src.ensemble_inference.load_landmark_predictor")
    def test_missing_camera_is_reported(self, load):
        load.return_value = LandmarkPredictor(baseline_checkpoint())
        with patch("cv2.VideoCapture") as camera:
            camera.return_value.isOpened.return_value = False
            with self.assertRaisesRegex(RuntimeError, "Cannot open webcam"):
                run_landmark_webcam("unused.pt")
            camera.return_value.release.assert_called_once()

    @patch("src.ensemble_inference.load_landmark_predictor")
    def test_webcam_no_hand_and_cleanup(self, load):
        load.return_value = LandmarkPredictor(baseline_checkpoint())
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        detection = {
            "features": np.zeros(LANDMARK_FEATURE_DIM, dtype=np.float32),
            "presence": np.zeros(2, dtype=np.float32),
            "handedness_reliable": np.zeros(2, dtype=np.float32),
            "positions": np.zeros(8, dtype=np.float32),
            "detected": False,
        }
        extractor = MagicMock()
        extractor.extract.return_value = detection
        extractor.draw.return_value = frame
        with patch("cv2.VideoCapture") as camera, \
             patch("src.landmark_extractor.HandLandmarkExtractor") as extractor_class, \
             patch("cv2.putText") as put_text, \
             patch("cv2.imshow"), patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows") as destroy:
            camera.return_value.isOpened.return_value = True
            camera.return_value.read.return_value = (True, frame)
            extractor_class.return_value.__enter__.return_value = extractor
            run_landmark_webcam("unused.pt")
            self.assertIn("No hands detected", put_text.call_args_list[0].args[1])
            camera.return_value.release.assert_called_once()
            extractor_class.return_value.__exit__.assert_called_once()
            destroy.assert_called_once()

    @patch("src.ensemble_inference.load_landmark_predictor")
    def test_webcam_frame_error_cleans_up(self, load):
        load.return_value = LandmarkPredictor(baseline_checkpoint())
        with patch("cv2.VideoCapture") as camera, \
             patch("src.landmark_extractor.HandLandmarkExtractor") as extractor, \
             patch("cv2.destroyAllWindows") as destroy:
            camera.return_value.isOpened.return_value = True
            camera.return_value.read.return_value = (False, None)
            with self.assertRaisesRegex(RuntimeError, "frame capture failed"):
                run_landmark_webcam("unused.pt")
            camera.return_value.release.assert_called_once()
            extractor.return_value.__exit__.assert_called_once()
            destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
