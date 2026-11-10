"""Inference for the landmark baseline and landmark/image ensemble."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from src.landmark_extractor import LANDMARK_FEATURE_DIM
from src.landmark_image_ensemble import (
    combine_probabilities,
    image_backbone,
    image_feature_dim,
    image_model,
    image_transform,
    landmark_model_from_state,
    landmark_model_logits,
)


class LandmarkPredictor(nn.Module):
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        super().__init__()
        mapping = checkpoint.get("class_to_idx")
        if not isinstance(mapping, dict) or sorted(mapping.values()) != list(range(len(mapping))) or not mapping:
            raise ValueError("Invalid checkpoint class mapping.")
        self.class_to_idx = mapping
        self.model_type = checkpoint.get("model_type")
        self.input_is_mirrored = checkpoint.get("input_is_mirrored", False)
        if not isinstance(self.input_is_mirrored, bool):
            raise ValueError("Invalid checkpoint mirrored-input setting.")
        self.models = nn.ModuleList()
        if self.model_type == "landmark_mlp":
            states = [checkpoint["model_state_dict"]]
        elif self.model_type == "robust_landmark_ensemble":
            states = checkpoint["landmark_states"]
            if not states:
                raise ValueError("Ensemble checkpoint has no landmark models.")
            self.image_weights = checkpoint.get("weights")
            feature_dim = image_feature_dim(self.image_weights)
            self.backbone = image_backbone(self.image_weights)
            self.backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
            self.image_head = image_model(len(mapping), feature_dim, checkpoint.get("image_head", "legacy"))
            self.image_head.load_state_dict(checkpoint["image_model_state_dict"], strict=True)
            mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]
            if mean.shape != (feature_dim,) or std.shape != (feature_dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Invalid image feature normalization.")
            self.register_buffer("feature_mean", mean)
            self.register_buffer("feature_std", std)
            self.image_weight = float(checkpoint["image_weight"])
            self.fallback = checkpoint["fallback"]
            if not 0 <= self.image_weight <= 1 or not isinstance(self.fallback, bool):
                raise ValueError("Invalid ensemble settings.")
        else:
            raise ValueError(f"Unsupported checkpoint type: {self.model_type!r}")
        for state in states:
            self.models.append(landmark_model_from_state(len(mapping), state))
        self.eval()

    def probabilities(self, inputs: torch.Tensor, images: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != LANDMARK_FEATURE_DIM + 12 or not torch.isfinite(inputs).all():
            raise ValueError("Invalid landmark inputs.")
        landmarks = torch.stack([landmark_model_logits(model, inputs).softmax(1) for model in self.models]).mean(0)
        if self.model_type == "landmark_mlp":
            return landmarks
        if images is None or images.ndim != 4 or images.shape[0] != inputs.shape[0]:
            raise ValueError("The ensemble requires one preprocessed image per landmark input.")
        images = self.image_head((self.backbone(images) - self.feature_mean) / self.feature_std).softmax(1)
        presence = inputs[:, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 2].sum(1) > 0
        return combine_probabilities(landmarks, images, presence, self.image_weight, self.fallback)


def load_landmark_predictor(path: str | Path, device: torch.device | None = None) -> LandmarkPredictor:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return LandmarkPredictor(checkpoint).to(device or torch.device("cpu")).eval()


def predict_frame(model: LandmarkPredictor, rgb: np.ndarray, detection: dict[str, Any]) -> dict[str, Any]:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB input with shape (height, width, 3).")
    values = np.concatenate([
        detection["features"],
        detection["presence"],
        detection["handedness_reliable"],
        detection["positions"],
    ]).astype(np.float32)
    device = next(model.parameters()).device
    inputs = torch.from_numpy(values).unsqueeze(0).to(device)
    image = None
    if model.model_type == "robust_landmark_ensemble":
        image = image_transform(model.image_weights)(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model.probabilities(inputs, image)[0].cpu().numpy()
    index = int(probabilities.argmax())
    names = {value: key for key, value in model.class_to_idx.items()}
    return {"label": names[index], "confidence": float(probabilities[index]), "probabilities": probabilities}


def run_landmark_webcam(
    checkpoint_path: str | Path, camera_index: int = 0, confidence_threshold: float = 0.5,
) -> None:
    """Run the selected model on local camera frames; press q or Escape to exit."""
    import cv2
    from src.landmark_extractor import HandLandmarkExtractor

    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    model = load_landmark_predictor(checkpoint_path)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open webcam {camera_index}; check camera permissions.")
    try:
        with HandLandmarkExtractor(static_image_mode=False, input_is_mirrored=model.input_is_mirrored) as extractor:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Webcam frame capture failed.")
                if model.input_is_mirrored:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detection = extractor.extract(rgb)
                prediction = predict_frame(model, rgb, detection)
                if not detection["detected"] and model.model_type == "landmark_mlp":
                    label = "No hands detected - no reliable landmark prediction"
                else:
                    note = "No hands | " if not detection["detected"] else ""
                    if prediction["confidence"] < confidence_threshold:
                        note += "Low confidence | "
                    label = f"{note}{prediction['label']} ({prediction['confidence']:.1%})"
                shown = cv2.cvtColor(extractor.draw(rgb, detection), cv2.COLOR_RGB2BGR)
                cv2.putText(shown, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
                cv2.putText(shown, "q / Esc: exit", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(f"ISL recognition - {model.model_type}", shown)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
