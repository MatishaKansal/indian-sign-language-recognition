"""MediaPipe hand + body landmark extraction, normalization, and cache writing.

Feature vector layout:
  [0:126]    two hands x 21 landmarks x (x, y, z), wrist-centered and scale-normalized
  [126:184]  per-hand fingertip distances and joint angles
  [184:230]  body-relative features from MediaPipe Pose
The presence / handedness masks are appended by the callers exactly as before.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
HAND_COUNT = 2
HAND_FEATURE_DIM = HAND_COUNT * LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK  # 126
FINGER_TIP_INDICES = (4, 8, 12, 16, 20)
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
FINGERTIP_DISTANCE_COUNT = len(FINGER_TIP_INDICES) * (len(FINGER_TIP_INDICES) - 1) // 2
FINGER_JOINT_ANGLE_COUNT = sum(len(chain) - 2 for chain in FINGER_CHAINS)
THUMB_RELATIVE_ANGLE_COUNT = len(FINGER_TIP_INDICES) - 1
SHAPE_FEATURES_PER_HAND = FINGERTIP_DISTANCE_COUNT + FINGER_JOINT_ANGLE_COUNT + THUMB_RELATIVE_ANGLE_COUNT
SHAPE_FEATURE_DIM = HAND_COUNT * SHAPE_FEATURES_PER_HAND  # 58

# Hand points used for body-relative geometry: wrist, thumb tip, index tip, middle tip, pinky tip.
KEY_POINTS = (0, 4, 8, 12, 20)
PER_HAND_BODY_DIM = len(KEY_POINTS) * 2 * 2 + 1  # 5 pts x 2 references x (dx, dy) + hand size = 21
BODY_FEATURE_DIM = HAND_COUNT * PER_HAND_BODY_DIM + 3 + 1  # + hand-to-hand (dx, dy, dist) + body-valid flag = 46
BODY_FEATURE_OFFSET = HAND_FEATURE_DIM + SHAPE_FEATURE_DIM
LANDMARK_FEATURE_DIM = BODY_FEATURE_OFFSET + BODY_FEATURE_DIM  # 230
LANDMARK_CACHE_FORMAT_VERSION = 4

POSE_NOSE, POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 0, 11, 12
MIN_SHOULDER_VISIBILITY = 0.5


def hand_shape_features(hand_norm: np.ndarray) -> np.ndarray:
    """Return scale- and rotation-invariant fingertip distances and finger angles."""
    if hand_norm.shape != (HAND_COUNT, LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK):
        raise ValueError(f"Expected normalized hand landmarks with shape {(HAND_COUNT, LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK)}.")
    features = np.zeros((HAND_COUNT, SHAPE_FEATURES_PER_HAND), dtype=np.float32)
    for slot, points in enumerate(hand_norm[:, :, :2]):
        points = points - points[0]
        scale = float(np.max(np.linalg.norm(points, axis=1)))
        if np.isfinite(scale) and scale > 1e-8:
            points = points / scale
        tips = points[list(FINGER_TIP_INDICES)]
        distances = [
            float(np.linalg.norm(tips[left] - tips[right]))
            for left in range(len(FINGER_TIP_INDICES))
            for right in range(left + 1, len(FINGER_TIP_INDICES))
        ]
        angles = []
        for chain in FINGER_CHAINS:
            for index in range(1, len(chain) - 1):
                previous, joint, following = points[list((chain[index - 1], chain[index], chain[index + 1]))]
                first, second = previous - joint, following - joint
                denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
                cosine = float(np.dot(first, second) / denominator) if denominator > 1e-8 else 1.0
                angles.append(float(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi))
        thumb = tips[0] - points[0]
        thumb_angles = []
        for tip in tips[1:]:
            other = tip - points[0]
            denominator = float(np.linalg.norm(thumb) * np.linalg.norm(other))
            thumb_angles.append(float(np.dot(thumb, other) / denominator) if denominator > 1e-8 else 0.0)
        features[slot] = np.asarray(distances + angles + thumb_angles, dtype=np.float32)
    return features.reshape(-1)


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> int:
    """Return a simple 64-bit average hash for duplicate screening."""
    resized = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32).reshape(-1)
    threshold = float(pixels.mean())
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= threshold)
    return value


def body_relative_features(
    hand_norm: np.ndarray,
    positions: np.ndarray,
    presence: np.ndarray,
    pose_xy: np.ndarray | None,
    pose_visibility: np.ndarray | None,
    width: int,
    height: int,
) -> np.ndarray:
    """Hand geometry expressed relative to the face and shoulders, in shoulder-width units.

    hand_norm : (2, 21, 3) wrist-centered, scale-normalized hand landmarks (slot-ordered)
    positions : (2, 4) per slot [wrist_x, wrist_y, wrist_z, scale] in MediaPipe image-normalized units
    presence  : (2,) hand presence mask
    pose_xy   : (33, 2) MediaPipe Pose landmarks in image-normalized units, or None if no body found
    Returns BODY_FEATURE_DIM floats; all zeros if the nose or shoulders are not reliably visible.
    """
    out = np.zeros(BODY_FEATURE_DIM, dtype=np.float32)
    if pose_xy is None or pose_visibility is None:
        return out
    wh = np.array([width, height], dtype=np.float32)
    left, right = pose_xy[POSE_LEFT_SHOULDER] * wh, pose_xy[POSE_RIGHT_SHOULDER] * wh
    visible = min(
        pose_visibility[POSE_NOSE],
        pose_visibility[POSE_LEFT_SHOULDER],
        pose_visibility[POSE_RIGHT_SHOULDER],
    ) >= MIN_SHOULDER_VISIBILITY
    shoulder_width = float(np.linalg.norm(left - right))
    if not visible or shoulder_width < 1e-3 * max(width, height):
        return out
    nose = pose_xy[POSE_NOSE] * wh
    mid = (left + right) / 2.0
    wrists = np.zeros((HAND_COUNT, 2), dtype=np.float32)
    for slot in range(HAND_COUNT):
        if not presence[slot]:
            continue
        scale = max(float(positions[slot, 3]), 1e-6)
        wrist_n = positions[slot, :2]
        points_px = (hand_norm[slot, :, :2] * scale + wrist_n) * wh  # all 21 landmarks, pixels
        wrists[slot] = points_px[0]
        key = points_px[list(KEY_POINTS)]
        block = np.concatenate([
            ((key - nose) / shoulder_width).reshape(-1),
            ((key - mid) / shoulder_width).reshape(-1),
            [np.max(np.linalg.norm(points_px - points_px[0], axis=1)) / shoulder_width],
        ])
        start = slot * PER_HAND_BODY_DIM
        out[start:start + PER_HAND_BODY_DIM] = block
    if presence[0] and presence[1]:
        delta = (wrists[1] - wrists[0]) / shoulder_width
        base = HAND_COUNT * PER_HAND_BODY_DIM
        out[base:base + 3] = [delta[0], delta[1], float(np.linalg.norm(delta))]
    out[-1] = 1.0
    return np.clip(out, -6.0, 6.0)


class HandLandmarkExtractor:
    """Extract two-hand landmarks, body-relative features, and presence/handedness masks."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = True,
        input_is_mirrored: bool = False,
        handedness_confidence_threshold: float = 0.7,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise ImportError("MediaPipe is required. Install the project requirements first.") from error
        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "hands") or not hasattr(solutions, "pose"):
            raise RuntimeError(
                "This extractor requires the MediaPipe Solutions Hands and Pose APIs. "
                "Install a compatible MediaPipe 0.10.x release."
            )
        self._hands = solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose = solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._draw = solutions.drawing_utils
        self._connections = solutions.hands.HAND_CONNECTIONS
        self.input_is_mirrored = input_is_mirrored
        self.handedness_confidence_threshold = handedness_confidence_threshold

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
        self._pose.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        """Wrist-center and scale one hand's coordinates; preserve relative geometry."""
        centered = points.astype(np.float32, copy=True)
        centered -= centered[0:1, :]
        scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        centered /= scale
        return centered

    def extract(self, rgb_image: np.ndarray) -> dict[str, Any]:
        """Return the feature vector, presence mask and handedness IDs."""
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or rgb_image.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB input with shape (height, width, 3).")
        contiguous = np.ascontiguousarray(rgb_image)
        height, width = contiguous.shape[:2]
        results = self._hands.process(contiguous)
        pose_detector = getattr(self, "_pose", None)
        pose_results = pose_detector.process(contiguous) if pose_detector is not None else None
        features = np.zeros((HAND_COUNT, LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK), dtype=np.float32)
        presence = np.zeros(HAND_COUNT, dtype=np.float32)
        handedness = np.zeros((HAND_COUNT, 2), dtype=np.float32)
        handedness_reliable = np.zeros(HAND_COUNT, dtype=np.float32)
        positions = np.zeros((HAND_COUNT, 4), dtype=np.float32)
        assigned: dict[int, tuple[float, np.ndarray, np.ndarray, bool]] = {}
        unassigned: list[tuple[np.ndarray, np.ndarray]] = []
        landmarks_list = results.multi_hand_landmarks or []
        handedness_list = results.multi_handedness or []
        for index, landmarks in enumerate(landmarks_list[:HAND_COUNT]):
            points = np.asarray([[item.x, item.y, item.z] for item in landmarks.landmark], dtype=np.float32)
            if points.shape != (LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK):
                continue
            scale = float(np.max(np.linalg.norm(points[:, :2] - points[0:1, :2], axis=1)))
            position = np.array([*points[0], scale], dtype=np.float32)
            points = self._normalize(points)
            category = None
            if index < len(handedness_list) and handedness_list[index].classification:
                category = handedness_list[index].classification[0]
            name = category.label.casefold() if category is not None else ""
            if not self.input_is_mirrored:
                name = "right" if name == "left" else "left" if name == "right" else name
            if name not in ("left", "right"):
                unassigned.append((points, position))
                continue
            hand_slot = 0 if name == "left" else 1
            confidence = float(category.score)
            if hand_slot in assigned:
                previous = assigned[hand_slot]
                if confidence > previous[0]:
                    unassigned.append((previous[1], previous[2]))
                    assigned[hand_slot] = (confidence, points, position, True)
                else:
                    unassigned.append((points, position))
            else:
                assigned[hand_slot] = (confidence, points, position, True)
        for points, position in unassigned:
            open_slots = [slot for slot in range(HAND_COUNT) if slot not in assigned]
            if not open_slots:
                break
            assigned[open_slots[0]] = (0.0, points, position, False)
        for hand_slot, (confidence, points, position, has_handedness) in assigned.items():
            features[hand_slot] = points
            positions[hand_slot] = position
            presence[hand_slot] = 1.0
            if has_handedness:
                handedness[hand_slot, hand_slot] = confidence
                handedness_reliable[hand_slot] = float(confidence >= self.handedness_confidence_threshold)

        pose_xy = pose_visibility = None
        if pose_results is not None and pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark
            pose_xy = np.asarray([[item.x, item.y] for item in pose_landmarks], dtype=np.float32)
            pose_visibility = np.asarray([item.visibility for item in pose_landmarks], dtype=np.float32)
        shape = hand_shape_features(features)
        body = body_relative_features(features, positions, presence, pose_xy, pose_visibility, width, height)
        return {
            "features": np.concatenate([features.reshape(-1), shape, body]).astype(np.float32),
            "positions": positions.reshape(-1),
            "presence": presence,
            "handedness": handedness.reshape(-1),
            "handedness_reliable": handedness_reliable,
            "detected": bool(presence.any()),
            "annotated": results,
        }

    def draw(self, rgb_image: np.ndarray, detection: dict[str, Any]) -> np.ndarray:
        """Return a copy with MediaPipe connections drawn for visual validation."""
        annotated = rgb_image.copy()
        results = detection.get("annotated")
        if results is not None and results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self._draw.draw_landmarks(annotated, landmarks, self._connections)
        return annotated


def extract_landmark_cache(
    records: list[dict[str, Any]],
    output_path: str | Path,
    min_detection_confidence: float = 0.5,
    input_is_mirrored: bool = False,
) -> dict[str, Any]:
    """Extract landmarks for indexed images, retaining label/path alignment."""
    items: dict[str, dict[str, Any]] = {}
    with HandLandmarkExtractor(
        min_detection_confidence=min_detection_confidence,
        input_is_mirrored=input_is_mirrored,
    ) as extractor:
        for index, record in enumerate(records, start=1):
            try:
                with Image.open(record["path"]) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                result = extractor.extract(rgb)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                LOGGER.exception("Landmark extraction failed for %s", record["relative_path"])
                raise RuntimeError(f"Could not extract landmarks for {record['relative_path']}: {error}") from error
            items[record["relative_path"]] = {
                "features": result["features"].astype(float).tolist(),
                "positions": result["positions"].astype(float).tolist(),
                "presence": result["presence"].astype(float).tolist(),
                "handedness": result["handedness"].astype(float).tolist(),
                "handedness_reliable": result["handedness_reliable"].astype(float).tolist(),
                "label": record["label"],
                "label_idx": record["label_idx"],
                "sha256": record["sha256"],
            }
            if index % 100 == 0:
                LOGGER.info("Extracted landmarks for %d/%d images.", index, len(records))
    payload = {
        "format_version": LANDMARK_CACHE_FORMAT_VERSION,
        "landmarks_per_hand": LANDMARKS_PER_HAND,
        "hand_count": HAND_COUNT,
        "hand_features_dim": HAND_FEATURE_DIM,
        "shape_features_dim": SHAPE_FEATURE_DIM,
        "body_features_offset": BODY_FEATURE_OFFSET,
        "feature_dim": LANDMARK_FEATURE_DIM,
        "input_is_mirrored": input_is_mirrored,
        "items": items,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(payload, handle)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, target)
    return payload
