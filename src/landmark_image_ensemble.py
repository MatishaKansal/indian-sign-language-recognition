"""CPU-friendly validation-selected ensemble of landmark and image neural networks."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as vision_transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

from src.config import project_path
from src.dataset import check_checkpoint_split, load_split_manifest, split_fingerprint, validate_split_groups
from src.landmark_extractor import (
    BODY_FEATURE_DIM,
    BODY_FEATURE_OFFSET,
    HAND_FEATURE_DIM,
    LANDMARK_CACHE_FORMAT_VERSION,
    LANDMARK_FEATURE_DIM,
    SHAPE_FEATURE_DIM,
)
from src.utils import set_seed

MOBILENET_WEIGHTS_NAME = "MobileNet_V3_Small_Weights.DEFAULT"
MOBILENET_LARGE_WEIGHTS_NAME = "MobileNet_V3_Large_Weights.DEFAULT"
BACKBONE_WEIGHTS = {
    "mobilenet_v3_small": MOBILENET_WEIGHTS_NAME,
    "mobilenet_v3_large": MOBILENET_LARGE_WEIGHTS_NAME,
}
IMAGE_FEATURE_DIMS = {
    MOBILENET_WEIGHTS_NAME: 576,
    MOBILENET_LARGE_WEIGHTS_NAME: 960,
}


def image_weights_name(backbone_name: str) -> str:
    try:
        return BACKBONE_WEIGHTS[backbone_name]
    except KeyError as error:
        raise ValueError(f"Unsupported image backbone: {backbone_name!r}") from error


def image_feature_dim(weights_name: str) -> int:
    try:
        return IMAGE_FEATURE_DIMS[weights_name]
    except KeyError as error:
        raise ValueError(f"Unsupported image model weights: {weights_name!r}") from error


def image_backbone(weights_name: str = MOBILENET_WEIGHTS_NAME, pretrained: bool = False) -> nn.Module:
    if weights_name == MOBILENET_WEIGHTS_NAME:
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        backbone.classifier = nn.Identity()
        return backbone
    if weights_name == MOBILENET_LARGE_WEIGHTS_NAME:
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        backbone.classifier = nn.Identity()
        return backbone
    image_feature_dim(weights_name)
    raise AssertionError("Unreachable image model configuration.")


def image_transform(weights_name: str = MOBILENET_WEIGHTS_NAME) -> Any:
    if weights_name == MOBILENET_WEIGHTS_NAME:
        return MobileNet_V3_Small_Weights.DEFAULT.transforms()
    if weights_name == MOBILENET_LARGE_WEIGHTS_NAME:
        return MobileNet_V3_Large_Weights.DEFAULT.transforms()
    image_feature_dim(weights_name)
    raise AssertionError("Unreachable image model configuration.")


class ImageRows(Dataset):
    """Load image rows in manifest order for ImageNet feature extraction."""

    def __init__(self, rows: list[dict[str, Any]], transform: Any) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.rows[index]["path"]
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("RGB"))
        except OSError as error:
            raise RuntimeError(f"Unable to read dataset image: {path}") from error


class LabeledImageRows(ImageRows):
    """Load augmented training images and their manifest labels."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return super().__getitem__(index), int(self.rows[index]["label_idx"])


def image_train_transform(weights_name: str = MOBILENET_WEIGHTS_NAME) -> Any:
    """Apply conservative, orientation-preserving augmentation for sign images."""
    normalization = image_transform(weights_name)
    return vision_transforms.Compose(
        [
            vision_transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            vision_transforms.RandomAffine(
                degrees=8,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                fill=(0, 0, 0),
            ),
            vision_transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean=normalization.mean, std=normalization.std),
        ]
    )


def configure_image_finetuning(backbone: nn.Module) -> list[nn.Parameter]:
    """Unfreeze only MobileNet's final feature block and projection."""
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    features = getattr(backbone, "features", None)
    if features is None or len(features) < 2:
        raise ValueError("Image backbone does not expose the expected feature stages.")
    trainable = []
    for layer in features[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return trainable


def landmark_model(num_classes: int, input_dim: int = LANDMARK_FEATURE_DIM + 12) -> nn.Module:
    """Build the landmark MLP, defaulting to the full feature set used by the improved model."""
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, num_classes),
    )


def landmark_model_from_state(num_classes: int, state: dict[str, torch.Tensor]) -> nn.Module:
    """Load a hand-only baseline or one of the supported improved landmark variants."""
    first_layer = state.get("0.weight")
    valid_dims = {
        HAND_FEATURE_DIM + 4,
        HAND_FEATURE_DIM + 12,
        HAND_FEATURE_DIM + 4 + 46,
        HAND_FEATURE_DIM + 12 + 46,
        HAND_FEATURE_DIM + 58 + 4,
        HAND_FEATURE_DIM + 58 + 12,
        LANDMARK_FEATURE_DIM + 4,
        LANDMARK_FEATURE_DIM + 12,
    }
    if first_layer is None or first_layer.ndim != 2 or first_layer.shape[1] not in valid_dims:
        raise ValueError("Landmark checkpoint has an unsupported input dimension.")
    model = landmark_model(num_classes, input_dim=first_layer.shape[1])
    model.load_state_dict(state, strict=True)
    return model


def landmark_feature_variant(inputs: torch.Tensor, variant: str) -> torch.Tensor:
    """Select a landmark feature ablation from the current feature layout."""
    full_width = LANDMARK_FEATURE_DIM + 12
    if inputs.ndim != 2 or inputs.shape[1] != full_width:
        raise ValueError(f"Expected landmark inputs with shape (batch, {full_width}).")
    features = inputs[:, :LANDMARK_FEATURE_DIM]
    masks = inputs[:, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 4]
    positions = inputs[:, LANDMARK_FEATURE_DIM + 4:]
    hands = features[:, :HAND_FEATURE_DIM]
    shape = features[:, HAND_FEATURE_DIM:BODY_FEATURE_OFFSET]
    full = features
    variants = {
        "hand_only": torch.cat((hands, masks), dim=1),
        "hand_only_position": torch.cat((hands, positions, masks), dim=1),
        "hand_plus_shape": torch.cat((hands, shape, masks), dim=1),
        "hand_plus_shape_position": torch.cat((hands, shape, positions, masks), dim=1),
        "hand_shape_pose": torch.cat((full, masks), dim=1),
        "hand_shape_pose_position": inputs,
    }
    try:
        return variants[variant]
    except KeyError as error:
        raise ValueError(f"Unsupported landmark feature variant: {variant!r}") from error


def landmark_model_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Map the current feature layout to a checkpoint's supported landmark variant."""
    full_width = LANDMARK_FEATURE_DIM + 12
    if inputs.ndim != 2 or inputs.shape[1] != full_width:
        raise ValueError(f"Expected landmark inputs with shape (batch, {full_width}).")
    input_dim = model[0].in_features
    dimensions = {
        HAND_FEATURE_DIM + 4: "hand_only",
        HAND_FEATURE_DIM + 12: "hand_only_position",
        HAND_FEATURE_DIM + 4 + BODY_FEATURE_DIM: "hand_only_pose_legacy",
        HAND_FEATURE_DIM + 12 + BODY_FEATURE_DIM: "hand_only_pose_position_legacy",
        HAND_FEATURE_DIM + SHAPE_FEATURE_DIM + 4: "hand_plus_shape",
        HAND_FEATURE_DIM + SHAPE_FEATURE_DIM + 12: "hand_plus_shape_position",
        LANDMARK_FEATURE_DIM + 4: "hand_shape_pose",
        LANDMARK_FEATURE_DIM + 12: "hand_shape_pose_position",
    }
    variant = dimensions.get(input_dim)
    if variant in ("hand_only_pose_legacy", "hand_only_pose_position_legacy"):
        features = inputs[:, :LANDMARK_FEATURE_DIM]
        body = features[:, BODY_FEATURE_OFFSET:]
        masks = inputs[:, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 4]
        if variant == "hand_only_pose_legacy":
            model_inputs = torch.cat((features[:, :HAND_FEATURE_DIM], body, masks), dim=1)
        else:
            model_inputs = torch.cat(
                (features[:, :HAND_FEATURE_DIM], body, inputs[:, LANDMARK_FEATURE_DIM + 4:], masks), dim=1
            )
    elif variant is None:
        model_inputs = None
    else:
        model_inputs = landmark_feature_variant(inputs, variant)
    if model_inputs is None:
        raise ValueError("Landmark model has an unsupported input dimension.")
    return model(model_inputs)


class GatedResidualImageHead(nn.Module):
    """Project image features, then apply a gated residual block before classification."""

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 384) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projection(features)
        refined = self.norm(projected + self.transform(projected) * self.gate(projected))
        return self.classifier(self.dropout(refined))


def image_model(
    num_classes: int,
    feature_dim: int = 576,
    head_type: str = "gated_residual",
) -> nn.Module:
    """Build the selected improved-model image head."""
    if head_type == "gated_residual":
        return GatedResidualImageHead(feature_dim, num_classes)
    if head_type == "legacy":
        return nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    raise ValueError(f"Unsupported image classifier head: {head_type!r}")


def combine_probabilities(
    landmarks: torch.Tensor, images: torch.Tensor, presence: torch.Tensor, image_weight: float, fallback: bool
) -> torch.Tensor:
    """Combine class probabilities, optionally using the image model for absent hands."""
    combined = (1.0 - image_weight) * landmarks + image_weight * images
    if fallback:
        combined = torch.where(presence.unsqueeze(1), combined, images)
    return combined


def run_experiment(
    config: dict[str, Any],
    manifest_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    baseline_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fine-tune the improved image branch and select its blend on validation only."""
    seed = int(config["project"].get("seed", 42))
    set_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest = load_split_manifest(manifest_path or project_path(config, "splits") / "split_manifest.json")
    cache_path = project_path(config, "landmarks") / "mediapipe_landmarks.json"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Landmark cache missing: {cache_path}. Run notebook 03 first.")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if (
        cache.get("format_version") != LANDMARK_CACHE_FORMAT_VERSION
        or cache.get("feature_dim") != LANDMARK_FEATURE_DIM
        or cache.get("input_is_mirrored", False) != bool(config["data"].get("images_are_mirrored", False))
    ):
        raise ValueError("Landmark cache settings do not match configuration; rerun notebook 03.")
    mapping = manifest["class_to_idx"]
    class_names = [name for name, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    if sorted(mapping.values()) != list(range(len(class_names))):
        raise ValueError("Class indices must be consecutive integers starting at zero.")
    splits = manifest["splits"]
    rows = [row for name in ("train", "validation", "test") for row in splits[name]]
    keys = [row["relative_path"] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(cache["items"]):
        raise ValueError("Split manifest and cache have duplicate, missing, or extra image keys.")
    validate_split_groups(manifest)
    for row in rows:
        item = cache["items"][row["relative_path"]]
        if (
            item.get("sha256") != row["sha256"]
            or item.get("label") != row["label"]
            or item.get("label_idx") != row["label_idx"]
            or mapping.get(row["label"]) != row["label_idx"]
        ):
            raise ValueError(f"Stale or misaligned landmark data: {row['relative_path']}")
        if (
            len(item["features"]) != LANDMARK_FEATURE_DIM
            or len(item["positions"]) != 8
            or len(item["presence"]) != 2
            or len(item["handedness_reliable"]) != 2
        ):
            raise ValueError(f"Invalid landmark feature shape: {row['relative_path']}")

    checkpoint_path = (
        Path(baseline_checkpoint_path)
        if baseline_checkpoint_path is not None
        else project_path(config, "checkpoints") / "landmark_mlp_best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Landmark model missing: {checkpoint_path}. Run notebook 04 first.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    check_checkpoint_split(checkpoint, manifest)
    if checkpoint.get("model_type") != "landmark_mlp" or checkpoint.get("class_to_idx") != mapping:
        raise ValueError("Landmark model type or class mapping does not match the split manifest.")
    if checkpoint.get("input_is_mirrored", False) != cache.get("input_is_mirrored", False):
        raise ValueError("Landmark model and cache use different mirrored-input settings.")
    landmark_net = landmark_model_from_state(len(class_names), checkpoint["model_state_dict"])
    landmark_net.eval()

    weights_name = image_weights_name(config.get("model", {}).get("backbone", "mobilenet_v3_small"))
    backbone = image_backbone(weights_name, pretrained=True).eval()
    backbone_parameters = configure_image_finetuning(backbone)

    landmark_inputs: dict[str, torch.Tensor] = {}
    labels: dict[str, torch.Tensor] = {}
    feature_started = time.perf_counter()
    for name in ("train", "validation"):
        split_rows = splits[name]
        landmark_inputs[name] = torch.tensor(
            [cache["items"][row["relative_path"]]["features"][:HAND_FEATURE_DIM] + cache["items"][row["relative_path"]]["presence"]
             + cache["items"][row["relative_path"]]["handedness_reliable"] for row in split_rows], dtype=torch.float32
        )
        labels[name] = torch.tensor([row["label_idx"] for row in split_rows], dtype=torch.long)
        if (landmark_inputs[name].shape != (len(split_rows), HAND_FEATURE_DIM + 4)
                or not torch.isfinite(landmark_inputs[name]).all()
                or set(labels[name].tolist()) != set(range(len(class_names)))):
            raise ValueError(f"Invalid feature values or incomplete class coverage in {name} split.")
        print(f"{name}: prepared {len(split_rows)} samples", flush=True)

    train_feature_loader = DataLoader(
        ImageRows(splits["train"], image_transform(weights_name)),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        train_features = torch.cat([backbone(batch) for batch in train_feature_loader])
    feature_seconds = time.perf_counter() - feature_started

    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp(min=1e-4)
    with torch.inference_mode():
        validation_landmarks = landmark_net(landmark_inputs["validation"]).softmax(dim=1)
    validation_presence = landmark_inputs["validation"][:, HAND_FEATURE_DIM:HAND_FEATURE_DIM + 2].sum(dim=1) > 0
    train_loader = DataLoader(
        LabeledImageRows(splits["train"], image_train_transform(weights_name)),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    head_type = config.get("model", {}).get("image_head", "gated_residual")
    net = image_model(len(class_names), image_feature_dim(weights_name), head_type)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 1e-5},
            {"params": net.parameters(), "lr": 5e-4},
        ],
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_accuracy = -1.0
    best_epoch = 0
    best_weight = 0.0
    best_fallback = False
    best_state = None
    best_backbone_state = None
    history = []
    training_started = time.perf_counter()
    stale_epochs = 0
    validation_loader = DataLoader(
        ImageRows(splits["validation"], image_transform(weights_name)),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    for epoch in range(1, 13):
        backbone.eval()
        net.train()
        for batch, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = net((backbone(batch) - mean) / std)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            image_probabilities = torch.cat(
                [
                    net((backbone(batch) - mean) / std).softmax(dim=1)
                    for batch in validation_loader
                ]
            )
        improved_this_epoch = False
        for weight in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
            for fallback in (False, True):
                combined = combine_probabilities(validation_landmarks, image_probabilities, validation_presence, weight, fallback)
                accuracy = (combined.argmax(dim=1) == labels["validation"]).float().mean().item()
                history.append({"epoch": epoch, "image_weight": weight, "fallback": fallback, "validation_accuracy": accuracy})
                if accuracy > best_accuracy:
                    best_accuracy, best_epoch, best_weight, best_fallback = accuracy, epoch, weight, fallback
                    best_state = copy.deepcopy(net.state_dict())
                    best_backbone_state = copy.deepcopy(backbone.state_dict())
                    improved_this_epoch = True
        stale_epochs = 0 if improved_this_epoch else stale_epochs + 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch}: best ensemble validation accuracy so far {best_accuracy:.2%}", flush=True)
        if stale_epochs >= 4:
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None or best_backbone_state is None:
        raise RuntimeError("No validation-selected image checkpoint was produced.")
    net.load_state_dict(best_state, strict=True)
    backbone.load_state_dict(best_backbone_state, strict=True)
    net.eval()
    backbone.eval()
    print(f"Validation selected: epoch {best_epoch}, image weight {best_weight:.1f}, no-hand fallback {best_fallback}; accuracy {best_accuracy:.2%}")

    output_dir = Path(checkpoint_dir) if checkpoint_dir is not None else project_path(config, "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_dir / "landmark_image_ensemble_best.pt"
    torch.save({
        "model_type": "landmark_image_ensemble", "class_to_idx": mapping, "landmark_model_state_dict": landmark_net.state_dict(),
        "image_model_state_dict": net.state_dict(), "backbone_state_dict": backbone.state_dict(), "feature_mean": mean,
        "feature_std": std, "weights": weights_name, "image_weight": best_weight,
        "no_hand_image_fallback": best_fallback, "validation_accuracy": best_accuracy, "epoch": best_epoch,
        "seed": seed, "input_is_mirrored": cache.get("input_is_mirrored", False), "split_fingerprint": split_fingerprint(manifest),
        "fine_tuned_feature_blocks": 2,
        "image_head": head_type,
    }, output_checkpoint)
    pd.DataFrame(history).to_csv(output_dir / "landmark_image_ensemble_validation.csv", index=False)
    return {
        "validation_accuracy": best_accuracy,
        "image_weight": best_weight,
        "no_hand_image_fallback": best_fallback,
        "epoch": best_epoch,
        "feature_extraction_seconds": feature_seconds,
        "training_seconds": training_seconds,
        "checkpoint": output_checkpoint,
    }
