"""Validation-selected landmark ensemble with the existing image classifier."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import project_path
from src.dataset import check_checkpoint_split, load_split_manifest, split_fingerprint, validate_split_groups
from src.landmark_extractor import (
    BODY_FEATURE_OFFSET,
    HAND_FEATURE_DIM,
    LANDMARK_CACHE_FORMAT_VERSION,
    LANDMARK_FEATURE_DIM,
    SHAPE_FEATURE_DIM,
    SHAPE_FEATURES_PER_HAND,
)
from src.landmark_image_ensemble import (
    ImageRows,
    combine_probabilities,
    image_backbone,
    image_feature_dim,
    image_model,
    image_transform,
    landmark_feature_variant,
    landmark_model,
    landmark_model_from_state,
    landmark_model_logits,
)
from src.utils import set_seed


def load_inputs(
    config: dict[str, Any], manifest_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    source_manifest = manifest_path or project_path(config, "splits") / "split_manifest.json"
    manifest = load_split_manifest(source_manifest)
    cache_path = project_path(config, "landmarks") / "mediapipe_landmarks.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if (
        cache.get("format_version") != LANDMARK_CACHE_FORMAT_VERSION
        or cache.get("feature_dim") != LANDMARK_FEATURE_DIM
        or cache.get("input_is_mirrored", False) != bool(config["data"].get("images_are_mirrored", False))
    ):
        raise ValueError("Landmark cache settings do not match configuration; rerun notebook 03.")
    mapping = manifest["class_to_idx"]
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Class indices must be consecutive integers starting at zero.")
    splits = manifest["splits"]
    rows = [row for name in ("train", "validation", "test") for row in splits[name]]
    keys = [row["relative_path"] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(cache["items"]):
        raise ValueError("Manifest and landmark cache have duplicate, missing, or extra image keys.")
    validate_split_groups(manifest)
    inputs, labels = {}, {}
    for name in ("train", "validation", "test"):
        features, targets = [], []
        for row in splits[name]:
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
            features.append(
                item["features"] + item["presence"] + item["handedness_reliable"] + item["positions"]
            )
            targets.append(row["label_idx"])
        inputs[name] = torch.tensor(features, dtype=torch.float32)
        labels[name] = torch.tensor(targets, dtype=torch.long)
        if inputs[name].shape[1] != LANDMARK_FEATURE_DIM + 12:
            raise ValueError(f"Invalid landmark input shape in {name}.")
        if not torch.isfinite(inputs[name]).all() or set(targets) != set(mapping.values()):
            raise ValueError(f"Invalid landmark values or incomplete class coverage in {name}.")
    return manifest, inputs, labels


def add_landmark_noise(inputs: torch.Tensor, std: float) -> torch.Tensor:
    if inputs.ndim != 2 or inputs.shape[1] != LANDMARK_FEATURE_DIM + 12:
        raise ValueError(f"Expected landmark inputs with shape (batch, {LANDMARK_FEATURE_DIM + 12}).")
    if std < 0:
        raise ValueError("Landmark noise standard deviation must be non-negative.")
    augmented = inputs.clone()
    hands = augmented[:, :HAND_FEATURE_DIM].reshape(-1, 2, 21, 3)
    present = augmented[:, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 2]
    hands.add_(torch.randn_like(hands) * std * present[:, :, None, None])
    shape = augmented[:, HAND_FEATURE_DIM:BODY_FEATURE_OFFSET].reshape(-1, 2, SHAPE_FEATURES_PER_HAND)
    shape.add_(torch.randn_like(shape) * std * present[:, :, None])
    body = augmented[:, BODY_FEATURE_OFFSET:LANDMARK_FEATURE_DIM]
    body_valid = body[:, -1:].clamp(0, 1)
    per_hand_end = 2 * 21
    body[:, :per_hand_end].reshape(-1, 2, 21).add_(
        torch.randn_like(body[:, :per_hand_end].reshape(-1, 2, 21))
        * std
        * present[:, :, None]
        * body_valid[:, :, None]
    )
    both_hands_present = (present.sum(1) == 2).to(body.dtype).unsqueeze(1) * body_valid
    body[:, per_hand_end:-1].add_(
        torch.randn_like(body[:, per_hand_end:-1]) * std * both_hands_present
    )
    return augmented


def _refit_landmark_model(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    classes: int,
    seed: int,
    noise: float,
    epochs: int,
    feature_variant: str,
) -> nn.Module:
    if epochs < 1:
        raise ValueError("Selected landmark training epoch must be positive.")
    set_seed(seed)
    input_dim = landmark_feature_variant(inputs[:1], feature_variant).shape[1]
    model = landmark_model(classes, input_dim=input_dim)
    loader = DataLoader(
        TensorDataset(inputs, labels), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    for _ in range(epochs):
        model.train()
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            augmented = landmark_feature_variant(add_landmark_noise(batch, noise), feature_variant)
            loss = nn.functional.cross_entropy(model(augmented), target)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def run_experiment(
    config: dict[str, Any],
    output_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    initial_checkpoint_path: str | Path | None = None,
    selection_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the improved ensemble on validation, then refit it on train and validation."""
    started = time.perf_counter()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest, inputs, labels = load_inputs(config, manifest_path)
    mapping = manifest["class_to_idx"]
    names = [name for name, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    old_path = (
        Path(initial_checkpoint_path)
        if initial_checkpoint_path is not None
        else project_path(config, "checkpoints") / "landmark_image_ensemble_best.pt"
    )
    old = torch.load(old_path, map_location="cpu", weights_only=True)
    check_checkpoint_split(old, manifest)
    if old.get("model_type") != "landmark_image_ensemble" or old.get("class_to_idx") != mapping or old.get("input_is_mirrored", False) != bool(config["data"].get("images_are_mirrored", False)):
        raise ValueError("Existing ensemble checkpoint does not match the data configuration.")

    original = landmark_model_from_state(len(names), old["landmark_model_state_dict"])
    original.eval()
    models = [original]
    validation_scores = []
    train_started = time.perf_counter()
    feature_variant = (
        selection_override.get("feature_variant", "hand_shape_pose_position")
        if selection_override is not None
        else "hand_shape_pose_position"
    )
    landmark_feature_variant(inputs["train"][:1], feature_variant)
    for seed, noise in ((42, 0.02), (43, 0.02), (44, 0.05), (45, 0.0)):
        set_seed(seed)
        input_dim = landmark_feature_variant(inputs["train"][:1], feature_variant).shape[1]
        model = landmark_model(len(names), input_dim=input_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loader = DataLoader(TensorDataset(inputs["train"], labels["train"]), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
        best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
        for epoch in range(1, 81):
            model.train()
            for batch, target in loader:
                optimizer.zero_grad(set_to_none=True)
                augmented = landmark_feature_variant(add_landmark_noise(batch, noise), feature_variant)
                loss = nn.functional.cross_entropy(model(augmented), target)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                validation_logits = landmark_model_logits(model, inputs["validation"])
                score = (validation_logits.argmax(1) == labels["validation"]).float().mean().item()
            if score > best_score:
                best_score, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                stale += 1
                if stale >= 15:
                    break
        if best_state is None:
            raise RuntimeError("Training did not produce a validation-selected model.")
        model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        validation_scores.append({"seed": seed, "noise": noise, "epoch": best_epoch, "validation_accuracy": best_score})
        print(f"Landmark model seed {seed}, noise {noise:.2f}: validation {best_score:.2%}, epoch {best_epoch}", flush=True)
    training_seconds = time.perf_counter() - train_started

    weights_name = old["weights"]
    image_head_type = old.get("image_head", "legacy")
    backbone = image_backbone(weights_name)
    backbone.load_state_dict(old["backbone_state_dict"])
    backbone.eval()
    image_net = image_model(len(names), image_feature_dim(weights_name), image_head_type)
    image_net.load_state_dict(old["image_model_state_dict"])
    image_net.eval()
    transform = image_transform(weights_name)

    def image_probabilities(name: str) -> torch.Tensor:
        loader = DataLoader(ImageRows(manifest["splits"][name], transform), batch_size=32, shuffle=False, num_workers=0)
        with torch.inference_mode():
            return torch.cat([image_net((backbone(batch) - old["feature_mean"]) / old["feature_std"]).softmax(1) for batch in loader])

    validation_images = image_probabilities("validation")
    with torch.inference_mode():
        validation_probs = [landmark_model_logits(model, inputs["validation"]).softmax(1) for model in models]
    presence = inputs["validation"][:, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 2].sum(1) > 0
    best = None
    selection_method = "validation"
    if selection_override is None:
        # Include the exact old combination, then select candidates using validation only.
        candidates = [([0], old["image_weight"], old["no_hand_image_fallback"])]
        for indices in ([0], [1], [2], [3], [4], [1, 2, 3, 4], [0, 1, 2, 3, 4]):
            for weight in (0.0, 0.1, 0.2, 0.4, 0.6):
                for fallback in (False, True):
                    candidates.append((indices, weight, fallback))
    else:
        indices = selection_override.get("indices")
        weight = selection_override.get("image_weight")
        fallback = selection_override.get("fallback")
        if (
            not isinstance(indices, list)
            or not indices
            or any(not isinstance(index, int) or index < 1 or index > len(models) - 1 for index in indices)
            or len(set(indices)) != len(indices)
            or not isinstance(weight, (int, float))
            or not 0 <= weight <= 1
            or not isinstance(fallback, bool)
        ):
            raise ValueError("Invalid training-signer cross-validation selection.")
        selection_method = "training_signer_cross_validation"
        candidates = [(indices, float(weight), fallback)]
    for indices, weight, fallback in candidates:
        landmark_probs = torch.stack([validation_probs[i] for i in indices]).mean(0)
        predicted = combine_probabilities(landmark_probs, validation_images, presence, weight, fallback).argmax(1)
        score = (predicted == labels["validation"]).float().mean().item()
        if best is None or score > best["validation_accuracy"]:
            best = {"indices": list(indices), "image_weight": weight, "fallback": fallback, "validation_accuracy": score}
    if best is None:
        raise RuntimeError("No validation candidate was selected.")
    print(f"Validation selection: {best}", flush=True)

    combined_inputs = torch.cat((inputs["train"], inputs["validation"]))
    combined_labels = torch.cat((labels["train"], labels["validation"]))
    final_models = []
    for model_index in best["indices"]:
        if model_index == 0:
            seed = int(old.get("seed", config["project"].get("seed", 42)))
            noise = 0.0
            epochs = int(old.get("epoch", 1))
        else:
            spec = validation_scores[model_index - 1]
            seed, noise, epochs = int(spec["seed"]), float(spec["noise"]), int(spec["epoch"])
        final_models.append(
            _refit_landmark_model(
                combined_inputs, combined_labels, len(names), seed, noise, epochs, feature_variant
            )
        )

    refit_rows = manifest["splits"]["train"] + manifest["splits"]["validation"]
    feature_loader = DataLoader(
        ImageRows(refit_rows, transform), batch_size=32, shuffle=False, num_workers=0,
    )
    with torch.inference_mode():
        refit_features = torch.cat([backbone(batch) for batch in feature_loader])
    refit_mean = refit_features.mean(dim=0)
    refit_std = refit_features.std(dim=0).clamp(min=1e-4)
    refit_x = (refit_features - refit_mean) / refit_std
    refit_y = combined_labels
    refit_epoch = int(old.get("epoch", 0))
    if refit_epoch < 1:
        raise ValueError("Image-head checkpoint is missing its validation-selected epoch.")
    set_seed(int(old.get("seed", config["project"].get("seed", 42))))
    final_image_net = image_model(len(names), image_feature_dim(weights_name), image_head_type)
    image_loader = DataLoader(
        TensorDataset(refit_x, refit_y), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(int(old.get("seed", 42))),
    )
    optimizer = torch.optim.AdamW(final_image_net.parameters(), lr=0.001, weight_decay=0.01)
    for _ in range(refit_epoch):
        final_image_net.train()
        for batch, target in image_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(final_image_net(batch), target)
            loss.backward()
            optimizer.step()
    final_image_net.eval()

    with torch.inference_mode():
        val_image_probs = image_probabilities("validation")
        val_landmark_probs = torch.stack(
            [landmark_model_logits(model, inputs["validation"]).softmax(1) for model in models]
        )
        selected_landmark_probs = torch.stack(
            [landmark_model_logits(models[i], inputs["validation"]).softmax(1) for i in best["indices"]]
        ).mean(0)
        selected_validation = combine_probabilities(
            selected_landmark_probs,
            val_image_probs,
            presence,
            best["image_weight"],
            best["fallback"],
        )
    if not torch.isfinite(selected_validation).all():
        raise RuntimeError("Validation selection produced non-finite probabilities.")
    checkpoint_dir = project_path(config, "checkpoints") if output_root is None else Path(output_root) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "robust_landmark_ensemble_best.pt"
    torch.save({
        "model_type": "robust_landmark_ensemble", "class_to_idx": mapping,
        "landmark_states": [model.state_dict() for model in final_models],
        "backbone_state_dict": backbone.state_dict(), "image_model_state_dict": final_image_net.state_dict(),
        "feature_mean": refit_mean, "feature_std": refit_std,
        "image_weight": best["image_weight"], "fallback": best["fallback"],
        "validation_accuracy": best["validation_accuracy"], "input_is_mirrored": old.get("input_is_mirrored", False),
        "feature_variant": feature_variant,
        "weights": weights_name, "split_fingerprint": split_fingerprint(manifest),
        "training_signers": sorted({row["signer_id"] for row in refit_rows}),
        "selection_method": selection_method,
        "image_head": image_head_type,
    }, path)
    pd.DataFrame(validation_scores).to_csv(checkpoint_dir / "robust_landmark_validation.csv", index=False)
    return {
        "checkpoint": path,
        "selection": best,
        "training_seconds": training_seconds,
        "total_seconds": time.perf_counter() - started,
    }
