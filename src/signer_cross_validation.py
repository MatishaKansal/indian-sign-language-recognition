"""Signer-wise validation for landmark models without using the test signer."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import project_path
from src.dataset import check_checkpoint_split, split_fingerprint
from src.landmark_extractor import LANDMARK_FEATURE_DIM
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
from src.metrics import save_evaluation
from src.robust_landmark_ensemble import add_landmark_noise, load_inputs
from src.utils import set_seed


def fold_indices(rows: list[dict[str, Any]], held_signer: str) -> tuple[list[int], list[int]]:
    """Keep the held signer and its duplicate groups out of the fold training set."""
    validation = [i for i, row in enumerate(rows) if row["signer_id"] == held_signer]
    if not validation:
        raise ValueError(f"No training rows for held-out signer {held_signer!r}.")
    held_groups = {rows[i]["duplicate_group"] for i in validation}
    training = [i for i, row in enumerate(rows) if row["signer_id"] != held_signer and row["duplicate_group"] not in held_groups]
    if not training:
        raise ValueError(f"No training rows remain for signer {held_signer!r}.")
    return training, validation


def train_models(
    x: torch.Tensor,
    y: torch.Tensor,
    classes: int,
    seed: int,
    noise: float,
    epochs: tuple[int, ...],
    feature_variant: str = "hand_shape_pose_position",
) -> dict[int, nn.Module]:
    set_seed(seed)
    input_dim = landmark_feature_variant(x[:1], feature_variant).shape[1]
    model = landmark_model(classes, input_dim=input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loader = DataLoader(
        TensorDataset(x, y), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    snapshots = {}
    for epoch in range(1, max(epochs) + 1):
        model.train()
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            augmented = landmark_feature_variant(add_landmark_noise(batch, noise), feature_variant)
            loss = nn.functional.cross_entropy(model(augmented), target)
            loss.backward()
            optimizer.step()
        if epoch in epochs:
            snapshot = copy.deepcopy(model)
            snapshot.eval()
            snapshots[epoch] = snapshot
    return snapshots


def image_probabilities(checkpoint: dict[str, Any], rows: list[dict[str, Any]], classes: int) -> torch.Tensor:
    weights_name = checkpoint["weights"]
    backbone = image_backbone(weights_name)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.eval()
    image_net = image_model(classes, image_feature_dim(weights_name), checkpoint.get("image_head", "legacy"))
    image_net.load_state_dict(checkpoint["image_model_state_dict"])
    image_net.eval()
    loader = DataLoader(
        ImageRows(rows, image_transform(weights_name)),
        batch_size=32, shuffle=False, num_workers=0,
    )
    with torch.inference_mode():
        return torch.cat([
            image_net((backbone(batch) - checkpoint["feature_mean"]) / checkpoint["feature_std"]).softmax(1)
            for batch in loader
        ])


def ensemble_probabilities(
    models: list[nn.Module], inputs: torch.Tensor, images: torch.Tensor, weight: float, fallback: bool,
) -> torch.Tensor:
    if not models:
        raise ValueError("An ensemble needs at least one landmark model.")
    with torch.inference_mode():
        landmarks = torch.stack([landmark_model_logits(model, inputs).softmax(1) for model in models]).mean(0)
        presence = inputs[:, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 2].sum(1) > 0
        return combine_probabilities(landmarks, images, presence, weight, fallback)


def run_experiment(config: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Select landmark settings on training signers, then compare on the existing validation signer."""
    started = time.perf_counter()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest, inputs, labels = load_inputs(config)
    if not manifest.get("split_mode", "").startswith("signer-independent"):
        raise ValueError("Signer cross-validation requires a signer-independent manifest; the image split shares signers.")
    mapping = manifest["class_to_idx"]
    names = [name for name, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    rows = manifest["splits"]["train"]
    signers = sorted({row["signer_id"] for row in rows})
    if len(signers) < 3:
        raise ValueError("At least three training signers are required for cross-validation.")
    seed = int(config["project"].get("seed", 42))
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    fold_results = []
    fold_classes = []
    for signer in signers:
        train_idx, val_idx = fold_indices(rows, signer)
        if set(labels["train"][train_idx].tolist()) != set(range(len(names))):
            raise ValueError(f"Fold excluding {signer} does not have all classes in training.")
        missing = sorted(set(names) - {rows[i]["label"] for i in val_idx})
        excluded = len(rows) - len(train_idx) - len(val_idx)
        print(f"Fold {signer}: train {len(train_idx)}, validation {len(val_idx)}, duplicate exclusions {excluded}, missing validation classes: {missing}", flush=True)
        for noise in (0.0, 0.02):
            models = train_models(inputs["train"][train_idx], labels["train"][train_idx], len(names), seed, noise, (30, 60))
            for epoch, model in models.items():
                with torch.inference_mode():
                    pred = landmark_model_logits(model, inputs["train"][val_idx]).argmax(1).numpy()
                target = labels["train"][val_idx].numpy()
                score = accuracy_score(target, pred)
                fold_results.append({"signer": signer, "noise": noise, "epochs": epoch, "accuracy": score,
                                     "samples": len(val_idx), "excluded_duplicate_rows": excluded,
                                     "missing_classes": ", ".join(missing)})
                report = classification_report(target, pred, labels=list(range(len(names))), target_names=names, output_dict=True, zero_division=0)
                for name in names:
                    fold_classes.append({"signer": signer, "noise": noise, "epochs": epoch, "class": name, **report[name]})
                print(f"  noise {noise:.2f}, epochs {epoch}: {score:.2%}", flush=True)
    folds = pd.DataFrame(fold_results)
    summary = folds.groupby(["noise", "epochs"])["accuracy"].agg(["mean", "min"]).reset_index()
    summary = summary.sort_values(["mean", "min", "noise", "epochs"], ascending=[False, False, True, True])
    chosen = summary.iloc[0]
    noise, epochs = float(chosen["noise"]), int(chosen["epochs"])
    folds.to_csv(output / "signer_folds.csv", index=False)
    pd.DataFrame(fold_classes).to_csv(output / "signer_folds_per_class.csv", index=False)
    summary.to_csv(output / "signer_cv_summary.csv", index=False)
    print(f"CV selection: noise {noise:.2f}, epochs {epochs}, mean signer accuracy {chosen['mean']:.2%}", flush=True)

    model = train_models(inputs["train"], labels["train"], len(names), seed, noise, (epochs,))[epochs]
    cp_path = project_path(config, "checkpoints") / "robust_landmark_ensemble_best.pt"
    cp = torch.load(cp_path, map_location="cpu", weights_only=True)
    check_checkpoint_split(cp, manifest)
    if (cp.get("model_type") != "robust_landmark_ensemble" or cp.get("class_to_idx") != mapping
        or cp.get("input_is_mirrored", False) != bool(config["data"].get("images_are_mirrored", False))):
        raise ValueError("Existing best checkpoint does not match the dataset.")
    old_models = []
    for state in cp["landmark_states"]:
        old_model = landmark_model_from_state(len(names), state)
        old_model.eval()
        old_models.append(old_model)
    validation_images = image_probabilities(cp, manifest["splits"]["validation"], len(names))
    baseline = ensemble_probabilities(old_models, inputs["validation"], validation_images, cp["image_weight"], cp["fallback"])
    best_accuracy = (baseline.argmax(1) == labels["validation"]).float().mean().item()
    selection = {"source": "existing", "weight": cp["image_weight"], "fallback": cp["fallback"], "validation_accuracy": best_accuracy}
    validation_results = [dict(selection)]
    for weight in (0.0, 0.2, 0.4):
        for fallback in (False, True):
            probs = ensemble_probabilities([model], inputs["validation"], validation_images, weight, fallback)
            accuracy = (probs.argmax(1) == labels["validation"]).float().mean().item()
            validation_results.append({"source": "cross_validated", "weight": weight, "fallback": fallback, "validation_accuracy": accuracy})
            if accuracy > selection["validation_accuracy"]:
                selection = {"source": "cross_validated", "weight": weight, "fallback": fallback, "validation_accuracy": accuracy}
    pd.DataFrame(validation_results).to_csv(output / "validation_candidates.csv", index=False)
    candidate_best = max(row["validation_accuracy"] for row in validation_results if row["source"] == "cross_validated")
    print(f"Best CV candidate validation accuracy: {candidate_best:.2%}", flush=True)
    print(f"Validation selection: {selection}; previous baseline {best_accuracy:.2%}", flush=True)
    selected_models = old_models if selection["source"] == "existing" else [model]
    test_images = image_probabilities(cp, manifest["splits"]["test"], len(names))
    predicted = ensemble_probabilities(selected_models, inputs["test"], test_images, selection["weight"], selection["fallback"]).argmax(1).numpy()
    targets = labels["test"].numpy()
    ids = list(range(len(names)))
    metrics = {
        "accuracy": float(accuracy_score(targets, predicted)),
        "macro_precision": float(precision_score(targets, predicted, labels=ids, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predicted, labels=ids, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predicted, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predicted, labels=ids, average="weighted", zero_division=0)),
        "test_samples": len(targets), "total_seconds": time.perf_counter() - started,
        "validation_accuracy": selection["validation_accuracy"], "selected_source": selection["source"],
        "evaluation_note": "Test and validation signers were previously used; near-duplicate screening was incomplete.",
    }
    result = {
        "metrics": metrics, "targets": targets.tolist(), "predictions": predicted.tolist(),
        "paths": [row["path"] for row in manifest["splits"]["test"]], "class_names": names,
        "confusion_matrix": confusion_matrix(targets, predicted, labels=ids),
        "classification_report": classification_report(targets, predicted, labels=ids, target_names=names, output_dict=True, zero_division=0),
    }
    path = output / "signer_cv_selected.pt"
    torch.save({
        "model_type": "robust_landmark_ensemble", "class_to_idx": mapping,
        "landmark_states": [selected.state_dict() for selected in selected_models],
        "backbone_state_dict": cp["backbone_state_dict"], "image_model_state_dict": cp["image_model_state_dict"],
        "feature_mean": cp["feature_mean"], "feature_std": cp["feature_std"],
        "image_weight": selection["weight"], "fallback": selection["fallback"],
        "input_is_mirrored": cp.get("input_is_mirrored", False), "weights": cp["weights"],
        "image_head": cp.get("image_head", "legacy"),
        "validation_accuracy": selection["validation_accuracy"], "selected_source": selection["source"],
        "cv_noise": noise, "cv_epochs": epochs, "cv_signers": signers, "split_fingerprint": split_fingerprint(manifest),
    }, path)
    saved = save_evaluation(result, "signer_cv_selected", output)
    return {"result": result, "folds": folds, "summary": summary, "checkpoint": path, "saved": saved}
