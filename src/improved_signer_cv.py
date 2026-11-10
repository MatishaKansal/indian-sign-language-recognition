"""Compare improved-model image branches using training-signers-only cross-validation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config, project_path
from src.dataset import load_split_manifest
from src.landmark_extractor import (
    LANDMARK_CACHE_FORMAT_VERSION,
    LANDMARK_FEATURE_DIM,
)
from src.landmark_image_ensemble import (
    LabeledImageRows,
    ImageRows,
    BACKBONE_WEIGHTS,
    combine_probabilities,
    configure_image_finetuning,
    image_backbone,
    image_feature_dim,
    image_model,
    image_train_transform,
    image_transform,
    image_weights_name,
    landmark_feature_variant,
    landmark_model,
    landmark_model_logits,
)
from src.robust_landmark_ensemble import add_landmark_noise
from src.signer_cross_validation import fold_indices, train_models
from src.utils import set_seed


def _training_landmarks(
    config: dict[str, Any], rows: list[dict[str, Any]], classes: int,
) -> torch.Tensor:
    cache_path = project_path(config, "landmarks") / "mediapipe_landmarks.json"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Landmark cache missing: {cache_path}. Run notebook 03 first.")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    expected_mirroring = bool(config["data"].get("images_are_mirrored", False))
    if (
        cache.get("format_version") != LANDMARK_CACHE_FORMAT_VERSION
        or cache.get("feature_dim") != LANDMARK_FEATURE_DIM
        or cache.get("input_is_mirrored", False) != expected_mirroring
    ):
        raise ValueError("Landmark cache settings do not match configuration; rerun notebook 03.")
    values = []
    for row in rows:
        item = cache["items"].get(row["relative_path"])
        if (
            item is None
            or item.get("sha256") != row["sha256"]
            or item.get("label") != row["label"]
            or item.get("label_idx") != row["label_idx"]
        ):
            raise ValueError(f"Stale or misaligned landmark data: {row['relative_path']}")
        if (
            len(item["features"]) != LANDMARK_FEATURE_DIM
            or len(item["positions"]) != 8
            or len(item["presence"]) != 2
            or len(item["handedness_reliable"]) != 2
        ):
            raise ValueError(f"Invalid landmark feature shape: {row['relative_path']}")
        values.append(item["features"] + item["presence"] + item["handedness_reliable"] + item["positions"])
    inputs = torch.tensor(values, dtype=torch.float32)
    if inputs.shape != (len(rows), LANDMARK_FEATURE_DIM + 12) or not torch.isfinite(inputs).all():
        raise ValueError("Invalid training landmark values.")
    if {row["label_idx"] for row in rows} != set(range(classes)):
        raise ValueError("Training signers do not contain every configured class.")
    return inputs


def _landmark_feature_variants(inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build nested feature ablations while keeping the four masks at the end."""
    variants = (
        "hand_only",
        "hand_only_position",
        "hand_plus_shape",
        "hand_plus_shape_position",
        "hand_shape_pose",
        "hand_shape_pose_position",
    )
    return {variant: landmark_feature_variant(inputs, variant) for variant in variants}


def run_feature_ablation(
    config: dict[str, Any], output_dir: str | Path | None = None, epochs: int = 45,
) -> dict[str, pd.DataFrame]:
    """Compare hand, hand-shape, and Pose features across training-signer folds only."""
    if epochs < 1:
        raise ValueError("Feature-ablation epochs must be positive.")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest = load_split_manifest(project_path(config, "splits") / "split_manifest.json")
    if not manifest.get("split_mode", "").startswith("signer-independent"):
        raise ValueError("Feature ablation requires a signer-independent split.")
    rows = manifest["splits"]["train"]
    names = [name for name, _ in sorted(manifest["class_to_idx"].items(), key=lambda pair: pair[1])]
    inputs = _training_landmarks(config, rows, len(names))
    variants = _landmark_feature_variants(inputs)
    labels = torch.tensor([row["label_idx"] for row in rows], dtype=torch.long)
    signers = sorted({row["signer_id"] for row in rows})
    if len(signers) < 3:
        raise ValueError("At least three training signers are required for feature ablation.")

    folds: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    seed = int(config["project"].get("seed", 42))
    for fold_number, signer in enumerate(signers):
        train_idx, validation_idx = fold_indices(rows, signer)
        if set(labels[train_idx].tolist()) != set(range(len(names))):
            raise ValueError(f"Fold excluding {signer} does not contain every class in training.")
        target = labels[validation_idx].numpy()
        missing = sorted(set(names) - {rows[index]["label"] for index in validation_idx})
        for variant_name, values in variants.items():
            fold_seed = seed + fold_number
            set_seed(fold_seed)
            model = landmark_model(len(names), input_dim=values.shape[1])
            loader = DataLoader(
                TensorDataset(values[train_idx], labels[train_idx]),
                batch_size=256,
                shuffle=True,
                generator=torch.Generator().manual_seed(fold_seed),
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            for _ in range(epochs):
                model.train()
                for batch, target_batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(batch), target_batch)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.inference_mode():
                predicted = model(values[validation_idx]).argmax(1).numpy()
            folds.append({
                "variant": variant_name,
                "signer": signer,
                "accuracy": float(accuracy_score(target, predicted)),
                "macro_f1": float(f1_score(target, predicted, labels=list(range(len(names))), average="macro", zero_division=0)),
                "samples": len(target),
                "missing_validation_classes": ", ".join(missing),
            })
            report = classification_report(
                target, predicted, labels=list(range(len(names))), target_names=names,
                output_dict=True, zero_division=0,
            )
            for class_name in names:
                classes.append({
                    "variant": variant_name,
                    "signer": signer,
                    "class": class_name,
                    **report[class_name],
                })
            print(f"Fold {signer}, {variant_name}: {folds[-1]['accuracy']:.2%}", flush=True)

    fold_frame = pd.DataFrame(folds)
    summary = (
        fold_frame.groupby("variant")
        .agg(
            mean_signer_accuracy=("accuracy", "mean"),
            min_signer_accuracy=("accuracy", "min"),
            mean_signer_macro_f1=("macro_f1", "mean"),
            folds=("signer", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["mean_signer_accuracy", "mean_signer_macro_f1", "min_signer_accuracy"],
            ascending=False,
        )
    )
    target_dir = Path(output_dir) if output_dir is not None else project_path(config, "predictions") / "feature_ablation_cv"
    target_dir.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(target_dir / "feature_ablation_folds.csv", index=False)
    summary.to_csv(target_dir / "feature_ablation_summary.csv", index=False)
    pd.DataFrame(classes).to_csv(target_dir / "feature_ablation_per_class.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print("Test signer was not used for feature selection or evaluation.", flush=True)
    return {"folds": fold_frame, "summary": summary, "per_class": pd.DataFrame(classes)}


def _image_probabilities(
    train_rows: list[dict[str, Any]],
    train_labels: torch.Tensor,
    validation_rows: list[dict[str, Any]],
    *,
    fine_tune: bool,
    epochs: int,
    seed: int,
    backbone_name: str = "mobilenet_v3_small",
    head_type: str = "gated_residual",
) -> torch.Tensor:
    set_seed(seed)
    weights_name = image_weights_name(backbone_name)
    backbone = image_backbone(weights_name, pretrained=True).eval()
    if fine_tune:
        backbone_parameters = configure_image_finetuning(backbone)
    else:
        backbone_parameters = []
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

    train_feature_loader = DataLoader(
        ImageRows(train_rows, image_transform(weights_name)),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        train_features = torch.cat([backbone(batch) for batch in train_feature_loader])
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp(min=1e-4)

    head = image_model(train_labels.max().item() + 1, image_feature_dim(weights_name), head_type)
    optimizer_groups = [{"params": head.parameters(), "lr": 5e-4 if fine_tune else 1e-3}]
    if fine_tune:
        optimizer_groups.insert(0, {"params": backbone_parameters, "lr": 1e-5})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05 if fine_tune else 0.0)
    if fine_tune:
        loader = DataLoader(
            LabeledImageRows(train_rows, image_train_transform(weights_name)),
            batch_size=16 if backbone_name == "mobilenet_v3_large" else 32,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
    else:
        loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=256,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

    for _ in range(epochs):
        backbone.eval()
        head.train()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            if fine_tune:
                features = backbone(images)
            else:
                features = images
            loss = criterion(head((features - mean) / std), labels)
            loss.backward()
            optimizer.step()

    validation_loader = DataLoader(
        ImageRows(validation_rows, image_transform(weights_name)),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    backbone.eval()
    head.eval()
    with torch.inference_mode():
        return torch.cat(
            [head((backbone(batch) - mean) / std).softmax(dim=1) for batch in validation_loader]
        )


def _landmark_probabilities(
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    validation_inputs: torch.Tensor,
    classes: int,
    seed: int,
) -> torch.Tensor:
    set_seed(seed)
    model = landmark_model(classes)
    loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    for _ in range(45):
        model.train()
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(add_landmark_noise(batch, 0.02)), target)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        return model(validation_inputs).softmax(dim=1)


def run_experiment(config: dict[str, Any], output_dir: str | Path | None = None) -> pd.DataFrame:
    """Measure frozen and fine-tuned improved branches on held-out training signers only."""
    started = time.perf_counter()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest = load_split_manifest(project_path(config, "splits") / "split_manifest.json")
    rows = manifest["splits"]["train"]
    names = [name for name, _ in sorted(manifest["class_to_idx"].items(), key=lambda pair: pair[1])]
    classes = len(names)
    signers = sorted({row["signer_id"] for row in rows})
    if len(signers) < 3:
        raise ValueError("At least three training signers are required for cross-validation.")
    inputs = _training_landmarks(config, rows, classes)
    labels = torch.tensor([row["label_idx"] for row in rows], dtype=torch.long)
    results: list[dict[str, Any]] = []

    for fold_number, signer in enumerate(signers):
        train_idx, validation_idx = fold_indices(rows, signer)
        if set(labels[train_idx].tolist()) != set(range(classes)):
            raise ValueError(f"Fold excluding {signer} does not have every class in training.")
        train_rows = [rows[index] for index in train_idx]
        validation_rows = [rows[index] for index in validation_idx]
        train_labels = labels[train_idx]
        validation_labels = labels[validation_idx]
        fold_seed = int(config["project"].get("seed", 42)) + fold_number
        landmark_probs = _landmark_probabilities(
            inputs[train_idx], train_labels, inputs[validation_idx], classes, fold_seed,
        )
        print(
            f"Fold {signer}: {len(train_idx)} train, {len(validation_idx)} validation images",
            flush=True,
        )
        for method, fine_tune, epochs in (
            ("frozen", False, 40),
            ("fine_tuned", True, 3),
        ):
            image_probs = _image_probabilities(
                train_rows,
                train_labels,
                validation_rows,
                fine_tune=fine_tune,
                epochs=epochs,
                seed=fold_seed,
            )
            presence = inputs[validation_idx, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 2].sum(1) > 0
            ensemble_probs = combine_probabilities(landmark_probs, image_probs, presence, 0.1, True)
            for model_name, probabilities in (
                (f"{method}_image", image_probs),
                (f"{method}_ensemble", ensemble_probs),
            ):
                predictions = probabilities.argmax(1).numpy()
                target = validation_labels.numpy()
                results.append(
                    {
                        "signer": signer,
                        "model": model_name,
                        "accuracy": float(accuracy_score(target, predictions)),
                        "macro_f1": float(f1_score(target, predictions, labels=list(range(classes)), average="macro", zero_division=0)),
                        "samples": len(target),
                    }
                )
            print(
                f"  {method}: image {results[-2]['accuracy']:.2%}, "
                f"ensemble {results[-1]['accuracy']:.2%}",
                flush=True,
            )

    fold_results = pd.DataFrame(results)
    summary = (
        fold_results.groupby("model")
        .agg(mean_signer_accuracy=("accuracy", "mean"), min_signer_accuracy=("accuracy", "min"),
             mean_signer_macro_f1=("macro_f1", "mean"), folds=("signer", "nunique"))
        .reset_index()
        .sort_values(["mean_signer_accuracy", "min_signer_accuracy"], ascending=False)
    )
    target_dir = Path(output_dir) if output_dir is not None else project_path(config, "predictions") / "improved_model_cv"
    target_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(target_dir / "training_signer_folds.csv", index=False)
    summary.to_csv(target_dir / "training_signer_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Cross-validation completed in {time.perf_counter() - started:.1f}s; test split was not used.", flush=True)
    return summary


def run_head_cv(config: dict[str, Any], output_dir: str | Path | None = None) -> pd.DataFrame:
    """Compare the legacy and gated-residual classifier heads on training signers only."""
    started = time.perf_counter()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest = load_split_manifest(project_path(config, "splits") / "split_manifest.json")
    rows = manifest["splits"]["train"]
    names = [name for name, _ in sorted(manifest["class_to_idx"].items(), key=lambda pair: pair[1])]
    classes = len(names)
    signers = sorted({row["signer_id"] for row in rows})
    if len(signers) < 3:
        raise ValueError("At least three training signers are required for cross-validation.")
    inputs = _training_landmarks(config, rows, classes)
    labels = torch.tensor([row["label_idx"] for row in rows], dtype=torch.long)
    results: list[dict[str, Any]] = []
    backbone_name = config.get("model", {}).get("backbone", "mobilenet_v3_small")

    for fold_number, signer in enumerate(signers):
        train_idx, validation_idx = fold_indices(rows, signer)
        if set(labels[train_idx].tolist()) != set(range(classes)):
            raise ValueError(f"Fold excluding {signer} does not have every class in training.")
        train_rows = [rows[index] for index in train_idx]
        validation_rows = [rows[index] for index in validation_idx]
        train_labels = labels[train_idx]
        validation_labels = labels[validation_idx]
        fold_seed = int(config["project"].get("seed", 42)) + fold_number
        landmark_probs = _landmark_probabilities(
            inputs[train_idx], train_labels, inputs[validation_idx], classes, fold_seed,
        )
        presence = inputs[validation_idx, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 2].sum(1) > 0
        print(f"Classifier-head fold {signer}", flush=True)
        for head_type in ("legacy", "gated_residual"):
            image_probs = _image_probabilities(
                train_rows,
                train_labels,
                validation_rows,
                fine_tune=False,
                epochs=40,
                seed=fold_seed,
                backbone_name=backbone_name,
                head_type=head_type,
            )
            ensemble_probs = combine_probabilities(landmark_probs, image_probs, presence, 0.1, True)
            for model_name, probabilities in (
                (f"{head_type}_image", image_probs),
                (f"{head_type}_ensemble", ensemble_probs),
            ):
                predictions = probabilities.argmax(1).numpy()
                target = validation_labels.numpy()
                results.append(
                    {
                        "signer": signer,
                        "model": model_name,
                        "accuracy": float(accuracy_score(target, predictions)),
                        "macro_f1": float(
                            f1_score(
                                target,
                                predictions,
                                labels=list(range(classes)),
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        "samples": len(target),
                    }
                )
            print(
                f"  {head_type}: image {results[-2]['accuracy']:.2%}, "
                f"ensemble {results[-1]['accuracy']:.2%}",
                flush=True,
            )

    fold_results = pd.DataFrame(results)
    summary = (
        fold_results.groupby("model")
        .agg(
            mean_signer_accuracy=("accuracy", "mean"),
            min_signer_accuracy=("accuracy", "min"),
            mean_signer_macro_f1=("macro_f1", "mean"),
            folds=("signer", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_signer_accuracy", "min_signer_accuracy"], ascending=False)
    )
    target_dir = Path(output_dir) if output_dir is not None else project_path(config, "predictions") / "improved_model_cv"
    target_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(target_dir / "classifier_head_folds.csv", index=False)
    summary.to_csv(target_dir / "classifier_head_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(
        f"Classifier-head cross-validation completed in {time.perf_counter() - started:.1f}s; "
        "validation/test signers were not used.",
        flush=True,
    )
    return summary


def run_architecture_cv(
    config: dict[str, Any], output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compare MobileNetV3-Small and Large on training-signers-only folds."""
    started = time.perf_counter()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest = load_split_manifest(project_path(config, "splits") / "split_manifest.json")
    rows = manifest["splits"]["train"]
    names = [name for name, _ in sorted(manifest["class_to_idx"].items(), key=lambda pair: pair[1])]
    classes = len(names)
    signers = sorted({row["signer_id"] for row in rows})
    if len(signers) < 3:
        raise ValueError("At least three training signers are required for cross-validation.")
    inputs = _training_landmarks(config, rows, classes)
    labels = torch.tensor([row["label_idx"] for row in rows], dtype=torch.long)
    results: list[dict[str, Any]] = []

    for fold_number, signer in enumerate(signers):
        train_idx, validation_idx = fold_indices(rows, signer)
        if set(labels[train_idx].tolist()) != set(range(classes)):
            raise ValueError(f"Fold excluding {signer} does not have every class in training.")
        train_rows = [rows[index] for index in train_idx]
        validation_rows = [rows[index] for index in validation_idx]
        train_labels = labels[train_idx]
        validation_labels = labels[validation_idx]
        fold_seed = int(config["project"].get("seed", 42)) + fold_number
        landmark_probs = _landmark_probabilities(
            inputs[train_idx], train_labels, inputs[validation_idx], classes, fold_seed,
        )
        presence = inputs[validation_idx, LANDMARK_FEATURE_DIM:LANDMARK_FEATURE_DIM + 2].sum(1) > 0
        print(f"Architecture fold {signer}", flush=True)
        for backbone_name in BACKBONE_WEIGHTS:
            image_probs = _image_probabilities(
                train_rows,
                train_labels,
                validation_rows,
                fine_tune=True,
                epochs=3,
                seed=fold_seed,
                backbone_name=backbone_name,
            )
            for model_name, probabilities in (
                (f"{backbone_name}_image", image_probs),
                (
                    f"{backbone_name}_ensemble",
                    combine_probabilities(landmark_probs, image_probs, presence, 0.1, True),
                ),
            ):
                predictions = probabilities.argmax(1).numpy()
                target = validation_labels.numpy()
                results.append(
                    {
                        "signer": signer,
                        "model": model_name,
                        "accuracy": float(accuracy_score(target, predictions)),
                        "macro_f1": float(
                            f1_score(
                                target,
                                predictions,
                                labels=list(range(classes)),
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        "samples": len(target),
                    }
                )
            print(
                f"  {backbone_name}: image {results[-2]['accuracy']:.2%}, "
                f"ensemble {results[-1]['accuracy']:.2%}",
                flush=True,
            )

    fold_results = pd.DataFrame(results)
    summary = (
        fold_results.groupby("model")
        .agg(
            mean_signer_accuracy=("accuracy", "mean"),
            min_signer_accuracy=("accuracy", "min"),
            mean_signer_macro_f1=("macro_f1", "mean"),
            folds=("signer", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_signer_accuracy", "min_signer_accuracy"], ascending=False)
    )
    target_dir = Path(output_dir) if output_dir is not None else project_path(config, "predictions") / "improved_model_cv"
    target_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(target_dir / "architecture_folds.csv", index=False)
    summary.to_csv(target_dir / "architecture_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(
        f"Architecture cross-validation completed in {time.perf_counter() - started:.1f}s; "
        "validation/test signers were not used.",
        flush=True,
    )
    return summary


def run_landmark_ensemble_cv(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    feature_variant: str = "hand_shape_pose_position",
) -> pd.DataFrame:
    """Compare fixed-size landmark ensembles across training signers only."""
    started = time.perf_counter()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest = load_split_manifest(project_path(config, "splits") / "split_manifest.json")
    rows = manifest["splits"]["train"]
    names = [name for name, _ in sorted(manifest["class_to_idx"].items(), key=lambda pair: pair[1])]
    classes = len(names)
    signers = sorted({row["signer_id"] for row in rows})
    if len(signers) < 3:
        raise ValueError("At least three training signers are required for cross-validation.")
    inputs = _training_landmarks(config, rows, classes)
    labels = torch.tensor([row["label_idx"] for row in rows], dtype=torch.long)
    seed_noise = ((42, 0.02), (43, 0.02), (44, 0.05), (45, 0.0))
    results: list[dict[str, Any]] = []

    for fold_number, signer in enumerate(signers):
        train_idx, validation_idx = fold_indices(rows, signer)
        if set(labels[train_idx].tolist()) != set(range(classes)):
            raise ValueError(f"Fold excluding {signer} does not have every class in training.")
        models = []
        for seed, noise in seed_noise:
            models.append(
                train_models(
                    inputs[train_idx],
                    labels[train_idx],
                    classes,
                    seed + fold_number,
                    noise,
                    (45,),
                    feature_variant,
                )[45]
            )
        with torch.inference_mode():
            member_probabilities = [
                landmark_model_logits(model, inputs[validation_idx]).softmax(1) for model in models
            ]
        validation_labels = labels[validation_idx].numpy()
        for member_count in (1, 2, 4):
            predictions = torch.stack(member_probabilities[:member_count]).mean(0).argmax(1).numpy()
            results.append(
                {
                    "signer": signer,
                    "model": f"landmark_{member_count}_member",
                    "feature_variant": feature_variant,
                    "accuracy": float(accuracy_score(validation_labels, predictions)),
                    "macro_f1": float(
                        f1_score(
                            validation_labels,
                            predictions,
                            labels=list(range(classes)),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "samples": len(validation_labels),
                }
            )
        print(
            f"Fold {signer}: "
            + ", ".join(
                f"{row['model']} {row['accuracy']:.2%}"
                for row in results[-3:]
            ),
            flush=True,
        )

    fold_results = pd.DataFrame(results)
    summary = (
        fold_results.groupby(["feature_variant", "model"])
        .agg(
            mean_signer_accuracy=("accuracy", "mean"),
            min_signer_accuracy=("accuracy", "min"),
            mean_signer_macro_f1=("macro_f1", "mean"),
            folds=("signer", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_signer_accuracy", "min_signer_accuracy"], ascending=False)
    )
    target_dir = Path(output_dir) if output_dir is not None else project_path(config, "predictions") / "improved_model_cv"
    target_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(target_dir / "landmark_ensemble_folds.csv", index=False)
    summary.to_csv(target_dir / "landmark_ensemble_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(
        f"Landmark ensemble cross-validation completed in {time.perf_counter() - started:.1f}s; "
        "test split was not used.",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run_experiment(load_config())
