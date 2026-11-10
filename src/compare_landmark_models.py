"""Compare saved baseline and improved models on the same manifest test images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from src.config import project_path
from src.dataset import check_checkpoint_split
from src.ensemble_inference import load_landmark_predictor
from src.landmark_image_ensemble import ImageRows, image_transform
from src.metrics import save_evaluation
from src.robust_landmark_ensemble import load_inputs


def compare_models(
    config: dict[str, Any],
    improved_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest, inputs, labels = load_inputs(config, manifest_path)
    rows, mapping = manifest["splits"]["test"], manifest["class_to_idx"]
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    targets = labels["test"].numpy()
    ids = list(range(len(names)))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, path, kind in (
        (
            "baseline",
            Path(baseline_path) if baseline_path is not None else project_path(config, "checkpoints") / "landmark_mlp_best.pt",
            "landmark_mlp",
        ),
        ("improved", Path(improved_path), "robust_landmark_ensemble"),
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        check_checkpoint_split(checkpoint, manifest)
        model = load_landmark_predictor(path)
        if model.model_type != kind or model.class_to_idx != mapping or model.input_is_mirrored != bool(config["data"].get("images_are_mirrored", False)):
            raise ValueError(f"Checkpoint type, mapping or mirroring does not match data: {path}")
        with torch.inference_mode():
            if kind == "landmark_mlp":
                probs = model.probabilities(inputs["test"])
            else:
                loader = DataLoader(
                    ImageRows(rows, image_transform(checkpoint["weights"])),
                    batch_size=32, shuffle=False, num_workers=0,
                )
                offset, batches = 0, []
                for images in loader:
                    batches.append(model.probabilities(inputs["test"][offset:offset + len(images)], images))
                    offset += len(images)
                if offset != len(rows):
                    raise RuntimeError("Image and landmark test inputs are misaligned.")
                probs = torch.cat(batches)
        predicted = probs.argmax(1).numpy()
        metrics = {
            "accuracy": float(accuracy_score(targets, predicted)),
            "macro_precision": float(precision_score(targets, predicted, labels=ids, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(targets, predicted, labels=ids, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(targets, predicted, labels=ids, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(targets, predicted, labels=ids, average="weighted", zero_division=0)),
            "test_samples": len(targets),
            "evaluation_note": (
                "Exploratory signer-independent evaluation; signer IDs are inferred from folder names and near-duplicate "
                "screening may miss matches."
                if manifest.get("split_mode", "").startswith("signer-independent")
                else "Exploratory evaluation; signers may overlap across splits and near-duplicate screening may miss matches."
            ),
        }
        result = {
            "metrics": metrics, "targets": targets.tolist(), "predictions": predicted.tolist(),
            "paths": [row["path"] for row in rows], "class_names": names,
            "confusion_matrix": confusion_matrix(targets, predicted, labels=ids),
            "classification_report": classification_report(targets, predicted, labels=ids, target_names=names, output_dict=True, zero_division=0),
        }
        save_evaluation(result, name, output)
        results[name] = result
    summary = pd.DataFrame([{"model": name, **r["metrics"]} for name, r in results.items()])
    summary.to_csv(output / "comparison_summary.csv", index=False)
    per_class = pd.DataFrame([
        {"model": model, "class": name, **result["classification_report"][name]}
        for model, result in results.items() for name in names
    ])
    per_class.to_csv(output / "comparison_per_class.csv", index=False)
    return {"summary": summary, "per_class": per_class, "results": results}
