"""Persist final test metrics, per-class reports, predictions, and confusion matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_evaluation(result: dict[str, Any], model_name: str, output_dir: str | Path) -> dict[str, Path]:
    """Save test metrics, per-class report, predictions, and confusion-matrix plot."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metric_path = output / f"{model_name}_test_metrics.json"
    metric_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    class_rows = [
        {"class": name, **values}
        for name, values in result["classification_report"].items()
        if name in result["class_names"]
    ]
    report_path = output / f"{model_name}_per_class.csv"
    pd.DataFrame(class_rows).to_csv(report_path, index=False)
    predictions_path = output / f"{model_name}_test_predictions.csv"
    pd.DataFrame(
        {
            "path": result["paths"],
            "true_label": [result["class_names"][index] for index in result["targets"]],
            "predicted_label": [result["class_names"][index] for index in result["predictions"]],
        }
    ).to_csv(predictions_path, index=False)
    confusion_path = output / f"{model_name}_confusion_matrix.png"
    figure, axis = plt.subplots(figsize=(max(8, len(result["class_names"]) * 0.55), 7))
    sns.heatmap(
        result["confusion_matrix"], annot=False, cmap="Blues", xticklabels=result["class_names"],
        yticklabels=result["class_names"], ax=axis,
    )
    axis.set(title=f"{model_name} — held-out test set", xlabel="Predicted label", ylabel="True label")
    plt.xticks(rotation=60, ha="right")
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)
    return {"metrics": metric_path, "per_class": report_path, "predictions": predictions_path, "confusion": confusion_path}
