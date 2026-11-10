"""Reproducibility, hardware inspection, logging, and plot helpers."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set common Python and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Return a supported accelerator if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_report() -> dict[str, Any]:
    """Report framework-visible accelerator and memory information."""
    device = get_device()
    report: dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": None,
        "gpu_total_memory_gib": None,
        "gpu_free_memory_gib": None,
        "mixed_precision_supported": device.type == "cuda",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free_memory, _ = torch.cuda.mem_get_info(device)
        report["gpu_name"] = properties.name
        report["gpu_total_memory_gib"] = round(properties.total_memory / (1024**3), 2)
        report["gpu_free_memory_gib"] = round(free_memory / (1024**3), 2)
    elif device.type == "mps":
        report["gpu_name"] = "Apple Metal Performance Shaders"
    return report


def select_batch_size(config: dict[str, Any]) -> int:
    """Choose a conservative starting batch size based on framework-visible hardware."""
    training = config.get("training", {})
    configured = int(training.get("batch_size", 16))
    if configured < 1:
        raise ValueError("training.batch_size must be positive.")
    if not training.get("auto_batch_size", True):
        return configured
    device = get_device()
    if device.type == "cuda":
        free_memory_gib = torch.cuda.mem_get_info(device)[0] / (1024**3)
        recommendation = (
            1 if free_memory_gib < 4 else
            2 if free_memory_gib < 8 else
            4 if free_memory_gib < 16 else
            8 if free_memory_gib < 32 else 16
        )
    elif device.type == "mps":
        recommendation = 4
    else:
        recommendation = 2
    return max(1, min(configured, recommendation))


def configure_logging(output_dir: str | Path, name: str = "isl") -> logging.Logger:
    """Configure console and file logging without duplicating handlers."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(directory / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    return logger


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Persist JSON-serializable metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def plot_history(history: list[dict[str, Any]], output_path: str | Path, title: str) -> Path:
    """Save train/validation loss and accuracy curves."""
    if not history:
        raise ValueError("Cannot plot empty training history.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, axis, label in (
        ("loss", axes[0], "Loss"),
        ("accuracy", axes[1], "Accuracy"),
    ):
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train")
        axis.plot(epochs, [row[f"validation_{metric}"] for row in history], label="Validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target
