"""Dataset indexing, leakage-aware splitting, and PyTorch dataset loaders."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import project_path, resolve_path
from src.landmark_extractor import LANDMARK_CACHE_FORMAT_VERSION
from src.transforms import build_transforms

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _read_metadata(config: dict[str, Any]) -> dict[str, str]:
    """Read optional image-to-signer metadata; returns empty when not configured."""
    data_config = config.get("data", {})
    metadata_path = resolve_path(config, data_config.get("metadata_file"))
    if metadata_path is None:
        return {}
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Configured metadata file does not exist: {metadata_path}")
    image_column = data_config.get("metadata_image_column", "image")
    signer_column = data_config.get("metadata_signer_column", "signer")
    root = resolve_path(config, data_config.get("dataset_root"))
    mapping: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or image_column not in reader.fieldnames or signer_column not in reader.fieldnames:
            raise ValueError(
                f"Metadata must contain columns {image_column!r} and {signer_column!r}: {metadata_path}"
            )
        for row in reader:
            image_value, signer_value = row.get(image_column), row.get(signer_column)
            if not image_value or not signer_value:
                continue
            image_path = Path(image_value)
            if root is not None and not image_path.is_absolute():
                image_path = root / image_path
            resolved_image = str(image_path.resolve())
            signer = signer_value.strip()
            previous_signer = mapping.get(resolved_image)
            if previous_signer is not None and previous_signer != signer:
                raise ValueError(
                    f"Conflicting signer IDs for metadata image {resolved_image}: "
                    f"{previous_signer!r} and {signer!r}"
                )
            mapping[resolved_image] = signer
    return mapping


def discover_images(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Index class-folder images, checking labels, readability, and exact duplicates."""
    root = resolve_path(config, config.get("data", {}).get("dataset_root"))
    if root is None:
        raise ValueError("Set data.dataset_root in configs/config.yaml to the extracted dataset root.")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    discovered_class_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.casefold())
    if not discovered_class_dirs:
        raise ValueError(f"No class subdirectories were found in dataset root: {root}")
    included_classes = config.get("data", {}).get("included_classes")
    if included_classes is not None:
        if not isinstance(included_classes, list) or not included_classes or len(included_classes) != len(set(included_classes)):
            raise ValueError("data.included_classes must be a non-empty list of unique class-folder names.")
        available = {path.name: path for path in discovered_class_dirs}
        missing = sorted(set(included_classes) - set(available))
        if missing:
            raise ValueError(f"Configured class folders are missing from the dataset: {missing}")
        class_dirs = sorted((available[name] for name in included_classes), key=lambda path: path.name.casefold())
    else:
        class_dirs = discovered_class_dirs
    expected = config.get("data", {}).get("expected_num_classes")
    if expected is not None and len(class_dirs) != int(expected):
        raise ValueError(f"Expected {expected} included class directories but found {len(class_dirs)}.")
    class_to_idx = {path.name: index for index, path in enumerate(class_dirs)}
    signer_mapping = _read_metadata(config)
    records: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    unreadable: list[str] = []
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
                digest = hashlib.sha256()
                with image_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                unreadable.append(f"{image_path}: {error}")
                continue
            checksum = digest.hexdigest()
            duplicate_of = seen_hashes.get(checksum)
            if duplicate_of is None:
                seen_hashes[checksum] = str(image_path.resolve())
            signer = signer_mapping.get(str(image_path.resolve()))
            records.append(
                {
                    "path": str(image_path.resolve()),
                    "relative_path": image_path.relative_to(root).as_posix(),
                    "label": class_dir.name,
                    "label_idx": class_to_idx[class_dir.name],
                    "sha256": checksum,
                    "duplicate_group": checksum,
                    "signer_id": signer,
                    "is_exact_duplicate": duplicate_of is not None,
                    "duplicate_of": duplicate_of,
                }
            )
    if unreadable:
        LOGGER.warning("Skipped %d unreadable image(s). First examples: %s", len(unreadable), unreadable[:5])
    if not records:
        raise ValueError(f"No readable supported image files found beneath {root}")
    observed = {record["label"] for record in records}
    empty_classes = sorted(set(class_to_idx) - observed)
    if empty_classes:
        raise ValueError(f"Class directories contain no readable images: {empty_classes}")
    LOGGER.info("Indexed %d images across %d classes.", len(records), len(class_to_idx))
    return records, class_to_idx


def _assign_duplicate_groups(records: list[dict[str, Any]]) -> None:
    """Merge exact duplicates and perceptual-hash near duplicates into split groups."""
    from src.landmark_extractor import perceptual_hash

    parent = list(range(len(records)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    hash_indices: dict[str, int] = {}
    for index, record in enumerate(records):
        digest = record["sha256"]
        if digest in hash_indices:
            union(index, hash_indices[digest])
        else:
            hash_indices[digest] = index

    hash_buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    hashes: list[int | None] = []
    mean_colors: list[np.ndarray | None] = []
    for index, record in enumerate(records):
        try:
            with Image.open(record["path"]) as image:
                value = perceptual_hash(image)
                small_rgb = np.asarray(image.convert("RGB").resize((16, 16)), dtype=np.float32)
                mean_color = small_rgb.mean(axis=(0, 1))
        except (OSError, UnidentifiedImageError, ValueError) as error:
            LOGGER.warning("Could not compute near-duplicate hash for %s: %s", record["relative_path"], error)
            value = None
            mean_color = None
        hashes.append(value)
        mean_colors.append(mean_color)
        if value is not None:
            for chunk_index in range(8):
                hash_buckets[(chunk_index, (value >> (chunk_index * 8)) & 0xFF)].add(index)

    candidates: set[tuple[int, int]] = set()
    max_candidates = 200_000
    truncated = False
    for bucket in hash_buckets.values():
        ordered = sorted(bucket)
        for offset, first in enumerate(ordered):
            for second in ordered[offset + 1 :]:
                candidates.add((first, second))
                if len(candidates) >= max_candidates:
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break
    if truncated:
        LOGGER.warning("Near-duplicate candidate cap reached; some near duplicates may remain undetected.")
    for first, second in candidates:
        first_hash, second_hash = hashes[first], hashes[second]
        first_color, second_color = mean_colors[first], mean_colors[second]
        if first_hash is not None and second_hash is not None and first_color is not None and second_color is not None:
            color_distance = float(np.linalg.norm(first_color - second_color))
            if (first_hash ^ second_hash).bit_count() <= 4 and color_distance <= 60:
                union(first, second)
    for index, record in enumerate(records):
        record["duplicate_group"] = f"dup-{find(index)}"


def _check_split_coverage(splits: dict[str, list[dict[str, Any]]], class_to_idx: dict[str, int]) -> None:
    all_classes = set(class_to_idx.values())
    for name, records in splits.items():
        actual = {record["label_idx"] for record in records}
        missing = sorted(all_classes - actual)
        if missing:
            missing_labels = [label for label, idx in class_to_idx.items() if idx in missing]
            raise ValueError(
                f"{name} split is missing classes {missing_labels}. More samples/groups per class "
                "or a different split strategy is needed; no split manifest was saved."
            )


def create_splits(
    records: list[dict[str, Any]],
    class_to_idx: dict[str, int],
    ratios: dict[str, float],
    seed: int,
    require_signer_independent: bool = False,
    strategy: str = "auto",
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Create reproducible duplicate-aware splits, optionally isolating signers."""
    if strategy not in {"auto", "image", "signer"}:
        raise ValueError(f"Unknown split strategy: {strategy!r}. Choose auto, image, or signer.")
    if strategy == "image" and require_signer_independent:
        raise ValueError("Image splitting cannot require signer independence.")
    names = ("train", "validation", "test")
    fractions = [float(ratios[name]) for name in names]
    if any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0, atol=1e-6):
        raise ValueError("Train/validation/test ratios must be positive and sum to 1.")
    _assign_duplicate_groups(records)
    signers = [record.get("signer_id") for record in records]
    signer_metadata_complete = bool(signers) and all(signers)
    signer_available = signer_metadata_complete and len(set(signers)) >= 3
    signer_independent = signer_available and strategy != "image"
    if (require_signer_independent or strategy == "signer") and not signer_independent:
        raise ValueError(
            "Signer-independent splitting was required, but verified signer IDs are incomplete "
            "or fewer than three distinct signers are available."
        )

    indices = np.arange(len(records))
    labels = np.asarray([record["label_idx"] for record in records])
    if signer_independent:
        parent = list(range(len(records)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        signer_first: dict[str, int] = {}
        duplicate_first: dict[str, int] = {}
        for index, record in enumerate(records):
            signer = str(record["signer_id"])
            duplicate = str(record["duplicate_group"])
            if signer in signer_first:
                union(index, signer_first[signer])
            else:
                signer_first[signer] = index
            if duplicate in duplicate_first:
                union(index, duplicate_first[duplicate])
            else:
                duplicate_first[duplicate] = index
        groups = np.asarray([f"signer-duplicate-{find(index)}" for index in range(len(records))])
        split_mode = "signer-independent, exact/near-duplicate-aware"
    else:
        groups = np.asarray([record["duplicate_group"] for record in records])
        split_mode = "duplicate-aware random" if len(set(groups)) < len(records) else "stratified random"
        if strategy != "image" and signer_metadata_complete and not signer_available:
            LOGGER.warning(
                "Signer metadata is complete but fewer than three distinct signers are available; "
                "signer-independent three-way splitting is not feasible."
            )
        LOGGER.warning(
            "Using %s splitting without signer isolation; evaluate unseen signers separately.",
            split_mode,
        )

    if len(set(groups)) == len(records):
        train_idx, remainder_idx = train_test_split(
            indices, test_size=fractions[1] + fractions[2], random_state=seed, stratify=labels
        )
        remainder_labels = labels[remainder_idx]
        val_share = fractions[1] / (fractions[1] + fractions[2])
        val_idx, test_idx = train_test_split(
            remainder_idx, test_size=1 - val_share, random_state=seed, stratify=remainder_labels
        )
    else:
        if len(set(groups)) < 3:
            raise ValueError("At least three independent signer/duplicate groups are required for three splits.")
        splitter = GroupShuffleSplit(n_splits=100, test_size=fractions[2], random_state=seed)
        candidates = list(splitter.split(indices, labels, groups))
        all_classes = set(class_to_idx.values())

        def candidate_score(
            candidate_indices: np.ndarray,
            held_out_indices: np.ndarray,
            expected_fraction: float,
            total_count: int,
        ) -> tuple[int, float, float]:
            held_out_classes = set(labels[held_out_indices])
            retained_classes = set(labels[candidate_indices])
            missing = len(all_classes - held_out_classes) + len(all_classes - retained_classes)
            fraction_error = abs(len(held_out_indices) / total_count - expected_fraction)
            expected_counts = np.bincount(labels[np.concatenate((candidate_indices, held_out_indices))], minlength=len(class_to_idx))
            actual_counts = np.bincount(labels[held_out_indices], minlength=len(class_to_idx))
            distribution_error = float(np.abs(actual_counts / len(held_out_indices) - expected_counts / total_count).sum())
            return missing, fraction_error, distribution_error

        train_val_idx, test_idx = min(
            candidates,
            key=lambda pair: candidate_score(pair[0], pair[1], fractions[2], len(records)),
        )
        train_val_groups = groups[train_val_idx]
        unique_groups = len(np.unique(train_val_groups))
        if unique_groups < 2:
            raise ValueError("Not enough independent groups remain for train/validation splitting.")
        val_share = fractions[1] / (fractions[0] + fractions[1])
        inner_splitter = GroupShuffleSplit(n_splits=100, test_size=val_share, random_state=seed + 1)
        inner_candidates = list(inner_splitter.split(train_val_idx, labels[train_val_idx], train_val_groups))
        relative_train_idx, relative_val_idx = min(
            inner_candidates,
            key=lambda pair: candidate_score(
                train_val_idx[pair[0]], train_val_idx[pair[1]], val_share, len(train_val_idx)
            ),
        )
        train_idx = train_val_idx[relative_train_idx]
        val_idx = train_val_idx[relative_val_idx]

    splits = {
        "train": [records[int(index)] for index in train_idx],
        "validation": [records[int(index)] for index in val_idx],
        "test": [records[int(index)] for index in test_idx],
    }
    _check_split_coverage(splits, class_to_idx)
    split_groups = {name: {record["duplicate_group"] for record in values} for name, values in splits.items()}
    if any(split_groups[first] & split_groups[second] for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Duplicate group leakage detected across splits.")
    if signer_independent:
        split_signers = {name: {record["signer_id"] for record in values} for name, values in splits.items()}
        if any(split_signers[a] & split_signers[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise RuntimeError("Signer leakage detected across splits.")
    return splits, split_mode


def save_split_manifest(
    splits: dict[str, list[dict[str, Any]]],
    class_to_idx: dict[str, int],
    split_mode: str,
    seed: int,
    config: dict[str, Any],
) -> Path:
    """Persist split rows and mapping as a reproducible JSON manifest."""
    output = project_path(config, "splits")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "split_manifest.json"
    payload = {
        "seed": seed,
        "split_mode": split_mode,
        "class_to_idx": class_to_idx,
        "splits": {
            name: [{key: value for key, value in row.items() if key != "path"} | {"path": row["path"]} for row in rows]
            for name, rows in splits.items()
        },
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata_path = project_path(config, "processed_data")
    metadata_path.mkdir(parents=True, exist_ok=True)
    with (metadata_path / "image_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["relative_path", "label", "label_idx", "sha256", "duplicate_group", "signer_id", "is_exact_duplicate", "duplicate_of"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rows in splits.values():
            writer.writerows(rows)
    return target


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a split manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}. Run notebook 02 first.")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"class_to_idx", "splits"}
    if not required.issubset(payload) or not {"train", "validation", "test"}.issubset(payload["splits"]):
        raise ValueError(f"Invalid split manifest structure: {manifest_path}")
    return payload


def validate_split_groups(manifest: dict[str, Any]) -> None:
    """Check that duplicate groups and, when required, signers do not cross splits."""
    splits = manifest["splits"]
    fields = ("signer_id", "duplicate_group") if manifest.get("split_mode", "").startswith("signer-independent") else ("duplicate_group",)
    for field in fields:
        groups = {
            name: {row.get(field) for row in splits[name]}
            for name in ("train", "validation", "test")
        }
        if any(None in values or "" in values for values in groups.values()):
            raise ValueError(f"Missing {field} in split manifest.")
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if groups[first] & groups[second]:
                raise ValueError(f"{field} overlap between splits.")


def split_fingerprint(manifest: dict[str, Any]) -> str:
    """Identify split membership and labels, independently of row ordering."""
    digest = hashlib.sha256()
    for name in ("train", "validation", "test"):
        rows = sorted(
            (row["relative_path"], row["sha256"], row["label_idx"])
            for row in manifest["splits"][name]
        )
        digest.update(json.dumps([name, rows], ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def check_checkpoint_split(checkpoint: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Reject checkpoints trained on a different partition of the dataset."""
    actual = checkpoint.get("split_fingerprint")
    if actual is None and manifest.get("split_mode", "").startswith("signer-independent"):
        LOGGER.warning("Legacy checkpoint has no split fingerprint; its split cannot be verified.")
        return
    if actual != split_fingerprint(manifest):
        raise ValueError("Checkpoint does not match the current split manifest. Retrain the model.")


class ISLImageDataset(Dataset):
    """Image dataset optionally paired with cached hand-landmark features."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        image_size: int = 224,
        train: bool = False,
        landmark_cache: str | Path | None = None,
    ) -> None:
        self.records = records
        self.transform = build_transforms(image_size=image_size, train=train)
        self.landmarks: dict[str, dict[str, Any]] = {}
        if landmark_cache is not None:
            cache_path = Path(landmark_cache)
            if not cache_path.is_file():
                raise FileNotFoundError(f"Landmark cache not found: {cache_path}. Run notebook 03 first.")
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("format_version") != LANDMARK_CACHE_FORMAT_VERSION:
                raise ValueError(f"Unsupported or stale landmark cache format in {cache_path}. Re-run notebook 03.")
            self.landmarks = payload["items"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            with Image.open(record["path"]) as image:
                tensor = self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Image became unreadable after indexing: {record['path']}") from error
        item: dict[str, Any] = {
            "image": tensor,
            "label": torch.tensor(record["label_idx"], dtype=torch.long),
            "path": record["path"],
            "relative_path": record["relative_path"],
        }
        if self.landmarks:
            cached = self.landmarks.get(record["relative_path"])
            if cached is None:
                raise KeyError(f"Landmark cache has no entry for {record['relative_path']}")
            item["landmarks"] = torch.tensor(cached["features"], dtype=torch.float32)
            item["presence"] = torch.tensor(cached["presence"], dtype=torch.float32)
            item["handedness_reliable"] = torch.tensor(
                cached.get("handedness_reliable", [0.0, 0.0]), dtype=torch.float32
            )
        return item


def make_dataloaders(
    manifest: dict[str, Any],
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    landmark_cache: str | Path | None = None,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Build train/validation/test loaders with augmentation on training only."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders = {}
    for split_name in ("train", "validation", "test"):
        dataset = ISLImageDataset(
            manifest["splits"][split_name],
            image_size=image_size,
            train=split_name == "train",
            landmark_cache=landmark_cache,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if split_name == "train" else None,
        )
    return loaders
