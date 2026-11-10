import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.dataset import check_checkpoint_split, create_splits, split_fingerprint, validate_split_groups


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {"a": 0, "b": 1, "c": 2}
        self.records = [
            {
                "label": label,
                "label_idx": index,
                "duplicate_group": f"{label}-{sample // 2}",
                "signer_id": f"user-{sample % 5}",
            }
            for label, index in self.mapping.items()
            for sample in range(200)
        ]
        self.ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}

    @patch("src.dataset._assign_duplicate_groups")
    def test_image_split_is_reproducible_complete_and_duplicate_aware(self, _assign):
        first, mode = create_splits(self.records, self.mapping, self.ratios, 42, strategy="image")
        second, _ = create_splits(self.records, self.mapping, self.ratios, 42, strategy="image")
        self.assertEqual(first, second)
        self.assertEqual(mode, "duplicate-aware random")
        for name, rows in first.items():
            self.assertEqual({row["label_idx"] for row in rows}, set(self.mapping.values()))
            self.assertAlmostEqual(len(rows) / len(self.records), self.ratios[name], delta=0.02)
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(
                {row["duplicate_group"] for row in first[left]}
                & {row["duplicate_group"] for row in first[right]}
            )
        self.assertTrue(
            {row["signer_id"] for row in first["train"]}
            & {row["signer_id"] for row in first["validation"]}
        )

    @patch("src.dataset._assign_duplicate_groups")
    def test_impossible_class_coverage_is_rejected(self, _assign):
        rows = [row for row in self.records if row["label"] != "c"] + [self.records[-1]]
        with self.assertRaisesRegex(ValueError, "missing classes"):
            create_splits(rows, self.mapping, self.ratios, 42, strategy="image")

    def test_conflicting_and_unknown_strategies_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot require signer independence"):
            create_splits(self.records, self.mapping, self.ratios, 42, True, strategy="image")
        with self.assertRaisesRegex(ValueError, "Unknown split strategy"):
            create_splits(self.records, self.mapping, self.ratios, 42, strategy="unknown")

    def test_checkpoint_must_match_image_split(self):
        manifest = {
            "split_mode": "duplicate-aware random",
            "splits": {
                name: [{"relative_path": name, "sha256": name, "label_idx": 0}]
                for name in ("train", "validation", "test")
            },
        }
        checkpoint = {"split_fingerprint": split_fingerprint(manifest)}
        check_checkpoint_split(checkpoint, manifest)
        with self.assertRaisesRegex(ValueError, "does not match"):
            check_checkpoint_split({}, manifest)
        manifest["splits"]["train"][0]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "does not match"):
            check_checkpoint_split(checkpoint, manifest)

    def test_split_group_validation(self):
        manifest = {
            "split_mode": "duplicate-aware random",
            "splits": {
                name: [{"duplicate_group": name, "signer_id": "shared"}]
                for name in ("train", "validation", "test")
            },
        }
        validate_split_groups(manifest)
        manifest["splits"]["test"][0]["duplicate_group"] = "train"
        with self.assertRaisesRegex(ValueError, "duplicate_group overlap"):
            validate_split_groups(manifest)
        manifest["splits"]["test"][0]["duplicate_group"] = None
        with self.assertRaisesRegex(ValueError, "Missing duplicate_group"):
            validate_split_groups(manifest)
        manifest["splits"]["test"][0]["duplicate_group"] = "test"
        manifest["split_mode"] = "signer-independent"
        with self.assertRaisesRegex(ValueError, "signer_id overlap"):
            validate_split_groups(manifest)
        for name, rows in manifest["splits"].items():
            rows[0]["signer_id"] = name
        validate_split_groups(manifest)

    def test_discover_images_uses_only_configured_class_folders(self):
        from src.dataset import discover_images

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for class_name in ("keep-a", "keep-b", "excluded"):
                class_dir = root / class_name
                class_dir.mkdir()
                Image.new("RGB", (4, 4), color="red").save(class_dir / "sample.png")
            config = {
                "_project_root": root,
                "data": {
                    "dataset_root": root,
                    "included_classes": ["keep-b", "keep-a"],
                    "expected_num_classes": 2,
                },
            }
            records, mapping = discover_images(config)
            self.assertEqual(mapping, {"keep-a": 0, "keep-b": 1})
            self.assertEqual({row["label"] for row in records}, {"keep-a", "keep-b"})

    def test_discover_images_rejects_missing_configured_class(self):
        from src.dataset import discover_images

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "present").mkdir()
            config = {
                "_project_root": root,
                "data": {
                    "dataset_root": root,
                    "included_classes": ["present", "missing"],
                    "expected_num_classes": 2,
                },
            }
            with self.assertRaisesRegex(ValueError, "class folders are missing"):
                discover_images(config)


if __name__ == "__main__":
    unittest.main()
