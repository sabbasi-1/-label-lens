import unittest
from pathlib import Path
from unittest.mock import patch

from yolo_reviewer.core import (
    Box,
    Dataset,
    DatasetLoadCancelled,
    ImageRecord,
    ReviewState,
    box_quality_issues,
    discover_dataset,
    label_path_for,
    load_labels,
    replace_class_in_records,
    save_labels,
    trash_record,
)

TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class CoreTests(unittest.TestCase):
    def test_label_path_mirrors_images_directory(self) -> None:
        root = TEST_TEMP_ROOT / "paths"
        root.mkdir(parents=True, exist_ok=True)
        image = root / "images" / "train" / "sample.jpg"
        self.assertEqual(
            label_path_for(image, root),
            root / "labels" / "train" / "sample.txt",
        )

    def test_load_labels_reports_structural_problems(self) -> None:
        root = TEST_TEMP_ROOT / "problems"
        root.mkdir(parents=True, exist_ok=True)
        label = root / "bad.txt"
        label.write_text(
            "0 0.5 0.5 0.2 0.2\n"
            "3 0.5 0.5 0.2 0.2\n"
            "0 1.1 0.5 0.2 0.2\n"
            "not a valid row\n",
            encoding="utf-8",
        )
        boxes, issues = load_labels(label, class_count=2)
        self.assertEqual(len(boxes), 3)
        kinds = {issue.kind for issue in issues}
        self.assertTrue({
            "unknown-class",
            "out-of-range",
            "out-of-bounds",
            "unsupported-row",
            "duplicate-box",
        }.issubset(kinds))

    def test_discover_yaml_dataset_and_save_backup(self) -> None:
        root = TEST_TEMP_ROOT / "dataset"
        image = root / "images" / "train" / "one.jpg"
        label = root / "labels" / "train" / "one.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"not decoded by core")
        label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        (root / "data.yaml").write_text(
            "path: .\ntrain: images/train\nnames: [cat, dog]\n", encoding="utf-8"
        )

        dataset = discover_dataset(root / "data.yaml")
        self.assertEqual(dataset.names, ["cat", "dog"])
        self.assertEqual(len(dataset.records), 1)
        dataset.records[0].boxes[0].class_id = 1
        save_labels(dataset.records[0])
        self.assertTrue(label.read_text(encoding="utf-8").startswith("1 "))
        self.assertTrue(
            label.with_suffix(".txt.bak").read_text(encoding="utf-8").startswith("0 ")
        )

    def test_discovery_reports_progress_and_can_be_cancelled(self) -> None:
        root = TEST_TEMP_ROOT / "progress"
        image_dir = root / "images" / "train"
        label_dir = root / "labels" / "train"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            (image_dir / f"{index}.jpg").write_bytes(b"image")
            (label_dir / f"{index}.txt").write_text(
                f"{index % 2} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        yaml_path = root / "data.yaml"
        yaml_path.write_text(
            "path: .\ntrain: images/train\nnames: [cat, dog]\n",
            encoding="utf-8",
        )
        updates: list[tuple[int, int, str]] = []
        dataset = discover_dataset(
            yaml_path,
            progress=lambda current, total, message: updates.append(
                (current, total, message)
            ),
        )
        self.assertEqual(len(dataset.records), 3)
        self.assertEqual(updates[-1][0:2], (3, 3))
        self.assertTrue(any("Index" in update[2] for update in updates))

        with self.assertRaises(DatasetLoadCancelled):
            discover_dataset(yaml_path, cancelled=lambda: True)

    def test_discovery_preserves_declared_dataset_splits(self) -> None:
        root = TEST_TEMP_ROOT / "splits"
        for split in ("train", "valid", "test"):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / f"{split}.jpg").write_bytes(b"image")
            (label_dir / f"{split}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        yaml_path = root / "data.yaml"
        yaml_path.write_text(
            "path: .\n"
            "train: images/train\n"
            "val: images/valid\n"
            "test: images/test\n"
            "names: [person]\n",
            encoding="utf-8",
        )
        dataset = discover_dataset(yaml_path)
        self.assertEqual(dataset.available_splits, ["train", "val", "test"])
        self.assertEqual(
            {record.image_path.stem: record.split for record in dataset.records},
            {"train": "train", "valid": "val", "test": "test"},
        )

    def test_review_state_round_trip(self) -> None:
        root = TEST_TEMP_ROOT / "state"
        root.mkdir(parents=True, exist_ok=True)
        path = root / ".yolo-review-state.json"
        ReviewState({"images/a.jpg"}, {"images/b.jpg"}, "images/a.jpg").save(path)
        loaded = ReviewState.load(path)
        self.assertEqual(loaded.reviewed, {"images/a.jpg"})
        self.assertEqual(loaded.flagged, {"images/b.jpg"})
        self.assertEqual(loaded.last_image, "images/a.jpg")

    def test_mixed_detection_and_polygon_rows_round_trip_safely(self) -> None:
        root = TEST_TEMP_ROOT / "unsupported"
        root.mkdir(parents=True, exist_ok=True)
        label = root / "polygon.txt"
        original = (
            "0 0.5 0.5 0.2 0.2\n"
            "1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        )
        label.write_text(original, encoding="utf-8")
        boxes, issues = load_labels(label, class_count=2)
        self.assertEqual(len(boxes), 2)
        self.assertIsNone(boxes[0].polygon)
        self.assertIsNotNone(boxes[1].polygon)
        self.assertNotIn("unsupported-row", {issue.kind for issue in issues})
        record = ImageRecord(root / "image.jpg", label, boxes, issues)
        boxes[1].class_id = 0
        boxes[1].x += 0.05
        save_labels(record, class_count=2)
        rows = label.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows[0].split()), 5)
        self.assertEqual(len(rows[1].split()), 9)
        values = [float(value) for value in rows[1].split()[1:]]
        self.assertAlmostEqual(values[0], 0.15)
        self.assertAlmostEqual(values[2], 0.95)

    def test_save_refuses_to_drop_truly_unsupported_rows(self) -> None:
        root = TEST_TEMP_ROOT / "malformed"
        root.mkdir(parents=True, exist_ok=True)
        label = root / "bad.txt"
        original = "0 0.1 0.1 0.2 0.2 extra\n"
        label.write_text(original, encoding="utf-8")
        boxes, issues = load_labels(label, class_count=1)
        record = ImageRecord(root / "image.jpg", label, boxes, issues)
        with self.assertRaises(ValueError):
            save_labels(record)
        self.assertEqual(label.read_text(encoding="utf-8"), original)

    def test_trash_record_moves_image_label_and_backup_recoverably(self) -> None:
        root = TEST_TEMP_ROOT / "trash"
        image = root / "train" / "images" / "one.jpg"
        label = root / "train" / "labels" / "one.txt"
        backup = label.with_suffix(".txt.bak")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        backup.write_text("backup\n", encoding="utf-8")
        record = ImageRecord(image, label)
        trash = trash_record(Dataset(root, ["person"], [record]), record)
        self.assertFalse(image.exists())
        self.assertFalse(label.exists())
        self.assertFalse(backup.exists())
        self.assertTrue((trash / "train" / "images" / "one.jpg").exists())
        self.assertTrue((trash / "train" / "labels" / "one.txt").exists())
        self.assertTrue((trash / "train" / "labels" / "one.txt.bak").exists())

    def test_bulk_class_replacement_saves_boxes_and_polygons(self) -> None:
        root = TEST_TEMP_ROOT / "bulk-replace"
        root.mkdir(parents=True, exist_ok=True)
        first_label = root / "first.txt"
        second_label = root / "second.txt"
        first_label.write_text(
            "1 0.5 0.5 0.2 0.2\n"
            "1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n",
            encoding="utf-8",
        )
        second_label.write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.7 0.7 0.1 0.1\n",
            encoding="utf-8",
        )
        first_boxes, first_issues = load_labels(first_label, 3)
        second_boxes, second_issues = load_labels(second_label, 3)
        records = [
            ImageRecord(root / "first.jpg", first_label, first_boxes, first_issues),
            ImageRecord(
                root / "second.jpg", second_label, second_boxes, second_issues
            ),
        ]
        file_count, annotation_count, backup = replace_class_in_records(
            records, 1, 2, 3, root
        )
        self.assertEqual((file_count, annotation_count), (2, 3))
        self.assertTrue((backup / "manifest.json").exists())
        self.assertEqual(
            len(list((backup / "labels").rglob("*.txt"))),
            2,
        )
        self.assertEqual(
            [box.class_id for record in records for box in record.boxes],
            [2, 2, 0, 2],
        )
        self.assertEqual(len(first_label.read_text(encoding="utf-8").splitlines()[1].split()), 9)

    def test_bulk_class_replacement_rolls_back_every_file_on_failure(self) -> None:
        root = TEST_TEMP_ROOT / "bulk-rollback"
        root.mkdir(parents=True, exist_ok=True)
        records: list[ImageRecord] = []
        originals: dict[Path, str] = {}
        for name in ("first", "second"):
            label = root / f"{name}.txt"
            original = "1 0.5 0.5 0.2 0.2\n"
            label.write_text(original, encoding="utf-8")
            boxes, issues = load_labels(label, 2)
            records.append(ImageRecord(root / f"{name}.jpg", label, boxes, issues))
            originals[label] = original
        real_save = save_labels
        call_count = 0

        def fail_second(record: ImageRecord, class_count: int = 0) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("simulated write failure")
            real_save(record, class_count)

        with patch(
            "yolo_reviewer.services.storage.save_labels",
            side_effect=fail_second,
        ):
            with self.assertRaises(OSError):
                replace_class_in_records(records, 1, 0, 2)
        self.assertEqual([record.boxes[0].class_id for record in records], [1, 1])
        for label, original in originals.items():
            self.assertEqual(label.read_text(encoding="utf-8"), original)

    def test_quality_checks_rank_unusual_geometry_as_suspicious(self) -> None:
        issues = box_quality_issues([
            Box(0, 0.5, 0.5, 0.005, 0.005),
            Box(0, 0.5, 0.5, 0.96, 0.96),
            Box(0, 0.5, 0.5, 0.8, 0.02),
        ])
        self.assertTrue({
            "tiny-box",
            "huge-box",
            "extreme-aspect-ratio",
        }.issubset({issue.kind for issue in issues}))

    def test_box_serialization_is_standard_yolo(self) -> None:
        self.assertEqual(
            Box(2, 0.5, 0.4, 0.3, 0.2).serialize(),
            "2 0.5 0.4 0.3 0.2",
        )


if __name__ == "__main__":
    unittest.main()
