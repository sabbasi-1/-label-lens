import unittest
from pathlib import Path

from yolo_reviewer.core import (
    Box,
    DatasetLoadCancelled,
    ReviewState,
    box_quality_issues,
    discover_dataset,
    label_path_for,
    load_labels,
    save_labels,
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

    def test_review_state_round_trip(self) -> None:
        root = TEST_TEMP_ROOT / "state"
        root.mkdir(parents=True, exist_ok=True)
        path = root / ".yolo-review-state.json"
        ReviewState({"images/a.jpg"}, {"images/b.jpg"}, "images/a.jpg").save(path)
        loaded = ReviewState.load(path)
        self.assertEqual(loaded.reviewed, {"images/a.jpg"})
        self.assertEqual(loaded.flagged, {"images/b.jpg"})
        self.assertEqual(loaded.last_image, "images/a.jpg")

    def test_save_refuses_to_drop_unsupported_rows(self) -> None:
        root = TEST_TEMP_ROOT / "unsupported"
        root.mkdir(parents=True, exist_ok=True)
        label = root / "polygon.txt"
        original = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        label.write_text(original, encoding="utf-8")
        boxes, issues = load_labels(label, class_count=1)
        from yolo_reviewer.core import ImageRecord

        record = ImageRecord(root / "image.jpg", label, boxes, issues)
        with self.assertRaises(ValueError):
            save_labels(record)
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
