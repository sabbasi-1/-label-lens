import unittest
from pathlib import Path

import yolo_reviewer.core as legacy_core
from yolo_reviewer.app import MainWindow as AppMainWindow
from yolo_reviewer.formats.yolo import load_labels
from yolo_reviewer.models import Box, Dataset, ImageRecord
from yolo_reviewer.services.review_session import ReviewSession
from yolo_reviewer.services.review_state import ReviewState
from yolo_reviewer.services.storage import save_labels
from yolo_reviewer.services.validation import validation_issues
from yolo_reviewer.ui.main_window import MainWindow


class ModuleBoundaryTests(unittest.TestCase):
    def test_legacy_core_reexports_modular_api(self) -> None:
        self.assertIs(legacy_core.Box, Box)
        self.assertIs(legacy_core.load_labels, load_labels)
        self.assertIs(legacy_core.save_labels, save_labels)
        self.assertIs(legacy_core.validation_issues, validation_issues)

    def test_app_reexports_modular_main_window(self) -> None:
        self.assertIs(AppMainWindow, MainWindow)

    def test_review_session_filters_without_qt_widgets(self) -> None:
        root = Path("dataset")
        records = [
            ImageRecord(
                root / "train" / "images" / "one.jpg",
                root / "train" / "labels" / "one.txt",
                [Box(0, 0.5, 0.5, 0.2, 0.2)],
                split="train",
            ),
            ImageRecord(
                root / "test" / "images" / "two.jpg",
                root / "test" / "labels" / "two.txt",
                [Box(1, 0.5, 0.5, 0.2, 0.2)],
                split="test",
            ),
        ]
        dataset = Dataset(root, ["helmet", "vest"], records)
        state = ReviewState(flagged={dataset.key(records[1])})
        session = ReviewSession(dataset, state)
        self.assertEqual(
            session.filtered_indices("Flagged", "all", set()),
            [1],
        )
        self.assertEqual(
            session.filtered_indices("All images", "train", {0}),
            [0],
        )


if __name__ == "__main__":
    unittest.main()
