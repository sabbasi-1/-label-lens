import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from yolo_reviewer.app import ClassPickerDialog, MainWindow
from yolo_reviewer.core import ReviewState, discover_dataset

TEST_ROOT = Path(__file__).parent / ".tmp" / "gui"


class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def make_dataset(
        self, root: Path, names: list[str], image_count: int = 1
    ) -> tuple[Path, list[Path]]:
        image_dir = root / "images" / "train"
        label_dir = root / "labels" / "train"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        labels: list[Path] = []
        image = QImage(320, 200, QImage.Format.Format_RGB32)
        image.fill(0x38404A)
        for index in range(image_count):
            self.assertTrue(image.save(str(image_dir / f"sample_{index}.png")))
            label = label_dir / f"sample_{index}.txt"
            label.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            labels.append(label)
        yaml_path = root / "data.yaml"
        yaml_path.write_text(
            "path: .\ntrain: images/train\nnames: "
            + repr(names).replace("'", '"')
            + "\n",
            encoding="utf-8",
        )
        return yaml_path, labels

    def open_test_window(self, yaml_path: Path) -> MainWindow:
        window = MainWindow()
        window.auto_save_check.setChecked(False)
        window.dataset = discover_dataset(yaml_path)
        window.state = ReviewState()
        window.visible_indices = list(range(len(window.dataset.records)))
        window.show_position(0)
        window.show()
        self.app.processEvents()
        return window

    def test_window_loads_image_and_relabels_selected_box(self) -> None:
        yaml_path, labels = self.make_dataset(
            TEST_ROOT / "relabel", ["cat", "dog"]
        )
        label_path = labels[0]
        window = self.open_test_window(yaml_path)
        screenshot = window.grab()
        self.assertFalse(screenshot.isNull())
        self.assertTrue(screenshot.save(str(TEST_ROOT / "gui-smoke.png")))
        window.canvas.selected = 0
        window.assign_class(1)

        self.assertEqual(window.current_record().boxes[0].class_id, 1)
        self.assertTrue(window.dirty)
        window.save_current()
        self.assertFalse(window.dirty)
        self.assertTrue(label_path.read_text(encoding="utf-8").startswith("1 "))
        window.close()

    def test_class_picker_searches_and_multidigit_shortcut_assigns(self) -> None:
        names = [f"class_{index}" for index in range(15)]
        dialog = ClassPickerDialog(names)
        dialog.search.setText("12")
        self.assertEqual(dialog.selected_class(), 12)
        dialog.close()

        yaml_path, _labels = self.make_dataset(TEST_ROOT / "shortcuts", names)
        window = self.open_test_window(yaml_path)
        window.canvas.selected = 0
        window.queue_digit("1")
        window.queue_digit("2")
        window.commit_digit_shortcut()
        self.assertEqual(window.current_record().boxes[0].class_id, 12)
        window.reload_current_labels()
        window.close()

    def test_draw_move_resize_delete_and_undo(self) -> None:
        yaml_path, _labels = self.make_dataset(
            TEST_ROOT / "geometry", ["cat", "dog"]
        )
        window = self.open_test_window(yaml_path)
        window.canvas.box_created.disconnect(window.show_class_picker)
        canvas = window.canvas

        window.draw_action.setChecked(True)
        image_rect = canvas._image_rect()
        start = QPoint(
            int(image_rect.left() + image_rect.width() * 0.1),
            int(image_rect.top() + image_rect.height() * 0.1),
        )
        end = QPoint(
            int(image_rect.left() + image_rect.width() * 0.25),
            int(image_rect.top() + image_rect.height() * 0.3),
        )
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(canvas, end, delay=5)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)
        self.assertEqual(len(window.current_record().boxes), 2)

        created = window.current_record().boxes[1]
        old_x = created.x
        center = canvas._box_rect(created).center().toPoint()
        moved_to = center + QPoint(30, 15)
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=center)
        self.assertEqual(canvas._interaction, "move")
        QTest.mouseMove(canvas, moved_to, delay=5)
        self.app.processEvents()
        self.assertTrue(canvas._moved)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=moved_to)
        self.assertGreater(created.x, old_x)

        old_width = created.width
        corner = canvas._box_rect(created).bottomRight().toPoint()
        larger_corner = corner + QPoint(25, 15)
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=corner)
        QTest.mouseMove(canvas, larger_corner, delay=5)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=larger_corner)
        self.assertGreater(created.width, old_width)

        canvas.selected = 1
        window.delete_selected()
        self.assertEqual(len(window.current_record().boxes), 1)
        window.undo()
        self.assertEqual(len(window.current_record().boxes), 2)
        window.redo()
        self.assertEqual(len(window.current_record().boxes), 1)
        window.undo()
        self.assertEqual(len(window.current_record().boxes), 2)
        window.reload_current_labels()
        window.close()

    def test_auto_save_writes_before_navigation(self) -> None:
        yaml_path, labels = self.make_dataset(
            TEST_ROOT / "autosave", ["cat", "dog"], image_count=2
        )
        window = self.open_test_window(yaml_path)
        window.auto_save_check.setChecked(True)
        window.canvas.selected = 0
        window.assign_class(1)
        window.navigate(1)
        self.assertEqual(window.position, 1)
        self.assertTrue(labels[0].read_text(encoding="utf-8").startswith("1 "))
        window.auto_save_check.setChecked(False)
        window.close()

    def test_class_filter_matches_any_requested_id_or_name(self) -> None:
        yaml_path, labels = self.make_dataset(
            TEST_ROOT / "filter", ["helmet", "vest", "person"], image_count=3
        )
        labels[0].write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        labels[1].write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        labels[2].write_text("2 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        window = self.open_test_window(yaml_path)
        window.class_filter_edit.setText("0, 2")
        window.apply_class_filter()
        self.assertEqual(len(window.visible_indices), 2)
        self.assertEqual(window.class_filter_ids, {0, 2})
        window.class_filter_edit.setText("vest")
        window.apply_class_filter()
        self.assertEqual(len(window.visible_indices), 1)
        self.assertEqual(window.class_filter_ids, {1})
        window.clear_class_filter()
        self.assertEqual(len(window.visible_indices), 3)
        window.close()

    def test_background_loader_completes_without_blocking_window(self) -> None:
        yaml_path, _labels = self.make_dataset(
            TEST_ROOT / "background", ["cat", "dog"], image_count=4
        )
        window = MainWindow()
        window.start_dataset_load(yaml_path)
        for _ in range(200):
            if window._loader_thread is None:
                break
            QTest.qWait(10)
        self.assertIsNone(window._loader_thread)
        self.assertIsNotNone(window.dataset)
        self.assertEqual(len(window.dataset.records), 4)
        window.close()


if __name__ == "__main__":
    unittest.main()
