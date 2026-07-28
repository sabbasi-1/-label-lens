import os
import unittest
from pathlib import Path
from unittest.mock import patch

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
        canvas = window.canvas

        window.set_draw_class(1)
        canvas.box_created.disconnect(window.handle_box_created)
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
        self.assertEqual(window.current_record().boxes[-1].class_id, 1)
        self.assertTrue(canvas.draw_mode)
        self.assertTrue(window.draw_action.isChecked())

        second_start = QPoint(
            int(image_rect.left() + image_rect.width() * 0.65),
            int(image_rect.top() + image_rect.height() * 0.15),
        )
        second_end = QPoint(
            int(image_rect.left() + image_rect.width() * 0.8),
            int(image_rect.top() + image_rect.height() * 0.35),
        )
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=second_start)
        QTest.mouseMove(canvas, second_end, delay=5)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=second_end)
        self.assertEqual(len(window.current_record().boxes), 3)
        self.assertEqual(window.current_record().boxes[-1].class_id, 1)
        self.assertTrue(canvas.draw_mode)

        window.queue_digit("0")
        window.commit_digit_shortcut()
        self.assertEqual(window.draw_class_id, 0)
        third_start = QPoint(
            int(image_rect.left() + image_rect.width() * 0.4),
            int(image_rect.top() + image_rect.height() * 0.65),
        )
        third_end = QPoint(
            int(image_rect.left() + image_rect.width() * 0.55),
            int(image_rect.top() + image_rect.height() * 0.85),
        )
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=third_start)
        QTest.mouseMove(canvas, third_end, delay=5)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=third_end)
        self.assertEqual(len(window.current_record().boxes), 4)
        self.assertEqual(window.current_record().boxes[-1].class_id, 0)

        window.select_action.setChecked(True)
        self.assertFalse(canvas.draw_mode)

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
        self.assertEqual(len(window.current_record().boxes), 3)
        window.undo()
        self.assertEqual(len(window.current_record().boxes), 4)
        window.redo()
        self.assertEqual(len(window.current_record().boxes), 3)
        window.undo()
        self.assertEqual(len(window.current_record().boxes), 4)
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
        window.draw_action.setChecked(True)
        window.navigate(1)
        self.assertEqual(window.position, 1)
        self.assertTrue(window.canvas.draw_mode)
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

    def test_split_filter_and_direct_position_jump(self) -> None:
        root = TEST_ROOT / "split-jump"
        image = QImage(320, 200, QImage.Format.Format_RGB32)
        image.fill(0x38404A)
        expected_counts = {"train": 3, "valid": 2, "test": 2}
        for split, count in expected_counts.items():
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                self.assertTrue(image.save(str(image_dir / f"{split}_{index}.png")))
                (label_dir / f"{split}_{index}.txt").write_text(
                    "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
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
        window = MainWindow()
        window.dataset = discover_dataset(yaml_path)
        window.state = ReviewState()
        window.populate_split_selector(window.dataset)
        window.rebuild_filter()

        test_index = window.split_combo.findData("test")
        window.split_combo.setCurrentIndex(test_index)
        self.assertEqual(len(window.visible_indices), 2)
        self.assertTrue(all(
            window.dataset.records[index].split == "test"
            for index in window.visible_indices
        ))
        window.jump_spin.setValue(2)
        window.jump_to_position()
        self.assertEqual(window.position, 1)

        train_index = window.split_combo.findData("train")
        window.split_combo.setCurrentIndex(train_index)
        self.assertEqual(len(window.visible_indices), 3)
        window.jump_spin.setValue(3)
        window.jump_to_position()
        self.assertEqual(window.position, 2)
        self.assertEqual(window.current_record().split, "train")
        window.close()

    def test_clickable_label_path_saves_then_opens_current_file(self) -> None:
        yaml_path, labels = self.make_dataset(
            TEST_ROOT / "open-label", ["cat", "dog"]
        )
        window = self.open_test_window(yaml_path)
        window.canvas.selected = 0
        window.assign_class(1)
        self.assertTrue(window.dirty)
        with patch(
            "yolo_reviewer.app.QDesktopServices.openUrl", return_value=True
        ) as open_url:
            window.open_current_label_file()
        self.assertFalse(window.dirty)
        self.assertTrue(labels[0].read_text(encoding="utf-8").startswith("1 "))
        open_url.assert_called_once()
        window.close()

    def test_each_new_box_offers_class_override_and_cancel_keeps_inherited(self) -> None:
        yaml_path, _labels = self.make_dataset(
            TEST_ROOT / "reuse-draw-class", ["cat", "dog"]
        )
        window = self.open_test_window(yaml_path)
        canvas = window.canvas
        image_rect = canvas._image_rect()

        def choose_or_keep(box_index: int, _position: QPoint) -> int | None:
            if window.draw_class_id is None:
                window.assign_class(1, box_index)
                return 1
            return None

        with patch.object(
            window, "show_class_picker", side_effect=choose_or_keep
        ) as picker:
            window.draw_action.setChecked(True)
            for left in (0.1, 0.6):
                start = QPoint(
                    int(image_rect.left() + image_rect.width() * left),
                    int(image_rect.top() + image_rect.height() * 0.1),
                )
                end = QPoint(
                    int(image_rect.left() + image_rect.width() * (left + 0.15)),
                    int(image_rect.top() + image_rect.height() * 0.3),
                )
                QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
                QTest.mouseMove(canvas, end, delay=5)
                QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)

        self.assertEqual(picker.call_count, 2)
        self.assertEqual(window.draw_class_id, 1)
        self.assertEqual(
            [box.class_id for box in window.current_record().boxes[-2:]],
            [1, 1],
        )

        newest = window.current_record().boxes[-1]
        old_x = newest.x
        center = canvas._box_rect(newest).center().toPoint()
        moved_to = center + QPoint(20, 10)
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseMove(canvas, moved_to, delay=5)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=moved_to)
        self.assertGreater(newest.x, old_x)
        self.assertTrue(canvas.draw_mode)

        overlap_start = canvas._box_rect(newest).topLeft().toPoint() + QPoint(8, 8)
        overlap_end = overlap_start + QPoint(20, 15)
        with patch.object(
            window, "show_class_picker", return_value=None
        ) as overlap_picker:
            QTest.mousePress(
                canvas,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.ShiftModifier,
                pos=overlap_start,
            )
            QTest.mouseMove(canvas, overlap_end, delay=5)
            QTest.mouseRelease(
                canvas,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.ShiftModifier,
                pos=overlap_end,
            )
        self.assertEqual(overlap_picker.call_count, 1)
        self.assertEqual(window.current_record().boxes[-1].class_id, 1)
        window.reload_current_labels()
        window.close()


if __name__ == "__main__":
    unittest.main()
