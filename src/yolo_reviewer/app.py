from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
    QThread,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .core import (
    AnnotationIssue,
    Box,
    Dataset,
    DatasetLoadCancelled,
    ImageRecord,
    ReviewState,
    discover_dataset,
    load_labels,
    save_labels,
    validation_issues,
)


def copy_boxes(boxes: list[Box]) -> list[Box]:
    return [Box(box.class_id, box.x, box.y, box.width, box.height) for box in boxes]


def class_color(class_id: int) -> QColor:
    return QColor.fromHsv((class_id * 67 + 15) % 360, 220, 255)


@dataclass
class HistoryEntry:
    record_index: int
    before: list[Box]
    after: list[Box]
    description: str


class DatasetLoader(QObject):
    progress = Signal(int, int, str)
    loaded = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, selected: Path) -> None:
        super().__init__()
        self.selected = selected
        self.cancel_requested = False

    @Slot()
    def run(self) -> None:
        try:
            dataset = discover_dataset(
                self.selected,
                progress=self.progress.emit,
                cancelled=lambda: self.cancel_requested,
            )
        except DatasetLoadCancelled:
            self.stopped.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.loaded.emit(dataset)


class ClassPickerDialog(QDialog):
    def __init__(
        self,
        names: list[str],
        current_class: int = -1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose class")
        self.resize(440, 520)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search by class name or ID...")
        self.classes = QListWidget()
        for class_id, name in enumerate(names):
            item = QListWidgetItem(f"{class_id}: {name}")
            item.setData(Qt.ItemDataRole.UserRole, class_id)
            self.classes.addItem(item)
            if class_id == current_class:
                self.classes.setCurrentItem(item)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._accept_visible)
        self.classes.itemDoubleClicked.connect(lambda _item: self.accept())

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Type a name or numeric class ID"))
        layout.addWidget(self.search)
        layout.addWidget(self.classes)
        layout.addWidget(buttons)
        self.search.setFocus()

    def _filter(self, text: str) -> None:
        query = text.strip().casefold()
        first_visible: QListWidgetItem | None = None
        for index in range(self.classes.count()):
            item = self.classes.item(index)
            class_id = str(item.data(Qt.ItemDataRole.UserRole))
            name = item.text().split(":", 1)[-1].strip().casefold()
            visible = not query or query in name or class_id.startswith(query)
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item
        if first_visible is not None:
            self.classes.setCurrentItem(first_visible)

    def _accept_visible(self) -> None:
        if self.classes.currentItem() and not self.classes.currentItem().isHidden():
            self.accept()

    def selected_class(self) -> int | None:
        item = self.classes.currentItem()
        if item is None or item.isHidden():
            return None
        return int(item.data(Qt.ItemDataRole.UserRole))


class ImageCanvas(QWidget):
    box_clicked = Signal(int, QPoint)
    selection_changed = Signal(int)
    edit_committed = Signal(object, str)
    box_created = Signal(int, QPoint)
    draw_finished = Signal()

    HANDLE_SIZE = 9.0
    MIN_BOX_SIZE = 0.002

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(500, 400)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.pixmap = QPixmap()
        self.boxes: list[Box] = []
        self.names: list[str] = []
        self.selected = -1
        self.draw_mode = False
        self._draw_rect = QRectF()
        self._interaction: str | None = None
        self._press_normalized = QPointF()
        self._start_box: Box | None = None
        self._before_boxes: list[Box] = []
        self._draw_start: QPointF | None = None
        self._draw_current: QPointF | None = None
        self._moved = False

    def set_image(self, path: Path, boxes: list[Box], names: list[str]) -> None:
        self.pixmap = QPixmap(str(path))
        self.boxes = boxes
        self.names = names
        self.selected = -1
        self._interaction = None
        self._draw_start = None
        self.update()

    def set_draw_mode(self, enabled: bool) -> None:
        self.draw_mode = enabled
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def _image_rect(self) -> QRectF:
        if self.pixmap.isNull():
            return QRectF()
        scale = min(
            self.width() / self.pixmap.width(),
            self.height() / self.pixmap.height(),
        )
        width, height = self.pixmap.width() * scale, self.pixmap.height() * scale
        return QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
        )

    def _box_rect(self, box: Box) -> QRectF:
        image = self._draw_rect if not self._draw_rect.isEmpty() else self._image_rect()
        return QRectF(
            image.left() + (box.x - box.width / 2) * image.width(),
            image.top() + (box.y - box.height / 2) * image.height(),
            box.width * image.width(),
            box.height * image.height(),
        )

    def _normalized(self, point: QPointF) -> QPointF | None:
        image = self._image_rect()
        if image.isEmpty() or not image.contains(point):
            return None
        return QPointF(
            min(1.0, max(0.0, (point.x() - image.left()) / image.width())),
            min(1.0, max(0.0, (point.y() - image.top()) / image.height())),
        )

    def _handle_at(self, point: QPointF, rectangle: QRectF) -> str | None:
        handles = {
            "nw": rectangle.topLeft(),
            "ne": rectangle.topRight(),
            "sw": rectangle.bottomLeft(),
            "se": rectangle.bottomRight(),
        }
        for name, center in handles.items():
            if QRectF(
                center.x() - self.HANDLE_SIZE,
                center.y() - self.HANDLE_SIZE,
                self.HANDLE_SIZE * 2,
                self.HANDLE_SIZE * 2,
            ).contains(point):
                return name
        return None

    def _select_at(self, point: QPointF) -> int:
        candidates: list[tuple[float, int]] = []
        for index, box in enumerate(self.boxes):
            rectangle = self._box_rect(box)
            if rectangle.contains(point):
                candidates.append((rectangle.width() * rectangle.height(), index))
        return min(candidates)[1] if candidates else -1

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17191d"))
        if self.pixmap.isNull():
            painter.setPen(QColor("#aab0ba"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Open a YOLO dataset to begin",
            )
            return
        self._draw_rect = self._image_rect()
        painter.drawPixmap(self._draw_rect.toRect(), self.pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index, box in enumerate(self.boxes):
            color = class_color(box.class_id)
            painter.setPen(QPen(color, 4 if index == self.selected else 2))
            rectangle = self._box_rect(box)
            painter.drawRect(rectangle)
            name = (
                self.names[box.class_id]
                if 0 <= box.class_id < len(self.names)
                else str(box.class_id)
            )
            text = f"{box.class_id}: {name}"
            metrics = painter.fontMetrics()
            label_rect = QRectF(
                rectangle.left(),
                max(self._draw_rect.top(), rectangle.top() - metrics.height() - 4),
                metrics.horizontalAdvance(text) + 10,
                metrics.height() + 4,
            )
            painter.fillRect(label_rect, color)
            painter.setPen(QColor("#111111"))
            painter.drawText(
                label_rect.adjusted(5, 0, -2, 0),
                Qt.AlignmentFlag.AlignVCenter,
                text,
            )
            if index == self.selected:
                painter.setBrush(QColor("#ffffff"))
                painter.setPen(QPen(QColor("#111111"), 1))
                for center in (
                    rectangle.topLeft(),
                    rectangle.topRight(),
                    rectangle.bottomLeft(),
                    rectangle.bottomRight(),
                ):
                    painter.drawRect(QRectF(center.x() - 5, center.y() - 5, 10, 10))

        if self._draw_start is not None and self._draw_current is not None:
            image = self._image_rect()
            start = QPointF(
                image.left() + self._draw_start.x() * image.width(),
                image.top() + self._draw_start.y() * image.height(),
            )
            current = QPointF(
                image.left() + self._draw_current.x() * image.width(),
                image.top() + self._draw_current.y() * image.height(),
            )
            preview = QRectF(start, current).normalized()
            painter.setPen(QPen(QColor("#ffffff"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(preview)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        normalized = self._normalized(event.position())
        if normalized is None:
            return
        if self.draw_mode:
            self._draw_start = normalized
            self._draw_current = normalized
            self._before_boxes = copy_boxes(self.boxes)
            self._interaction = "draw"
            return

        handle = None
        if 0 <= self.selected < len(self.boxes):
            handle = self._handle_at(event.position(), self._box_rect(self.boxes[self.selected]))
        if handle is None:
            self.selected = self._select_at(event.position())
            self.selection_changed.emit(self.selected)
        if self.selected < 0:
            self.update()
            return
        if handle is None:
            handle = self._handle_at(event.position(), self._box_rect(self.boxes[self.selected]))
        self._interaction = f"resize-{handle}" if handle else "move"
        self._press_normalized = normalized
        self._start_box = copy_boxes([self.boxes[self.selected]])[0]
        self._before_boxes = copy_boxes(self.boxes)
        self._moved = False
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        normalized = self._normalized(event.position())
        if normalized is None or self._interaction is None:
            return
        if self._interaction == "draw":
            self._draw_current = normalized
            self.update()
            return
        if self._start_box is None or not (0 <= self.selected < len(self.boxes)):
            return
        box = self.boxes[self.selected]
        start = self._start_box
        dx = normalized.x() - self._press_normalized.x()
        dy = normalized.y() - self._press_normalized.y()
        if abs(dx) + abs(dy) > 0.0005:
            self._moved = True
        if self._interaction == "move":
            box.x = min(1 - start.width / 2, max(start.width / 2, start.x + dx))
            box.y = min(1 - start.height / 2, max(start.height / 2, start.y + dy))
        else:
            left = start.x - start.width / 2
            right = start.x + start.width / 2
            top = start.y - start.height / 2
            bottom = start.y + start.height / 2
            handle = self._interaction.removeprefix("resize-")
            if "w" in handle:
                left = min(right - self.MIN_BOX_SIZE, max(0.0, normalized.x()))
            if "e" in handle:
                right = max(left + self.MIN_BOX_SIZE, min(1.0, normalized.x()))
            if "n" in handle:
                top = min(bottom - self.MIN_BOX_SIZE, max(0.0, normalized.y()))
            if "s" in handle:
                bottom = max(top + self.MIN_BOX_SIZE, min(1.0, normalized.y()))
            box.x = (left + right) / 2
            box.y = (top + bottom) / 2
            box.width = right - left
            box.height = bottom - top
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._interaction is None:
            return
        if self._interaction == "draw":
            start, end = self._draw_start, self._draw_current
            self._interaction = None
            self._draw_start = None
            self._draw_current = None
            if start is not None and end is not None:
                left, right = sorted((start.x(), end.x()))
                top, bottom = sorted((start.y(), end.y()))
                if (
                    right - left >= self.MIN_BOX_SIZE
                    and bottom - top >= self.MIN_BOX_SIZE
                ):
                    self.boxes.append(
                        Box(
                            0,
                            (left + right) / 2,
                            (top + bottom) / 2,
                            right - left,
                            bottom - top,
                        )
                    )
                    self.selected = len(self.boxes) - 1
                    self.selection_changed.emit(self.selected)
                    self.edit_committed.emit(self._before_boxes, "Added box")
                    self.box_created.emit(
                        self.selected, event.globalPosition().toPoint()
                    )
            self.draw_finished.emit()
            self.update()
            return

        interaction = self._interaction
        self._interaction = None
        if self._moved:
            action = "Resized box" if interaction.startswith("resize-") else "Moved box"
            self.edit_committed.emit(self._before_boxes, action)
        elif self.selected >= 0:
            self.box_clicked.emit(self.selected, event.globalPosition().toPoint())
        self.update()


class MainWindow(QMainWindow):
    UNSAFE_ISSUES = {"unsupported-row", "malformed-row", "unreadable-label"}

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Annotation Reviewer")
        self.resize(1280, 820)
        self.settings = QSettings("BZNSPACE", "YOLOAnnotationReviewer")
        self.dataset: Dataset | None = None
        self.state = ReviewState()
        self.visible_indices: list[int] = []
        self.position = -1
        self.dirty = False
        self.undo_stack: list[HistoryEntry] = []
        self.redo_stack: list[HistoryEntry] = []
        self.class_filter_ids: set[int] = set()
        self._loader_thread: QThread | None = None
        self._loader: DatasetLoader | None = None
        self._load_progress_dialog: QProgressDialog | None = None
        self._digit_buffer = ""
        self._digit_timer = QTimer(self)
        self._digit_timer.setSingleShot(True)
        self._digit_timer.setInterval(650)
        self._digit_timer.timeout.connect(self.commit_digit_shortcut)

        self.canvas = ImageCanvas()
        self.canvas.box_clicked.connect(self.show_class_picker)
        self.canvas.box_created.connect(self.show_class_picker)
        self.canvas.selection_changed.connect(self.select_canvas_box)
        self.canvas.edit_committed.connect(self.canvas_edit_committed)
        self.canvas.draw_finished.connect(self.finish_draw_mode)

        self.annotation_list = QListWidget()
        self.annotation_list.currentRowChanged.connect(self.select_from_list)
        self.annotation_list.itemDoubleClicked.connect(
            lambda _item: self.open_selected_picker()
        )
        self.issue_list = QListWidget()
        self.issue_list.itemClicked.connect(self.select_issue_box)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All images", "Unreviewed", "Flagged", "Suspicious"])
        self.filter_combo.currentIndexChanged.connect(self.rebuild_filter)
        self.class_filter_edit = QLineEdit()
        self.class_filter_edit.setPlaceholderText("e.g. 0, 3, helmet")
        self.class_filter_edit.returnPressed.connect(self.apply_class_filter)
        self.class_filter_apply = QPushButton("Apply")
        self.class_filter_apply.clicked.connect(self.apply_class_filter)
        self.class_filter_clear = QPushButton("Clear")
        self.class_filter_clear.clicked.connect(self.clear_class_filter)
        self.class_filter_status = QLabel("Showing all classes")
        self.class_filter_status.setWordWrap(True)
        self.reviewed_check = QCheckBox("Reviewed")
        self.flagged_check = QCheckBox("Flagged")
        self.reviewed_check.clicked.connect(self.update_review_state)
        self.flagged_check.clicked.connect(self.update_review_state)
        self.auto_save_check = QCheckBox("Auto-save when changing images")
        self.auto_save_check.setChecked(
            self.settings.value("auto_save", False, type=bool)
        )
        self.auto_save_check.toggled.connect(
            lambda enabled: self.settings.setValue("auto_save", enabled)
        )
        self.counter = QLabel("No dataset open")
        self.edit_hint = QLabel(
            "Click: relabel  •  Drag: move  •  Corner: resize  •  N: new  •  Del: delete"
        )
        self.edit_hint.setWordWrap(True)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(QLabel("Queue"))
        side_layout.addWidget(self.filter_combo)
        side_layout.addWidget(QLabel("Only images containing any of these classes"))
        side_layout.addWidget(self.class_filter_edit)
        class_filter_buttons = QHBoxLayout()
        class_filter_buttons.addWidget(self.class_filter_apply)
        class_filter_buttons.addWidget(self.class_filter_clear)
        side_layout.addLayout(class_filter_buttons)
        side_layout.addWidget(self.class_filter_status)
        side_layout.addWidget(self.counter)
        side_layout.addWidget(self.reviewed_check)
        side_layout.addWidget(self.flagged_check)
        side_layout.addWidget(self.auto_save_check)
        side_layout.addWidget(self.edit_hint)
        side_layout.addWidget(QLabel("Annotations (double-click to relabel)"))
        side_layout.addWidget(self.annotation_list, 2)
        side_layout.addWidget(QLabel("Automatic checks (click to select box)"))
        side_layout.addWidget(self.issue_list, 1)

        splitter = QSplitter()
        splitter.addWidget(self.canvas)
        splitter.addWidget(side)
        splitter.setSizes([950, 330])
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self._build_toolbar()
        self._build_shortcuts()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.open_action = QAction("Open dataset", self)
        self.open_action.triggered.connect(self.open_dataset)
        toolbar.addAction(self.open_action)
        toolbar.addSeparator()
        previous = QAction("Previous", self)
        previous.triggered.connect(lambda: self.navigate(-1))
        toolbar.addAction(previous)
        following = QAction("Next", self)
        following.triggered.connect(lambda: self.navigate(1))
        toolbar.addAction(following)
        toolbar.addSeparator()
        self.draw_action = QAction("New box", self)
        self.draw_action.setCheckable(True)
        self.draw_action.toggled.connect(self.canvas.set_draw_mode)
        toolbar.addAction(self.draw_action)
        delete_action = QAction("Delete box", self)
        delete_action.triggered.connect(self.delete_selected)
        toolbar.addAction(delete_action)
        toolbar.addSeparator()
        save_action = QAction("Save labels", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_current)
        toolbar.addAction(save_action)

    def _action(self, shortcut: str, callback) -> None:
        action = QAction(self)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        self.addAction(action)

    def _build_shortcuts(self) -> None:
        self._action("Left", lambda: self.navigate(-1))
        self._action("Right", lambda: self.navigate(1))
        self._action("R", self.toggle_reviewed)
        self._action("F", self.toggle_flagged)
        self._action("N", lambda: self.draw_action.setChecked(True))
        self._action("Delete", self.delete_selected)
        self._action("Ctrl+Z", self.undo)
        self._action("Ctrl+Y", self.redo)
        self._action("Ctrl+Shift+Z", self.redo)
        self._action("Return", self.commit_or_open_picker)
        self._action("Ctrl+L", self.open_selected_picker)
        self._action("Escape", self.cancel_transient_mode)
        for digit in range(10):
            self._action(
                str(digit),
                lambda checked=False, value=digit: self.queue_digit(str(value)),
            )

    def open_dataset(self) -> None:
        if self.dataset is not None and not self.maybe_save():
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Choose data.yaml (Cancel to choose a folder)",
            "",
            "YOLO dataset (*.yaml *.yml)",
        )
        if not selected:
            selected = QFileDialog.getExistingDirectory(
                self, "Choose dataset directory"
            )
        if not selected:
            return
        self.start_dataset_load(Path(selected))

    def start_dataset_load(self, selected: Path) -> None:
        if self._loader_thread is not None:
            return
        self.open_action.setEnabled(False)
        progress_dialog = QProgressDialog(
            "Reading dataset configuration...",
            "Cancel",
            0,
            0,
            self,
        )
        progress_dialog.setWindowTitle("Opening YOLO dataset")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)

        thread = QThread(self)
        loader = DatasetLoader(selected)
        loader.moveToThread(thread)
        thread.started.connect(loader.run)
        loader.progress.connect(self.update_load_progress)
        loader.loaded.connect(self.dataset_loaded)
        loader.failed.connect(self.dataset_load_failed)
        loader.stopped.connect(self.dataset_load_stopped)
        loader.loaded.connect(thread.quit)
        loader.failed.connect(thread.quit)
        loader.stopped.connect(thread.quit)
        loader.loaded.connect(loader.deleteLater)
        loader.failed.connect(loader.deleteLater)
        loader.stopped.connect(loader.deleteLater)
        thread.finished.connect(self.finish_dataset_load)
        thread.finished.connect(thread.deleteLater)
        progress_dialog.canceled.connect(
            lambda: self.request_dataset_load_cancel()
        )

        self._loader_thread = thread
        self._loader = loader
        self._load_progress_dialog = progress_dialog
        progress_dialog.show()
        thread.start()

    @Slot(int, int, str)
    def update_load_progress(self, current: int, total: int, message: str) -> None:
        dialog = self._load_progress_dialog
        if dialog is None:
            return
        dialog.setLabelText(message)
        if total <= 0:
            dialog.setRange(0, 0)
        else:
            dialog.setRange(0, total)
            dialog.setValue(current)

    def request_dataset_load_cancel(self) -> None:
        if self._loader is not None:
            self._loader.cancel_requested = True
        if self._load_progress_dialog is not None:
            self._load_progress_dialog.setLabelText("Cancelling dataset load...")

    @Slot(object)
    def dataset_loaded(self, dataset: Dataset) -> None:
        if not dataset.records:
            QMessageBox.warning(
                self, "No images", "No supported images were discovered."
            )
            return
        self.dataset = dataset
        self.state = ReviewState.load(dataset.state_path)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.class_filter_ids.clear()
        self.class_filter_edit.clear()
        self.class_filter_status.setText("Showing all classes")
        self.dirty = False
        self.rebuild_filter()
        target = next(
            (
                i
                for i, record_index in enumerate(self.visible_indices)
                if dataset.key(dataset.records[record_index])
                == self.state.last_image
            ),
            0,
        )
        self.show_position(target)
        self.statusBar().showMessage(
            f"Loaded {len(dataset.records):,} images and "
            f"{sum(len(record.boxes) for record in dataset.records):,} annotations",
            8000,
        )

    @Slot(str)
    def dataset_load_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Could not open dataset", message)

    @Slot()
    def dataset_load_stopped(self) -> None:
        self.statusBar().showMessage("Dataset loading cancelled", 5000)

    @Slot()
    def finish_dataset_load(self) -> None:
        if self._load_progress_dialog is not None:
            self._load_progress_dialog.close()
            self._load_progress_dialog.deleteLater()
        self._load_progress_dialog = None
        self._loader = None
        self._loader_thread = None
        self.open_action.setEnabled(True)

    def current_record(self) -> ImageRecord | None:
        if not self.dataset or not (0 <= self.position < len(self.visible_indices)):
            return None
        return self.dataset.records[self.visible_indices[self.position]]

    def current_record_index(self) -> int:
        if not (0 <= self.position < len(self.visible_indices)):
            return -1
        return self.visible_indices[self.position]

    def apply_class_filter(self) -> None:
        if not self.dataset:
            return
        tokens = [
            token
            for token in re.split(r"[\s,;]+", self.class_filter_edit.text().strip())
            if token
        ]
        selected: set[int] = set()
        invalid: list[str] = []
        names_by_key = {
            name.casefold(): class_id
            for class_id, name in enumerate(self.dataset.names)
        }
        for token in tokens:
            if token.lstrip("-").isdigit():
                class_id = int(token)
                if 0 <= class_id < len(self.dataset.names):
                    selected.add(class_id)
                else:
                    invalid.append(token)
            else:
                class_id = names_by_key.get(token.casefold())
                if class_id is None:
                    invalid.append(token)
                else:
                    selected.add(class_id)
        if invalid:
            QMessageBox.warning(
                self,
                "Unknown classes",
                "These class IDs or names were not found: " + ", ".join(invalid),
            )
            return
        self.class_filter_ids = selected
        if selected:
            labels = [
                f"{class_id}: {self.dataset.names[class_id]}"
                for class_id in sorted(selected)
            ]
            self.class_filter_status.setText(
                "Any of: " + ", ".join(labels)
            )
        else:
            self.class_filter_status.setText("Showing all classes")
        self.rebuild_filter()

    def clear_class_filter(self) -> None:
        self.class_filter_edit.clear()
        self.class_filter_ids.clear()
        self.class_filter_status.setText("Showing all classes")
        self.rebuild_filter()

    def rebuild_filter(self) -> None:
        if not self.dataset:
            return
        current = self.current_record()
        mode = self.filter_combo.currentText()
        indices: list[int] = []
        for index, record in enumerate(self.dataset.records):
            key = self.dataset.key(record)
            include = (
                mode == "All images"
                or (mode == "Unreviewed" and key not in self.state.reviewed)
                or (mode == "Flagged" and key in self.state.flagged)
                or (mode == "Suspicious" and bool(record.issues))
            )
            if include and record.contains_any_class(self.class_filter_ids):
                indices.append(index)
        self.visible_indices = indices
        if not indices:
            self.position = -1
            self.counter.setText("No images match this filter")
            return
        if current is not None:
            current_index = self.dataset.records.index(current)
            position = indices.index(current_index) if current_index in indices else 0
        else:
            position = min(max(self.position, 0), len(indices) - 1)
        self.show_position(position)

    def maybe_save(self) -> bool:
        if not self.dirty:
            return True
        if self.auto_save_check.isChecked():
            return self.save_current()
        choice = QMessageBox.question(
            self,
            "Unsaved annotation changes",
            "Save changes to the current label file?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Save:
            return self.save_current()
        self.reload_current_labels()
        return True

    def navigate(self, offset: int) -> None:
        if not self.visible_indices or not self.maybe_save():
            return
        self.show_position(
            max(0, min(len(self.visible_indices) - 1, self.position + offset))
        )

    def show_position(self, position: int) -> None:
        if not self.dataset or not self.visible_indices:
            return
        self.position = position
        record = self.current_record()
        if not record:
            return
        self.cancel_transient_mode()
        self.canvas.set_image(record.image_path, record.boxes, self.dataset.names)
        self.refresh_sidebar()
        key = self.dataset.key(record)
        self.reviewed_check.blockSignals(True)
        self.flagged_check.blockSignals(True)
        self.reviewed_check.setChecked(key in self.state.reviewed)
        self.flagged_check.setChecked(key in self.state.flagged)
        self.reviewed_check.blockSignals(False)
        self.flagged_check.blockSignals(False)
        self.state.last_image = key
        self.state.save(self.dataset.state_path)
        self.update_dirty_display()
        self.statusBar().showMessage(str(record.image_path))

    def update_dirty_display(self) -> None:
        record = self.current_record()
        suffix = "  [unsaved]" if self.dirty else ""
        if record:
            self.counter.setText(
                f"{self.position + 1} / {len(self.visible_indices)}"
                f"  -  {record.image_path.name}{suffix}"
            )
        self.setWindowTitle(
            f"YOLO Annotation Reviewer{' *' if self.dirty else ''}"
        )
        self.filter_combo.setEnabled(not self.dirty)
        self.class_filter_edit.setEnabled(not self.dirty)
        self.class_filter_apply.setEnabled(not self.dirty)
        self.class_filter_clear.setEnabled(not self.dirty)

    def _refresh_current_issues(self) -> None:
        record = self.current_record()
        if not record or not self.dataset:
            return
        preserved = [
            issue for issue in record.issues if issue.kind in self.UNSAFE_ISSUES
        ]
        record.issues = preserved + validation_issues(
            record.boxes, len(self.dataset.names)
        )

    def refresh_sidebar(self) -> None:
        record = self.current_record()
        if not record or not self.dataset:
            return
        selected = self.canvas.selected
        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()
        for index, box in enumerate(record.boxes):
            name = (
                self.dataset.names[box.class_id]
                if 0 <= box.class_id < len(self.dataset.names)
                else "UNKNOWN"
            )
            self.annotation_list.addItem(
                f"{index + 1}.  [{box.class_id}] {name}"
                f"  ({box.width:.3f} x {box.height:.3f})"
            )
        self.annotation_list.setCurrentRow(selected)
        self.annotation_list.blockSignals(False)
        self.issue_list.clear()
        for issue in record.issues:
            item = QListWidgetItem(f"{issue.kind}: {issue.message}")
            item.setData(Qt.ItemDataRole.UserRole, issue.box_index)
            self.issue_list.addItem(item)

    def select_canvas_box(self, row: int) -> None:
        self.annotation_list.blockSignals(True)
        self.annotation_list.setCurrentRow(row)
        self.annotation_list.blockSignals(False)

    def select_from_list(self, row: int) -> None:
        if row >= 0:
            self.canvas.selected = row
            self.canvas.update()

    def select_issue_box(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(index, int) and index >= 0:
            self.canvas.selected = index
            self.select_canvas_box(index)
            self.canvas.update()

    def show_class_picker(
        self, box_index: int, global_position: QPoint | None = None
    ) -> None:
        record = self.current_record()
        if not self.dataset or not record or not (0 <= box_index < len(record.boxes)):
            return
        dialog = ClassPickerDialog(
            self.dataset.names, record.boxes[box_index].class_id, self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            class_id = dialog.selected_class()
            if class_id is not None:
                self.assign_class(class_id, box_index)

    def open_selected_picker(self) -> None:
        if self.canvas.selected >= 0:
            self.show_class_picker(self.canvas.selected)

    def commit_or_open_picker(self) -> None:
        if self._digit_buffer:
            self.commit_digit_shortcut()
        else:
            self.open_selected_picker()

    def queue_digit(self, digit: str) -> None:
        if self.canvas.selected < 0 or not self.dataset:
            return
        self._digit_buffer += digit
        self._digit_timer.start()
        self.statusBar().showMessage(
            f"Class shortcut: {self._digit_buffer}  (Enter to apply now)"
        )

    def commit_digit_shortcut(self) -> None:
        if not self._digit_buffer:
            return
        value = int(self._digit_buffer)
        self._digit_buffer = ""
        self._digit_timer.stop()
        if not self.dataset or value >= len(self.dataset.names):
            self.statusBar().showMessage(f"Class ID {value} does not exist", 4000)
            return
        self.assign_class(value)

    def cancel_transient_mode(self) -> None:
        self._digit_buffer = ""
        self._digit_timer.stop()
        if hasattr(self, "draw_action"):
            self.draw_action.setChecked(False)

    def assign_class(self, class_id: int, box_index: int | None = None) -> None:
        record = self.current_record()
        if (
            not record
            or not self.dataset
            or class_id < 0
            or class_id >= len(self.dataset.names)
        ):
            return
        index = self.canvas.selected if box_index is None else box_index
        if not (0 <= index < len(record.boxes)):
            return
        old = record.boxes[index].class_id
        if old == class_id:
            return
        before = copy_boxes(record.boxes)
        record.boxes[index].class_id = class_id
        self.canvas.selected = index
        self.commit_change(before, f"Changed box {index + 1}: {old} -> {class_id}")

    def canvas_edit_committed(self, before: list[Box], description: str) -> None:
        self.commit_change(before, description)

    def commit_change(self, before: list[Box], description: str) -> None:
        record = self.current_record()
        record_index = self.current_record_index()
        if not record or record_index < 0:
            return
        self.undo_stack.append(
            HistoryEntry(record_index, before, copy_boxes(record.boxes), description)
        )
        self.redo_stack.clear()
        self.dirty = True
        self._refresh_current_issues()
        self.refresh_sidebar()
        self.canvas.update()
        self.update_dirty_display()
        self.statusBar().showMessage(description)

    def delete_selected(self) -> None:
        record = self.current_record()
        index = self.canvas.selected
        if not record or not (0 <= index < len(record.boxes)):
            return
        before = copy_boxes(record.boxes)
        record.boxes.pop(index)
        self.canvas.selected = min(index, len(record.boxes) - 1)
        self.commit_change(before, f"Deleted box {index + 1}")

    def finish_draw_mode(self) -> None:
        self.draw_action.setChecked(False)

    def _apply_history(self, entry: HistoryEntry, boxes: list[Box]) -> None:
        if entry.record_index != self.current_record_index():
            self.statusBar().showMessage(
                "Navigate back to the edited image to undo or redo it", 4000
            )
            return
        record = self.current_record()
        if not record:
            return
        record.boxes[:] = copy_boxes(boxes)
        self.canvas.boxes = record.boxes
        self.canvas.selected = min(self.canvas.selected, len(record.boxes) - 1)
        self.dirty = True
        self._refresh_current_issues()
        self.refresh_sidebar()
        self.canvas.update()
        self.update_dirty_display()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        entry = self.undo_stack[-1]
        if entry.record_index != self.current_record_index():
            self.statusBar().showMessage(
                "The latest undo belongs to another image", 4000
            )
            return
        self.undo_stack.pop()
        self.redo_stack.append(entry)
        self._apply_history(entry, entry.before)
        self.statusBar().showMessage(f"Undo: {entry.description}")

    def redo(self) -> None:
        if not self.redo_stack:
            return
        entry = self.redo_stack[-1]
        if entry.record_index != self.current_record_index():
            return
        self.redo_stack.pop()
        self.undo_stack.append(entry)
        self._apply_history(entry, entry.after)
        self.statusBar().showMessage(f"Redo: {entry.description}")

    def save_current(self) -> bool:
        record = self.current_record()
        if not record:
            return True
        try:
            save_labels(record, len(self.dataset.names) if self.dataset else 0)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self.dirty = False
        self.refresh_sidebar()
        self.update_dirty_display()
        self.statusBar().showMessage(f"Saved {record.label_path}", 5000)
        return True

    def reload_current_labels(self) -> None:
        if not self.dataset:
            return
        record = self.current_record()
        if not record:
            return
        record.boxes, record.issues = load_labels(
            record.label_path, len(self.dataset.names)
        )
        self.canvas.boxes = record.boxes
        self.canvas.selected = -1
        self.dirty = False
        self.undo_stack = [
            entry
            for entry in self.undo_stack
            if entry.record_index != self.current_record_index()
        ]
        self.redo_stack.clear()
        self.refresh_sidebar()
        self.canvas.update()
        self.update_dirty_display()

    def update_review_state(self) -> None:
        record = self.current_record()
        if not record or not self.dataset:
            return
        key = self.dataset.key(record)
        if self.reviewed_check.isChecked():
            self.state.reviewed.add(key)
        else:
            self.state.reviewed.discard(key)
        if self.flagged_check.isChecked():
            self.state.flagged.add(key)
        else:
            self.state.flagged.discard(key)
        self.state.save(self.dataset.state_path)

    def toggle_reviewed(self) -> None:
        self.reviewed_check.setChecked(not self.reviewed_check.isChecked())
        self.update_review_state()

    def toggle_flagged(self) -> None:
        self.flagged_check.setChecked(not self.flagged_check.isChecked())
        self.update_review_state()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._loader_thread is not None:
            self.request_dataset_load_cancel()
            self.statusBar().showMessage(
                "Cancelling dataset load before closing...", 5000
            )
            event.ignore()
            return
        if self.maybe_save():
            event.accept()
        else:
            event.ignore()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Annotation Reviewer")
    window = MainWindow()
    window.show()
    return app.exec()
