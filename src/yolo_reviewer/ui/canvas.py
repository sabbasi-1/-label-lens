from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget

from ..models import Box, copy_boxes


def class_color(class_id: int) -> QColor:
    return QColor.fromHsv((class_id * 67 + 15) % 360, 220, 255)


class ImageCanvas(QWidget):
    box_clicked = Signal(int, QPoint)
    selection_changed = Signal(int)
    edit_committed = Signal(object, str)
    box_created = Signal(int, QPoint)

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
        self.new_box_class_id = 0
        self._draw_rect = QRectF()
        self._interaction: str | None = None
        self._press_normalized = QPointF()
        self._start_box: Box | None = None
        self._before_boxes: list[Box] = []
        self._draw_start: QPointF | None = None
        self._draw_current: QPointF | None = None
        self._moved = False

    def set_image(
        self,
        path: Path,
        boxes: list[Box],
        names: list[str],
    ) -> None:
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
            Qt.CursorShape.CrossCursor
            if enabled
            else Qt.CursorShape.ArrowCursor
        )

    def _image_rect(self) -> QRectF:
        if self.pixmap.isNull():
            return QRectF()
        scale = min(
            self.width() / self.pixmap.width(),
            self.height() / self.pixmap.height(),
        )
        width = self.pixmap.width() * scale
        height = self.pixmap.height() * scale
        return QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
        )

    def _box_rect(self, box: Box) -> QRectF:
        image = (
            self._draw_rect
            if not self._draw_rect.isEmpty()
            else self._image_rect()
        )
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
            min(
                1.0,
                max(0.0, (point.x() - image.left()) / image.width()),
            ),
            min(
                1.0,
                max(0.0, (point.y() - image.top()) / image.height()),
            ),
        )

    def _handle_at(
        self,
        point: QPointF,
        rectangle: QRectF,
    ) -> str | None:
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
                candidates.append((
                    rectangle.width() * rectangle.height(),
                    index,
                ))
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
            rectangle = self._box_rect(box)
            if box.polygon is not None:
                image = self._draw_rect
                polygon = QPolygonF([
                    QPointF(
                        image.left() + x * image.width(),
                        image.top() + y * image.height(),
                    )
                    for x, y in box.polygon_points()
                ])
                painter.setPen(
                    QPen(color, 4 if index == self.selected else 2)
                )
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolygon(polygon)
                painter.setPen(QPen(
                    color,
                    2 if index == self.selected else 1,
                    Qt.PenStyle.DashLine,
                ))
            else:
                painter.setPen(
                    QPen(color, 4 if index == self.selected else 2)
                )
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
                max(
                    self._draw_rect.top(),
                    rectangle.top() - metrics.height() - 4,
                ),
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
                    painter.drawRect(QRectF(
                        center.x() - 5,
                        center.y() - 5,
                        10,
                        10,
                    ))

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
            painter.setPen(QPen(
                QColor("#ffffff"),
                2,
                Qt.PenStyle.DashLine,
            ))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(preview)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        normalized = self._normalized(event.position())
        if normalized is None:
            return
        if self.draw_mode:
            force_draw = bool(
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            )
            if not force_draw and 0 <= self.selected < len(self.boxes):
                selected_rect = self._box_rect(
                    self.boxes[self.selected]
                )
                handle = self._handle_at(
                    event.position(),
                    selected_rect,
                )
                if (
                    handle is not None
                    or selected_rect.contains(event.position())
                ):
                    self._interaction = (
                        f"resize-{handle}"
                        if handle is not None
                        else "move"
                    )
                    self._press_normalized = normalized
                    self._start_box = self.boxes[self.selected].copy()
                    self._before_boxes = copy_boxes(self.boxes)
                    self._moved = False
                    self.update()
                    return
            self._draw_start = normalized
            self._draw_current = normalized
            self._before_boxes = copy_boxes(self.boxes)
            self._interaction = "draw"
            return

        handle = None
        if 0 <= self.selected < len(self.boxes):
            handle = self._handle_at(
                event.position(),
                self._box_rect(self.boxes[self.selected]),
            )
        if handle is None:
            self.selected = self._select_at(event.position())
            self.selection_changed.emit(self.selected)
        if self.selected < 0:
            self.update()
            return
        if handle is None:
            handle = self._handle_at(
                event.position(),
                self._box_rect(self.boxes[self.selected]),
            )
        self._interaction = f"resize-{handle}" if handle else "move"
        self._press_normalized = normalized
        self._start_box = self.boxes[self.selected].copy()
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
        if (
            self._start_box is None
            or not (0 <= self.selected < len(self.boxes))
        ):
            return
        box = self.boxes[self.selected]
        start = self._start_box
        dx = normalized.x() - self._press_normalized.x()
        dy = normalized.y() - self._press_normalized.y()
        if abs(dx) + abs(dy) > 0.0005:
            self._moved = True
        if self._interaction == "move":
            box.x = min(
                1 - start.width / 2,
                max(start.width / 2, start.x + dx),
            )
            box.y = min(
                1 - start.height / 2,
                max(start.height / 2, start.y + dy),
            )
        else:
            left = start.x - start.width / 2
            right = start.x + start.width / 2
            top = start.y - start.height / 2
            bottom = start.y + start.height / 2
            handle = self._interaction.removeprefix("resize-")
            if "w" in handle:
                left = min(
                    right - self.MIN_BOX_SIZE,
                    max(0.0, normalized.x()),
                )
            if "e" in handle:
                right = max(
                    left + self.MIN_BOX_SIZE,
                    min(1.0, normalized.x()),
                )
            if "n" in handle:
                top = min(
                    bottom - self.MIN_BOX_SIZE,
                    max(0.0, normalized.y()),
                )
            if "s" in handle:
                bottom = max(
                    top + self.MIN_BOX_SIZE,
                    min(1.0, normalized.y()),
                )
            box.x = (left + right) / 2
            box.y = (top + bottom) / 2
            box.width = right - left
            box.height = bottom - top
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._interaction is None
        ):
            return
        if self._interaction == "draw":
            start = self._draw_start
            end = self._draw_current
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
                    self.boxes.append(Box(
                        self.new_box_class_id,
                        (left + right) / 2,
                        (top + bottom) / 2,
                        right - left,
                        bottom - top,
                    ))
                    self.selected = len(self.boxes) - 1
                    self.selection_changed.emit(self.selected)
                    self.edit_committed.emit(
                        self._before_boxes,
                        "Added box",
                    )
                    self.box_created.emit(
                        self.selected,
                        event.globalPosition().toPoint(),
                    )
            self.update()
            return

        interaction = self._interaction
        self._interaction = None
        if self._moved:
            action = (
                "Resized box"
                if interaction.startswith("resize-")
                else "Moved box"
            )
            self.edit_committed.emit(self._before_boxes, action)
        elif self.selected >= 0:
            self.box_clicked.emit(
                self.selected,
                event.globalPosition().toPoint(),
            )
        self.update()
