from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import ImageRecord
from ..services.review_session import suggested_filename_pattern


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
            visible = (
                not query
                or query in name
                or class_id.startswith(query)
            )
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item
        if first_visible is not None:
            self.classes.setCurrentItem(first_visible)

    def _accept_visible(self) -> None:
        item = self.classes.currentItem()
        if item is not None and not item.isHidden():
            self.accept()

    def selected_class(self) -> int | None:
        item = self.classes.currentItem()
        if item is None or item.isHidden():
            return None
        return int(item.data(Qt.ItemDataRole.UserRole))


class BulkClassReplaceDialog(QDialog):
    def __init__(
        self,
        names: list[str],
        record: ImageRecord,
        current_class: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Replace annotation class")
        self.resize(560, 260)
        self.source_combo = QComboBox()
        self.target_combo = QComboBox()
        for class_id, name in enumerate(names):
            label = f"{class_id}: {name}"
            self.source_combo.addItem(label, class_id)
            self.target_combo.addItem(label, class_id)
        source_index = self.source_combo.findData(current_class)
        self.source_combo.setCurrentIndex(max(0, source_index))
        if len(names) > 1:
            self.target_combo.setCurrentIndex(
                1 if self.source_combo.currentIndex() == 0 else 0
            )

        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Current image", "current")
        self.scope_combo.addItem(
            f"Current split ({record.split})",
            "split",
        )
        self.scope_combo.addItem(
            f"Current image folder ({record.image_path.parent.name})",
            "folder",
        )
        self.scope_combo.addItem(
            "Images matching a filename pattern",
            "pattern",
        )
        self.pattern_edit = QLineEdit(
            suggested_filename_pattern(record.image_path.name)
        )
        self.pattern_edit.setPlaceholderText("Example: frame_*.jpg")
        self.pattern_edit.setEnabled(False)
        self.scope_combo.currentIndexChanged.connect(
            lambda: self.pattern_edit.setEnabled(
                self.scope_combo.currentData() == "pattern"
            )
        )
        explanation = QLabel(
            "Use * to match any characters and ? to match one character. "
            "A precise image and annotation count will be shown before files "
            "are changed."
        )
        explanation.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Replace:", self.source_combo)
        form.addRow("With:", self.target_combo)
        form.addRow("Scope:", self.scope_combo)
        form.addRow("Filename pattern:", self.pattern_edit)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(explanation)
        layout.addWidget(buttons)

    def values(self) -> tuple[int, int, str, str]:
        return (
            int(self.source_combo.currentData()),
            int(self.target_combo.currentData()),
            str(self.scope_combo.currentData()),
            self.pattern_edit.text().strip(),
        )


__all__ = [
    "BulkClassReplaceDialog",
    "ClassPickerDialog",
    "suggested_filename_pattern",
]
