from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .ui.dialogs import ClassPickerDialog, suggested_filename_pattern
from .ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Annotation Reviewer")
    window = MainWindow()
    window.show()
    return app.exec()


__all__ = [
    "ClassPickerDialog",
    "MainWindow",
    "main",
    "suggested_filename_pattern",
]
