from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..services.dataset_loader import (
    DatasetLoadCancelled,
    discover_dataset,
)


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
