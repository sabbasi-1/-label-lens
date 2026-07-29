from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float
    polygon: tuple[tuple[float, float], ...] | None = None

    def copy(self) -> "Box":
        return Box(
            self.class_id,
            self.x,
            self.y,
            self.width,
            self.height,
            self.polygon,
        )

    def serialize(self) -> str:
        if self.polygon is not None:
            left = self.x - self.width / 2
            top = self.y - self.height / 2
            coordinates = (
                coordinate
                for point in self.polygon
                for coordinate in (
                    left + point[0] * self.width,
                    top + point[1] * self.height,
                )
            )
            return f"{self.class_id} " + " ".join(
                f"{coordinate:.8g}" for coordinate in coordinates
            )
        return (
            f"{self.class_id} {self.x:.8g} {self.y:.8g} "
            f"{self.width:.8g} {self.height:.8g}"
        )

    def polygon_points(self) -> tuple[tuple[float, float], ...]:
        if self.polygon is None:
            return ()
        left = self.x - self.width / 2
        top = self.y - self.height / 2
        return tuple(
            (left + x * self.width, top + y * self.height)
            for x, y in self.polygon
        )


def copy_boxes(boxes: list[Box]) -> list[Box]:
    return [box.copy() for box in boxes]


@dataclass
class AnnotationIssue:
    kind: str
    message: str
    box_index: int | None = None


@dataclass
class ImageRecord:
    image_path: Path
    label_path: Path
    boxes: list[Box] = field(default_factory=list)
    issues: list[AnnotationIssue] = field(default_factory=list)
    split: str = "other"

    def contains_any_class(self, class_ids: set[int]) -> bool:
        return not class_ids or any(
            box.class_id in class_ids for box in self.boxes
        )


@dataclass
class Dataset:
    root: Path
    names: list[str]
    records: list[ImageRecord]

    @property
    def state_path(self) -> Path:
        return self.root / ".yolo-review-state.json"

    def key(self, record: ImageRecord) -> str:
        try:
            return record.image_path.relative_to(self.root).as_posix()
        except ValueError:
            return str(record.image_path)

    @property
    def available_splits(self) -> list[str]:
        order = {"train": 0, "val": 1, "test": 2, "other": 3}
        return sorted(
            {record.split for record in self.records},
            key=lambda split: (order.get(split, 99), split),
        )
