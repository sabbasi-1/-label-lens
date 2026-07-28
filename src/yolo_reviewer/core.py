from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


class DatasetLoadCancelled(Exception):
    """Raised when a dataset discovery operation is cancelled."""


@dataclass
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float
    polygon: tuple[tuple[float, float], ...] | None = None

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
        return not class_ids or any(box.class_id in class_ids for box in self.boxes)


@dataclass
class ReviewState:
    reviewed: set[str] = field(default_factory=set)
    flagged: set[str] = field(default_factory=set)
    last_image: str | None = None

    @classmethod
    def load(cls, path: Path) -> "ReviewState":
        if not path.exists():
            return cls()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                reviewed=set(value.get("reviewed", [])),
                flagged=set(value.get("flagged", [])),
                last_image=value.get("last_image"),
            )
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "reviewed": sorted(self.reviewed),
            "flagged": sorted(self.flagged),
            "last_image": self.last_image,
        }
        atomic_write(path, json.dumps(data, indent=2) + "\n", make_backup=False)


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


def _class_names(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict):
        pairs = sorted(((int(key), str(value)) for key, value in raw.items()))
        if not pairs:
            return []
        names = [str(i) for i in range(pairs[-1][0] + 1)]
        for index, value in pairs:
            names[index] = value
        return names
    return []


def _images_from_source(source: Path, base: Path) -> Iterable[Path]:
    source = source if source.is_absolute() else (base / source)
    if source.is_dir():
        yield from (
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif source.is_file() and source.suffix.lower() == ".txt":
        for line in source.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                path = Path(value)
                path = path if path.is_absolute() else (base / path)
                if path.is_file():
                    yield path
    elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        yield source


def label_path_for(image_path: Path, root: Path) -> Path:
    parts = list(image_path.parts)
    image_positions = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if image_positions:
        parts[image_positions[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        relative = Path(image_path.name)
    return (root / "labels" / relative).with_suffix(".txt")


def load_labels(path: Path, class_count: int) -> tuple[list[Box], list[AnnotationIssue]]:
    boxes: list[Box] = []
    issues: list[AnnotationIssue] = []
    if not path.exists():
        return boxes, [AnnotationIssue("missing-label", f"Missing label file: {path.name}")]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return boxes, [AnnotationIssue("unreadable-label", str(exc))]

    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        is_detection = len(fields) == 5
        is_segmentation = len(fields) >= 7 and len(fields) % 2 == 1
        if not is_detection and not is_segmentation:
            issues.append(AnnotationIssue(
                "unsupported-row",
                f"Line {line_number}: expected a YOLO box or polygon, "
                f"found {len(fields)} values",
            ))
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            issues.append(AnnotationIssue(
                "malformed-row", f"Line {line_number}: values are not numeric"
            ))
            continue
        if is_detection:
            box = Box(class_id, *values)
        else:
            points = list(zip(values[::2], values[1::2]))
            left = min(point[0] for point in points)
            right = max(point[0] for point in points)
            top = min(point[1] for point in points)
            bottom = max(point[1] for point in points)
            width = right - left
            height = bottom - top
            relative_points = tuple(
                (
                    (x - left) / width if width else 0.5,
                    (y - top) / height if height else 0.5,
                )
                for x, y in points
            )
            box = Box(
                class_id,
                (left + right) / 2,
                (top + bottom) / 2,
                width,
                height,
                relative_points,
            )
        boxes.append(box)
    issues.extend(validation_issues(boxes, class_count))
    return boxes, issues


def validation_issues(boxes: list[Box], class_count: int = 0) -> list[AnnotationIssue]:
    issues: list[AnnotationIssue] = []
    for box_index, box in enumerate(boxes):
        values = (box.x, box.y, box.width, box.height)
        class_id = box.class_id
        if class_id < 0 or (class_count and class_id >= class_count):
            issues.append(AnnotationIssue(
                "unknown-class",
                f"Box {box_index + 1}: class ID {class_id} is not defined",
                box_index,
            ))
        if box.width <= 0 or box.height <= 0:
            issues.append(AnnotationIssue(
                "degenerate-box",
                f"Box {box_index + 1}: width and height must be positive",
                box_index,
            ))
        if any(value < 0 or value > 1 for value in values):
            issues.append(AnnotationIssue(
                "out-of-range",
                f"Box {box_index + 1}: normalized values must be between 0 and 1",
                box_index,
            ))
        if (
            box.x - box.width / 2 < 0
            or box.y - box.height / 2 < 0
            or box.x + box.width / 2 > 1
            or box.y + box.height / 2 > 1
        ):
            issues.append(AnnotationIssue(
                "out-of-bounds",
                f"Box {box_index + 1}: box extends beyond the image",
                box_index,
            ))
    issues.extend(box_quality_issues(boxes))
    return issues


def _iou(a: Box, b: Box) -> float:
    ax1, ay1 = a.x - a.width / 2, a.y - a.height / 2
    ax2, ay2 = a.x + a.width / 2, a.y + a.height / 2
    bx1, by1 = b.x - b.width / 2, b.y - b.height / 2
    bx2, by2 = b.x + b.width / 2, b.y + b.height / 2
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def box_quality_issues(boxes: list[Box]) -> list[AnnotationIssue]:
    issues: list[AnnotationIssue] = []
    for index, box in enumerate(boxes):
        area = box.width * box.height
        if 0 < area < 0.0001:
            issues.append(AnnotationIssue(
                "tiny-box",
                f"Box {index + 1}: covers less than 0.01% of the image",
                index,
            ))
        if area > 0.9:
            issues.append(AnnotationIssue(
                "huge-box",
                f"Box {index + 1}: covers more than 90% of the image",
                index,
            ))
        short_side = min(box.width, box.height)
        long_side = max(box.width, box.height)
        if short_side > 0 and long_side / short_side > 15:
            issues.append(AnnotationIssue(
                "extreme-aspect-ratio",
                f"Box {index + 1}: aspect ratio exceeds 15:1",
                index,
            ))
    for i, first in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            second = boxes[j]
            overlap = _iou(first, second)
            if overlap >= 0.98:
                issues.append(AnnotationIssue(
                    "duplicate-box",
                    f"Boxes {i + 1} and {j + 1} appear duplicated",
                    i,
                ))
            elif overlap >= 0.85:
                issues.append(AnnotationIssue(
                    "high-overlap",
                    f"Boxes {i + 1} and {j + 1} overlap by {overlap:.0%}",
                    i,
                ))
    return issues


def _find_yaml(directory: Path) -> Path | None:
    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _normalize_split(split: str) -> str:
    lowered = split.casefold()
    if lowered in {"val", "valid", "validation"}:
        return "val"
    if lowered in {"train", "training"}:
        return "train"
    if lowered in {"test", "testing"}:
        return "test"
    return "other"


def _infer_split(image_path: Path, root: Path) -> str:
    try:
        parts = image_path.relative_to(root).parts
    except ValueError:
        parts = image_path.parts
    for part in parts:
        split = _normalize_split(part)
        if split != "other":
            return split
    return "other"


def discover_dataset(
    selected: Path,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> Dataset:
    def report(current: int, total: int, message: str) -> None:
        if progress is not None:
            progress(current, total, message)

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise DatasetLoadCancelled()

    report(0, 0, "Reading dataset configuration...")
    selected = selected.resolve()
    yaml_path = selected if selected.is_file() else _find_yaml(selected)
    names: list[str] = []
    image_splits: dict[Path, str] = {}

    if yaml_path:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        names = _class_names(config.get("names"))
        declared_root = Path(str(config.get("path", ".")))
        root = declared_root if declared_root.is_absolute() else yaml_path.parent / declared_root
        root = root.resolve()
        report(0, 0, "Discovering image files...")
        for split_key in ("train", "val", "valid", "test"):
            raw = config.get(split_key)
            split = _normalize_split(split_key)
            sources = raw if isinstance(raw, list) else ([raw] if raw else [])
            for source in sources:
                for path in _images_from_source(Path(str(source)), root):
                    image_splits.setdefault(path, split)
                    if len(image_splits) % 1000 == 0:
                        report(
                            len(image_splits),
                            0,
                            f"Discovered {len(image_splits):,} images...",
                        )
                        check_cancelled()
    else:
        root = selected if selected.is_dir() else selected.parent
        report(0, 0, "Discovering image files...")
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and "labels" not in {part.lower() for part in path.parts}
                and ".label-lens-trash" not in {
                    part.lower() for part in path.parts
                }
            ):
                image_splits.setdefault(path, _infer_split(path, root))
                if len(image_splits) % 1000 == 0:
                    report(
                        len(image_splits),
                        0,
                        f"Discovered {len(image_splits):,} images...",
                    )
                    check_cancelled()

    check_cancelled()
    ordered_paths = sorted(image_splits, key=lambda value: str(value).lower())
    total = len(ordered_paths)
    report(0, total, f"Indexing annotations for {total:,} images...")
    records: list[ImageRecord] = []
    maximum_class = -1

    def load_record(image_path: Path) -> ImageRecord:
        label_path = label_path_for(image_path, root)
        boxes, issues = load_labels(label_path, len(names))
        return ImageRecord(
            image_path,
            label_path,
            boxes,
            issues,
            image_splits[image_path],
        )

    worker_count = min(8, max(2, os.cpu_count() or 2))
    batch_size = worker_count * 16
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="yolo-index"
    ) as executor:
        for batch_start in range(0, total, batch_size):
            check_cancelled()
            batch = ordered_paths[batch_start:batch_start + batch_size]
            for offset, record in enumerate(executor.map(load_record, batch), start=1):
                records.append(record)
                if record.boxes:
                    maximum_class = max(
                        maximum_class,
                        max(box.class_id for box in record.boxes),
                    )
                current = batch_start + offset
                if current == total or current % 25 == 0:
                    report(
                        current,
                        total,
                        f"Indexed {current:,} of {total:,} images",
                    )
                check_cancelled()

    if not names and maximum_class >= 0:
        names = [f"class_{index}" for index in range(maximum_class + 1)]
    report(total, total, f"Loaded {total:,} images")
    return Dataset(root, names, records)


def atomic_write(path: Path, content: str, *, make_backup: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if make_backup and path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def save_labels(record: ImageRecord, class_count: int = 0) -> None:
    unsafe_kinds = {"unsupported-row", "malformed-row", "unreadable-label"}
    unsafe = [issue for issue in record.issues if issue.kind in unsafe_kinds]
    if unsafe:
        raise ValueError(
            "This label file contains rows that are not standard YOLO detection "
            "annotations. Saving is blocked to avoid deleting unsupported data. "
            "See the Automatic checks panel."
        )
    content = "".join(f"{box.serialize()}\n" for box in record.boxes)
    atomic_write(record.label_path, content)
    record.issues = validation_issues(record.boxes, class_count)


def trash_record(dataset: Dataset, record: ImageRecord) -> Path:
    """Move an image, its label, and label backup to recoverable local trash."""
    trash_root = dataset.root / ".label-lens-trash" / uuid.uuid4().hex
    sources = [
        path
        for path in (
            record.image_path,
            record.label_path,
            record.label_path.with_suffix(record.label_path.suffix + ".bak"),
        )
        if path.exists()
    ]
    moved: list[tuple[Path, Path]] = []
    try:
        for source in sources:
            try:
                relative = source.relative_to(dataset.root)
            except ValueError:
                relative = Path("external") / source.name
            destination = trash_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
    except BaseException:
        for source, destination in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(source))
        raise
    return trash_root
