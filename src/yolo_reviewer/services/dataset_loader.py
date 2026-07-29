from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import yaml

from ..formats.yolo import label_path_for, load_labels
from ..models import Dataset, ImageRecord

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


class DatasetLoadCancelled(Exception):
    """Raised when a dataset discovery operation is cancelled."""


def _class_names(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict):
        pairs = sorted(
            (int(key), str(value)) for key, value in raw.items()
        )
        if not pairs:
            return []
        names = [str(index) for index in range(pairs[-1][0] + 1)]
        for index, value in pairs:
            names[index] = value
        return names
    return []


def _images_from_source(source: Path, base: Path) -> Iterable[Path]:
    source = source if source.is_absolute() else (base / source)
    if source.is_dir():
        yield from (
            path
            for path in source.rglob("*")
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
        root = (
            declared_root
            if declared_root.is_absolute()
            else yaml_path.parent / declared_root
        ).resolve()
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
            lowered_parts = {part.lower() for part in path.parts}
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and "labels" not in lowered_parts
                and ".label-lens-trash" not in lowered_parts
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
    ordered_paths = sorted(
        image_splits, key=lambda value: str(value).lower()
    )
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
        max_workers=worker_count,
        thread_name_prefix="yolo-index",
    ) as executor:
        for batch_start in range(0, total, batch_size):
            check_cancelled()
            batch = ordered_paths[batch_start:batch_start + batch_size]
            for offset, record in enumerate(
                executor.map(load_record, batch),
                start=1,
            ):
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
        names = [
            f"class_{index}" for index in range(maximum_class + 1)
        ]
    report(total, total, f"Loaded {total:,} images")
    return Dataset(root, names, records)
