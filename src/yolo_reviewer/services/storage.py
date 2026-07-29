from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

from ..models import AnnotationIssue, Box, Dataset, ImageRecord, copy_boxes
from .validation import validation_issues

UNSAFE_ISSUE_KINDS = {
    "unsupported-row",
    "malformed-row",
    "unreadable-label",
}


def atomic_write(
    path: Path, content: str, *, make_backup: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if make_backup and path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
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
    unsafe = [
        issue
        for issue in record.issues
        if issue.kind in UNSAFE_ISSUE_KINDS
    ]
    if unsafe:
        raise ValueError(
            "This label file contains rows that are not standard YOLO detection "
            "or segmentation annotations. Saving is blocked to avoid deleting "
            "unsupported data. See the Automatic checks panel."
        )
    content = "".join(f"{box.serialize()}\n" for box in record.boxes)
    atomic_write(record.label_path, content)
    record.issues = validation_issues(record.boxes, class_count)


def replace_class_in_records(
    records: Iterable[ImageRecord],
    source_class: int,
    target_class: int,
    class_count: int = 0,
    backup_root: Path | None = None,
) -> tuple[int, int, Path | None]:
    """Replace a class across records and roll every file back on failure."""
    if source_class == target_class:
        raise ValueError("Source and replacement classes must be different.")
    affected = [
        record
        for record in records
        if any(box.class_id == source_class for box in record.boxes)
    ]
    unsafe_records = [
        record
        for record in affected
        if any(
            issue.kind in UNSAFE_ISSUE_KINDS for issue in record.issues
        )
    ]
    if unsafe_records:
        raise ValueError(
            f"{len(unsafe_records)} affected label file(s) contain unsupported "
            "or malformed rows. No files were changed."
        )

    snapshots: list[
        tuple[ImageRecord, list[Box], list[AnnotationIssue], str | None]
    ] = []
    annotation_count = 0
    for record in affected:
        original = (
            record.label_path.read_text(encoding="utf-8")
            if record.label_path.exists()
            else None
        )
        snapshots.append((
            record,
            copy_boxes(record.boxes),
            list(record.issues),
            original,
        ))
        annotation_count += sum(
            box.class_id == source_class for box in record.boxes
        )

    operation_backup = _write_bulk_backup(
        snapshots,
        source_class,
        target_class,
        annotation_count,
        backup_root,
    )
    written: list[tuple[ImageRecord, str | None]] = []
    try:
        for record, _boxes, _issues, original in snapshots:
            for box in record.boxes:
                if box.class_id == source_class:
                    box.class_id = target_class
            written.append((record, original))
            save_labels(record, class_count)
    except BaseException as exc:
        for record, boxes, issues, _original in snapshots:
            record.boxes[:] = boxes
            record.issues = issues
        rollback_errors: list[str] = []
        for record, original in reversed(written):
            try:
                if original is None:
                    record.label_path.unlink(missing_ok=True)
                else:
                    atomic_write(
                        record.label_path, original, make_backup=False
                    )
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{record.label_path.name}: {rollback_exc}"
                )
        if rollback_errors:
            raise RuntimeError(
                f"Bulk replacement failed ({exc}) and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    return len(affected), annotation_count, operation_backup


def _write_bulk_backup(
    snapshots: list[
        tuple[ImageRecord, list[Box], list[AnnotationIssue], str | None]
    ],
    source_class: int,
    target_class: int,
    annotation_count: int,
    backup_root: Path | None,
) -> Path | None:
    if not snapshots or backup_root is None:
        return None
    operation_backup = (
        backup_root / ".label-lens-bulk-backups" / uuid.uuid4().hex
    )
    entries: list[dict[str, str]] = []
    for record, _boxes, _issues, original in snapshots:
        if original is None:
            continue
        try:
            relative = record.label_path.relative_to(backup_root)
        except ValueError:
            relative = Path("external") / record.label_path.name
        backup_path = operation_backup / "labels" / relative
        atomic_write(backup_path, original, make_backup=False)
        entries.append({
            "original": str(record.label_path),
            "backup": str(backup_path),
        })
    manifest = {
        "source_class": source_class,
        "target_class": target_class,
        "annotation_count": annotation_count,
        "files": entries,
    }
    atomic_write(
        operation_backup / "manifest.json",
        json.dumps(manifest, indent=2) + "\n",
        make_backup=False,
    )
    return operation_backup


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
