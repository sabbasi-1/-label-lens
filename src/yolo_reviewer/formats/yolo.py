from __future__ import annotations

from pathlib import Path

from ..models import AnnotationIssue, Box
from ..services.validation import validation_issues


def label_path_for(image_path: Path, root: Path) -> Path:
    parts = list(image_path.parts)
    image_positions = [
        index
        for index, part in enumerate(parts)
        if part.lower() == "images"
    ]
    if image_positions:
        parts[image_positions[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        relative = Path(image_path.name)
    return (root / "labels" / relative).with_suffix(".txt")


def load_labels(
    path: Path, class_count: int
) -> tuple[list[Box], list[AnnotationIssue]]:
    boxes: list[Box] = []
    issues: list[AnnotationIssue] = []
    if not path.exists():
        return boxes, [
            AnnotationIssue(
                "missing-label",
                f"Missing label file: {path.name}",
            )
        ]
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
                "malformed-row",
                f"Line {line_number}: values are not numeric",
            ))
            continue
        if is_detection:
            box = Box(class_id, *values)
        else:
            box = _polygon_box(class_id, values)
        boxes.append(box)
    issues.extend(validation_issues(boxes, class_count))
    return boxes, issues


def _polygon_box(class_id: int, values: list[float]) -> Box:
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
    return Box(
        class_id,
        (left + right) / 2,
        (top + bottom) / 2,
        width,
        height,
        relative_points,
    )
