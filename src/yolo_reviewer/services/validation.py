from __future__ import annotations

from ..models import AnnotationIssue, Box


def validation_issues(
    boxes: list[Box], class_count: int = 0
) -> list[AnnotationIssue]:
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
    for first_index, first in enumerate(boxes):
        for second_index in range(first_index + 1, len(boxes)):
            second = boxes[second_index]
            overlap = _iou(first, second)
            if overlap >= 0.98:
                issues.append(AnnotationIssue(
                    "duplicate-box",
                    f"Boxes {first_index + 1} and "
                    f"{second_index + 1} appear duplicated",
                    first_index,
                ))
            elif overlap >= 0.85:
                issues.append(AnnotationIssue(
                    "high-overlap",
                    f"Boxes {first_index + 1} and {second_index + 1} "
                    f"overlap by {overlap:.0%}",
                    first_index,
                ))
    return issues
