"""Backward-compatible exports for the modular Label Lens core.

New code should import from ``models``, ``formats``, or ``services`` directly.
This facade keeps the original public API working for existing callers.
"""

from .formats.yolo import label_path_for, load_labels
from .models import AnnotationIssue, Box, Dataset, ImageRecord, copy_boxes
from .services.dataset_loader import (
    IMAGE_EXTENSIONS,
    CancelCallback,
    DatasetLoadCancelled,
    ProgressCallback,
    discover_dataset,
)
from .services.review_state import ReviewState
from .services.storage import (
    UNSAFE_ISSUE_KINDS,
    atomic_write,
    replace_class_in_records,
    save_labels,
    trash_record,
)
from .services.validation import box_quality_issues, validation_issues

__all__ = [
    "AnnotationIssue",
    "Box",
    "CancelCallback",
    "Dataset",
    "DatasetLoadCancelled",
    "IMAGE_EXTENSIONS",
    "ImageRecord",
    "ProgressCallback",
    "ReviewState",
    "UNSAFE_ISSUE_KINDS",
    "atomic_write",
    "box_quality_issues",
    "copy_boxes",
    "discover_dataset",
    "label_path_for",
    "load_labels",
    "replace_class_in_records",
    "save_labels",
    "trash_record",
    "validation_issues",
]
