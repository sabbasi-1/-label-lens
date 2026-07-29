from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase

from ..models import Dataset, ImageRecord
from .review_state import ReviewState


@dataclass
class ReviewSession:
    """UI-independent queue and scope calculations for a loaded dataset."""

    dataset: Dataset
    state: ReviewState

    def filtered_indices(
        self,
        mode: str,
        selected_split: str,
        class_ids: set[int],
    ) -> list[int]:
        indices: list[int] = []
        for index, record in enumerate(self.dataset.records):
            key = self.dataset.key(record)
            review_matches = (
                mode == "All images"
                or (
                    mode == "Unreviewed"
                    and key not in self.state.reviewed
                )
                or (
                    mode == "Flagged"
                    and key in self.state.flagged
                )
                or (mode == "Suspicious" and bool(record.issues))
            )
            split_matches = (
                selected_split == "all"
                or record.split == selected_split
            )
            if (
                review_matches
                and split_matches
                and record.contains_any_class(class_ids)
            ):
                indices.append(index)
        return indices

    def records_for_scope(
        self,
        current: ImageRecord,
        scope: str,
        filename_pattern: str = "",
    ) -> tuple[list[ImageRecord], str]:
        if scope == "current":
            return [current], "the current image"
        if scope == "split":
            records = [
                record
                for record in self.dataset.records
                if record.split == current.split
            ]
            return records, f"the {current.split} split"
        if scope == "folder":
            records = [
                record
                for record in self.dataset.records
                if record.image_path.parent == current.image_path.parent
            ]
            return records, f"folder {current.image_path.parent}"
        if scope == "pattern":
            pattern = filename_pattern.casefold()
            records = [
                record
                for record in self.dataset.records
                if fnmatchcase(record.image_path.name.casefold(), pattern)
            ]
            return records, f'filename pattern "{filename_pattern}"'
        return [], "unknown scope"


def suggested_filename_pattern(filename: str) -> str:
    pattern = re.sub(
        r"(?i)(?<=\.rf\.)[0-9a-f]{16,}",
        "*",
        filename,
    )
    pattern = re.sub(r"\d+", "*", pattern)
    return re.sub(r"\*+", "*", pattern)
