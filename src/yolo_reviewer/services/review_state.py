from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .storage import atomic_write


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
        atomic_write(
            path,
            json.dumps(data, indent=2) + "\n",
            make_backup=False,
        )
