"""Local knowledge base (JSON). Item entries override category defaults; both are plain data you can extend."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, categories: Optional[dict] = None, items: Optional[dict] = None):
        self.categories, self.items = categories or {}, items or {}

    @classmethod
    def from_file(cls, path: Path) -> "KnowledgeBase":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(d.get("categories", {}), d.get("items", {}))
        except (OSError, ValueError) as exc:
            log.warning("knowledge base unavailable (%s); using generic descriptions only", exc)
            return cls()

    def item(self, name: str) -> Optional[dict]:
        return self.items.get(name)

    def category(self, name: str) -> dict:
        return self.categories.get(name, {})
