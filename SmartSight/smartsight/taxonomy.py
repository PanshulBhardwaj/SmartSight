"""Vocabulary (Tier 2 labels) and taxonomy (categories + detector-label mapping). Pure data-driven: edit the JSON files."""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

CATEGORY_OTHER = "Other"


@dataclass(frozen=True)
class VocabEntry:
    name: str
    category: str
    subcategory: str
    used_for: str = ""
    aliases: tuple = ()


class Mapped(NamedTuple):
    name: str
    category: str
    subcategory: str
    generic: bool          # True for vague detector classes ("bottle", "cup") that Tier 2 should refine


def _norm(text: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    t = re.sub(r"\b(my|the|a|an|some|of)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _singular(t: str) -> str:
    return t[:-1] if t.endswith("s") and not t.endswith("ss") and len(t) > 3 else t


class Vocabulary:
    def __init__(self, entries: list[VocabEntry]):
        self.entries = entries
        self._index: dict[str, VocabEntry] = {}
        for e in entries:
            for n in (e.name, *e.aliases):
                self._index.setdefault(_singular(_norm(n)), e)

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        out = []
        for g in data["groups"]:
            for it in g["items"]:
                out.append(VocabEntry(it["name"], g["category"], g["subcategory"], g.get("used_for", ""), tuple(it.get("aliases", ()))))
        return cls(out)

    @classmethod
    def empty(cls) -> "Vocabulary":
        return cls([])

    def content_hash(self) -> str:
        return hashlib.md5(json.dumps([(e.name, e.aliases, e.category) for e in self.entries]).encode()).hexdigest()[:12]

    def get(self, name: str) -> Optional[VocabEntry]:
        return self._index.get(_singular(_norm(name)))

    def find(self, text: str, cutoff: float = 0.82) -> Optional[VocabEntry]:
        """Match free text ('red water bottle', 'cups', 'shampoo') to a vocabulary entry."""
        q = _singular(_norm(text))
        if not q:
            return None
        if q in self._index:
            return self._index[q]
        # the query contains a known name ("red water bottle" contains "water bottle"): prefer the longest
        hits = [(len(k), e) for k, e in self._index.items() if re.search(rf"\b{re.escape(k)}\b", q)]
        if hits:
            return max(hits, key=lambda h: h[0])[1]
        # a known name contains the query as a whole word ("shampoo" -> "shampoo bottle"): prefer shortest
        hits = [(len(k), e) for k, e in self._index.items() if re.search(rf"\b{re.escape(q)}\b", k)]
        if hits:
            return min(hits, key=lambda h: h[0])[1]
        m = difflib.get_close_matches(q, list(self._index), n=1, cutoff=cutoff)
        return self._index[m[0]] if m else None


class Taxonomy:
    def __init__(self, categories: dict, detector_map: dict):
        self.categories = categories            # name -> [subcategories]
        self._map = detector_map

    @classmethod
    def from_file(cls, path: Path) -> "Taxonomy":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({c["name"]: c["subcategories"] for c in d["categories"]}, d["detector_map"])

    @classmethod
    def minimal(cls) -> "Taxonomy":
        return cls({CATEGORY_OTHER: [CATEGORY_OTHER]}, {})

    def map_detector(self, label: str) -> Mapped:
        v = self._map.get(label)
        if v:
            return Mapped(v["name"], v["category"], v["subcategory"], bool(v.get("generic")))
        return Mapped(label, CATEGORY_OTHER, CATEGORY_OTHER, True)

    def is_valid(self, category: str, subcategory: str) -> bool:
        return subcategory in self.categories.get(category, ())
