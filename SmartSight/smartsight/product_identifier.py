"""Product interpretation with an explicit evidence priority.

1 readable label  >  2 barcode record checked against label  >  3 database record alone
>  4 general knowledge of a detected object/category  >  5 uncertainty.
A database record never overrides what the package says; mismatches are reported as conflicts.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from .food_label_parser import front_text
from .product_lookup import ProductRecord
from .utils import InfoState, OcrLine

KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "data" / "product_knowledge.json"
_STOP = {"the", "and", "with", "for", "pack", "net", "pvt", "ltd", "new", "original"}
_NOISE = re.compile(r"\d|mrp|rs\.?|₹|\bexp|\bmfg|batch|fssai|kcal|protein|sugar|\bnet\b", re.I)


@dataclass
class ProductInfo:
    state: InfoState
    level: str                       # specific | category | object | unknown
    name: Optional[str] = None
    category: Optional[str] = None
    purpose: Optional[str] = None
    caution: Optional[str] = None
    source: str = "none"             # label+barcode | barcode-database | label | detector | none
    object_label: Optional[str] = None
    name_guess: Optional[str] = None  # most prominent label text; NOT a verified name
    text_found: bool = False
    conflict: bool = False
    notes: list = field(default_factory=list)


def load_knowledge(path: Path = KNOWLEDGE_PATH) -> list[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [{"category": "shampoo", "keywords": ["shampoo"], "purpose": "Shampoo is used to wash hair."}]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t not in _STOP}


def guess_product_name(lines: Sequence[OcrLine], min_conf: float = 0.6) -> Optional[str]:
    cands = [l for l in lines if l.conf >= min_conf and sum(c.isalpha() for c in l.text) >= 3 and not _NOISE.search(l.text)]
    if not cands:
        return None
    best = max(cands, key=lambda l: (l.height, len(l.text) if not l.height else 0))
    return best.text.strip()


class ProductIdentifier:
    def __init__(self, knowledge: Optional[list[dict]] = None, min_conf: float = 0.6):
        self.knowledge = knowledge if knowledge is not None else load_knowledge()
        self.min_conf = min_conf
        self._kw = sorted(((kw.lower(), e) for e in self.knowledge for kw in e["keywords"]), key=lambda x: -len(x[0]))

    def identify(self, objects: Sequence[str] = (), lines: Sequence[OcrLine] = (),
                 record: Optional[ProductRecord] = None) -> ProductInfo:
        good = [l for l in lines if l.conf >= self.min_conf]
        full_text = " ".join(l.text for l in good)
        ocr_tokens = _tokens(full_text)
        obj = objects[0] if objects else None
        info = ProductInfo(InfoState.UNKNOWN, "unknown", object_label=obj,
                           text_found=len(ocr_tokens) >= 2, name_guess=guess_product_name(self._front_lines(lines), self.min_conf))

        # 2/3: barcode record, verified against label text
        if record:
            rtoks = _tokens(f"{record.brand} {record.name}")
            overlap = self._overlap(rtoks, ocr_tokens)
            if rtoks and overlap >= 0.5:
                info.state, info.level, info.source = InfoState.CONFIRMED, "specific", "label+barcode"
                info.name = f"{record.brand} {record.name}".strip() if record.brand and record.brand.lower() not in record.name.lower() else record.name
                info.purpose = record.purpose or None
                info.category = record.category or None
                info.notes.append(f"record from {record.source} agrees with printed label text")
                self._add_category(info, front_text(good))
                return info
            if len(ocr_tokens) >= 4:
                info.conflict, info.state, info.source = True, InfoState.TENTATIVE, "barcode-database"
                info.notes.append(f"barcode record ({record.source}) says '{record.name}' but the label text does not match")
                return info
            info.state, info.level, info.source = InfoState.TENTATIVE, "specific", "barcode-database"
            info.name, info.category, info.purpose = record.name, record.category or None, record.purpose or None
            info.notes.append(f"from {record.source} record only; label could not confirm it")
            self._add_category(info, front_text(good))
            return info

        # 1/4: category from front-of-pack label text
        if self._add_category(info, front_text(good)):
            info.state, info.level, info.source = InfoState.TENTATIVE, "category", "label"
            return info

        # 4: detector class alone
        if obj:
            info.state, info.level, info.source = InfoState.TENTATIVE, "object", "detector"
            return info
        return info

    @staticmethod
    def _front_lines(lines: Sequence[OcrLine]) -> list[OcrLine]:
        keep = set(front_text(lines).splitlines())
        return [l for l in lines if l.text in keep]

    def _add_category(self, info: ProductInfo, text: str) -> bool:
        low = text.lower()
        for kw, entry in self._kw:
            if re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", low):
                info.category = info.category or entry["category"]
                info.purpose = info.purpose or entry.get("purpose")
                info.caution = entry.get("caution")
                return True
        return False

    @staticmethod
    def _overlap(rtoks: set[str], otoks: set[str]) -> float:
        if not rtoks:
            return 0.0
        hit = sum(1 for t in rtoks if t in otoks or difflib.get_close_matches(t, otoks, n=1, cutoff=0.85))
        return hit / len(rtoks)
