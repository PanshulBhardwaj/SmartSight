"""Nutrition tables whose labels and numbers sit in separate columns.

OCR boxes are grouped into rows by y-coordinate; each label is paired with the numbers on its row, and numbers are
assigned to the 'per 100 g' / 'per serving' columns by x position. Output: 'Protein 7 grams per serving'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Optional, Sequence

from .utils import OcrLine

I = re.I
_UNIT_WORDS = {"g": "grams", "gm": "grams", "mg": "milligrams", "mcg": "micrograms", "µg": "micrograms", "kcal": "kilocalories",
               "cal": "calories", "kj": "kilojoules", "ml": "millilitres"}
_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(kcal|kj|cal|mcg|µg|mg|gm|g|ml|%)?(?![a-z])", I)
_NUTRIENTS = [      # ordered: specific before general
    ("saturated_fat", r"saturated|sat\.?\s*fat"), ("trans_fat", r"trans"), ("added_sugars", r"added\s+sugar"),
    ("sugars", r"sugars?"), ("fat", r"fat"), ("energy", r"energy|calories"), ("protein", r"protein"),
    ("carbohydrate", r"carbohydrate|carbs?"), ("fibre", r"fib(?:re|er)"), ("sodium", r"sodium"), ("salt", r"\bsalt\b"),
]
_HDR = [("per 100 grams", r"per\s*100\s*(?:g|gm)\b"), ("per 100 millilitres", r"per\s*100\s*ml\b"),
        ("per serving", r"per\s*(?:serving|serve|portion)\b|serving\s*size"), ("ignore", r"%\s*(?:rda|dv|nrv|daily)|rda|\bnrv\b")]


def nutrition_phrase(name: str, value: float, unit: Optional[str], basis: Optional[str] = None) -> str:
    """'Protein 7 grams per serving'."""
    u = _UNIT_WORDS.get((unit or "").lower(), unit or "")
    return " ".join(x for x in (name.replace("_", " ").capitalize(), f"{value:g}", u, basis) if x)


@dataclass
class Cell:
    nutrient: str
    value: float
    unit: Optional[str]
    basis: Optional[str]


@dataclass
class TableResult:
    cells: list = field(default_factory=list)
    columns: list = field(default_factory=list)        # [(basis, x)]
    rows: int = 0

    @property
    def phrases(self) -> list:
        order = {"per serving": 0, "per 100 grams": 1, "per 100 millilitres": 2, None: 3}
        out = []
        by_nut: dict = {}
        for c in self.cells:
            by_nut.setdefault(c.nutrient, []).append(c)
        for cs in by_nut.values():
            for c in sorted(cs, key=lambda c: order.get(c.basis, 3)):
                out.append(nutrition_phrase(c.nutrient, c.value, c.unit, c.basis))
        return out


def _cy(l: OcrLine) -> float:
    return (l.box[1] + l.box[3]) / 2


def group_rows(lines: Sequence[OcrLine], tol: float = 0.6) -> list:
    """Group boxed OCR lines into visual rows (top to bottom), each sorted left to right."""
    boxed = sorted((l for l in lines if l.box), key=_cy)
    if not boxed:
        return []
    h = median(max(1, l.box[3] - l.box[1]) for l in boxed)
    rows, cur, cur_y = [], [], None
    for l in boxed:
        y = _cy(l)
        if cur and abs(y - cur_y) > tol * h:
            rows.append(sorted(cur, key=lambda l: l.box[0])); cur = []
        cur.append(l)
        cur_y = sum(_cy(x) for x in cur) / len(cur)
    if cur:
        rows.append(sorted(cur, key=lambda l: l.box[0]))
    return rows


def merge_rows(lines: Sequence[OcrLine]) -> list:
    """One OcrLine per visual row (text joined left to right). Lines without boxes pass through unchanged."""
    out = [l for l in lines if not l.box]
    for row in group_rows(lines):
        box = (min(l.box[0] for l in row), min(l.box[1] for l in row), max(l.box[2] for l in row), max(l.box[3] for l in row))
        out.append(OcrLine(" ".join(l.text.strip() for l in row), min(l.conf for l in row), box))
    return out


def _x_at(line: OcrLine, pos: int, n: int) -> float:
    x1, _, x2, _ = line.box
    return x1 + (pos + 0.5) / max(n, 1) * (x2 - x1)


def _header_cols(row) -> list:
    cols = []
    for l in row:
        for basis, rx in _HDR:
            for m in re.finditer(rx, l.text, I):
                cols.append((basis, _x_at(l, (m.start() + m.end()) // 2, len(l.text))))
    return cols


def _nutrient(label: str) -> Optional[str]:
    for key, rx in _NUTRIENTS:
        if re.search(rx, label, I):
            return key
    return None


def parse_table(lines: Sequence[OcrLine]) -> TableResult:
    rows = group_rows(lines)
    res = TableResult(rows=len(rows))
    cols: list = []
    for row in rows:                                                       # header: the row naming the columns
        c = _header_cols(row)
        if len(c) >= len(cols) and c:
            cols = c
    res.columns = [(b, x) for b, x in cols if b != "ignore"]
    for row in rows:
        if _header_cols(row):
            continue
        label_parts, tokens = [], []
        for l in row:
            first = _NUM.search(l.text)
            head = l.text[:first.start()] if first else l.text
            if not tokens and head.strip():
                label_parts.append(head)
            for m in _NUM.finditer(l.text):
                tokens.append((float(m.group(1).replace(",", ".")), (m.group(2) or "").lower() or None, _x_at(l, (m.start() + m.end()) // 2, len(l.text))))
            if first is not None and not tokens:
                pass
        key = _nutrient(" ".join(label_parts))
        if key is None or not tokens:
            continue
        for value, unit, x in tokens:
            if unit == "%":
                continue                                                    # %RDA / %DV are not nutrient amounts
            basis = None
            if cols:
                basis, cx = min(cols, key=lambda c: abs(c[1] - x))
                if basis == "ignore":
                    continue
            if key == "energy" and unit is None:
                unit = "kcal"
            res.cells.append(Cell(key, value, unit, basis))
    return res
