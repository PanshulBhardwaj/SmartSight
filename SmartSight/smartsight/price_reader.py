"""Printed-price extraction from OCR text.

Rules: a number is a price only with currency or price-label context. Dates, weights,
barcodes and nutrition values are rejected. Original OCR text is preserved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from .utils import InfoState, OcrLine, TextSource, as_lines

_CUR = r"(?:₹|(?<![a-z])rs\.?|(?<![a-z])inr|(?<![a-z])rupees?)"
_NUM = r"(?P<int>\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(?P<dec>\d{1,2}))?(?!\d)"
_SEP = r"[\s:\-=.]{0,4}"
_UNIT_AFTER = re.compile(r"^\s?(?:g|gm|gms|grams?|kg|ml|l|ltr|mg|%|kcal|kj)(?![a-z])", re.I)
_DATE_AFTER = re.compile(r"^[/\-.]\d{1,2}[/\-.]\d{2,4}")
_CUR_RE = re.compile(rf"(?P<cur>{_CUR})\s*{_NUM}", re.I)
_SUFFIX_RE = re.compile(rf"(?<![\d.,]){_NUM}\s*/\s*-")
_PREF = {"product": ("mrp", "sale", "printed"), "receipt": ("total", "printed", "sale", "mrp")}
KIND_PHRASE = {"mrp": "maximum retail price", "sale": "sale price", "printed": "price",
               "total": "total", "unit": "unit price"}


def _groups(mode: str):
    g = [("unit", r"unit\s+price|price\s+per\s+\w+"),
         ("mrp", r"maximum\s+retail\s+price|m\.?\s?r\.?\s?p"),
         ("sale", r"sale\s+price|selling\s+price|offer\s+price|our\s+price|special\s+price|discounted\s+price")]
    if mode == "receipt":
        g.append(("total", r"grand\s+total|total\s+amount|amount\s+due|net\s+amount|total"))
    g.append(("printed", r"price"))
    return g


def _label_re(mode: str):
    alt = "|".join(f"(?P<{k}>{v})" for k, v in _groups(mode))
    return re.compile(rf"(?<![a-z])(?:{alt})(?![a-z])\s*(?:\([^)]{{0,40}}\))?{_SEP}(?:(?P<cur>{_CUR})\s*)?{_NUM}", re.I)


def _label_only_re(mode: str):
    alt = "|".join(v for _, v in _groups(mode))
    return re.compile(rf"^\s*(?:{alt})[\s:\-=.]*$", re.I)


@dataclass(frozen=True)
class PriceCandidate:
    amount: Decimal
    kind: str            # mrp | sale | printed | total | unit
    raw: str             # matched text
    source_text: str     # the original OCR text the match came from
    ocr_confidence: Optional[float]
    state: InfoState
    reason: str


@dataclass
class PriceResult:
    state: InfoState
    best: Optional[PriceCandidate]
    candidates: list = field(default_factory=list)
    rejected: list = field(default_factory=list)   # (raw, why)
    saw_numbers: bool = False
    conflict: bool = False

    def of_kind(self, kind: str) -> list:
        return [c for c in self.candidates if c.kind == kind]


def extract_prices(source: TextSource, *, mode: str = "product", min_confidence: float = 0.7) -> PriceResult:
    lines = as_lines(source)
    units = _merge_label_lines(lines, mode)
    cands: list[PriceCandidate] = []
    rejected: list = []
    for text, conf in units:
        _scan(text, conf, mode, min_confidence, cands, rejected)
    uniq: dict = {}
    for c in cands:
        k = (c.amount, c.kind)
        if k not in uniq or _rank(c) > _rank(uniq[k]):
            uniq[k] = c
    cands = list(uniq.values())
    best, conflict = _choose(cands, mode)
    saw = any(re.search(r"\d", t) for t, _ in units)
    state = best.state if best else InfoState.UNKNOWN
    if conflict:
        state = InfoState.TENTATIVE
    return PriceResult(state, best, cands, rejected, saw and best is None, conflict)


def _rank(c: PriceCandidate):
    return (c.state == InfoState.CONFIRMED, c.ocr_confidence or 0.0)


def _merge_label_lines(lines: list[OcrLine], mode: str) -> list[tuple[str, Optional[float]]]:
    """A line that is only a label ('MRP') is joined with the next line."""
    label_only = _label_only_re(mode)
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if label_only.match(ln.text) and i + 1 < len(lines):
            nxt = lines[i + 1]
            out.append((f"{ln.text.strip()} {nxt.text.strip()}", min(ln.conf, nxt.conf)))
            i += 2
        else:
            out.append((ln.text, ln.conf))
            i += 1
    return out


def _scan(text, conf, mode, min_conf, out, rejected):
    found: dict = {}
    lab = _label_re(mode)
    for m in lab.finditer(text):
        kind = next(k for k, v in m.groupdict().items() if k in KIND_PHRASE and v)
        found[m.start("int")] = (m, kind, bool(m.group("cur")), False)
    for m in _CUR_RE.finditer(text):
        found.setdefault(m.start("int"), (m, "printed", True, False))
    for m in _SUFFIX_RE.finditer(text):
        found.setdefault(m.start("int"), (m, "printed", False, True))
    for m, kind, has_cur, suffix_only in found.values():
        raw = m.group(0).strip()
        digits = m.group("int").replace(",", "")
        after = text[m.end():m.end() + 6]
        if len(digits) > 7:
            rejected.append((raw, "too many digits (barcode or batch number)"))
            continue
        if _DATE_AFTER.match(text[m.end("int"):]):
            rejected.append((raw, "looks like a date"))
            continue
        if not has_cur and _UNIT_AFTER.match(after):
            rejected.append((raw, "followed by a weight/volume unit"))
            continue
        amount = Decimal(digits + ("." + m.group("dec") if m.group("dec") else ""))
        if amount <= 0:
            rejected.append((raw, "zero amount"))
            continue
        conf_ok = conf is None or conf >= min_conf
        if suffix_only:
            state, why = InfoState.TENTATIVE, "only a '/-' suffix"
        elif has_cur or kind in ("mrp", "sale"):
            state = InfoState.CONFIRMED if conf_ok else InfoState.TENTATIVE
            why = "currency symbol" if has_cur else f"{kind} label"
            if not conf_ok:
                why += "; low OCR confidence"
        else:
            state, why = InfoState.TENTATIVE, "price word without currency symbol"
        out.append(PriceCandidate(amount, kind, raw, text, conf, state, why))


def _choose(cands, mode):
    for kind in _PREF[mode]:
        group = [c for c in cands if c.kind == kind]
        if not group:
            continue
        group.sort(key=_rank, reverse=True)
        strong = {c.amount for c in group if c.state == InfoState.CONFIRMED}
        return group[0], len(strong) > 1
    return None, False
