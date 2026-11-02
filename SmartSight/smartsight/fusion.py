"""Fuse Tier 1 (detector), Tier 2 (zero-shot), OCR text and barcode evidence into one Recognition."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .taxonomy import CATEGORY_OTHER, Mapped

SOURCE_WEIGHT = {"barcode": 1.0, "OCR text": 1.0, "zero-shot model": 1.0, "detector": 0.95}


@dataclass
class Recognition:
    label: str
    category: str
    subcategory: str
    confidence: float
    source: str
    alternatives: list = field(default_factory=list)   # [(label, score)]
    evidence: list = field(default_factory=list)       # readable lines, one per contributing source
    generic: bool = False                              # True if only a vague detector class ("bottle") supports it
    track_id: int = 0
    box: Optional[tuple] = None
    specific_product: bool = False                     # named from a barcode record or readable product name


@dataclass
class _Cand:
    label: str
    category: str
    subcategory: str
    conf: float
    source: str
    generic: bool = False


def fuse(det: Optional[tuple], det_conf: float, zs: Sequence, zs_threshold: float = 0.30, ocr: Optional[tuple] = None,
         barcode: Optional[tuple] = None) -> Recognition:
    """det: Mapped | None; zs: [Candidate] best first; ocr: (VocabEntry, conf) | None; barcode: (name, category, subcategory, conf) | None."""
    if det is not None and det.category == "Person":
        return Recognition("person", "Person", "Person", det_conf, "detector", evidence=[f"detector: person ({det_conf:.0%})"])
    cands: list[_Cand] = []
    if det is not None:
        cands.append(_Cand(det.name, det.category, det.subcategory, det_conf, "detector", det.generic))
    if zs and zs[0].score >= zs_threshold and zs[0].category != "Person":
        t = zs[0]
        cands.append(_Cand(t.label, t.category, t.subcategory, t.score, "zero-shot model"))
    if ocr is not None:
        e, c = ocr
        cands.append(_Cand(e.name, e.category, e.subcategory, c, "OCR text"))
    if barcode is not None:
        n, cat, sub, c = barcode
        cands.append(_Cand(n, cat, sub, c, "barcode"))
    if not cands:
        alts = [(c.label, c.score) for c in zs[:3]]
        return Recognition("unknown object", CATEGORY_OTHER, CATEGORY_OTHER, 0.0, "none", alts)

    def eff(c: _Cand) -> float:
        return c.conf * SOURCE_WEIGHT[c.source] * (0.6 if c.generic else 1.0)

    win = max(cands, key=eff)
    agree = [c for c in cands if c is not win and c.category == win.category and (c.label == win.label or c.generic or win.generic or c.subcategory == win.subcategory)]
    conf = win.conf
    for c in agree:                                  # independent sources agreeing -> noisy-or boost
        conf = 1 - (1 - conf) * (1 - c.conf * 0.8)
    clash = [c for c in cands if c is not win and c not in agree and c.conf >= 0.5 and not (c.generic or win.generic)]
    if clash:
        conf *= 0.85
    conf = min(conf, 0.99)
    sources = [win.source] + [c.source for c in agree]
    alts, seen = [], {win.label}
    for c in cands:
        if c.label not in seen:
            alts.append((c.label, c.conf)); seen.add(c.label)
    for c in zs[:3]:
        if c.label not in seen:
            alts.append((c.label, c.score)); seen.add(c.label)
    evidence = [f"{c.source}: {c.label} ({c.conf:.0%})" for c in cands]
    only_generic = all(c.generic for c in cands if c.source == "detector") and all(c.source == "detector" for c in cands) and win.generic
    return Recognition(win.label, win.category, win.subcategory, conf, ", ".join(dict.fromkeys(sources)), alts[:3], evidence,
                       generic=only_generic, specific_product=win.source == "barcode")
