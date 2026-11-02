"""Veg / non-veg mark detection: a coloured square outline with a filled shape (circle or triangle) inside.

Colour decides the answer (green = vegetarian, red/brown = non-vegetarian); the nested-square structure is what makes a
candidate trustworthy, so plain red or green blobs, logos and solid rectangles are rejected.
Limitation: needs the mark to be at least ~14 px wide in the image. Tested on synthetic images, not yet on real packs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VegMark:
    label: Optional[str]            # "vegetarian" | "non-vegetarian" | None
    confidence: float               # structure x colour purity; an uncalibrated score
    box: Optional[tuple] = None


def _candidate_score(n: np.ndarray) -> float:
    """n: 48x48 boolean mask of one candidate. Score 0..1 for 'square outline with a filled shape inside'."""
    side = 6                                                           # outer band, 12.5 % of the side
    cov = [n[:side, :].any(axis=0).mean(), n[-side:, :].any(axis=0).mean(), n[:, :side].any(axis=1).mean(), n[:, -side:].any(axis=1).mean()]
    ring = min(cov)
    if ring < 0.85:
        return 0.0
    inner = float(n[12:36, 12:36].mean())
    if not 0.25 <= inner <= 1.0:
        return 0.0
    gap = float(np.mean([n[7:11, 7:41].mean(), n[37:41, 7:41].mean(), n[7:41, 7:11].mean(), n[7:41, 37:41].mean()]))
    if gap > 0.45 or n.mean() > 0.85:                                  # solid rectangle / blob
        return 0.0
    ys, xs = np.nonzero(n[10:38, 10:38])
    if len(xs) == 0:
        return 0.0
    cy, cx = ys.mean() + 10, xs.mean() + 10
    centred = 1.0 - min(1.0, (abs(cx - 24) + abs(cy - 24)) / 14.0)     # the inner shape sits near the middle
    return 0.35 * ring + 0.35 * min(1.0, inner / 0.5) + 0.2 * (1.0 - gap) + 0.1 * centred


def classify_veg_mark(img, min_side: int = 14) -> VegMark:
    import cv2
    if img is None or getattr(img, "size", 0) == 0:
        return VegMark(None, 0.0)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = (S > 70) & (V > 40)
    masks = {"vegetarian": colored & (H >= 38) & (H <= 90), "non-vegetarian": colored & ((H <= 22) | (H >= 165))}
    best = VegMark(None, 0.0)
    for name, mask in masks.items():
        m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if min(bw, bh) < min_side or max(bw, bh) > 0.9 * min(h, w) or not 0.8 <= bw / bh <= 1.25:
                continue
            sub = m[y:y + bh, x:x + bw] > 0
            n = cv2.resize(sub.astype(np.uint8) * 255, (48, 48), interpolation=cv2.INTER_AREA) > 100
            score = _candidate_score(n)
            if score <= 0:
                continue
            hue = H[y:y + bh, x:x + bw][sub].astype(np.int32)
            med = np.median(hue)
            diff = np.minimum(np.abs(hue - med), 180 - np.abs(hue - med))
            purity = float((diff <= 12).mean())
            conf = min(0.99, score * purity)
            if conf > best.confidence:
                best = VegMark(name, conf, (x, y, x + bw, y + bh))
    return best if best.confidence >= 0.5 else VegMark(None, best.confidence, None)
