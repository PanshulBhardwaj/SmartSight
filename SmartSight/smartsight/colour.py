"""Dominant colour name of an object crop (simple HSV rules; indicative only)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def dominant_colour(crop, min_share: float = 0.30) -> Optional[str]:
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    import cv2
    h, w = crop.shape[:2]
    core = crop[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]       # centre of the object, away from background
    if core.size == 0:
        core = crop
    core = cv2.resize(core, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    hue, sat, val = hsv[:, 0] * 2, hsv[:, 1], hsv[:, 2]                         # hue in degrees
    names = np.empty(len(hsv), dtype=object)
    names[:] = "grey"
    names[val < 55] = "black"
    chroma = (sat >= 70) & (val >= 55)
    names[(sat < 40) & (val >= 190)] = "white"
    for lo, hi, n in ((0, 15, "red"), (15, 40, "orange"), (40, 70, "yellow"), (70, 170, "green"), (170, 260, "blue"),
                      (260, 300, "purple"), (300, 345, "pink"), (345, 361, "red")):
        names[chroma & (hue >= lo) & (hue < hi)] = n
    brown = chroma & (hue >= 10) & (hue < 40) & (val < 160)
    names[brown] = "brown"
    vals, counts = np.unique(names, return_counts=True)
    best = int(np.argmax(counts))
    return str(vals[best]) if counts[best] / len(names) >= min_share else "multicoloured"
