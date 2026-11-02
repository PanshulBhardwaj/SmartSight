"""Cheap frame-quality checks (sharpness via Laplacian variance, mean brightness)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QualityReport:
    ok: bool
    sharpness: float = 0.0
    brightness: float = 0.0
    issues: list = field(default_factory=list)   # subset of: invalid, dark, blurry


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return (frame[..., :3].astype(np.float32) @ np.array([0.114, 0.587, 0.299], dtype=np.float32))
    return frame.astype(np.float32)


def sharpness(frame: np.ndarray) -> float:
    import cv2
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def assess_frame(frame, min_sharpness: float = 60.0, min_brightness: float = 50.0) -> QualityReport:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        return QualityReport(False, issues=["invalid"])
    try:
        bright = float(_gray(frame).mean())
        sharp = sharpness(frame)
    except Exception:  # noqa: BLE001
        return QualityReport(False, issues=["invalid"])
    issues = []
    if bright < min_brightness:
        issues.append("dark")
    if sharp < min_sharpness:
        issues.append("blurry")
    return QualityReport(not issues, sharp, bright, issues)
