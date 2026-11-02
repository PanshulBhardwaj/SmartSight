"""Lightweight IoU tracker (stable object IDs) with per-track label voting (temporal smoothing)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    label: str
    conf: float
    box: Box
    streak: int = 1          # consecutive frames seen
    hits: int = 1            # total frames seen
    misses: int = 0
    votes: deque = field(default_factory=lambda: deque(maxlen=8))

    def vote(self) -> Tuple[str, float]:
        """Confidence-weighted majority label over the recent window -> (label, share 0..1)."""
        if not self.votes:
            return self.label, 0.0
        score: dict = {}
        for lab, c in self.votes:
            score[lab] = score.get(lab, 0.0) + c
        best = max(score, key=score.get)
        return best, score[best] / (sum(score.values()) or 1.0)


def _mean_conf(track, label: str) -> float:
    cs = [c for l, c in track.votes if l == label]
    return sum(cs) / len(cs) if cs else track.conf


Track.mean_conf = _mean_conf   # type: ignore[attr-defined]


class IouTracker:
    def __init__(self, iou_same: float = 0.3, iou_diff: float = 0.5, max_misses: int = 6, vote_window: int = 8,
                 smoothing: float = 0.6):
        self.iou_same, self.iou_diff, self.max_misses = iou_same, iou_diff, max_misses
        self.vote_window, self.alpha = vote_window, smoothing
        self._tracks: dict = {}
        self._next = 1

    @property
    def tracks(self) -> list:
        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, dets: Sequence) -> list:
        """dets: objects with .label .confidence .box. Returns the tracks seen in this frame."""
        pairs = []
        for t in self._tracks.values():
            for j, d in enumerate(dets):
                v = iou(t.box, d.box)
                if v >= (self.iou_same if t.label == d.label else self.iou_diff):
                    pairs.append((v, t.id, j))
        pairs.sort(reverse=True)
        used_t, used_d, seen = set(), set(), []
        for v, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j)
            t, d = self._tracks[tid], dets[j]
            a = self.alpha
            t.box = tuple(a * n + (1 - a) * o for n, o in zip(d.box, t.box))
            t.label, t.conf = d.label, d.confidence
            t.streak += 1; t.hits += 1; t.misses = 0
            t.votes.append((d.label, d.confidence))
            seen.append(t)
        for j, d in enumerate(dets):
            if j in used_d:
                continue
            t = Track(self._next, d.label, d.confidence, tuple(float(x) for x in d.box),
                      votes=deque([(d.label, d.confidence)], maxlen=self.vote_window))
            self._tracks[t.id] = t
            self._next += 1
            seen.append(t)
        for tid in list(self._tracks):
            if tid not in used_t and self._tracks[tid] not in seen:
                t = self._tracks[tid]
                t.misses += 1
                t.streak = 0
                if t.misses > self.max_misses:
                    del self._tracks[tid]
        return seen
