"""Position, distance and framing cues from bounding boxes, plus Find mode.

Distance is a HEURISTIC from how much of the frame the object fills (bigger objects look 'closer'); it is not metric depth.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def area_fraction(box, w: int, h: int) -> float:
    return max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / float(w * h)


def horizontal_zone(box, w: int) -> str:
    cx = _centre(box)[0] / w
    return "far left" if cx < 0.2 else "left" if cx < 0.4 else "centre" if cx < 0.6 else "right" if cx < 0.8 else "far right"


def distance_phrase(box, w: int, h: int) -> str:
    a = area_fraction(box, w, h)
    return "very close" if a >= 0.45 else "about arm's length" if a >= 0.15 else "a few steps away" if a >= 0.04 else "far away"


def position_phrase(box, w: int, h: int) -> str:
    """'on your left, about arm's length' / 'centred and close'."""
    zone, dist = horizontal_zone(box, w), distance_phrase(box, w, h)
    where = {"far left": "far to your left", "left": "on your left", "centre": "centred",
             "right": "on your right", "far right": "far to your right"}[zone]
    if zone == "centre" and dist in ("very close", "about arm's length"):
        return "centred and close"
    return f"{where}, {dist}"


def short_position(box, w: int) -> str:
    return {"far left": "far left", "left": "on your left", "centre": "in the centre", "right": "on your right",
            "far right": "far right"}[horizontal_zone(box, w)]


def edges_cut(box, w: int, h: int, margin: float = 0.012) -> list:
    out = []
    if box[0] <= w * margin: out.append("left")
    if box[2] >= w * (1 - margin): out.append("right")
    if box[1] <= h * margin: out.append("top")
    if box[3] >= h * (1 - margin): out.append("bottom")
    return out


def framing_hint(box, w: int, h: int, tol: float = 0.15) -> Optional[str]:
    """Spoken advice to improve framing, or None if it is fine. 'Move' = move the camera."""
    a = area_fraction(box, w, h)
    cut = edges_cut(box, w, h)
    if a > 0.85:
        return "It fills the whole view. Move the camera back a little."
    cx, cy = _centre(box)
    dx, dy = (cx - w / 2) / w, (cy - h / 2) / h
    steps = []
    if dx < -tol: steps.append("slightly left")
    elif dx > tol: steps.append("slightly right")
    if dy < -tol: steps.append("up")
    elif dy > tol: steps.append("down")
    if cut:
        side = " and ".join(cut)
        advice = f" Move the camera {' and '.join(steps)}." if steps else " Move the camera back a little."
        return f"The object is cut off at the {side} edge.{advice}"
    if steps:
        return f"Move the camera {' and '.join(steps)} to centre it."
    return None


# ------------------------------------------------------------------ Find mode
@dataclass
class FindUpdate:
    speech: Optional[str] = None
    beep: Optional[tuple] = None        # (frequency_hz, repeat_interval_s) while the object is visible
    earcon: Optional[str] = None        # "found" | "lost"
    state: str = "searching"            # searching | seen | centred


class FindMode:
    """'find my cup': audio cues get higher and faster as the object nears the centre of the frame."""
    CENTRED = 0.14

    def __init__(self, target: str, matcher: Callable[[object], bool], clock: Callable[[], float] = time.monotonic):
        self.target, self._match, self._clock = target, matcher, clock
        self.state, self._last_search_msg, self._last_seen = "searching", -1e9, -1e9

    @staticmethod
    def centre_distance(box, w: int, h: int) -> float:
        cx, cy = _centre(box)
        return min(1.0, (((cx - w / 2) / (w / 2)) ** 2 + ((cy - h / 2) / (h / 2)) ** 2) ** 0.5)

    def update(self, items: Sequence, w: int, h: int) -> FindUpdate:
        """items: [(recognition, box)] currently visible."""
        now = self._clock()
        hits = [(r, b) for r, b in items if self._match(r)]
        if not hits:
            if self.state != "searching" and now - self._last_seen > 1.5:
                self.state = "searching"
                return FindUpdate("I lost your %s. Move the camera slowly." % self.target, None, "lost", "searching")
            if self.state == "searching" and now - self._last_search_msg > 10:
                first = self._last_search_msg < -1e8
                self._last_search_msg = now
                return FindUpdate(None if first else f"Still looking for your {self.target}.", None, None, "searching")
            return FindUpdate(state=self.state)
        rec, box = max(hits, key=lambda x: area_fraction(x[1], w, h))
        self._last_seen = now
        d = self.centre_distance(box, w, h)
        freq, interval = int(400 + 900 * (1 - d)), 0.12 + 0.9 * d
        if d <= self.CENTRED:
            if self.state != "centred":
                self.state = "centred"
                return FindUpdate(f"Found your {self.target}, centred, {distance_phrase(box, w, h)}.", (1500, 0.12), "found", "centred")
            return FindUpdate(beep=(1500, 0.12), state="centred")
        if self.state in ("searching", "centred"):
            self.state = "seen"
            return FindUpdate(f"I can see your {self.target}, {position_phrase(box, w, h)}. Beeps get faster as it nears the centre.",
                              (freq, interval), "found", "seen")
        return FindUpdate(beep=(freq, interval), state="seen")
