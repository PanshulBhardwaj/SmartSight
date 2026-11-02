"""Announce only stable, changed labels, with a cooldown so the user is never flooded."""
from __future__ import annotations

import time
from typing import Callable


class AnnouncementDebouncer:
    def __init__(self, stable_frames: int = 4, cooldown: float = 6.0, global_gap: float = 1.5,
                 clock: Callable[[], float] = time.monotonic):
        self.stable_frames, self.cooldown, self.global_gap, self._clock = stable_frames, cooldown, global_gap, clock
        self._last_label: dict = {}       # track id -> label last announced for it
        self._label_time: dict = {}       # label -> time last announced (any track)
        self._last_any = -1e9

    def update(self, key, label: str, streak: int) -> bool:
        """True if (key, label) should be spoken now. Call only for tracks currently seen."""
        if streak < self.stable_frames:
            return False
        now = self._clock()
        if self._last_label.get(key) == label:
            return False                                   # unchanged since we last told the user
        if now - self._last_any < self.global_gap:
            return False                                   # never two announcements back to back
        if now - self._label_time.get(label, -1e9) < self.cooldown and key not in self._last_label:
            return False                                   # same label announced recently for another object
        self._last_label[key] = label
        self._label_time[label] = now
        self._last_any = now
        return True

    def retain(self, live_keys) -> list:
        """Forget tracks that no longer exist; returns the ids that were forgotten (they were announced earlier)."""
        gone = [k for k in self._last_label if k not in set(live_keys)]
        for k in gone:
            del self._last_label[k]
        return gone

    def reset(self) -> None:
        self._last_label.clear(); self._label_time.clear(); self._last_any = -1e9
