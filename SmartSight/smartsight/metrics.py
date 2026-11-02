"""Small helpers for measuring FPS and latency percentiles."""
from __future__ import annotations

import time
from collections import deque
from typing import Callable


class Percentiles:
    def __init__(self, maxlen: int = 1000):
        self._v: deque = deque(maxlen=maxlen)

    def add(self, x: float) -> None:
        self._v.append(x)

    def p(self, q: float) -> float:
        if not self._v:
            return 0.0
        s = sorted(self._v)
        return s[min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))]

    def summary(self) -> dict:
        return {"n": len(self._v), "p50": self.p(50), "p95": self.p(95), "mean": (sum(self._v) / len(self._v)) if self._v else 0.0}


class FpsMeter:
    def __init__(self, window: float = 1.0, clock: Callable[[], float] = time.perf_counter):
        self._t: deque = deque()
        self._w, self._clock = window, clock

    def tick(self) -> float:
        now = self._clock()
        self._t.append(now)
        while self._t and now - self._t[0] > self._w:
            self._t.popleft()
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0
