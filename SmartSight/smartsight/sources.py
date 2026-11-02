"""Synthetic video source (tests, benchmarks, demos without a camera). Implements the CameraSource interface."""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class SyntheticSource:
    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0, seed: int = 0):
        self.w, self.h, self.fps = width, height, fps
        self._rng = np.random.default_rng(seed)
        self._base = self._rng.integers(60, 200, (height, width, 3), dtype=np.uint8)   # textured, sharp, bright
        self._lock = threading.Lock()
        self._frame = None
        self._ts, self._seq = 0.0, 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running = False
        self.disconnected = False
        self.error = None
        self.index = 0

    def start(self) -> None:
        self._stop.clear(); self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="synthetic-cam")
        self._thread.start()

    def _run(self) -> None:
        period, n = 1.0 / self.fps, 0
        nxt = time.perf_counter()
        while not self._stop.is_set():
            f = self._base.copy()
            x = int((n * 7) % (self.w - 260))
            f[200:460, x:x + 260] = (30, 200, 240)                       # a "moving object"
            with self._lock:
                self._frame, self._ts, self._seq = f, time.perf_counter(), n + 1
            n += 1
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def read(self):
        m = self.read_meta()
        return None if m is None else m[0]

    def read_meta(self, copy: bool = True):
        with self._lock:
            if self._frame is None:
                return None
            return (self._frame.copy() if copy else self._frame), self._ts, self._seq

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.running = False
