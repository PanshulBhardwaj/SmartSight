"""Threaded webcam capture that always exposes the latest frame and releases the device on stop."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


def _backend():
    import cv2
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def list_cameras(max_index: int = 5) -> list[int]:
    """Probe camera indices. Slow-ish; call once and cache."""
    try:
        import cv2
    except ImportError:
        return []
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _backend())
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append(i)
        finally:
            cap.release()
    return found


class Camera:
    MAX_FAILURES = 30

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index, self.width, self.height = index, width, height
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self.running = False
        self.disconnected = False
        self.error: Optional[str] = None

    def start(self) -> None:
        if self.running:
            return
        try:
            import cv2
        except ImportError as exc:
            raise CameraError("OpenCV is not installed (pip install opencv-python)") from exc
        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Camera {self.index} could not be opened (missing, in use, or permission denied)")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # less USB bandwidth, lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                              # do not queue stale frames
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap, self._frame, self.error, self.disconnected = cap, None, None, False
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        fails = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size:
                fails = 0
                with self._lock:
                    self._frame = frame
                    self._ts = time.perf_counter()
                    self._seq += 1
            else:
                fails += 1
                if fails >= self.MAX_FAILURES:
                    self.disconnected, self.error = True, "Camera stopped delivering frames"
                    log.error(self.error)
                    break
                time.sleep(0.05)
        self.running = False

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def read_meta(self, copy: bool = True):
        """(frame, capture_time, sequence) or None. copy=False shares the buffer (never mutated, only replaced)."""
        with self._lock:
            if self._frame is None:
                return None
            return (self._frame.copy() if copy else self._frame), self._ts, self._seq

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.running = False
        with self._lock:
            self._frame = None
