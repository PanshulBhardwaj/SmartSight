"""Short, distinct audio cues: ready, locked, lost, error, found, plus a tunable beep for Find mode."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Sequence, Tuple

log = logging.getLogger(__name__)

PATTERNS: dict[str, Sequence[Tuple[int, int]]] = {       # (frequency Hz, duration ms)
    "ready": ((600, 120), (900, 160)),
    "locked": ((1000, 70),),
    "lost": ((500, 110), (350, 150)),
    "error": ((300, 250), (250, 350)),
    "found": ((1200, 110), (1500, 160)),
}


class NullBackend:
    def beep(self, freq: int, ms: int) -> None:
        pass


class RecordingBackend(NullBackend):
    """Used by tests and as a visible fallback."""
    def __init__(self):
        self.played: list = []

    def beep(self, freq: int, ms: int) -> None:
        self.played.append((freq, ms))


class WinsoundBackend:
    def __init__(self):
        import winsound  # Windows only
        self._ws = winsound

    def beep(self, freq: int, ms: int) -> None:
        self._ws.Beep(max(37, min(32767, int(freq))), max(10, int(ms)))


def make_backend():
    try:
        return WinsoundBackend()
    except Exception as exc:  # noqa: BLE001  (not Windows)
        log.info("earcons: winsound unavailable (%s); sound cues disabled", exc)
        return NullBackend()


class Earcons:
    def __init__(self, backend=None, enabled: bool = True):
        self.backend, self.enabled = backend or NullBackend(), enabled
        self._lock = threading.Lock()                      # cues never overlap each other
        self.available = not isinstance(self.backend, NullBackend)

    def play(self, name: str) -> None:
        if not self.enabled or name not in PATTERNS:
            return
        threading.Thread(target=self._run, args=(PATTERNS[name],), daemon=True, name="earcon").start()

    def beep(self, freq: int, ms: int = 60) -> None:
        if self.enabled:
            threading.Thread(target=self._run, args=(((freq, ms),),), daemon=True, name="earcon").start()

    def _run(self, seq) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            for f, ms in seq:
                self.backend.beep(f, ms)
        except Exception:  # noqa: BLE001
            log.debug("earcon failed", exc_info=True)
        finally:
            self._lock.release()
