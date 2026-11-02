"""Thread-safe, priority-aware speech queue with a pluggable backend."""
from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional, Protocol

log = logging.getLogger(__name__)


class Priority(IntEnum):
    CRITICAL = 0   # errors, camera failure
    USER = 1       # answers to an explicit request
    AUTO = 2       # automatic announcements (droppable, interruptible)


class SpeechBackend(Protocol):
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...


class ConsoleBackend:
    """Fallback when no TTS engine is installed: prints instead of speaking."""
    def __init__(self, **_):
        pass

    def speak(self, text: str) -> None:
        print(f"[SPEAK] {text}", flush=True)

    def stop(self) -> None:
        pass


class Pyttsx3Backend:
    """Offline Windows (SAPI5) speech. A fresh engine per utterance is the most reliable way
    to keep stop() working with pyttsx3 across threads."""
    def __init__(self, rate: int = 170, volume: float = 1.0):
        import pyttsx3  # noqa: F401  (raises ImportError if missing)
        self.rate, self.volume = rate, volume
        self._engine = None

    def speak(self, text: str) -> None:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        self._engine = engine
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            self._engine = None

    def stop(self) -> None:
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                log.debug("pyttsx3 stop failed", exc_info=True)


def make_backend(rate: int = 170, volume: float = 1.0) -> SpeechBackend:
    try:
        return Pyttsx3Backend(rate, volume)
    except Exception as exc:  # noqa: BLE001
        log.warning("pyttsx3 unavailable (%s); speech will be printed to the console", exc)
        return ConsoleBackend()


@dataclass(order=True)
class _Item:
    priority: int
    seq: int
    text: Optional[str] = field(compare=False, default=None)


class SpeechController:
    def __init__(self, backend: SpeechBackend, dedupe_seconds: float = 5.0, max_pending_auto: int = 2,
                 clock: Callable[[], float] = time.monotonic):
        self.backend = backend
        self.dedupe_seconds = dedupe_seconds
        self.max_pending_auto = max_pending_auto
        self._clock = clock
        self._q: "queue.PriorityQueue[_Item]" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending = 0
        self._pending_auto = 0
        self._current: Optional[int] = None
        self._recent: dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self.last_text: Optional[str] = None
        self.history: deque = deque(maxlen=20)

    # -- public API ---------------------------------------------------
    def say(self, text: str, priority: Priority = Priority.AUTO) -> bool:
        """Queue text. Returns False if dropped (duplicate or too many pending automatic items)."""
        text = (text or "").strip()
        if not text:
            return False
        interrupt = False
        with self._lock:
            if self._closed:
                return False
            now = self._clock()
            if priority == Priority.AUTO:
                last = self._recent.get(text)
                if self._pending_auto >= self.max_pending_auto or (last is not None and now - last < self.dedupe_seconds):
                    return False
                self._pending_auto += 1
            self._recent[text] = now
            interrupt = priority < Priority.AUTO and self._current == Priority.AUTO
            self._pending += 1
            self._q.put(_Item(int(priority), next(self._seq), text))
            self._ensure_thread()
        if interrupt:
            self.backend.stop()
        return True

    def stop(self) -> None:
        """Discard everything queued and cut off current speech."""
        with self._cond:
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if item.text is None:
                    continue
                self._pending -= 1
                if item.priority == Priority.AUTO:
                    self._pending_auto -= 1
            self._cond.notify_all()
        self.backend.stop()

    def repeat(self) -> bool:
        if not self.last_text:
            return False
        return self.say(self.last_text, Priority.USER)

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._pending > 0

    def wait_idle(self, timeout: float = 5.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._pending == 0, timeout)

    def shutdown(self) -> None:
        self.stop()
        with self._lock:
            self._closed = True
            t = self._thread
            self._q.put(_Item(-1, next(self._seq), None))
        if t:
            t.join(timeout=3)

    # -- worker -------------------------------------------------------
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="speech", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item.text is None:
                return
            with self._lock:
                self._current = item.priority
                if item.priority == Priority.AUTO:
                    self._pending_auto -= 1
                self.last_text = item.text
                self.history.append(item.text)
            try:
                self.backend.speak(item.text)
            except Exception:  # noqa: BLE001
                log.exception("speech backend failed")
            finally:
                with self._cond:
                    self._current = None
                    self._pending -= 1
                    self._cond.notify_all()
