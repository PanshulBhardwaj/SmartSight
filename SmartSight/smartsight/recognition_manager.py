"""Runs Tier 2 asynchronously, once per stable tracked object, caches by track id, and fuses the evidence."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from .fusion import Recognition, fuse
from .metrics import Percentiles
from .recognizer import center_box, crop_box
from .taxonomy import Taxonomy, Vocabulary

log = logging.getLogger(__name__)


@dataclass
class _State:
    zs: list = field(default_factory=list)
    pending: bool = False
    attempts: int = 0
    last_try: float = -1e9


class RecognitionManager:
    RETRY_SECONDS = 3.0
    MAX_ATTEMPTS = 2

    def __init__(self, recognizer, taxonomy: Taxonomy, vocab: Vocabulary, settings, async_tiers: bool = True,
                 clock: Callable[[], float] = time.monotonic):
        self.recognizer, self.tax, self.vocab, self.s = recognizer, taxonomy, vocab, settings
        self._async, self._clock = async_tiers, clock
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tier2") if (recognizer and async_tiers) else None
        self._lock = threading.Lock()
        self._state: dict = {}
        self._extra: dict = {}                 # track id -> (ocr tuple, barcode tuple) supplied by on-demand analysis
        self.zs_ms = Percentiles()
        self.calls = 0
        # centre-of-frame fallback (when nothing is detected)
        self._center = _State()
        self._center_result: tuple = (0.0, [])
        self._empty_frames = 0
        self._last_center_key = None

    # ------------------------------------------------------------------
    def update(self, frame, tracks) -> None:
        """Call once per detection cycle with the tracks seen in this frame."""
        if self.recognizer is None:
            return
        live = {t.id for t in tracks}
        with self._lock:
            for tid in [k for k in self._state if k not in live and k > 0]:
                self._state.pop(tid, None); self._extra.pop(tid, None)
        for t in tracks:
            if t.streak < self.s.stable_frames:
                continue
            label = t.vote()[0]
            if self.tax.map_detector(label).category == "Person":
                continue                                      # never analyse people
            st = self._state.setdefault(t.id, _State())
            if self._should_try(st):
                self._submit(st, frame, t.box, t.id)
        if tracks:
            self._empty_frames = 0
        else:
            self._empty_frames += 1
            if self._empty_frames >= self.s.stable_frames:
                self._maybe_center(frame)

    def _should_try(self, st: _State) -> bool:
        if st.pending or st.attempts >= self.MAX_ATTEMPTS:
            return False
        if st.attempts == 0:
            return True
        top = st.zs[0].score if st.zs else 0.0
        return top < self.s.zero_shot_threshold and self._clock() - st.last_try > self.RETRY_SECONDS

    def _submit(self, st: _State, frame, box, key) -> None:
        crop = crop_box(frame, box)
        if crop is None:
            return
        st.pending, st.attempts, st.last_try = True, st.attempts + 1, self._clock()
        if self._exec is None:
            self._run(st, crop)
        else:
            self._exec.submit(self._run, st, crop)

    def _run(self, st: _State, crop) -> None:
        t0 = time.perf_counter()
        try:
            res = self.recognizer.classify(crop, 3)
        except Exception:  # noqa: BLE001
            log.exception("zero-shot recognizer failed")
            res = []
        ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            st.zs, st.pending = res, False
            self.zs_ms.add(ms)
            self.calls += 1

    def _maybe_center(self, frame) -> None:
        from .cloud_vision import frame_key
        now = self._clock()
        if self._center.pending or now - self._center.last_try < 2.0:
            return
        key = frame_key(frame)
        if key == self._last_center_key:
            return
        self._last_center_key = key
        self._submit(self._center, frame, center_box(frame), 0)
        self._center_result_pending = True

    # ------------------------------------------------------------------
    def set_evidence(self, track_id: int, ocr=None, barcode=None) -> None:
        with self._lock:
            self._extra[track_id] = (ocr, barcode)

    def recognition(self, track) -> Recognition:
        label, _share = track.vote()
        mapped = self.tax.map_detector(label)
        conf = track.mean_conf(label)
        with self._lock:
            st = self._state.get(track.id)
            zs = list(st.zs) if st else []
            ocr, barcode = self._extra.get(track.id, (None, None))
        r = fuse(mapped, conf, zs, self.s.zero_shot_threshold, ocr, barcode)
        r.track_id, r.box = track.id, tuple(track.box)
        return r

    def center_recognition(self) -> Optional[Recognition]:
        with self._lock:
            zs = list(self._center.zs)
        if not zs:
            return None
        r = fuse(None, 0.0, zs, self.s.zero_shot_threshold)
        r.track_id = -1
        return r if r.source != "none" else None

    def classify_now(self, crop):
        """Synchronous Tier 2 for an explicit user request (runs in the command thread, never the detection thread)."""
        if self.recognizer is None or crop is None:
            return []
        t0 = time.perf_counter()
        try:
            res = self.recognizer.classify(crop, 3)
        except Exception:  # noqa: BLE001
            log.exception("zero-shot recognizer failed")
            return []
        self.zs_ms.add((time.perf_counter() - t0) * 1000)
        return res

    def reset(self) -> None:
        with self._lock:
            self._state.clear(); self._extra.clear()
            self._center = _State(); self._last_center_key = None; self._empty_frames = 0

    def shutdown(self) -> None:
        if self._exec:
            self._exec.shutdown(wait=False, cancel_futures=True)
