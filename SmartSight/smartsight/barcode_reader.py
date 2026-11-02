"""Barcode decoding (pyzbar, with OpenCV's detector as fallback) and repeat suppression."""
from __future__ import annotations

import logging
import time
from typing import Callable

log = logging.getLogger(__name__)


class BarcodeReader:
    def __init__(self, cooldown_seconds: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.cooldown, self._clock = cooldown_seconds, clock
        self._seen: dict[str, float] = {}
        self._pyzbar = None
        try:
            from pyzbar import pyzbar
            self._pyzbar = pyzbar
        except Exception as exc:  # noqa: BLE001  (ImportError or missing zbar DLL)
            log.warning("pyzbar unavailable (%s); trying OpenCV barcode detector", exc)

    def read(self, frame) -> list[str]:
        """All decodable barcode strings in the frame (no suppression)."""
        codes: list[str] = []
        if self._pyzbar is not None:
            try:
                codes = [b.data.decode("utf-8", "ignore") for b in self._pyzbar.decode(frame)]
            except Exception:  # noqa: BLE001
                log.debug("pyzbar decode failed", exc_info=True)
        if not codes:
            try:
                import cv2
                det = cv2.barcode.BarcodeDetector()
                ok, decoded, *_ = det.detectAndDecode(frame)
                if ok:
                    codes = [d for d in decoded if d]
            except Exception:  # noqa: BLE001  (cv2.barcode missing in some builds)
                pass
        return list(dict.fromkeys(c for c in codes if c))

    def new_only(self, codes: list[str]) -> list[str]:
        now = self._clock()
        fresh = [c for c in codes if now - self._seen.get(c, -1e9) > self.cooldown]
        for c in codes:
            self._seen[c] = now if c in fresh else self._seen[c]
        return fresh

    def reset(self) -> None:
        self._seen.clear()
