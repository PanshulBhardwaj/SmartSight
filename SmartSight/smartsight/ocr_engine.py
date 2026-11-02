"""OCR wrapper: PaddleOCR (preferred) -> EasyOCR -> Tesseract, with preprocessing and a small cache.

Engine APIs differ between versions; every backend is wrapped defensively and failures
degrade to the next engine rather than crashing.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from .utils import OcrLine

log = logging.getLogger(__name__)


class OcrUnavailable(RuntimeError):
    pass


def preprocess(frame):
    """Upscale small images, boost local contrast. Returns a BGR image (engines expect colour)."""
    import cv2
    h, w = frame.shape[:2]
    img = frame
    if max(h, w) < 1000:
        s = 1000 / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), (img.shape[1] / w)


def _box(poly, scale):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return (int(min(xs) / scale), int(min(ys) / scale), int(max(xs) / scale), int(max(ys) / scale))


class _Paddle:
    name = "paddleocr"

    def __init__(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OcrUnavailable("paddleocr not installed") from exc
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)   # 2.x signature
        except (TypeError, ValueError):
            self.ocr = PaddleOCR(lang="en")                                       # 3.x signature
        self._v3 = hasattr(self.ocr, "predict")

    def read(self, img, scale):
        out = []
        if self._v3:
            for page in self.ocr.predict(img):
                for t, sc, poly in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
                    out.append(OcrLine(t, float(sc), _box(poly, scale)))
        else:
            for page in self.ocr.ocr(img, cls=True) or []:
                for poly, (t, sc) in page or []:
                    out.append(OcrLine(t, float(sc), _box(poly, scale)))
        return out


class _Easy:
    name = "easyocr"

    def __init__(self, gpu=False):
        try:
            import easyocr
        except ImportError as exc:
            raise OcrUnavailable("easyocr not installed") from exc
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, img, scale):
        return [OcrLine(t, float(c), _box(poly, scale)) for poly, t, c in self.reader.readtext(img)]


class _Tess:
    name = "tesseract"

    def __init__(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except Exception as exc:  # noqa: BLE001
            raise OcrUnavailable(f"tesseract not available: {exc}") from exc
        self.t = pytesseract

    def read(self, img, scale):
        """Tesseract reports words; group them into text lines so downstream parsers see whole lines."""
        d = self.t.image_to_data(img, output_type=self.t.Output.DICT)
        groups: dict = {}
        for i, txt in enumerate(d["text"]):
            if not txt.strip() or float(d["conf"][i]) < 0:
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            groups.setdefault(key, []).append((x, y, x + w, y + h, txt.strip(), float(d["conf"][i]) / 100))
        out = []
        for words in groups.values():
            words.sort(key=lambda w: w[0])
            box = (int(min(w[0] for w in words) / scale), int(min(w[1] for w in words) / scale),
                   int(max(w[2] for w in words) / scale), int(max(w[3] for w in words) / scale))
            out.append(OcrLine(" ".join(w[4] for w in words), sum(w[5] for w in words) / len(words), box))
        return out


class OcrEngine:
    def __init__(self, preferred: str = "auto", gpu: bool = False, cache_seconds: float = 20.0):
        order = {"auto": ["paddle", "easyocr", "tesseract"]}.get(preferred, [preferred])
        makers = {"paddle": _Paddle, "easyocr": lambda: _Easy(gpu), "tesseract": _Tess}
        self.backend, errors = None, []
        for name in order:
            try:
                self.backend = makers[name]()
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
                log.warning("OCR engine %s unavailable: %s", name, exc)
        if self.backend is None:
            raise OcrUnavailable("No OCR engine could be loaded. " + " | ".join(errors))
        self.name = self.backend.name
        self._cache: dict[str, tuple[float, list]] = {}
        self._ttl = cache_seconds

    def read(self, frame) -> list[OcrLine]:
        key = self._key(frame)
        hit = self._cache.get(key)
        now = time.monotonic()
        if hit and now - hit[0] < self._ttl:
            return hit[1]
        img, scale = preprocess(frame)
        try:
            lines = [l for l in self.backend.read(img, scale) if l.text.strip()]
        except Exception as exc:  # noqa: BLE001
            log.exception("OCR failed")
            raise OcrUnavailable(f"OCR failed: {exc}") from exc
        self._cache = {k: v for k, v in self._cache.items() if now - v[0] < self._ttl}
        self._cache[key] = (now, lines)
        return lines

    def clear_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def _key(frame) -> str:
        import cv2
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)) // 16
        return hashlib.md5(small.tobytes()).hexdigest()
