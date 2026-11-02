"""Swappable component contracts. Implementations are chosen in config / factory; nothing else needs to change."""
from __future__ import annotations

from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

from .utils import OcrLine


@runtime_checkable
class CameraSource(Protocol):
    """Anything that yields frames. read_meta() -> (frame, capture_time_perf_counter, sequence_number) or None."""
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self): ...
    def read_meta(self, copy: bool = True) -> Optional[Tuple[object, float, int]]: ...


@runtime_checkable
class Detector(Protocol):
    """Tier 1. detect(frame, conf) -> list[Detection] (label, confidence, box in frame pixels)."""
    def detect(self, frame, conf: float = 0.5) -> list: ...


@runtime_checkable
class Recognizer(Protocol):
    """Tier 2. classify(crop) -> list[Candidate] best first (label, score in 0..1, category, subcategory)."""
    name: str
    def classify(self, crop, top_k: int = 3) -> list: ...


@runtime_checkable
class InfoProvider(Protocol):
    """Supplies facts about a recognised object (knowledge base, barcode DB, AI description)."""
    name: str
    def lookup(self, label: str, category: str) -> Optional[dict]: ...


@runtime_checkable
class OCREngine(Protocol):
    name: str
    def read(self, frame) -> Sequence[OcrLine]: ...
    def clear_cache(self) -> None: ...


@runtime_checkable
class Speaker(Protocol):
    """Low-level speech backend used by SpeechController (queueing/priorities live there)."""
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...


@runtime_checkable
class Describer(Protocol):
    """Tier 3. describe(jpeg, hints) -> CloudResult-like (status, data). Cloud today; a local VLM can implement it."""
    def describe(self, jpeg: bytes, hints=None): ...
