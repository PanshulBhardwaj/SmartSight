"""Builds a real Pipeline from Settings, degrading gracefully when optional components are missing."""
from __future__ import annotations

import logging
from pathlib import Path

from config.settings import ROOT, Settings
from .barcode_reader import BarcodeReader
from .camera import Camera
from .pipeline import Pipeline
from .product_lookup import LocalProductStore, OpenFoodFactsLookup
from .speech_engine import SpeechController, make_backend

log = logging.getLogger(__name__)


def build_pipeline(s: Settings) -> tuple[Pipeline, dict]:
    status: dict[str, str] = {}
    detector = ocr = None
    try:
        from .detector import YoloDetector
        detector = YoloDetector(s.model_path, s.gpu, s.detect_imgsz)
        status["detector"] = f"ok ({s.model_path})"
    except Exception as exc:  # noqa: BLE001
        status["detector"] = f"unavailable: {exc}"
    try:
        from .ocr_engine import OcrEngine
        ocr = OcrEngine(s.ocr_engine, s.gpu)
        status["ocr"] = f"ok ({ocr.name})"
    except Exception as exc:  # noqa: BLE001
        status["ocr"] = f"unavailable: {exc}"
    from .recognizer import make_recognizer
    from .taxonomy import Taxonomy, Vocabulary
    try:
        vocab = Vocabulary.from_file(s.path("vocabulary_path"))
    except Exception as exc:  # noqa: BLE001
        vocab = Vocabulary.empty(); status["vocabulary"] = f"unavailable: {exc}"
    try:
        tax = Taxonomy.from_file(s.path("taxonomy_path"))
    except Exception as exc:  # noqa: BLE001
        tax = Taxonomy.minimal(); status["taxonomy"] = f"unavailable: {exc}"
    recognizer, msg = make_recognizer(s, vocab)
    status["zero_shot"] = msg
    status["vocabulary_items"] = str(len(vocab.entries))
    from .earcons import Earcons, make_backend as make_earcon_backend
    earcons = Earcons(make_earcon_backend(), s.earcons)
    status["earcons"] = "on" if earcons.available and s.earcons else "off (no Windows sound API or disabled)"
    barcode = BarcodeReader()
    backend = make_backend(s.speech_rate, s.speech_volume)
    status["speech"] = type(backend).__name__
    speech = SpeechController(backend, dedupe_seconds=s.repeat_suppression_seconds)
    store = LocalProductStore(ROOT / "data" / "products.json")
    lookup = OpenFoodFactsLookup(ROOT / "data" / "cache" / "lookups.json") if s.online_lookup else None
    import os
    from .cloud_vision import CloudVision
    cloud = CloudVision(os.environ.get("ANTHROPIC_API_KEY"), s.cloud_model, s.cloud_web_search)
    status["online_ai"] = ("key found" if cloud.configured else "NO API KEY") + \
        ("" if s.cloud_consent else "; waiting for consent") + ("" if s.cloud_ai else "; paused")
    for k, v in status.items():
        log.info("%s: %s", k, v)
    return Pipeline(s, Camera(s.camera_index), detector, ocr, barcode, store, speech, lookup, cloud=cloud, recognizer=recognizer, taxonomy=tax, vocab=vocab,
                    earcons=earcons, persist=True, component_status=status), status
