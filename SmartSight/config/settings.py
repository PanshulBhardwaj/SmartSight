"""Typed settings. Load order: defaults -> config.yaml -> data/user_config.yaml -> legacy data/settings.json -> SMARTSIGHT_* env."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
USER_CONFIG_PATH = ROOT / "data" / "user_config.yaml"
LEGACY_JSON_PATH = ROOT / "data" / "settings.json"
DEFAULT_SETTINGS_PATH = USER_CONFIG_PATH            # where save() writes by default
VERBOSITY_LEVELS = ("concise", "standard", "detailed")
OCR_ENGINES = ("auto", "paddle", "easyocr", "tesseract")

_RANGES = {
    "confidence_threshold": (0.05, 0.95), "detection_fps": (0.2, 60.0), "stable_frames": (1, 30),
    "speech_rate": (80, 300), "speech_volume": (0.0, 1.0), "ocr_min_confidence": (0.1, 0.99),
    "price_min_confidence": (0.1, 0.99), "repeat_suppression_seconds": (1.0, 600.0),
    "object_absence_seconds": (0.5, 60.0), "capture_samples": (1, 8), "camera_index": (0, 20),
    "detect_imgsz": (160, 1280), "preview_width": (160, 1920), "cloud_min_interval": (1.0, 600.0),
    "cloud_max_per_minute": (1, 60), "infer_max_side": (160, 1920), "display_target_fps": (5, 120),
    "tracker_iou": (0.05, 0.95), "tracker_max_misses": (1, 60), "vote_window": (1, 60),
    "zero_shot_threshold": (0.01, 0.99), "cloud_low_confidence": (0.05, 0.99), "detail_level": (1, 3),
    "confidence_speak_threshold": (0.05, 0.99), "announce_cooldown": (0.0, 600.0),
}


@dataclass
class Settings:
    # camera & speed
    camera_index: int = 0
    detection_fps: float = 8.0
    infer_max_side: int = 640
    detect_imgsz: int = 416
    gpu: bool = False
    detector_half: bool = True
    preview_width: int = 640
    display_target_fps: int = 30
    # tier 1
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.45
    stable_frames: int = 4
    tracker_iou: float = 0.3
    tracker_max_misses: int = 6
    vote_window: int = 8
    ignore_classes: list = field(default_factory=list)
    # tier 2
    zero_shot_enabled: bool = True
    zero_shot_model: str = "ViT-B-32"
    zero_shot_pretrained: str = "laion2b_s34b_b79k"
    zero_shot_threshold: float = 0.30
    vocabulary_path: str = "data/vocabulary.json"
    taxonomy_path: str = "data/taxonomy.json"
    knowledge_path: str = "data/knowledge.json"
    # tier 3
    cloud_ai: bool = True               # configured on; nothing is sent until cloud_consent is True
    cloud_consent: bool = False
    cloud_auto: bool = True
    cloud_low_confidence: float = 0.5
    cloud_web_search: bool = True
    cloud_model: str = "claude-sonnet-5-5"
    cloud_min_interval: float = 10.0
    cloud_max_per_minute: int = 5
    # speech & announcements
    verbosity: str = "standard"
    detail_level: int = 1
    speech_rate: int = 170
    speech_volume: float = 1.0
    speak_percent: bool = True
    confidence_speak_threshold: float = 0.60
    announce_cooldown: float = 6.0
    auto_announce: bool = True
    earcons: bool = True
    position_cues: bool = True
    repeat_suppression_seconds: float = 30.0
    object_absence_seconds: float = 3.0
    # text reading
    ocr_engine: str = "auto"
    ocr_min_confidence: float = 0.3
    price_min_confidence: float = 0.7
    min_sharpness: float = 30.0
    min_brightness: float = 40.0
    capture_samples: int = 3
    # optional integrations
    online_lookup: bool = False
    voice_input: bool = False
    vosk_model_path: str = "models/vosk-model-small-en-us-0.15"
    show_preview: bool = True
    high_contrast: bool = True

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None, env: Optional[Mapping[str, str]] = None) -> "Settings":
        warnings: list[str] = []
        files = [Path(path)] if path else [CONFIG_PATH, USER_CONFIG_PATH, LEGACY_JSON_PATH]
        data: dict[str, Any] = {}
        for f in files:
            data.update(_read_file(f, warnings))
        env = os.environ if env is None else env
        for f in fields(cls):
            key = "SMARTSIGHT_" + f.name.upper()
            if key in env:
                data[f.name] = env[key]
        s = cls()
        for f in fields(cls):
            if f.name not in data:
                continue
            try:
                setattr(s, f.name, _coerce(data[f.name], getattr(s, f.name)))
            except (TypeError, ValueError):
                warnings.append(f"Invalid value for {f.name!r}; using default")
        warnings += s.sanitize()
        for w in warnings:
            log.warning(w)
        s.load_warnings = warnings  # type: ignore[attr-defined]
        return s

    def sanitize(self) -> list[str]:
        out = []
        for name, (lo, hi) in _RANGES.items():
            v = getattr(self, name)
            if not lo <= v <= hi:
                new = min(max(v, lo), hi)
                out.append(f"{name}={v} out of range [{lo}, {hi}]; using {new}")
                setattr(self, name, type(v)(new))
        if self.verbosity not in VERBOSITY_LEVELS:
            out.append(f"verbosity {self.verbosity!r} invalid; using 'standard'")
            self.verbosity = "standard"
        if self.ocr_engine not in OCR_ENGINES:
            out.append(f"ocr_engine {self.ocr_engine!r} invalid; using 'auto'")
            self.ocr_engine = "auto"
        return out

    def path(self, name: str) -> Path:
        """Resolve a path-valued setting relative to the project root."""
        p = Path(getattr(self, name))
        return p if p.is_absolute() else ROOT / p

    def save(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path else USER_CONFIG_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            import yaml
            p.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def _read_file(f: Path, warnings: list) -> dict:
    if not f.exists():
        return {}
    try:
        text = f.read_text(encoding="utf-8")
        if f.suffix.lower() == ".json":
            loaded = json.loads(text)
        else:
            import yaml
            loaded = yaml.safe_load(text)
    except ImportError:
        warnings.append(f"PyYAML not installed; could not read {f.name} (pip install pyyaml)")
        return {}
    except Exception as exc:  # noqa: BLE001  (OSError, JSON/YAML errors)
        warnings.append(f"Could not read {f.name} ({exc}); using defaults")
        return {}
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        warnings.append(f"{f.name} is not a mapping; ignored")
        return {}
    return loaded


def _coerce(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return list(value)
    return str(value)
