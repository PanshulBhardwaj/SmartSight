"""YOLO object detection (Ultralytics). A generic COCO model knows ~80 classes, NOT product brands."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Tuple

log = logging.getLogger(__name__)


class DetectorUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])


class YoloDetector:
    def __init__(self, model_path: str = "yolo11n.pt", use_gpu: bool = False, imgsz: int = 416):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailable("ultralytics is not installed (pip install ultralytics)") from exc
        try:
            self.model = YOLO(model_path)   # downloads known weights on first use (needs internet once)
        except Exception as exc:  # noqa: BLE001
            raise DetectorUnavailable(f"Could not load model {model_path!r}: {exc}") from exc
        self.imgsz = imgsz
        self.device = "cpu"
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = 0
            except ImportError:
                pass
        self.names = self.model.names

    def detect(self, frame, conf: float = 0.5) -> list[Detection]:
        res = self.model.predict(frame, conf=conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append(Detection(self.names[int(b.cls[0])], float(b.conf[0]), (x1, y1, x2, y2)))
        return sorted(out, key=lambda d: -d.area)


def draw_detections(frame, detections: Sequence[Detection]):
    """Preview only. Labels carry the model confidence (uncalibrated, not a probability of being correct)."""
    import cv2
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out
