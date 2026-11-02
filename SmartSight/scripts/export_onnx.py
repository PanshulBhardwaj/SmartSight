"""Export the YOLO model for faster CPU inference. Then set  model_path: yolo11n.onnx  in data/user_config.yaml.

  python scripts/export_onnx.py                       # ONNX (needs requirements-onnx.txt for onnxruntime)
  python scripts/export_onnx.py --format openvino     # Intel OpenVINO
  python scripts/export_onnx.py --half                # FP16 (GPU only)
UNTESTED here (needs ultralytics weights); measure with scripts/benchmark_latency.py before and after.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import Settings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--format", default="onnx", choices=["onnx", "openvino"])
ap.add_argument("--half", action="store_true")
a = ap.parse_args()
s = Settings.load()
from ultralytics import YOLO  # noqa: E402
out = YOLO(s.model_path).export(format=a.format, imgsz=s.detect_imgsz, half=a.half)
print("Exported to:", out, "\nNow set model_path to that path.")
