"""Measure display FPS, capture->screen latency (p50/p95), per-tier inference time and end-to-end label latency.

  python scripts/benchmark_latency.py                 # real camera + real models on THIS computer (what you want)
  python scripts/benchmark_latency.py --synthetic     # synthetic video + a SIMULATED detector (--sim-infer-ms): checks the
                                                      # pipeline's own overhead / non-blocking design, NOT your hardware speed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import Settings  # noqa: E402
from smartsight.live import run_live  # noqa: E402
from smartsight.pipeline import Pipeline  # noqa: E402
from smartsight.product_lookup import LocalProductStore  # noqa: E402
from smartsight.speech_engine import SpeechController  # noqa: E402


class TimingBackend:
    """Silent speech backend that records when the first words would have been spoken."""
    def __init__(self): self.first = None; self.texts = []
    STARTUP = ("SmartSight is ready", "Online AI", "Object detection is not available", "Self check")

    def speak(self, text):
        if not text.startswith(self.STARTUP):            # ignore greetings; we want the first object label
            self.first = self.first or time.perf_counter()
            self.texts.append(text)
    def stop(self): pass


class SimDetector:
    def __init__(self, ms): self.ms = ms
    def detect(self, frame, conf=0.5):
        from smartsight.detector import Detection
        time.sleep(self.ms / 1000)
        return [Detection("bottle", 0.9, (int(frame.shape[1] * .1), int(frame.shape[0] * .25), int(frame.shape[1] * .35), int(frame.shape[0] * .65)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sim-infer-ms", type=float, default=60.0)
    a = ap.parse_args()
    s = Settings.load(env={}) if not a.synthetic else Settings.load(path="/none", env={})
    s.cloud_ai = False                                   # never send anything during a benchmark
    backend = TimingBackend()
    if a.synthetic:
        from smartsight.sources import SyntheticSource
        class NoOcr:
            def read(self, f): return []
            def clear_cache(self): pass
        class NoBar:
            def read(self, f): return []
            def reset(self): pass
        pipe = Pipeline(s, SyntheticSource(1280, 720, 30), SimDetector(a.sim_infer_ms), NoOcr(), NoBar(),
                        LocalProductStore(), SpeechController(backend), sleep=time.sleep)
        print(f"SYNTHETIC mode: 1280x720@30 fps source, simulated detector {a.sim_infer_ms:.0f} ms")
    else:
        from smartsight.factory import build_pipeline
        pipe, status = build_pipeline(s)
        pipe.speech = SpeechController(backend)
        print("components:", status)
    t_start = time.perf_counter()
    if not pipe.start():
        print("camera failed:", pipe.camera_status); return 1
    frames = int(a.seconds * s.display_target_fps)
    stats = run_live(pipe, s, headless_frames=frames * 4)   # headless: measures loop + latency without drawing
    time.sleep(0.3)
    pipe.close()
    d = pipe.detect_ms.summary(); l = pipe.latency.summary()
    print(f"\ndisplay FPS            : {stats['display_fps']:.1f}   (target >= 24)")
    c = stats["capture_to_screen_ms"]
    print(f"capture->screen (ms)   : p50 {c['p50']:.1f}   p95 {c['p95']:.1f}   (target p95 < 150)   [headless: excludes drawing/imshow]")
    print(f"Tier 1 detect (ms)     : mean {d['mean']:.1f}  p50 {d['p50']:.1f}  p95 {d['p95']:.1f}   n={d['n']}")
    print(f"capture->detection (ms): p50 {l['p50']*1000:.1f}  p95 {l['p95']*1000:.1f}")
    for k in ("ocr_ms", "zeroshot_ms"):
        if k in pipe.metrics: print(f"{k:23}: last {pipe.metrics[k]:.1f}")
    if backend.first:
        print(f"first spoken label     : {(backend.first - t_start):.2f} s after start   (target < 1.5)   -> {backend.texts[0]!r}")
    else:
        print("first spoken label     : none spoken (no object detected/stable)")


if __name__ == "__main__":
    sys.exit(main() or 0)
