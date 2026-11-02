"""Live Mode: native OpenCV window. The display loop only reads the newest frame and draws the latest results;
it never waits for inference, OCR or the network. Also provides real keyboard shortcuts.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

from .metrics import FpsMeter, Percentiles
from .voice_commands import Command, parse_command

log = logging.getLogger(__name__)

KEYS = {"i": Command.IDENTIFY, "m": Command.MORE, "t": Command.READ_TEXT, "r": Command.REPEAT,
        "d": Command.DESCRIBE, "s": Command.SCENE, "c": Command.CATEGORY, "f": Command.FIND,
        "+": Command.FASTER, "=": Command.FASTER, "-": Command.SLOWER, "v": Command.VERBOSITY,
        "h": Command.HELP, "p": Command.PRICE, "g": Command.INGREDIENTS, "e": Command.EXPIRY,
        "a": Command.SCAN_AGAIN, "o": Command.ASK_AI, " ": Command.STOP, "x": Command.STOP}
HELP_LINES = ("Keys: i identify, m more, t read text, r repeat, d describe, s scene scan, f find, c category, "
              "plus or minus speech speed, v verbosity, p price, g ingredients, e expiry, a scan again, "
              "o ask online AI, space stop speaking, h help, q quit.")


def draw_hud(frame, pipe, fps: float, lat_ms: float, high_contrast: bool = True):
    import cv2
    out = frame.copy()
    for tid, label, conf, box in pipe.overlay_tracks():
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        txt = f"#{tid} {label} {conf:.2f}"
        cv2.putText(out, txt, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, txt, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    bar = f"FPS {fps:4.1f} | latency {lat_ms:4.0f} ms | Online AI: {pipe.cloud_status().split(':')[0]} | h = help, q = quit"
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0) if high_contrast else (60, 60, 60), -1)
    cv2.putText(out, bar, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    return out


def _typed_commands(pipe, stop: threading.Event) -> None:
    """Console thread: type phrases such as 'what is this' or 'find my cup'."""
    while not stop.is_set():
        try:
            line = input()
        except EOFError:
            return
        if line.strip().lower() in ("q", "quit", "exit"):
            stop.set(); return
        cmd, arg = parse_command(line)
        if cmd:
            pipe.request(cmd, arg)


def run_live(pipe, settings, window: str = "SmartSight Live (h = help, q = quit)", headless_frames: Optional[int] = None) -> dict:
    """Blocking. Returns display statistics. headless_frames: run N loop iterations without a window (benchmark/test)."""
    cv2 = None
    if headless_frames is None:
        import cv2  # noqa: F811
    stop = threading.Event()
    if headless_frames is None:
        threading.Thread(target=_typed_commands, args=(pipe, stop), daemon=True, name="typed").start()
    fps, e2e = FpsMeter(), Percentiles()
    period = 1.0 / settings.display_target_fps
    last_seq, shown = -1, 0
    try:
        while not stop.is_set() and not pipe.quit_event.is_set():
            t0 = time.perf_counter()
            m = pipe.camera.read_meta(copy=False)
            if m is not None and m[2] != last_seq:
                frame, ts, last_seq = m
                f = fps.tick()
                if cv2 is not None:
                    img = draw_hud(frame, pipe, f, (time.perf_counter() - ts) * 1000, settings.high_contrast)
                    cv2.imshow(window, img)
                e2e.add((time.perf_counter() - ts) * 1000)          # capture -> screen, ms
                shown += 1
                if headless_frames is not None and shown >= headless_frames:
                    break
            if cv2 is not None:
                k = cv2.waitKey(1) & 0xFF
                if k != 255:
                    ch = chr(k).lower() if k < 128 else ""
                    if ch == "q" or k == 27:
                        break
                    if ch in KEYS:
                        pipe.request(KEYS[ch])
            else:
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    finally:
        stop.set()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return {"display_fps": fps.fps, "capture_to_screen_ms": e2e.summary(), "frames": shown}