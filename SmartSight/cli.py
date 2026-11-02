"""Console interface (screen-reader friendly), secondary to Live Mode (python live.py). Run: python cli.py

Type a shortcut letter or a phrase such as "what is this" and press Enter.
  i identify   d describe   p price   g ingredients   f full ingredients
  e expiry     l label      r repeat  s stop speaking   a scan again   o ask online AI   q quit

`python cli.py --preview` opens a smooth camera window (much less lag than the browser) and the same keys
work while the window is focused.
"""
from __future__ import annotations

import logging
import sys

from config.settings import ROOT, Settings
from smartsight.cli_setup import first_run_setup, start_voice
from smartsight.factory import build_pipeline
from smartsight.utils import setup_logging

from smartsight.live import KEYS  # noqa: E402  (same shortcut letters as Live Mode)
from smartsight.voice_commands import parse_command  # noqa: E402


def run_window(pipe) -> None:
    import cv2
    while True:
        frame = pipe.preview_frame()
        if frame is not None:
            cv2.imshow("SmartSight (q to quit)", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        ch = chr(k).lower()
        if ch == "q":
            break
        if ch in KEYS:
            pipe.request(KEYS[ch])
    cv2.destroyAllWindows()


def main() -> None:
    setup_logging(logging.INFO, ROOT / "logs" / "smartsight.log")
    settings = Settings.load()
    first_run_setup(settings)
    pipe, status = build_pipeline(settings)
    print(__doc__)
    for k, v in status.items():
        print(f"  {k}: {v}")
    pipe.start()
    voice = start_voice(pipe)
    try:
        if "--preview" in sys.argv:
            return run_window(pipe)
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if text.lower() in ("q", "quit", "exit"):
                break
            cmd, arg = (KEYS.get(text.lower()), None) if text.lower() in KEYS else parse_command(text)
            if cmd:
                pipe.request(cmd, arg)
            elif text:
                print("Not understood. Try: i, p, g, e, l, r, s, a, q")
    except KeyboardInterrupt:
        pass
    finally:
        if voice:
            voice.stop()
        pipe.close()   # always releases the camera


if __name__ == "__main__":
    main()
