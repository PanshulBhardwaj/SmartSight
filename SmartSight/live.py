"""Live Mode (primary interface): smooth OpenCV window with real keyboard shortcuts. Run: python live.py"""
from __future__ import annotations

import logging
import sys

from config.settings import ROOT, Settings
from smartsight.cli_setup import first_run_setup, start_voice
from smartsight.factory import build_pipeline
from smartsight.live import HELP_LINES, run_live
from smartsight.utils import setup_logging


def main() -> int:
    setup_logging(logging.INFO, ROOT / "logs" / "smartsight.log")
    settings = Settings.load()
    first_run_setup(settings)
    pipe, status = build_pipeline(settings)
    print("SmartSight Live Mode"); [print(f"  {k}: {v}") for k, v in status.items()]
    print(HELP_LINES, "\nYou can also type phrases here, e.g. 'what is this', 'find my cup'.")
    if not pipe.start():
        print("Camera could not be started:", pipe.camera_status)
        pipe.close()
        return 1
    voice = start_voice(pipe)
    try:
        run_live(pipe, settings)
    except KeyboardInterrupt:
        pass
    finally:
        if voice:
            voice.stop()
        pipe.close()                 # always releases the camera
    return 0


if __name__ == "__main__":
    sys.exit(main())
