"""Plain-language errors with the fix, for speech and the screen."""
from __future__ import annotations

import urllib.error


def explain(exc: BaseException, what: str = "") -> str:
    """Return one or two short sentences: what went wrong and how to fix it."""
    name = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    if name == "CameraError" or "camera" in low:
        return ("I could not use the camera. Close other programs that use it, check that it is plugged in, "
                "or choose another camera in settings.")
    if name == "DetectorUnavailable" or "ultralytics" in low:
        return "Object detection is not available. Run pip install ultralytics, or check your internet for the first model download."
    if name == "OcrUnavailable" or "ocr" in low:
        return "Text reading is not available. Run pip install easyocr to enable it."
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return "I could not reach the internet. Check your connection. SmartSight still works offline for common objects and text."
    if isinstance(exc, PermissionError):
        return "SmartSight was denied access to a file or device. Check Windows privacy settings for the camera and microphone."
    if isinstance(exc, FileNotFoundError):
        return "A required file is missing. Check the README for the model download step."
    if isinstance(exc, MemoryError):
        return "The computer ran out of memory. Close other programs, or choose a smaller model in the settings."
    return f"Something went wrong{(' while ' + what) if what else ''}. Please try again. Details are in the log file."
