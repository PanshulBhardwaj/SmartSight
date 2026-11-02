"""Runs app.py headlessly with Streamlit's AppTest. No camera/YOLO here, so this checks the UI
builds, labels exist, and button presses don't crash when components are unavailable."""
import pytest

pytest.importorskip("streamlit")
from pathlib import Path  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")

REQUIRED = ["Start Camera", "Stop Camera", "What is this? (Identify Item)", "More details", "Read text", "Read Price", "Read Ingredients",
            "Read Expiry Date", "Describe Product", "Scene scan: what is around me", "Category", "Repeat Last Announcement", "Stop Speaking",
            "Faster speech", "Slower speech", "Change verbosity", "Help: list commands"]


def test_app_renders_with_required_controls_and_survives_clicks():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.button]
    for r in REQUIRED:
        assert r in labels, r
    for r in ("Start Camera", "What is this? (Identify Item)", "More details", "Help: list commands", "Read Price", "Stop Speaking", "Stop Camera"):
        next(b for b in at.button if b.label == r).click().run()
        assert not at.exception, (r, at.exception)
    assert any("Camera status" in m.value for m in at.markdown)
