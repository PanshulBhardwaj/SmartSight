"""Real OCR on rendered images (Tesseract backend). Skipped if Tesseract/Pillow are unavailable.
These are SYNTHETIC clean renders, not photos of real packaging; they prove the plumbing, not field accuracy."""
import numpy as np
import pytest

pytest.importorskip("pytesseract")
PIL = pytest.importorskip("PIL.Image")
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from smartsight.food_label_parser import FieldStatus, parse_food_label  # noqa: E402
from smartsight.price_reader import extract_prices  # noqa: E402
from smartsight.utils import InfoState  # noqa: E402

try:
    from smartsight.ocr_engine import OcrEngine, OcrUnavailable
    ENGINE = OcrEngine("tesseract")
except Exception:  # noqa: BLE001
    pytest.skip("tesseract binary not available", allow_module_level=True)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def render(lines, size=34, blur=0.0, width=900):
    font = ImageFont.truetype(FONT, size)
    img = Image.new("RGB", (width, 60 + len(lines) * (size + 18)), "white")
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((30, 30 + i * (size + 18)), t, fill="black", font=font)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.array(img)[:, :, ::-1].copy()   # RGB -> BGR


def test_reads_price_from_label_beside_other_numbers():
    lines = ENGINE.read(render(["Net Wt 120 g", "Mfg: 12/05/2025", "MRP Rs. 99.00 (Incl. of all taxes)"]))
    r = extract_prices(lines)
    assert r.best is not None and str(r.best.amount) == "99.00"
    assert r.state == InfoState.CONFIRMED


def test_weight_and_date_alone_not_a_price():
    r = extract_prices(ENGINE.read(render(["Net Wt 250 g", "Best before 12/05/2026", "Batch 4567"])))
    assert r.best is None


def test_reads_ingredients_and_expiry():
    lbl = parse_food_label(ENGINE.read(render([
        "Ingredients: Potato, Palm Oil, Salt, Spices.", "Contains: Milk", "Best before 6 months from manufacture"])))
    assert lbl.get("ingredients").status == FieldStatus.PRESENT
    assert "Potato" in lbl.get("ingredients").value
    assert lbl.get("expiry").status == FieldStatus.PRESENT


def test_heavy_blur_does_not_produce_a_confident_price():
    r = extract_prices(ENGINE.read(render(["MRP Rs. 99.00"], size=24, blur=6)))
    assert r.state != InfoState.CONFIRMED or str(r.best.amount) == "99.00"   # never a *wrong* confirmed price
