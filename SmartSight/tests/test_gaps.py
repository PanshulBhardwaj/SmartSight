"""M5: veg-dot classifier and split-column nutrition tables (synthetic images / synthetic OCR boxes)."""
import cv2
import numpy as np
import pytest
from smartsight.food_label_parser import FieldStatus, parse_food_label
from smartsight.nutrition_table import group_rows, merge_rows, nutrition_phrase, parse_table
from smartsight.utils import OcrLine
from smartsight.vegmark import classify_veg_mark

GREEN, RED, BROWN, BLUE = (0, 150, 0), (0, 0, 200), (30, 70, 140), (200, 60, 0)


def canvas(w=320, h=240, noise=False, seed=0):
    if noise:
        g = np.random.default_rng(seed).integers(90, 230, (h, w), dtype=np.uint8)
        return np.dstack([g, g, g])                                     # sharp grey texture, no colour
    return np.full((h, w, 3), 255, np.uint8)


def mark(img, colour, size=40, shape="circle", xy=(30, 30), thick=None):
    x, y = xy
    t = thick or max(2, size // 12)
    cv2.rectangle(img, (x, y), (x + size, y + size), (255, 255, 255), -1)   # clean white tile like on real packs
    cv2.rectangle(img, (x, y), (x + size, y + size), colour, t)
    cx, cy, r = x + size // 2, y + size // 2, int(size * 0.3)
    if shape == "circle":
        cv2.circle(img, (cx, cy), r, colour, -1)
    else:
        cv2.fillPoly(img, [np.array([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)])], colour)
    return img


# ---------------- veg-dot ----------------
@pytest.mark.parametrize("colour,shape,expected", [(GREEN, "circle", "vegetarian"), (RED, "circle", "non-vegetarian"),
                                                    (BROWN, "circle", "non-vegetarian"), (BROWN, "triangle", "non-vegetarian"),
                                                    (GREEN, "triangle", "vegetarian")])
def test_marks_by_colour_and_inner_shape(colour, shape, expected):
    r = classify_veg_mark(mark(canvas(), colour, shape=shape))
    assert r.label == expected and r.confidence >= 0.7 and r.box is not None


@pytest.mark.parametrize("size", [20, 24, 40, 80, 150])
def test_marks_at_different_scales(size):
    assert classify_veg_mark(mark(canvas(400, 300), GREEN, size)).label == "vegetarian"
    assert classify_veg_mark(mark(canvas(400, 300), RED, size)).label == "non-vegetarian"


def test_mark_survives_jpeg_blur_and_noisy_background():
    img = mark(canvas(noise=True), GREEN, 48)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    img = cv2.GaussianBlur(cv2.imdecode(buf, 1), (3, 3), 0.8)
    r = classify_veg_mark(img)
    assert r.label == "vegetarian" and r.confidence >= 0.5


def test_mark_found_among_other_colourful_things():
    img = mark(canvas(noise=True), RED, 44, xy=(200, 120))
    cv2.rectangle(img, (20, 20), (120, 60), BLUE, -1)                        # blue logo
    cv2.putText(img, "MAGGI", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    assert classify_veg_mark(img).label == "non-vegetarian"


@pytest.mark.parametrize("name", ["blue_mark", "solid_red_rect", "solid_green_square", "ring_only", "filled_circle_only", "empty", "colour_noise"])
def test_rejects_look_alikes(name):
    img = canvas()
    if name == "blue_mark": mark(img, BLUE)
    elif name == "solid_red_rect": cv2.rectangle(img, (20, 20), (120, 120), RED, -1)
    elif name == "solid_green_square": cv2.rectangle(img, (20, 20), (80, 80), GREEN, -1)
    elif name == "ring_only": cv2.rectangle(img, (20, 20), (80, 80), GREEN, 4)
    elif name == "filled_circle_only": cv2.circle(img, (60, 60), 20, RED, -1)
    elif name == "colour_noise": img = np.random.default_rng(3).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    r = classify_veg_mark(img)
    assert r.label is None


def test_degenerate_inputs():
    assert classify_veg_mark(None).label is None
    assert classify_veg_mark(np.zeros((0, 0, 3), np.uint8)).label is None
    assert classify_veg_mark(np.zeros((10, 10, 3), np.uint8)).label is None


# ---------------- split-column tables ----------------
def L(text, x, y, w=90, h=18, conf=0.9):
    return OcrLine(text, conf, (x, y, x + w, y + h))


TABLE = [L("Nutritional Information", 40, 10, 300), L("per 100 g", 380, 40), L("per serving", 520, 41),
         L("Energy", 40, 80), L("520 kcal", 380, 83), L("156 kcal", 520, 79),
         L("Protein", 40, 110), L("6.5 g", 380, 112), L("2 g", 520, 108),
         L("Total Fat", 40, 140), L("32 g", 380, 141), L("9.6 g", 520, 142),
         L("Saturated Fat", 40, 170), L("14 g", 380, 172), L("4.2 g", 520, 171),
         L("Total Sugars", 40, 200), L("2 g", 380, 201), L("0.6 g", 520, 199),
         L("Sodium", 40, 230), L("650 mg", 380, 231), L("195 mg", 520, 232),
         L("Contains permitted colours", 40, 270, 300)]


def test_rows_are_grouped_by_y_with_jitter():
    rows = group_rows(TABLE)
    labelled = [[l.text for l in r] for r in rows]
    assert ["Protein", "6.5 g", "2 g"] in labelled and ["Energy", "520 kcal", "156 kcal"] in labelled


def test_split_columns_paired_per_serving_and_per_100g():
    t = parse_table(TABLE)
    ph = t.phrases
    assert "Protein 2 grams per serving" in ph and "Protein 6.5 grams per 100 grams" in ph
    assert "Energy 156 kilocalories per serving" in ph and "Sodium 195 milligrams per serving" in ph
    assert "Saturated fat 4.2 grams per serving" in ph and "Fat 9.6 grams per serving" in ph and "Sugars 0.6 grams per serving" in ph
    assert ph.index("Protein 2 grams per serving") < ph.index("Protein 6.5 grams per 100 grams")      # per serving is read first
    assert not any("permitted" in p for p in ph)


def test_header_in_one_wide_line_and_rda_column_ignored():
    lines = [L("Per 100 g   Per serving   %RDA", 300, 10, 400),
             L("Protein", 30, 50), L("20 g", 330, 51, 40), L("6 g", 480, 50, 40), L("12 %", 640, 50, 40)]
    t = parse_table(lines)
    assert t.phrases == ["Protein 6 grams per serving", "Protein 20 grams per 100 grams"]


def test_single_column_without_header():
    t = parse_table([L("Protein", 30, 50), L("7 g", 300, 52), L("Carbohydrate", 30, 80), L("45.2 g", 300, 79)])
    assert t.phrases == ["Protein 7 grams", "Carbohydrate 45.2 grams"]


def test_table_edge_cases():
    assert parse_table([]).phrases == [] and parse_table([OcrLine("Protein 7 g", 0.9)]).phrases == []     # no boxes -> caller falls back
    assert parse_table([L("Calcium", 30, 50), L("120 mg", 300, 50)]).phrases == []                          # unknown nutrient ignored
    assert nutrition_phrase("saturated_fat", 4.2, "g", "per serving") == "Saturated fat 4.2 grams per serving"


def test_food_parser_now_reads_split_columns_without_a_header():
    """The documented gap: label and number as separate OCR boxes on the same row."""
    lines = [L("Protein", 30, 50), L("7 g", 300, 52), L("Total Fat", 30, 80), L("12.5 g", 300, 79), L("Sodium", 30, 110), L("480 mg", 300, 111)]
    label = parse_food_label(lines)
    assert label.get("nutrition").status == FieldStatus.PRESENT
    assert label.nutrients["protein"].value == 7 and label.nutrients["fat"].value == 12.5 and label.nutrients["sodium"].unit == "mg"


def test_merge_rows_keeps_unboxed_lines():
    out = merge_rows([OcrLine("free text", 0.9), L("Protein", 30, 50), L("7 g", 300, 52)])
    assert [l.text for l in out] == ["free text", "Protein 7 g"]


# ---------------- end to end: IDENTIFY -> MORE reads the veg mark and the table ----------------
def test_pipeline_reads_veg_mark_and_split_table_into_the_card():
    from config.settings import Settings
    from smartsight.detector import Detection
    from smartsight.pipeline import Pipeline
    from smartsight.product_lookup import LocalProductStore
    from smartsight.speech_engine import SpeechController
    from smartsight.voice_commands import Command as C
    from tests.fakes import FakeBackend
    from tests.test_pipeline import FakeCamera, FakeOcr, NoBarcode

    img = mark(canvas(640, 480, noise=True), GREEN, 56, xy=(60, 50))

    class Det:
        def detect(self, f, conf=0.5): return [Detection("donut", 0.9, (30, 20, 600, 450))]

    s = Settings.load(path="/none", env={}); s.capture_samples = 1; s.cloud_ai = False
    cam = FakeCamera(img); cam.started = True; b = FakeBackend()
    p = Pipeline(s, cam, Det(), FakeOcr([]), NoBarcode(), LocalProductStore(), SpeechController(b), sleep=lambda t: None, async_tiers=False)
    p.ocr.lines = TABLE + [L("Ingredients: Wheat flour, sugar, palm oil.", 40, 300, 400), L("Best before 6 months from manufacture", 40, 330, 400)]
    p.process_command(C.IDENTIFY); p.process_command(C.MORE); p.speech.wait_idle(2)
    t = b.spoken[-1]
    assert "Vegetarian mark" in t and "Protein 2 grams per serving" in t and "Ingredients: Wheat flour, sugar, palm oil" in t
    assert "Expiry or best before: 6 months from manufacture" in t and "Caution" in t
