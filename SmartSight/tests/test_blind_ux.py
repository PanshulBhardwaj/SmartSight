import pytest
from config.settings import ROOT, Settings
from smartsight.announce import Announcer, LOW
from smartsight.card import InformationCard
from smartsight.debounce import AnnouncementDebouncer
from smartsight.earcons import Earcons, RecordingBackend
from smartsight.fusion import Recognition
from smartsight.spatial import (FindMode, distance_phrase, edges_cut, framing_hint, horizontal_zone, position_phrase)
from smartsight.taxonomy import Vocabulary
from smartsight.utils import OcrLine
from tests.fakes import FakeClock

W, H = 1000, 800
VOCAB = Vocabulary.from_file(ROOT / "data" / "vocabulary.json")


def S(**kw):
    s = Settings.load(path="/none", env={})
    for k, v in kw.items(): setattr(s, k, v)
    return s


def R(label="shampoo bottle", conf=0.8, cat="Packaged Goods", sub="Personal Care Product", **kw):
    return Recognition(label, cat, sub, conf, "zero-shot model", **kw)


# ---------- spatial ----------
@pytest.mark.parametrize("box,zone", [((0, 300, 100, 500), "far left"), ((250, 300, 350, 500), "left"), ((450, 300, 550, 500), "centre"),
                                       ((650, 300, 750, 500), "right"), ((900, 300, 990, 500), "far right")])
def test_horizontal_zones(box, zone):
    assert horizontal_zone(box, W) == zone


def test_position_and_distance_phrases():
    assert position_phrase((100, 200, 400, 600), W, H) == "on your left, about arm's length"
    assert position_phrase((250, 100, 750, 700), W, H) == "centred and close"
    assert distance_phrase((0, 0, 900, 700), W, H) == "very close"
    assert distance_phrase((10, 10, 60, 60), W, H) == "far away"
    assert position_phrase((700, 300, 800, 400), W, H).startswith("on your right")


def test_edge_cut_and_framing_hints():
    assert edges_cut((0, 100, 200, 300), W, H) == ["left"]
    assert edges_cut((300, 100, 500, 300), W, H) == []
    assert "cut off at the left edge" in framing_hint((0, 300, 250, 600), W, H)
    assert framing_hint((100, 300, 300, 500), W, H) == "Move the camera slightly left to centre it."   # camera follows the object
    assert framing_hint((700, 100, 900, 250), W, H) == "Move the camera slightly right and up to centre it."
    assert framing_hint((400, 300, 600, 500), W, H) is None
    assert "back a little" in framing_hint((0, 0, 1000, 800), W, H)


# ---------- find mode ----------
def test_find_mode_state_machine_and_cues():
    clock = FakeClock()
    fm = FindMode("cup", lambda r: r.label == "cup", clock)
    assert fm.update([], W, H).speech is None                                    # first search tick: silent
    clock.advance(11)
    assert "Still looking for your cup" in fm.update([], W, H).speech
    far = (R("cup"), (50, 300, 150, 500))
    u = fm.update([far], W, H)
    assert "I can see your cup" in u.speech and u.earcon == "found" and u.beep
    f1, i1 = u.beep
    nearer = (R("cup"), (300, 300, 400, 500))
    u2 = fm.update([nearer], W, H)
    assert u2.speech is None and u2.beep[0] > f1 and u2.beep[1] < i1             # higher and faster as it approaches the centre
    centred = (R("cup"), (450, 300, 550, 500))
    u3 = fm.update([centred], W, H)
    assert "Found your cup, centred" in u3.speech and u3.state == "centred"
    clock.advance(0.5); assert fm.update([centred], W, H).speech is None        # does not repeat
    clock.advance(3)
    lost = fm.update([], W, H)
    assert "lost your cup" in lost.speech and lost.earcon == "lost"
    assert fm.update([(R("mug"), (450, 300, 550, 500))], W, H).beep is None       # other objects ignored


# ---------- debouncer ----------
def test_debouncer_waits_for_stability_and_announces_once():
    c = FakeClock(); d = AnnouncementDebouncer(4, 6.0, 1.5, c)
    assert [d.update(1, "cup", n) for n in (1, 2, 3)] == [False] * 3
    assert d.update(1, "cup", 4) is True
    for n in range(5, 200):
        c.advance(0.1); assert d.update(1, "cup", n) is False                    # stationary: silent forever


def test_debouncer_announces_label_changes_but_not_flicker_back_to_back():
    c = FakeClock(); d = AnnouncementDebouncer(2, 6.0, 1.5, c)
    assert d.update(1, "bottle", 5)
    c.advance(0.5); assert not d.update(1, "shampoo bottle", 6)                   # too soon after the last announcement
    c.advance(1.5); assert d.update(1, "shampoo bottle", 7)                       # refinement is announced
    c.advance(5); assert not d.update(1, "shampoo bottle", 8)


def test_debouncer_cooldown_for_same_label_on_a_new_track_and_forgetting():
    c = FakeClock(); d = AnnouncementDebouncer(2, 6.0, 1.0, c)
    assert d.update(1, "cup", 3)
    c.advance(2); assert not d.update(2, "cup", 3)                                # another cup, inside the cooldown
    c.advance(5); assert d.update(2, "cup", 3)
    assert d.retain([2]) == [1]
    c.advance(2); assert d.update(1, "cup", 3) is False                           # same label announced 2 s ago: still cooling down
    c.advance(5); assert d.update(1, "cup", 3) is True                            # after the cooldown


# ---------- earcons ----------
def test_earcons_play_distinct_patterns_and_can_be_disabled():
    import time
    b = RecordingBackend(); e = Earcons(b)
    for name in ("ready", "locked", "lost", "error", "found"):
        e.play(name); time.sleep(0.05)
    assert len({tuple(x) for x in [b.played]}) == 1 and len(b.played) == 2 + 1 + 2 + 2 + 2
    b2 = RecordingBackend(); Earcons(b2, enabled=False).play("ready"); time.sleep(0.05)
    assert b2.played == []
    Earcons(b2).play("nonsense")                                                  # unknown cue: no crash


# ---------- announcements ----------
def test_level1_states_category_and_percent():
    a = Announcer(S())
    t = a.level1(R("shampoo bottle", 0.8), "on your left, about arm's length")
    assert t.startswith("Packaged goods, shampoo bottle, 80 percent sure.") and "On your left" in t and t.endswith("Say more for details.")


def test_level1_hedges_below_threshold_and_says_not_sure_when_low():
    a = Announcer(S())
    mid = a.level1(R("mug", 0.45, "Household Item", "Kitchenware"))
    assert mid.startswith("I think this is a mug, in household item, 45 percent sure.")
    low = a.level1(R("mug", 0.2, "Household Item", "Kitchenware", alternatives=[("cup", 0.15), ("glass", 0.1)]))
    assert low.startswith("I'm not sure. It could be mug, cup or glass.") and "Category: household item" in low
    assert "move it closer" in a.level1(Recognition("unknown object", "Other", "Other", 0.0, "none"))


def test_level1_options():
    r = R("shampoo bottle", 0.8)
    assert "percent" not in Announcer(S(speak_percent=False)).level1(r)
    assert Announcer(S(verbosity="concise")).level1(r, "on your left") == "Packaged goods, shampoo bottle, 80 percent sure."
    assert "On your left" not in Announcer(S(position_cues=False)).level1(r, "on your left")
    assert "Caution" in Announcer(S()).level1(R("angle grinder", 0.9, "Machine and Appliance", "Power Equipment"), hazard=True)


def test_person_is_only_position():
    t = Announcer(S()).level1(Recognition("person", "Person", "Person", 0.9, "detector"), "on your right, about arm's length")
    assert t == "Person, on your right, about arm's length." and "percent" not in t


def test_level2_reads_the_whole_card_and_respects_verbosity():
    card = InformationCard("Peppy Crisps", "Packaged Goods", "Packaged Snack", 0.85, "A packaged snack.", used_for="a ready-to-eat snack",
                           attributes={"colour": "red", "brand": "Peppy", "size": "80 g"}, text_read=["PEPPY", "CRISPS"],
                           safety_notes=["Check the printed label for ingredients, allergens and the expiry date."],
                           fields={"ingredients": "Potato, Palm Oil, Salt", "allergens_printed": "Contains: Milk",
                                   "nutrition": ["Protein 7 grams per serving"], "expiry_or_best_before": "12/2026",
                                   "veg_mark": {"mark": "vegetarian", "confidence": 0.9}}, sources=["zero-shot model", "OCR"], position="on your left")
    std = Announcer(S()).level2(card)
    for piece in ("Peppy Crisps. Packaged goods, packaged snack.", "85 percent sure", "Used for: a ready-to-eat snack", "Colour: red", "Brand: Peppy",
                  "Ingredients: Potato, Palm Oil, Salt", "Printed allergen text: Contains: Milk", "Protein 7 grams per serving",
                  "Expiry or best before: 12/2026", "Vegetarian mark, 90 percent sure", "Text I can read: PEPPY. CRISPS.", "Caution: Check the printed label"):
        assert piece in std, piece
    assert "Sources" not in std and "Sources: zero-shot model, OCR." in Announcer(S(verbosity="detailed")).level2(card)
    con = Announcer(S(verbosity="concise")).level2(card)
    assert "Colour" not in con and "Ingredients" not in con and "Caution" in con and len(con) < len(std)


def test_level2_hedges_unsure_cards_and_includes_ai_with_estimate_wording():
    card = InformationCard("mug", "Household Item", "Kitchenware", 0.4, "A mug.", alternatives=[("cup", 0.3)],
                           ai={"description": "A ceramic mug.", "typical_uses": "Drinking tea.", "similar_items": ["cup"],
                               "approx_price_inr": {"low": 150, "high": 400, "basis": "x"}, "sources": ["amazon.in"]})
    t = Announcer(S()).level2(card)
    assert "not fully sure; it could also be cup" in t and "percent sure" not in t
    assert "The online AI adds: A ceramic mug." in t and "one hundred and fifty to four hundred rupees" in t and "not a printed price" in t


def test_text_chunks_reading_order_and_unclear_note():
    lines = [OcrLine("second line", 0.9, (0, 80, 50, 100)), OcrLine("first line", 0.9, (0, 10, 50, 30)), OcrLine("junk", 0.2, (0, 40, 50, 60))]
    ch = Announcer(S()).text_chunks(lines, 0.6)
    assert ch[0].startswith("Text I can read: first line") and "second line" in ch[0] and "junk" not in " ".join(ch) and "too unclear" in ch[-1]
    assert Announcer(S()).text_chunks([], 0.6) == []
    long = [OcrLine("word " * 20 + ".", 0.9, (0, i * 30, 50, i * 30 + 20)) for i in range(10)]
    assert all(len(c) < 500 for c in Announcer(S()).text_chunks(long, 0.6)) and len(Announcer(S()).text_chunks(long, 0.6)) > 2


def test_scene_scan_groups_by_category_left_to_right():
    a = Announcer(S())
    items = [(R("mobile phone", 0.8, "Electronics", "Phone and Tablet"), (800, 300, 900, 500)),
             (R("bottle", 0.7, "Household Item", "Kitchenware"), (50, 300, 150, 500)),
             (R("cup", 0.7, "Household Item", "Kitchenware"), (450, 300, 550, 500))]
    t = a.scene(items, W)
    assert t.startswith("I can see 3 objects, listed from left to right.")
    assert t.index("Household item: a bottle far left and a cup in the centre.") < t.index("Electronics: a mobile phone far right.")
    assert "do not see any objects" in a.scene([], W)


def test_help_lists_commands_and_category_phrase():
    h = Announcer(S()).help()
    for w in ("what is this", "more", "read text", "repeat", "describe", "scan", "find", "category", "faster", "slower", "verbosity", "help", "quit"):
        assert w in h
    assert Announcer(S()).category(R()) == "Category: packaged goods, personal care product."
