from decimal import Decimal
from smartsight.food_label_parser import parse_food_label
from smartsight.frame_quality import QualityReport
from smartsight.narration import (NarrationEngine, ObjectAnnouncer, ObjectStabilizer, TEMPLATES, describe_objects, tpl)
from smartsight.price_reader import extract_prices
from smartsight.product_identifier import ProductInfo
from smartsight.utils import InfoState
from tests.fakes import FakeClock


def test_stabilizer_requires_consecutive_frames():
    st = ObjectStabilizer(3, misses=1)
    assert st.update(["cup"]) == {} and st.update(["cup"]) == {}
    assert st.update(["cup"]) == {"cup": 1}


def test_stabilizer_tolerates_one_miss_then_resets():
    st = ObjectStabilizer(2, misses=1)
    st.update(["cup"]); st.update(["cup"])
    st.update([])                            # one missed cycle is tolerated
    assert st.update(["cup"]) == {"cup": 1}
    st.update([]); st.update([])             # two misses -> history dropped
    assert st.update(["cup"]) == {}


def test_duplicate_announcements_suppressed_while_stationary():
    clock = FakeClock()
    a = ObjectAnnouncer(3.0, clock)
    spoken = []
    for _ in range(50):                      # object stays in view for 50 cycles
        spoken += a.update({"bottle": 1}); clock.advance(0.3)
    assert spoken == ["bottle"]


def test_reannounce_after_leaving_and_returning():
    clock = FakeClock(); a = ObjectAnnouncer(3.0, clock)
    assert a.update({"bottle": 1}) == ["bottle"]
    clock.advance(5); a.update({})
    clock.advance(1)
    assert a.update({"bottle": 1}) == ["bottle"]


def test_new_object_announced_while_other_stays_quiet():
    clock = FakeClock(); a = ObjectAnnouncer(3.0, clock)
    a.update({"bottle": 1}); clock.advance(0.3)
    assert a.update({"bottle": 1, "cup": 1}) == ["cup"]


def test_describe_objects():
    assert describe_objects({"bottle": 1}) == "a bottle"
    assert describe_objects({"cup": 2, "apple": 1}) == "2 cups and an apple"


def test_all_templates_have_all_verbosities_and_format():
    for key, variants in TEMPLATES.items():
        assert set(variants) == {"concise", "standard", "detailed"}, key


def test_price_wording_follows_state():
    n = NarrationEngine()
    c = n.price(extract_prices("MRP Rs. 120/-"))
    assert c.state == InfoState.CONFIRMED and "one hundred and twenty rupees" in c.text and "maximum retail price" in c.text
    t = n.price(extract_prices("Price 45"))
    assert t.state == InfoState.TENTATIVE and "not fully sure" in t.text
    u = n.price(extract_prices("Net Wt 120 g"))
    assert u.state == InfoState.UNKNOWN and "cannot confirm" in u.text
    none = n.price(extract_prices(""))
    assert "cannot read a price" in none.text


def test_price_conflict_wording():
    assert "more than one price" in NarrationEngine().price(extract_prices("MRP Rs. 150\nMRP Rs. 160")).text


def test_unknown_product_never_overclaims():
    n = NarrationEngine()
    out = n.identify(ProductInfo(InfoState.UNKNOWN, "unknown", object_label="cell phone"))
    assert "cannot identify it confidently" in out.text and out.state == InfoState.UNKNOWN
    assert "do not see an item" in n.identify(ProductInfo(InfoState.UNKNOWN, "unknown")).text


def test_category_is_tentative_and_hedged():
    n = NarrationEngine().identify(ProductInfo(InfoState.TENTATIVE, "category", category="shampoo",
                                               purpose="Shampoo is used to wash hair.", object_label="bottle"))
    assert n.text.startswith("This appears to be a shampoo bottle.") and n.state == InfoState.TENTATIVE


def test_conflict_not_asserted():
    n = NarrationEngine().identify(ProductInfo(InfoState.TENTATIVE, "unknown", conflict=True))
    assert "do not match" in n.text and n.state == InfoState.UNKNOWN


def test_verbosity_changes_length():
    p = ProductInfo(InfoState.TENTATIVE, "category", category="shampoo", purpose="Shampoo is used to wash hair.")
    lens = [len(NarrationEngine(v).identify(p).text) for v in ("concise", "standard", "detailed")]
    assert lens[0] < lens[1] <= lens[2]


def test_ingredients_truncated_with_full_option():
    label = parse_food_label("Ingredients: " + ", ".join(f"item{i}" for i in range(20)) + ".\nNet Wt 100 g")
    n = NarrationEngine()
    short = n.ingredients(label)
    assert "item7" in short.text and "item8" not in short.text and "12 more" in short.text
    assert "item19" in n.ingredients(label, full=True).text


def test_ingredients_missing_and_unreadable():
    n = NarrationEngine()
    assert "cannot find an ingredient list" in n.ingredients(parse_food_label("PEPPY CRISPS")).text
    assert "cannot read it clearly" in n.ingredients(parse_food_label("Ingredients: ## ~~")).text


def test_low_ocr_confidence_flagged_in_ingredients():
    from smartsight.utils import OcrLine
    label = parse_food_label([OcrLine("Ingredients: salt, oil, sugar", 0.3)])
    out = NarrationEngine().ingredients(label, min_conf=0.6)
    assert out.state == InfoState.TENTATIVE and "misread" in out.text


def test_expiry_states():
    n = NarrationEngine()
    assert n.expiry(parse_food_label("EXP 12/2026")).state == InfoState.CONFIRMED
    assert n.expiry(parse_food_label("Best before")).state == InfoState.UNKNOWN
    assert "cannot find an expiry" in n.expiry(parse_food_label("hello")).text


def test_quality_messages():
    n = NarrationEngine()
    assert "too dark" in n.quality(QualityReport(False, issues=["dark", "blurry"])).text
    assert "blurry" in n.quality(QualityReport(False, issues=["blurry"])).text
