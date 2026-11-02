from smartsight.product_identifier import ProductIdentifier
from smartsight.product_lookup import LocalProductStore, OpenFoodFactsLookup, ProductRecord
from smartsight.utils import InfoState, OcrLine
import json, urllib.error

PI = ProductIdentifier()
L = lambda *t: [OcrLine(x, 0.9, (0, 0, 10, 20 + i * 5)) for i, x in enumerate(t)]
REC = ProductRecord("8901", "Classic Salted Crisps", "Peppy", source="local")


def test_category_from_label_is_tentative():
    p = PI.identify(["bottle"], L("HERBAL SHAMPOO"))
    assert p.level == "category" and p.category == "shampoo" and p.state == InfoState.TENTATIVE


def test_ingredient_words_do_not_set_category():
    p = PI.identify(["bottle"], L("Ingredients: water, salt, sugar, milk solids, tea extract"))
    assert p.level == "object" and p.category is None


def test_unknown_when_nothing_seen():
    assert PI.identify([], []).state == InfoState.UNKNOWN


def test_unknown_object_without_text():
    p = PI.identify(["cell phone"], [])
    assert p.level == "object" and p.name is None


def test_barcode_matching_label_confirms():
    p = PI.identify([], L("PEPPY", "Classic Salted Crisps"), REC)
    assert p.state == InfoState.CONFIRMED and p.level == "specific" and "Peppy" in p.name and not p.conflict


def test_barcode_conflicting_with_label():
    p = PI.identify([], L("Totally Different Brand", "Fresh Orange Juice Drink"), REC)
    assert p.conflict and p.name is None and p.state == InfoState.TENTATIVE


def test_barcode_with_little_text_is_database_only():
    p = PI.identify([], [], REC)
    assert p.level == "specific" and p.state == InfoState.TENTATIVE and p.source == "barcode-database"


def test_ocr_typo_tolerated():
    p = PI.identify([], L("PEPPY", "Clasic Salted Crsps"), REC)
    assert p.state == InfoState.CONFIRMED


def test_low_confidence_text_ignored():
    p = PI.identify(["bottle"], [OcrLine("SHAMPOO", 0.2)])
    assert p.category is None


def test_name_guess_is_front_text_only():
    p = PI.identify([], [OcrLine("PEPPY", 0.9, (0, 0, 50, 40)), OcrLine("Ingredients: water", 0.9, (0, 0, 50, 80))])
    assert p.name_guess == "PEPPY" and p.name is None


def test_local_store(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"barcode": 123456, "name": "Tea Box"}]))
    assert LocalProductStore(f).get("123456").name == "Tea Box"
    assert LocalProductStore(tmp_path / "missing.json").get("1") is None
    bad = tmp_path / "bad.json"; bad.write_text("{not json")
    assert LocalProductStore(bad).get("1") is None


def _ok(_u, _t):
    return json.dumps({"status": 1, "product": {"product_name": "Tea", "brands": "Acme,Other", "categories": "Drinks"}}).encode()


def test_lookup_found_and_cached(tmp_path):
    calls = []
    def fetch(u, t): calls.append(u); return _ok(u, t)
    lk = OpenFoodFactsLookup(tmp_path / "c.json", fetch)
    r = lk.fetch("8901234567890"); assert r.status == "found" and r.record.brand == "Acme"
    assert lk.fetch("8901234567890").status == "cached" and len(calls) == 1
    assert OpenFoodFactsLookup(tmp_path / "c.json", fetch).fetch("8901234567890").status == "cached"   # persisted


def test_lookup_failures(tmp_path):
    def boom(u, t): raise urllib.error.URLError("offline")
    assert OpenFoodFactsLookup(None, boom).fetch("8901234567890").status == "network_error"
    assert OpenFoodFactsLookup(None, lambda u, t: b"<html>").fetch("8901234567890").status == "invalid_response"
    assert OpenFoodFactsLookup(None, lambda u, t: b'{"status": 0}').fetch("8901234567890").status == "not_found"
    assert OpenFoodFactsLookup(None, _ok).fetch("12").status == "invalid_response"


def test_lookup_rate_limit_backoff():
    from tests.fakes import FakeClock
    clock = FakeClock(); n = []
    def f(u, t):
        n.append(1); raise urllib.error.HTTPError(u, 429, "slow down", {}, None)
    lk = OpenFoodFactsLookup(None, f, clock=clock, backoff_seconds=60)
    assert lk.fetch("8901234567890").status == "rate_limited"
    assert lk.fetch("8901234567891").status == "rate_limited" and len(n) == 1   # no second request during backoff
