import numpy as np
import pytest
from config.settings import ROOT
from smartsight import safety
from smartsight.card import CardBuilder
from smartsight.colour import dominant_colour
from smartsight.fusion import Recognition, fuse
from smartsight.knowledge import KnowledgeBase
from smartsight.product_lookup import ProductRecord
from smartsight.taxonomy import Taxonomy, Vocabulary
from smartsight.utils import OcrLine

VOCAB = Vocabulary.from_file(ROOT / "data" / "vocabulary.json")
TAX = Taxonomy.from_file(ROOT / "data" / "taxonomy.json")
KB = KnowledgeBase.from_file(ROOT / "data" / "knowledge.json")
CB = CardBuilder(KB, VOCAB)
L = lambda *t: [OcrLine(x, 0.9, (0, 20 * i, 100, 20 * i + 15)) for i, x in enumerate(t)]


def rec(label, conf=0.8, source="zero-shot model", **kw):
    e = VOCAB.find(label)
    return Recognition(label, e.category if e else "Other", e.subcategory if e else "Other", conf, source, **kw)


# ---------- safety ----------
@pytest.mark.parametrize("text,removed", [
    ("It is safe to eat.", True), ("This product is allergen-free.", True), ("Take 2 tablets twice a day.", True),
    ("The dosage is 500 mg every 6 hours.", True), ("Nut free and gluten free.", True), ("It cures headaches.", True),
    ("A reusable bottle for drinks. It keeps drinks cold.", False), ("Keep a safe distance.", False),
])
def test_ai_text_sanitiser(text, removed):
    clean, r = safety.sanitize_ai_text(text)
    assert r is removed and (not removed or not safety.violates(clean))


def test_sanitiser_keeps_the_harmless_sentences():
    clean, r = safety.sanitize_ai_text("A white tablet strip. It is safe to take. Sold in pharmacies.")
    assert r and clean == "A white tablet strip. Sold in pharmacies."


# ---------- colour ----------
@pytest.mark.parametrize("bgr,name", [((0, 0, 220), "red"), ((200, 40, 0), "blue"), ((30, 180, 30), "green"), ((0, 200, 230), "yellow"),
                                       ((10, 10, 10), "black"), ((240, 240, 240), "white"), ((20, 80, 150), "brown")])
def test_colour_names(bgr, name):
    img = np.zeros((100, 100, 3), np.uint8); img[:] = bgr
    assert dominant_colour(img) == name


def test_colour_edge_cases():
    assert dominant_colour(None) is None
    mix = np.zeros((100, 100, 3), np.uint8)
    cols = [(0, 0, 220), (200, 40, 0), (30, 180, 30), (0, 200, 230)]
    for i in range(100):
        for j in range(100):
            mix[i, j] = cols[((i // 10) + (j // 10)) % 4]
    assert dominant_colour(mix) == "multicoloured"


# ---------- knowledge ----------
def test_every_category_has_defaults_and_kb_items_exist_in_vocabulary():
    for c in {e.category for e in VOCAB.entries}:
        assert c in KB.categories, c
    for name in KB.items:
        assert VOCAB.get(name) is not None, name


# ---------- cards by category ----------
def test_machine_card_has_operation_parts_and_caution():
    c = CB.build(rec("microwave oven"))
    assert c.category == "Machine and Appliance" and c.fields["typically_operated"] and "door" in c.fields["main_parts_or_controls"]
    assert any("metal" in n.lower() for n in c.safety_notes) and c.hazardous and "knowledge base" in c.sources


def test_medicine_card_never_gives_purpose_or_dose_and_points_to_pharmacist():
    c = CB.build(rec("medicine strip"), L("PARACARE 500", "Exp: 12/2026"))
    assert c.category == "Medicine and Health"
    assert any("pharmacist" in n for n in c.safety_notes)
    import re
    assert not re.search(r"\d+\s?(mg|ml|tablet)", " ".join(c.safety_notes) + str(c.fields)) and not safety.violates(str(c.fields))
    assert c.fields["expiry_or_best_before"] == "12/2026" and "pharmacist" in c.fields["purpose"]


def test_food_card_reads_label_fields_without_inventing():
    lines = L("PEPPY CRISPS", "Ingredients: Potato, Palm Oil, Salt.", "Contains: Milk", "Protein 6.5 g", "Best before 6 months from manufacture", "Net Wt 80 g")
    c = CB.build(rec("packet of chips"), lines)
    f = c.fields
    assert f["ingredients"].startswith("Potato") and f["allergens_printed"] == "Contains: Milk"
    assert f["expiry_or_best_before"] == "6 months from manufacture" and "Protein 6.5 grams" in " ".join(f["nutrition"])
    assert c.attributes["size"] == "80 g" and c.attributes["brand"] == "PEPPY CRISPS (as printed)"
    assert any("allergen" in n.lower() for n in c.safety_notes) and any("expiry" in n.lower() for n in c.safety_notes)
    assert "OCR" in c.sources


def test_missing_label_fields_are_absent_not_invented():
    c = CB.build(rec("packet of chips"), L("PEPPY CRISPS"))
    assert "ingredients" not in c.fields and "allergens_printed" not in c.fields and "expiry_or_best_before" not in c.fields


def test_barcode_record_gives_name_and_brand():
    r = fuse(None, 0, [], barcode=("Classic Salted Crisps", "Packaged Goods", "Packaged Snack", 0.95))
    c = CB.build(r, barcode_record=ProductRecord("1", "Classic Salted Crisps", "Peppy"))
    assert c.name == "Classic Salted Crisps" and c.attributes["brand"] == "Peppy" and "barcode" in c.sources


def test_person_card_is_minimal_and_never_identifies():
    c = CB.build(rec("person", source="detector"), L("NAME TAG John Smith"), colour="red", position="on your left")
    assert c.category == "Person" and c.text_read == [] and "never identified" in c.fields["note"]
    assert "brand" not in c.attributes and c.used_for == ""


def test_currency_and_documents():
    c = CB.build(rec("500 rupee note"), L("500", "RESERVE BANK OF INDIA"))
    assert c.category == "Currency" and any("genuine" in n for n in c.safety_notes) and c.fields["printed_text"]
    d = CB.build(rec("notebook"), L("Maths", "Class 8"))
    assert d.category == "Document and Text" and d.text_read[:2] == ["Maths", "Class 8"]


def test_text_read_is_in_reading_order():
    lines = [OcrLine("second", 0.9, (0, 60, 50, 75)), OcrLine("first", 0.9, (0, 10, 50, 25)), OcrLine("third", 0.9, (0, 110, 50, 125))]
    assert CB.build(rec("notebook"), lines).text_read == ["first", "second", "third"]


def test_ai_description_is_sanitised_marked_and_cannot_rename_confirmed_item():
    ai = {"can_identify": True, "item": "steel bottle", "description": "A shiny bottle. It is safe to drink from.",
          "typical_uses": "Keeps drinks cold.", "similar_items": ["flask", "dosage 5 mg daily"], "price": {"low": 300, "high": 700, "basis": "x"}, "sources": ["amazon.in"]}
    c = CB.build(rec("cup", 0.9, "detector"), ai=ai)
    assert c.name == "cup" and "AI description" in c.sources
    assert c.ai["description"] == "A shiny bottle." and c.ai["similar_items"] == ["flask"]
    assert safety.AI_REMOVED_NOTE in c.safety_notes


def test_card_has_all_required_fields_for_every_vocabulary_entry_and_never_claims_safety():
    for e in VOCAB.entries:
        c = CB.build(rec(e.name))
        assert c.name and c.category == e.category and c.subcategory == e.subcategory and c.summary and c.sources
        blob = " ".join([c.summary, c.description, c.used_for, *c.safety_notes, str(c.fields)])
        assert not safety.violates(blob), (e.name, blob)
        if e.category == "Medicine and Health":
            assert any("pharmacist" in n for n in c.safety_notes), e.name
