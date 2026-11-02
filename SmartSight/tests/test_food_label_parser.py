from smartsight.food_label_parser import FieldStatus as F, front_text, parse_food_label, split_ingredients

LABEL = """PEPPY CRISPS
Ingredients: Potato (68%), Palm Oil, Salt (1.5%), Spices
& Condiments (Onion Powder, Chilli).
Contains: Wheat, Milk
Nutrition Information Per 100 g
Energy 520 kcal
Protein 6.5 g
Total Fat 32 g
Saturated Fat 14 g
Sugars 2 g
Sodium 650 mg
Serving size: 30 g
Mfg: 03/2025 Best before 6 months from manufacture
Store in a cool dry place.
VEG"""


def test_full_label():
    l = parse_food_label(LABEL)
    assert l.get("ingredients").status == F.PRESENT
    assert l.ingredient_items[:3] == ["Potato (68%)", "Palm Oil", "Salt (1.5%)"]
    assert len(l.ingredient_items) == 4        # bracketed commas not split
    assert l.get("allergens").value == "Contains: Wheat, Milk"
    assert l.nutrients["protein"].value == 6.5 and l.nutrients["sodium"].unit == "mg"
    assert l.nutrients["fat"].value == 32 and l.nutrients["saturated_fat"].value == 14
    assert l.basis == "per 100 g"
    assert l.get("serving_size").value == "30 g"
    assert l.get("expiry").value == "6 months from manufacture"
    assert l.get("manufacture").value == "03/2025"
    assert l.get("storage").status == F.PRESENT
    assert l.get("diet_marking").value == "vegetarian"
    assert "Potato (68%)" in l.raw_text            # original text preserved


def test_missing_vs_unreadable():
    l = parse_food_label("PEPPY CRISPS\nMRP Rs 10")
    for name in ("ingredients", "allergens", "nutrition", "expiry", "storage"):
        assert l.get(name).status == F.MISSING
    l2 = parse_food_label("Ingredients: ~~ ## ..\nBest before\nNutrition Information")
    assert l2.get("ingredients").status == F.UNREADABLE
    assert l2.get("expiry").status == F.UNREADABLE
    assert l2.get("nutrition").status == F.UNREADABLE


def test_no_invented_allergens_or_ingredients():
    l = parse_food_label("Ingredients: Wheat flour, sugar, milk solids")
    assert l.get("allergens").status == F.MISSING       # not inferred from ingredients
    assert l.get("expiry").status == F.MISSING


def test_no_per_serving_calculation():
    l = parse_food_label("Nutrition Information Per 100 g\nProtein 10 g\nServing size: 30 g")
    assert l.nutrients["protein"].value == 10          # reported as printed, not scaled


def test_non_veg_not_mistaken_for_veg():
    assert parse_food_label("Non-Vegetarian").get("diet_marking").value == "non-vegetarian"
    assert parse_food_label("Contains no marking").get("diet_marking").status == F.MISSING


def test_added_sugars_separate():
    l = parse_food_label("Total Sugars 12 g\nAdded Sugars 8 g")
    assert l.nutrients["sugars"].value == 12 and l.nutrients["added_sugars"].value == 8


def test_expiry_formats():
    assert parse_food_label("EXP: 12/2026").get("expiry").value == "12/2026"
    assert parse_food_label("Use by 05 Jan 2027").get("expiry").status == F.PRESENT
    assert parse_food_label("Expiry Date 15-08-2026").get("expiry").value == "15-08-2026"


def test_split_ingredients():
    assert split_ingredients("a, b (c, d), e") == ["a", "b (c, d)", "e"]


def test_front_text_excludes_ingredients_and_nutrition():
    t = front_text(LABEL)
    assert "PEPPY" in t and "Palm Oil" not in t and "Protein" not in t and "Milk" not in t


def test_mean_confidence():
    from smartsight.utils import OcrLine
    l = parse_food_label([OcrLine("Ingredients: salt, oil", 0.4), OcrLine("MRP 5", 0.6)])
    assert abs(l.mean_confidence - 0.5) < 1e-9
