from decimal import Decimal
import pytest
from smartsight.price_reader import extract_prices
from smartsight.utils import InfoState, OcrLine, amount_to_words


@pytest.mark.parametrize("text,amount,kind", [
    ("MRP Rs. 120/-", "120", "mrp"),
    ("M.R.P. ₹ 1,250.50 (Incl. of all taxes)", "1250.50", "mrp"),
    ("Maximum Retail Price: Rs.35", "35", "mrp"),
    ("Rs.45", "45", "printed"),
    ("Sale Price ₹89", "89", "sale"),
    ("MRP\n₹80", "80", "mrp"),
    ("MRP 99", "99", "mrp"),
])
def test_valid_prices(text, amount, kind):
    r = extract_prices(text)
    assert r.best and r.best.amount == Decimal(amount) and r.best.kind == kind
    assert r.state == InfoState.CONFIRMED


@pytest.mark.parametrize("text", [
    "Net Wt 120 g", "Net weight 500 ml", "Best before 12/05/2025", "Mfg 12.05.2025",
    "8901234567890", "Batch No 4567", "Protein 12 g", "Energy 250 kcal", "Total Fat 32 g", "Use by 05-2026",
])
def test_non_prices_rejected(text):
    r = extract_prices(text)
    assert r.best is None and r.state == InfoState.UNKNOWN


def test_barcode_after_price_word_is_rejected():
    r = extract_prices("Price 8901234567890")
    assert r.best is None and r.rejected


def test_date_after_label_rejected():
    assert extract_prices("Price 12/05/2025").best is None


def test_price_beside_weight_and_date():
    r = extract_prices("Net Wt 120 g\nMfg 12.05.2025\nMRP Rs. 99/-")
    assert r.best.amount == Decimal("99")


def test_price_word_without_currency_is_tentative():
    r = extract_prices("Price 45")
    assert r.state == InfoState.TENTATIVE


def test_low_ocr_confidence_downgrades():
    r = extract_prices([OcrLine("MRP Rs. 120", 0.3)], min_confidence=0.7)
    assert r.state == InfoState.TENTATIVE


def test_suffix_only_is_tentative():
    r = extract_prices("120/-")
    assert r.best and r.state == InfoState.TENTATIVE


def test_mrp_and_sale_kept_separate_and_mrp_preferred():
    r = extract_prices("MRP Rs. 150\nSale Price Rs. 120")
    assert {c.kind: c.amount for c in r.candidates} == {"mrp": Decimal("150"), "sale": Decimal("120")}
    assert r.best.kind == "mrp"


def test_conflicting_mrp_values():
    r = extract_prices("MRP Rs. 150\nMRP Rs. 160")
    assert r.conflict and r.state == InfoState.TENTATIVE


def test_receipt_total_only_in_receipt_mode():
    assert extract_prices("Total 450.00", mode="product").best is None
    assert extract_prices("Total 450.00", mode="receipt").best.kind == "total"
    assert extract_prices("Total Fat 32 g", mode="receipt").best is None


def test_raw_text_preserved():
    r = extract_prices("MRP Rs. 120/-")
    assert r.best.source_text == "MRP Rs. 120/-"


def test_empty_and_garbage_input():
    assert extract_prices("").state == InfoState.UNKNOWN
    assert extract_prices("@@## !!").best is None


def test_words():
    assert amount_to_words(Decimal("120")) == "one hundred and twenty rupees"
    assert amount_to_words(Decimal("1")) == "one rupee"
    assert amount_to_words(Decimal("1250.50")) == "one thousand two hundred and fifty rupees and fifty paise"
    assert amount_to_words(Decimal("125000")) == "one lakh twenty five thousand rupees"
