"""Rule-based food-label parsing. Never infers absent content; keeps raw OCR text.

Every field is PRESENT, MISSING (no sign of it in the text) or UNREADABLE (a header was
seen but no usable value followed). Nutrition values are reported as printed; nothing
is converted between per-100g and per-serving.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .utils import OcrLine, TextSource, as_lines

I = re.I
_STOP = (r"nutrition(?:al)?\s+(?:information|facts|value|info)|allergen|may\s+contain|manufactured|marketed|"
         r"mfd|mfg|best\s+before|use\s+by|expiry|exp\b|storage|store\s+in|net\s+(?:wt|weight|quantity|qty)|"
         r"fssai|mrp|m\.r\.p|customer\s+care|serving\s+size")
_ING = re.compile(rf"[il1]ngredient[s5]?\s*[:\-]?\s*(?P<v>.*?)(?=\b(?:{_STOP})|\n\s*contains\b|\Z)", I | re.S)
_ING_HEADER = re.compile(r"[il1]ngredient", I)
_ALLERGEN = re.compile(r"(?:allergen[^:\n]{0,30}[:\-]\s*(?P<a>[^\n]+))|(?:^|\n)\s*(?P<b>(?:may\s+contain|contains)\s*[:\-]?\s*[^\n]+)", I)
_NUT_HEADER = re.compile(r"nutrition(?:al)?\s+(?:information|facts|value|info)", I)
_SERVING = re.compile(r"serv(?:ing|e)\s*size\s*[:\-]?\s*(?P<v>\d+(?:\.\d+)?\s?(?:g|gm|ml)\b(?:\s*\([^)]{0,20}\))?)", I)
_SERVING_HDR = re.compile(r"serv(?:ing|e)\s*size", I)
_BASIS = re.compile(r"per\s*(?:100\s*(?:g|gm|ml)|serv(?:e|ing)|portion)", I)
_DATE = (r"(?:\d{1,2}\s?[/\-.]\s?\d{1,2}\s?[/\-.]\s?\d{2,4}|\d{1,2}\s?[/\-.]\s?\d{4}|"
         r"(?:\d{1,2}\s?)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,.\-/]*\d{2,4})")
_PERIOD = (r"\d+\s*(?:days?|weeks?|months?|years?)(?:\s+from\s+(?:the\s+)?(?:date\s+of\s+)?"
           r"(?:manufacture|manufacturing|packing|packaging|mfg|pkg))?")
_EXP_KEY = r"(?<![a-z])(?:best\s+before|use\s+by|use\s+before|expiry(?:\s+date)?|expires?|exp\.?(?:\s+date)?)(?![a-z])"
_MFG_KEY = (r"(?<![a-z])(?:mfg\.?(?:\s+date)?|mfd\.?(?:\s+date)?|manufactur(?:ed|ing)\s+date|date\s+of\s+manufacture|"
            r"pkd\.?|packed\s+on|packing\s+date|date\s+of\s+packing|pkg\.?\s+date)(?![a-z])")
_STORAGE = re.compile(r"(?:^|\n|\s)(?P<v>(?:store|storage|keep\s+(?:in|refrigerated|away))[^\n]{0,90})", I)
_WARN = re.compile(r"(?:^|\n)\s*(?P<v>(?:warning|caution|keep\s+out\s+of\s+reach|not\s+for\s+use|do\s+not)[^\n]{0,120})", I)


class FieldStatus(Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNREADABLE = "unreadable"


@dataclass
class LabelField:
    status: FieldStatus
    value: Optional[str] = None
    raw: Optional[str] = None


@dataclass
class Nutrient:
    value: float
    unit: str
    raw: str


@dataclass
class FoodLabel:
    raw_text: str
    mean_confidence: Optional[float]
    fields: dict = field(default_factory=dict)
    nutrients: dict = field(default_factory=dict)
    basis: Optional[str] = None
    ingredient_items: list = field(default_factory=list)

    def get(self, name: str) -> LabelField:
        return self.fields.get(name, LabelField(FieldStatus.MISSING))


_G = r"g|gm"
_NUTRIENTS = [
    ("energy", r"energy", r"kcal|kj|cal"),
    ("protein", r"protein", r"g|mg"),
    ("carbohydrate", r"carbohydrates?|carbs?", r"g|mg"),
    ("added_sugars", r"added\s+sugars?", _G),
    ("sugars", r"(?<!added\s)(?:total\s+)?sugars?", _G),
    ("fat", r"(?<!saturated\s)(?<!trans\s)(?<!unsaturated\s)(?:total\s+)?fat", _G),
    ("saturated_fat", r"saturated\s+(?:fat|fatty\s+acids)", _G),
    ("trans_fat", r"trans\s+fat", _G),
    ("fibre", r"(?:dietary\s+)?fib(?:re|er)", _G),
    ("sodium", r"sodium", r"mg|g"),
    ("salt", r"salt", r"g|mg"),
]


def _clean(s: str) -> str:
    return " ".join(s.split()).strip(" ,;:.-")


def _garbled(s: str) -> bool:
    if len(s) < 3:
        return True
    return sum(c.isalnum() or c.isspace() for c in s) / len(s) < 0.6


def split_ingredients(text: str) -> list[str]:
    """Split on commas/semicolons outside brackets."""
    items, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            items.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    items.append("".join(cur).strip())
    return [i for i in items if i]


def parse_food_label(source: TextSource) -> FoodLabel:
    lines = as_lines(source)
    text = "\n".join(l.text for l in lines)
    confs = [l.conf for l in lines]
    label = FoodLabel(text, sum(confs) / len(confs) if confs else None)
    f = label.fields

    # ingredients
    ing_text = ""
    m = _ING.search(text)
    if m:
        v = _clean(m.group("v"))
        ing_text = m.group(0)
        if _garbled(v):
            f["ingredients"] = LabelField(FieldStatus.UNREADABLE, None, m.group(0))
        else:
            f["ingredients"] = LabelField(FieldStatus.PRESENT, v, m.group(0))
            label.ingredient_items = split_ingredients(v)
    elif _ING_HEADER.search(text):
        f["ingredients"] = LabelField(FieldStatus.UNREADABLE)
    else:
        f["ingredients"] = LabelField(FieldStatus.MISSING)

    # allergens (only what is printed)
    parts = []
    for am in _ALLERGEN.finditer(text):
        parts.append(_clean(am.group("a") or am.group("b")))
    if parts:
        joined = "; ".join(parts)
        f["allergens"] = LabelField(FieldStatus.UNREADABLE if _garbled(joined) else FieldStatus.PRESENT, joined, joined)
    elif re.search(r"allergen", text, I):
        f["allergens"] = LabelField(FieldStatus.UNREADABLE)
    else:
        f["allergens"] = LabelField(FieldStatus.MISSING)

    # nutrition (ingredient text removed so e.g. "salt" there is not mistaken for a nutrient row)
    nut_text = text.replace(ing_text, " ") if ing_text else text
    if any(l.box for l in lines):                    # split-column tables: pair labels with numbers on the same visual row
        from .nutrition_table import merge_rows
        nut_text = "\n".join(r.text for r in merge_rows([l for l in lines if not _ING_HEADER.search(l.text)]))
    for key, kw, units in _NUTRIENTS:
        rx = re.compile(rf"(?<![a-z])(?:{kw})[^\d\n]{{0,15}}?(?P<n>\d+(?:\.\d+)?)\s*(?P<u>{units})(?![a-z])", I)
        ms = list(rx.finditer(nut_text))
        if not ms:
            continue
        pick = next((x for x in ms if key == "energy" and x.group("u").lower() == "kcal"), ms[0])
        label.nutrients[key] = Nutrient(float(pick.group("n")), pick.group("u").lower(), pick.group(0).strip())
    bm = _BASIS.search(nut_text)
    label.basis = bm.group(0).lower() if bm and label.nutrients else None
    if label.nutrients:
        f["nutrition"] = LabelField(FieldStatus.PRESENT)
    elif _NUT_HEADER.search(text):
        f["nutrition"] = LabelField(FieldStatus.UNREADABLE)
    else:
        f["nutrition"] = LabelField(FieldStatus.MISSING)

    sm = _SERVING.search(text)
    if sm:
        f["serving_size"] = LabelField(FieldStatus.PRESENT, _clean(sm.group("v")), sm.group(0))
    elif _SERVING_HDR.search(text):
        f["serving_size"] = LabelField(FieldStatus.UNREADABLE)
    else:
        f["serving_size"] = LabelField(FieldStatus.MISSING)

    f["expiry"] = _dated(text, _EXP_KEY)
    f["manufacture"] = _dated(text, _MFG_KEY)

    st = _STORAGE.search(text)
    f["storage"] = (LabelField(FieldStatus.PRESENT, _clean(st.group("v")), st.group("v"))
                    if st else LabelField(FieldStatus.MISSING))

    if re.search(r"non[\s\-]?veg(?:etarian)?", text, I):
        f["diet_marking"] = LabelField(FieldStatus.PRESENT, "non-vegetarian", None)
    elif re.search(r"(?<![a-z])(?:vegetarian|veg)(?![a-z])", text, I):
        f["diet_marking"] = LabelField(FieldStatus.PRESENT, "vegetarian", None)
    else:
        f["diet_marking"] = LabelField(FieldStatus.MISSING)  # symbols are not read by OCR

    ws = [_clean(w.group("v")) for w in _WARN.finditer(text)]
    f["warnings"] = LabelField(FieldStatus.PRESENT, "; ".join(ws)) if ws else LabelField(FieldStatus.MISSING)
    return label


def _dated(text: str, key: str) -> LabelField:
    m = re.search(rf"(?P<k>{key})[^0-9a-z]{{0,8}}(?P<v>{_DATE}|{_PERIOD})", text, I)
    if m:
        return LabelField(FieldStatus.PRESENT, _clean(m.group("v")), m.group(0))
    if re.search(key, text, I):
        return LabelField(FieldStatus.UNREADABLE)
    return LabelField(FieldStatus.MISSING)


_NUTR_LINE = re.compile(r"\b(protein|fat|sugars?|salt|sodium|energy|carbohydrates?|fib(?:re|er)|kcal|serving|per\s+100)\b", I)
_SECTION_START = re.compile(rf"^\s*(?:{_STOP}|contains|allergen)", I)


def front_text(source: TextSource) -> str:
    """Text excluding ingredient / nutrition / allergen lines, for category matching.
    Prevents 'salt', 'sugar' or 'milk' in an ingredient list from being read as the product type."""
    out, in_ing = [], False
    for ln in as_lines(source):
        t = ln.text
        if _ING_HEADER.search(t):
            in_ing = True
            continue
        if in_ing and _SECTION_START.match(t):
            in_ing = False
        if in_ing or _NUTR_LINE.search(t) or re.search(r"allergen|may contain", t, I) or re.match(r"\s*contains\b", t, I):
            continue
        out.append(t)
    return "\n".join(out)
