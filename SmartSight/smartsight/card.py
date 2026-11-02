"""Information Card: everything SmartSight knows about one recognised object, with sources.

Data priority: local knowledge base, then OCR / barcode, then (optional) AI description. Every field notes where it
came from; AI text is sanitised so it never makes safety, allergen or dosage claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import safety
from .food_label_parser import FieldStatus, parse_food_label
from .fusion import Recognition
from .knowledge import KnowledgeBase
from .nutrition_table import nutrition_phrase
from .product_identifier import guess_product_name
from .taxonomy import Vocabulary
from .utils import OcrLine

FOOD_SUBCATS = {"Packaged Snack", "Packaged Drink", "Grocery Staple", "Fruit", "Vegetable", "Cooked Food", "Beverage"}
LABEL_CATS = {"Packaged Goods", "Food and Drink", "Medicine and Health"}
_NET = re.compile(r"net\s*(?:wt|weight|qty|quantity|contents?)\.?[:\s]*(\d+(?:\.\d+)?\s?(?:g|gm|kg|ml|l|ltr))\b", re.I)


@dataclass
class InformationCard:
    name: str
    category: str
    subcategory: str
    confidence: float
    summary: str
    description: str = ""
    used_for: str = ""
    attributes: dict = field(default_factory=dict)       # colour, brand, size, material
    text_read: list = field(default_factory=list)
    safety_notes: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    fields: dict = field(default_factory=dict)           # category-specific
    position: str = ""
    alternatives: list = field(default_factory=list)
    ai: Optional[dict] = None                            # sanitised AI description, if used
    hazardous: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class CardBuilder:
    def __init__(self, kb: KnowledgeBase, vocab: Vocabulary, ocr_min_conf: float = 0.6):
        self.kb, self.vocab, self.ocr_min_conf = kb, vocab, ocr_min_conf

    def build(self, rec: Recognition, lines: Sequence[OcrLine] = (), barcode_record=None, ai: Optional[dict] = None,
              colour: Optional[str] = None, veg_mark: Optional[tuple] = None, nutrition_rows: Optional[list] = None,
              position: str = "") -> InformationCard:
        entry = self.vocab.find(rec.label)
        item = self.kb.item(rec.label) or (self.kb.item(entry.name) if entry else None) or {}
        cat_kb = self.kb.category(rec.category)
        name = barcode_record.name if barcode_record and rec.specific_product else rec.label
        sources = [s.strip() for s in rec.source.split(",") if s.strip() and s != "none"]
        used_kb = bool(item or cat_kb)
        if used_kb:
            sources.append("knowledge base")

        summary = item.get("summary") or f"{name}, {cat_kb.get('summary', 'an object')}."
        description = item.get("description") or cat_kb.get("description", "")
        used_for = item.get("used_for") or (entry.used_for if entry else "") or ""
        card = InformationCard(name, rec.category, rec.subcategory, rec.confidence, summary, description, used_for,
                               position=position, alternatives=list(rec.alternatives))
        card.hazardous = bool(item.get("hazard"))

        # attributes
        if colour:
            card.attributes["colour"] = colour
        if item.get("material"):
            card.attributes["material"] = item["material"]
        good = [l for l in lines if l.conf >= self.ocr_min_conf and len(l.text.strip()) >= 2]
        text = "\n".join(l.text for l in good)
        if barcode_record and barcode_record.brand:
            card.attributes["brand"] = barcode_record.brand
        elif rec.category in LABEL_CATS:
            guess = guess_product_name(good, self.ocr_min_conf)
            if guess:
                card.attributes["brand"] = f"{guess} (as printed)"
        m = _NET.search(text)
        if m:
            card.attributes["size"] = m.group(1)
        # text read, top-to-bottom then left-to-right
        card.text_read = [l.text.strip() for l in sorted(good, key=lambda l: ((l.box[1] // 20, l.box[0]) if l.box else (0, 0)))][:20]
        if good:
            sources.append("OCR")
        if barcode_record:
            sources.append("barcode")

        # safety (category + item); never claims safety
        notes = list(cat_kb.get("safety", []))
        if item.get("safety"):
            notes.append(item["safety"])
        card.safety_notes = list(dict.fromkeys(notes))

        # category-specific fields
        if rec.category == "Person":
            card.fields["note"] = "Only presence and position are reported. People are never identified."
            card.description, card.used_for, card.text_read = cat_kb.get("description", ""), "", []
            card.attributes.pop("brand", None)
        if item.get("operation"):
            card.fields["typically_operated"] = item["operation"]
        if item.get("parts"):
            card.fields["main_parts_or_controls"] = item["parts"]
        if rec.category in ("Food and Drink", "Packaged Goods", "Medicine and Health"):
            self._label_fields(card, rec, parse_food_label(list(good)), veg_mark, nutrition_rows)
        if rec.category == "Currency":
            card.fields["printed_text"] = card.text_read[:4]

        # optional AI description (clearly marked; sanitised)
        if ai and ai.get("can_identify"):
            clean = {}
            removed_any = False
            for k in ("item", "description", "typical_uses"):
                t, r = safety.sanitize_ai_text(ai.get(k) or "")
                clean[k] = t; removed_any |= r
            clean["similar_items"] = [x for x in ai.get("similar_items", []) if not safety.violates(x)][:3]
            clean["approx_price_inr"] = ai.get("price")
            clean["sources"] = ai.get("sources", [])
            card.ai = clean
            sources.append("AI description")
            if not card.description and clean["description"]:
                card.description = clean["description"]
            if not card.used_for and clean["typical_uses"]:
                card.used_for = clean["typical_uses"]
            if removed_any:
                card.safety_notes.append(safety.AI_REMOVED_NOTE)
        card.sources = list(dict.fromkeys(sources))
        return card

    def _label_fields(self, card: InformationCard, rec: Recognition, label, veg_mark, nutrition_rows) -> None:
        f = card.fields
        food = rec.category == "Food and Drink" or rec.subcategory in FOOD_SUBCATS
        med = rec.category == "Medicine and Health"
        exp = label.get("expiry")
        if exp.status == FieldStatus.PRESENT:
            f["expiry_or_best_before"] = exp.value
            card.safety_notes.append(safety.EXPIRY_NOTE)
        elif exp.status == FieldStatus.UNREADABLE:
            f["expiry_or_best_before"] = "present but unreadable"
        if med:
            card.safety_notes.append(safety.CHECK_LABEL)
            f["purpose"] = "not stated by SmartSight; read the label or ask a pharmacist"
            return
        if not food:
            return
        ing = label.get("ingredients")
        if ing.status == FieldStatus.PRESENT:
            f["ingredients"] = ing.value
        elif ing.status == FieldStatus.UNREADABLE:
            f["ingredients"] = "present but unreadable"
        al = label.get("allergens")
        if al.status == FieldStatus.PRESENT:
            f["allergens_printed"] = al.value
            card.safety_notes.append(safety.ALLERGEN_NOTE)
        if nutrition_rows:
            f["nutrition"] = list(nutrition_rows)
        elif label.nutrients:
            f["nutrition"] = [nutrition_phrase(k, n.value, n.unit, label.basis) for k, n in label.nutrients.items()]
        diet = label.get("diet_marking")
        if veg_mark and veg_mark[0]:
            f["veg_mark"] = {"mark": veg_mark[0], "confidence": round(veg_mark[1], 2)}
        elif diet.status == FieldStatus.PRESENT:
            f["veg_mark"] = {"mark": diet.value + " (printed text)", "confidence": None}
        if label.get("storage").status == FieldStatus.PRESENT:
            f["storage"] = label.get("storage").value
