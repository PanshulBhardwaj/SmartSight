"""Spoken forms: Level 1 (automatic, one short sentence with category), Level 2 (full Information Card on 'more'),
Level 3 (all text on 'read text'), scene scan, help. Wording is hedged whenever confidence is low.

The percentage is the recogniser's UNCALIBRATED score, not a probability of being right (config: speak_percent).
"""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Optional, Sequence

from .card import InformationCard
from .fusion import Recognition
from .narration import article, plural
from .spatial import short_position
from .utils import OcrLine, amount_to_words, number_to_words

HELP_TEXT = ("You can say or type: what is this. more, for the full details. read text. repeat. describe. scan, to hear what is "
             "around you. find followed by an object, for example find my cup. category. faster. slower. verbosity. price. "
             "ingredients. expiry. ask AI. stop. help. quit. In Live Mode the keys are i, m, t, r, d, s, f, c, plus, minus, v, "
             "p, g, e, a, o, space, h and q.")
LOW = 0.35        # below this the answer is "I'm not sure"


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def cat_phrase(category: str) -> str:
    return cap(category.lower())


def pct(conf: float) -> int:
    return max(1, min(99, round(conf * 100)))


class Announcer:
    def __init__(self, settings):
        self.s = settings

    # ---------------- Level 1 ----------------
    def level1(self, rec: Recognition, position: str = "", hazard: bool = False, update: bool = False) -> str:
        v, thr = self.s.verbosity, self.s.confidence_speak_threshold
        sure = f" {pct(rec.confidence)} percent sure" if self.s.speak_percent else ""
        cat = cat_phrase(rec.category)
        pos = f" {cap(position)}." if position and self.s.position_cues and v != "concise" else ""
        if rec.category == "Person":
            return f"Person, {position or 'in view'}."
        if rec.label == "unknown object" or rec.confidence <= 0:
            alts = ", or ".join(a for a, _ in rec.alternatives[:2])
            return ("I'm not sure what this is." + (f" It could be {alts}." if alts else "") +
                    " Please move it closer or show another side." + pos)
        pre = "Now I think it is " if update else ""
        if rec.confidence >= thr:
            lead = f"{pre}{cat}, {rec.label}{',' + sure if sure else ''}." if not update else f"{pre}{article(rec.label)}, {cat.lower()}{',' + sure if sure else ''}."
        elif rec.confidence >= LOW:
            lead = f"I think this is {article(rec.label)}, in {cat.lower()}{',' + sure if sure else ''}."
        else:
            names = [rec.label] + [a for a, _ in rec.alternatives[:2]]
            lead = (f"I'm not sure. It could be {', '.join(names[:-1]) + ' or ' + names[-1] if len(names) > 1 else names[0]}. "
                    f"Category: {cat.lower()}.")
        out = lead + pos
        if hazard:
            out += " Caution: take care with this item."
        if v != "concise":
            out += " Say more for details."
        return out.strip()

    # ---------------- Level 2 ----------------
    def level2(self, card: InformationCard) -> str:
        v = self.s.verbosity
        thr = self.s.confidence_speak_threshold
        out = [f"{cap(card.name)}. {cat_phrase(card.category)}, {card.subcategory.lower()}."]
        if card.confidence < thr and card.confidence > 0:
            alts = ", ".join(a for a, _ in card.alternatives[:2])
            out.append("I'm not fully sure" + (f"; it could also be {alts}." if alts else "."))
        elif self.s.speak_percent and card.confidence > 0 and card.category != "Person":
            out.append(f"{pct(card.confidence)} percent sure.")
        if card.summary and card.summary.lower().rstrip(".") != card.name.lower():
            out.append(card.summary if card.summary.endswith(".") else card.summary + ".")
        if card.used_for:
            out.append(f"Used for: {card.used_for}.")
        if card.position and self.s.position_cues:
            out.append(f"It is {card.position}.")
        if v != "concise":
            for k, label in (("colour", "Colour"), ("brand", "Brand"), ("size", "Size"), ("material", "Material")):
                if card.attributes.get(k):
                    out.append(f"{label}: {card.attributes[k]}.")
            out += self._fields(card, full=(v == "detailed"))
            if card.text_read:
                n = 5 if v == "standard" else 20
                out.append("Text I can read: " + ". ".join(card.text_read[:n]) + ".")
                if len(card.text_read) > n:
                    out.append("Say read text to hear all of it.")
        if card.ai:
            out.append(self.ai_text(card.ai))
        notes = card.safety_notes if v != "concise" else card.safety_notes[:1]
        out += [f"Caution: {n}" for n in notes]
        if v == "detailed" and card.sources:
            out.append("Sources: " + ", ".join(card.sources) + ".")
        return re.sub(r"\s+", " ", " ".join(out)).strip()

    def _fields(self, card: InformationCard, full: bool) -> list:
        f, out = card.fields, []
        if f.get("ingredients"):
            items = re.split(r",\s*(?![^()]*\))", f["ingredients"])
            out.append("Ingredients: " + ", ".join(items if full else items[:8]) + ("." if full or len(items) <= 8 else ". There are more; say ingredients for the full list."))
        if f.get("allergens_printed"):
            out.append(f"Printed allergen text: {f['allergens_printed']}.")
        if f.get("nutrition"):
            out.append("Nutrition as printed: " + ". ".join(f["nutrition"]) + ".")
        if f.get("expiry_or_best_before"):
            out.append(f"Expiry or best before: {f['expiry_or_best_before']}.")
        vm = f.get("veg_mark")
        if vm:
            conf = f", {pct(vm['confidence'])} percent sure" if vm.get("confidence") and self.s.speak_percent else ""
            out.append(f"{cap(vm['mark'])}{conf}." if "mark" in vm["mark"] or "text" in vm["mark"] else f"{cap(vm['mark'])} mark{conf}.")
        if f.get("storage"):
            out.append(f"Storage: {f['storage']}.")
        if f.get("purpose"):
            out.append(f"Purpose: {f['purpose']}.")
        if f.get("typically_operated"):
            out.append(f"How it is usually operated: {f['typically_operated']}")
        if f.get("main_parts_or_controls"):
            out.append("Main parts or controls: " + ", ".join(f["main_parts_or_controls"]) + ".")
        return out

    def ai_text(self, ai: dict) -> str:
        parts = ["The online AI adds:"]
        if ai.get("description"): parts.append(ai["description"])
        if ai.get("typical_uses"): parts.append(ai["typical_uses"])
        if ai.get("similar_items"): parts.append("Similar items include " + ", ".join(ai["similar_items"]) + ".")
        p = ai.get("approx_price_inr")
        if p:
            parts.append(f"Approximate online price: about {number_to_words(p['low'])} to {amount_to_words(Decimal(p['high']))}. "
                         "This is not a printed price, and prices vary by store.")
        if ai.get("sources") and self.s.verbosity == "detailed":
            parts.append("Prices came from " + ", ".join(ai["sources"]) + ".")
        parts.append("This is an AI estimate, so please double-check anything important.")
        return " ".join(parts)

    # ---------------- Level 3 ----------------
    def text_chunks(self, lines: Sequence[OcrLine], min_conf: float, max_chars: int = 320) -> list:
        good = [l for l in lines if l.conf >= min_conf and len(l.text.strip()) >= 1]
        if not good:
            return []
        ordered = sorted(good, key=lambda l: ((l.box[1] // 25, l.box[0]) if l.box else (0, 0)))
        text = " . ".join(l.text.strip() for l in ordered)
        chunks, cur = [], ""
        for piece in re.split(r"(?<=[.;:])\s+| \. ", text):
            if cur and len(cur) + len(piece) > max_chars:
                chunks.append(cur.strip()); cur = ""
            cur += piece + " "
        if cur.strip():
            chunks.append(cur.strip())
        low = len(lines) - len(good)
        chunks[0] = "Text I can read: " + chunks[0]
        if low > 0:
            chunks.append("Some text was too unclear to read. Move closer or add light to read the rest.")
        return chunks

    # ---------------- scene, category, help ----------------
    def category(self, rec: Recognition) -> str:
        return f"Category: {cat_phrase(rec.category).lower()}, {rec.subcategory.lower()}."

    def scene(self, items: Sequence, w: int) -> str:
        """items: [(Recognition, box)] -> objects listed left to right, grouped by category."""
        if not items:
            return "I do not see any objects. Point the camera around slowly and try again."
        items = sorted(items, key=lambda x: (x[1][0] + x[1][2]) / 2)[:8]
        groups: dict = {}
        for r, b in items:
            groups.setdefault(r.category, []).append((r, b))
        parts = []
        for cat, its in groups.items():
            names = [f"{article(r.label)} {short_position(b, w)}" for r, b in its]
            parts.append(f"{cat_phrase(cat)}: " + (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) + ".")
        return f"I can see {len(items)} object{'s' if len(items) != 1 else ''}, listed from left to right. " + " ".join(parts)

    def help(self) -> str:
        return HELP_TEXT
