"""SmartSpeak Intelligence: confidence-aware narration.

* All wording lives in TEMPLATES (no speech strings scattered through the code).
* Wording follows the evidence state: CONFIRMED / TENTATIVE / UNKNOWN.
* ObjectStabilizer requires an object to persist across detection cycles before it is announced.
* ObjectAnnouncer speaks once per appearance; a stationary object stays quiet until it leaves and returns.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from .food_label_parser import FieldStatus, FoodLabel
from .frame_quality import QualityReport
from .price_reader import KIND_PHRASE, PriceResult
from .product_identifier import ProductInfo
from decimal import Decimal

from .utils import InfoState, OcrLine, amount_to_words, number_to_words

C, S, D = "concise", "standard", "detailed"

TEMPLATES: dict[str, dict[str, str]] = {
    "obj_new": {C: "{objs}.", S: "I can see {objs}.", D: "I can see {objs}. Press Identify Item to hear about its label."},
    "id_specific_confirmed": {C: "This is {name}.", S: "This is {name}. {purpose}", D: "This is {name}. {purpose} {extra}"},
    "id_specific_database": {C: "Barcode suggests {name}.", S: "The barcode matches {name}, but I could not confirm it from the label.",
                             D: "The barcode matches {name}, but I could not confirm it from the label. Please bring the label closer."},
    "id_conflict": {C: "The barcode and label do not match.", S: "The barcode record and the label text do not match, so I cannot confirm what this is.",
                    D: "The barcode record and the label text do not match, so I cannot confirm what this is. Please ask someone you trust to check it."},
    "id_category": {C: "This appears to be {phrase}.", S: "This appears to be {phrase}. {purpose}",
                    D: "This appears to be {phrase}. {purpose} {extra}"},
    "id_object": {C: "{Obj}.", S: "I can see {obj}. Please hold it closer so I can read its label.",
                  D: "I can see {obj}. I cannot tell what the product is yet. Please hold it closer so I can read its label."},
    "id_unknown_obj": {C: "Unidentified item.", S: "I can see {obj}, but I cannot identify it confidently. Please move it closer to the camera.",
                       D: "I can see {obj}, but I cannot identify it confidently. Please move it closer to the camera and make sure there is good light."},
    "id_nothing": {C: "Nothing clear.", S: "I do not see an item clearly. Please point the camera at it.",
                   D: "I do not see an item clearly. Please point the camera at it and hold it steady."},
    "text_hint": {C: "", S: "I can see text on the label, but I need a clearer view to confirm the product details.", D: "I can see text on the label, but I need a clearer view to confirm the product details."},
    "price_confirmed": {C: "{kind} {words}.", S: "The label clearly shows the {kind} as {words}.", D: "The label clearly shows the {kind} as {words}. The text I read was: {raw}."},
    "price_tentative": {C: "Maybe {words}.", S: "I think the {kind} is {words}, but I am not fully sure. Please move the camera closer to confirm.",
                        D: "I think the {kind} is {words}, but I am not fully sure. The text I read was: {raw}. Please move the camera closer to confirm."},
    "price_conflict": {C: "More than one price seen.", S: "I can see more than one price: {list}. Please move the camera closer to the one you want.",
                       D: "I can see more than one price: {list}. Please move the camera closer to the one you want."},
    "price_number_only": {C: "Number seen, not a price.", S: "I can see a number on the label, but I cannot confirm whether it is the price. Please move the camera closer.",
                          D: "I can see a number on the label, but I cannot confirm whether it is the price. Please move the camera closer."},
    "price_none": {C: "No price readable.", S: "I cannot read a price. Please move the camera closer or improve the lighting.",
                   D: "I cannot read a price. Please move the camera closer, hold it steady, or improve the lighting."},
    "ing_present": {C: "Ingredients: {text}", S: "The ingredients read: {text}", D: "The ingredients read: {text}"},
    "ing_more": {C: "There are {n} more.", S: "There are {n} more ingredients. Say read the full ingredients to hear them all.",
                 D: "There are {n} more ingredients. Say read the full ingredients to hear them all."},
    "ing_unreadable": {C: "Ingredients unreadable.", S: "I can see an ingredient list but cannot read it clearly. Please move the package closer or improve the lighting.",
                       D: "I can see an ingredient list but cannot read it clearly. Please move the package closer, hold it steady, or improve the lighting."},
    "ing_missing": {C: "No ingredients in view.", S: "I cannot find an ingredient list in view. Try turning the package around.",
                    D: "I cannot find an ingredient list in view. Try turning the package around, then ask again."},
    "ocr_uncertain": {C: "Some words may be wrong.", S: "Some of the text may be misread.", D: "Some of the text may be misread, so please double-check anything important."},
    "allergen": {C: "Allergen text: {text}.", S: "The allergen statement reads: {text}.", D: "The allergen statement reads: {text}. I only report what is printed."},
    "expiry_present": {C: "Expiry: {value}.", S: "The expiry text reads: {value}.", D: "The expiry text reads: {value}. Please verify dates yourself or ask someone to confirm."},
    "expiry_mfg": {C: "", S: "", D: "The manufacturing text reads: {value}."},
    "expiry_unreadable": {C: "Expiry unreadable.", S: "I can see an expiry or best-before mark, but I cannot read it clearly. Please move the package closer.",
                          D: "I can see an expiry or best-before mark, but I cannot read it clearly. Please move the package closer and improve the lighting."},
    "expiry_missing": {C: "No expiry seen.", S: "I cannot find an expiry date in view. Try turning the package around.",
                       D: "I cannot find an expiry date in view. Try turning the package around, or check the lid and base."},
    "label_text": {C: "{text}", S: "The label reads: {text}", D: "The label reads: {text}"},
    "label_none": {C: "No text readable.", S: "I cannot read any text. Please move the item closer or improve the lighting.",
                   D: "I cannot read any text. Please move the item closer, hold it steady, or improve the lighting."},
    "q_dark": {C: "Too dark.", S: "It is too dark to read. Please improve the lighting.", D: "It is too dark to read. Please improve the lighting or move to a brighter place."},
    "q_blur": {C: "Image blurry.", S: "The image is blurry. Please hold the item steady.", D: "The image is blurry. Please hold the item steady and a little further back, then try again."},
    "q_bad": {C: "No usable image.", S: "I cannot get a usable image from the camera.", D: "I cannot get a usable image from the camera. Please check that it is connected and not covered."},
    "scan_again": {C: "Scanning again.", S: "Scanning again.", D: "Scanning again. I will describe what I see in a moment."},
    "camera_error": {C: "Camera problem.", S: "I could not use the camera. Please check that it is connected and not in use by another program.",
                     D: "I could not use the camera. Please check that it is connected and not in use by another program, then press Start Camera."},
    "started": {C: "Ready.", S: "SmartSight is ready. The camera is on.", D: "SmartSight is ready. The camera is on. Use the buttons or say what is this, read the price, or read the ingredients."},
    "stopped": {C: "Camera off.", S: "The camera is off.", D: "The camera is off and has been released."},
    "cloud_off": {C: "Online AI is off.", S: "Online AI is turned off. You can enable it in settings.", D: "Online AI is turned off. You can enable it in the settings panel, where you will also be asked for permission."},
    "cloud_consent": {C: "Permission needed.", S: "Online AI needs your permission to send a photo. Please tick the consent box in settings.", D: "Online AI needs your permission to send a photo. Please tick the consent box in settings. Nothing has been sent."},
    "cloud_nokey": {C: "No AI key.", S: "Online AI is not set up because no API key was found.", D: "Online AI is not set up because no API key was found. Nothing has been sent."},
    "cloud_notice": {C: "Sending photo.", S: "Sending one photo to the online AI now. This may take a few seconds.", D: "Sending one photo to the online AI service now. It is not saved on this computer. This may take a few seconds."},
    "cloud_error": {C: "AI unreachable.", S: "I could not reach the online AI. Please try again later.", D: "I could not reach the online AI. Please check the internet connection and try again."},
    "cloud_auth": {C: "AI key rejected.", S: "The online AI key was not accepted.", D: "The online AI key was not accepted. Please check the key."},
    "cloud_rate": {C: "AI busy.", S: "The online AI is busy. Please wait a minute and try again.", D: "The online AI is busy. Please wait a minute and try again."},
    "cloud_unknown": {C: "AI could not tell.", S: "The online AI could not tell what this is either. Please move the item closer or show another side.", D: "The online AI could not tell what this is either. Please move the item closer, show another side, or improve the lighting."},
    "cloud_ok": {C: "{name}. {price_short}", S: "The online AI thinks this is {name}. {description} {uses} {similar} {price} This is an AI estimate, so please double-check anything important.",
                 D: "The online AI thinks this is {name}. {description} {uses} {similar} {price} {basis} {sources} This is an AI estimate from one photo, so please double-check anything important."},
    "cloud_similar": {C: "", S: "Similar items include {items}.", D: "Similar items include {items}."},
    "cloud_sources": {C: "", S: "", D: "The price information came from {hosts}."},
    "cloud_wait": {C: "Looking online.", S: "Looking that up online.", D: "Looking that up online. This can take a few seconds."},
    "cloud_waiting_consent": {C: "AI needs permission.", S: "Online AI is waiting for your permission to send photos. Please agree in the settings panel.",
                              D: "Online AI is waiting for your permission to send photos. Until you agree, SmartSight uses only the local models. Please agree in the settings panel."},
    "cloud_price": {C: "", S: "Approximate online price: about {rng}. This is not a printed price, and prices vary by store.", D: "Approximate online price: about {rng}. This is not a price printed on the item, and it varies by store and over time."},
    "cloud_price_short": {C: "About {rng} online.", S: "", D: ""},
    "cloud_noprice": {C: "", S: "I could not estimate a price.", D: "I could not estimate a price for this item."},
    "no_detector": {C: "", S: "Object detection is not available, but I can still read labels.", D: "Object detection is not available, so I can only read text and barcodes."},
}

_PLURALS = {"knife": "knives", "person": "people", "sheep": "sheep", "scissors": "scissors"}
_FRIENDLY = {"tv": "television", "cell phone": "mobile phone", "remote": "remote control", "laptop": "laptop",
             "wine glass": "glass", "potted plant": "potted plant"}


@dataclass
class Narration:
    text: str
    state: InfoState
    kind: str = ""


def tpl(key: str, verbosity: str, **kw) -> str:
    t = TEMPLATES[key].get(verbosity) or TEMPLATES[key][S]
    kw.setdefault("extra", "")
    out = t.format(**kw)
    return re.sub(r"\s+", " ", out).strip()


def friendly(label: str) -> str:
    return _FRIENDLY.get(label, label)


def article(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def plural(noun: str) -> str:
    return _PLURALS.get(noun) or (noun + "es" if noun.endswith(("s", "x", "ch", "sh")) else noun + "s")


def describe_objects(counts: dict) -> str:
    parts = []
    for label, n in counts.items():
        name = friendly(label)
        parts.append(article(name) if n == 1 else f"{n} {plural(name)}")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1] if parts else ""


class ObjectStabilizer:
    """A label becomes stable after `n_frames` consecutive detection cycles (tolerating `misses` gaps)."""
    def __init__(self, n_frames: int = 4, misses: int = 1):
        self.n, self.misses = max(1, n_frames), misses
        self._hits: dict[str, int] = {}
        self._gap: dict[str, int] = {}

    def update(self, labels: Iterable[str]) -> dict:
        counts: dict[str, int] = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        for l in counts:
            self._hits[l] = self._hits.get(l, 0) + 1
            self._gap[l] = 0
        for l in list(self._hits):
            if l not in counts:
                self._gap[l] += 1
                if self._gap[l] > self.misses:
                    del self._hits[l], self._gap[l]
        return {l: c for l, c in counts.items() if self._hits.get(l, 0) >= self.n}

    def reset(self) -> None:
        self._hits.clear(); self._gap.clear()


class ObjectAnnouncer:
    """Announce each object once per appearance. Re-announce only after it has been absent for `absence_seconds`."""
    def __init__(self, absence_seconds: float = 3.0, clock: Callable[[], float] = time.monotonic):
        self.absence, self._clock = absence_seconds, clock
        self._announced: dict[str, float] = {}   # label -> last time seen

    def update(self, stable: dict) -> list[str]:
        now = self._clock()
        new = []
        for label in stable:
            last = self._announced.get(label)
            if last is None or now - last > self.absence:
                new.append(label)
            self._announced[label] = now
        return new

    def reset(self) -> None:
        self._announced.clear()


class NarrationEngine:
    def __init__(self, verbosity: str = S, ingredient_limits: Optional[dict] = None):
        self.verbosity = verbosity
        self.limits = ingredient_limits or {C: 5, S: 8, D: 15}

    def _t(self, key: str, **kw) -> str:
        return tpl(key, self.verbosity, **kw)

    def new_objects(self, counts: dict) -> Optional[Narration]:
        if not counts:
            return None
        return Narration(self._t("obj_new", objs=describe_objects(counts)), InfoState.TENTATIVE, "object")

    def identify(self, p: ProductInfo, describe: bool = False) -> Narration:
        v = D if describe else self.verbosity
        t = lambda k, **kw: tpl(k, v, **kw)
        obj = article(friendly(p.object_label)) if p.object_label else ""
        hint = t("text_hint") if p.text_found and p.level in ("category", "object", "unknown") else ""
        if p.conflict:
            return Narration(t("id_conflict"), InfoState.UNKNOWN, "identify")
        if p.level == "specific" and p.state == InfoState.CONFIRMED:
            extra = p.caution or ""
            return Narration(t("id_specific_confirmed", name=p.name, purpose=p.purpose or "", extra=extra), p.state, "identify")
        if p.level == "specific":
            return Narration(t("id_specific_database", name=p.name), InfoState.TENTATIVE, "identify")
        if p.level == "category":
            phrase = f"{article(p.category + ' ' + friendly(p.object_label))}" if p.object_label else p.category
            text = t("id_category", phrase=phrase, purpose=p.purpose or "", extra=p.caution or "")
            if v == D and p.name_guess:
                text += f" The most prominent text reads: {p.name_guess}."
            return Narration(text, InfoState.TENTATIVE, "identify")
        if p.level == "object":
            return Narration((t("id_object", obj=obj, Obj=friendly(p.object_label).capitalize()) + " " + hint).strip(),
                             InfoState.TENTATIVE, "identify")
        if p.object_label:
            return Narration(t("id_unknown_obj", obj=obj), InfoState.UNKNOWN, "identify")
        return Narration(t("id_nothing"), InfoState.UNKNOWN, "identify")

    def price(self, r: PriceResult) -> Narration:
        if r.conflict:
            parts = [f"{KIND_PHRASE[c.kind]} {amount_to_words(c.amount)}" for c in r.candidates if c.kind != "unit"]
            return Narration(self._t("price_conflict", list=", ".join(parts)), InfoState.TENTATIVE, "price")
        if r.best:
            b = r.best
            key = "price_confirmed" if b.state == InfoState.CONFIRMED else "price_tentative"
            return Narration(self._t(key, kind=KIND_PHRASE[b.kind], words=amount_to_words(b.amount), raw=b.raw), b.state, "price")
        if r.saw_numbers:
            return Narration(self._t("price_number_only"), InfoState.UNKNOWN, "price")
        return Narration(self._t("price_none"), InfoState.UNKNOWN, "price")

    def ingredients(self, label: FoodLabel, full: bool = False, min_conf: float = 0.6) -> Narration:
        f = label.get("ingredients")
        if f.status == FieldStatus.UNREADABLE:
            return Narration(self._t("ing_unreadable"), InfoState.UNKNOWN, "ingredients")
        if f.status == FieldStatus.MISSING:
            return Narration(self._t("ing_missing"), InfoState.UNKNOWN, "ingredients")
        items = label.ingredient_items
        limit = len(items) if full else self.limits[self.verbosity]
        text = self._t("ing_present", text=", ".join(items[:limit]) + ".")
        if len(items) > limit:
            text += " " + self._t("ing_more", n=len(items) - limit)
        low_conf = label.mean_confidence is not None and label.mean_confidence < min_conf
        if low_conf:
            text += " " + self._t("ocr_uncertain")
        al = label.get("allergens")
        if al.status == FieldStatus.PRESENT and self.verbosity != C:
            text += " " + self._t("allergen", text=al.value)
        return Narration(text, InfoState.TENTATIVE if low_conf else InfoState.CONFIRMED, "ingredients")

    def expiry(self, label: FoodLabel) -> Narration:
        f = label.get("expiry")
        if f.status == FieldStatus.PRESENT:
            text = self._t("expiry_present", value=f.value)
            m = label.get("manufacture")
            if m.status == FieldStatus.PRESENT and self.verbosity == D:
                text += " " + self._t("expiry_mfg", value=m.value)
            return Narration(text, InfoState.CONFIRMED, "expiry")
        key = "expiry_unreadable" if f.status == FieldStatus.UNREADABLE else "expiry_missing"
        return Narration(self._t(key), InfoState.UNKNOWN, "expiry")

    def label(self, lines: Sequence[OcrLine], min_conf: float = 0.6, max_chars: int = 400) -> Narration:
        good = [l.text.strip() for l in lines if l.conf >= min_conf and len(l.text.strip()) >= 2]
        if not good:
            return Narration(self._t("label_none"), InfoState.UNKNOWN, "label")
        text = " ".join(good)
        if len(text) > max_chars and self.verbosity != D:
            text = text[:max_chars].rsplit(" ", 1)[0]
        dropped = len(lines) - len(good)
        state = InfoState.TENTATIVE if dropped > len(lines) / 2 else InfoState.CONFIRMED
        return Narration(self._t("label_text", text=text), state, "label")

    def cloud(self, res) -> Narration:
        key = {"no_key": "cloud_nokey", "unauthorized": "cloud_auth", "rate_limited": "cloud_rate",
               "network_error": "cloud_error", "invalid_response": "cloud_error", "error": "cloud_error"}.get(res.status)
        if key:
            return Narration(self._t(key), InfoState.UNKNOWN, "cloud")
        d = res.data
        if not d["can_identify"]:
            return Narration(self._t("cloud_unknown"), InfoState.UNKNOWN, "cloud")
        name = d["item"]
        if d["brand"] and d["brand"].lower() not in name.lower():
            name = f"{d['brand']} {name}"
        pr = d["price"]
        if pr:
            rng = f"{number_to_words(pr['low'])} to {amount_to_words(Decimal(pr['high']))}"
            price, short = self._t("cloud_price", rng=rng), self._t("cloud_price_short", rng=rng)
            basis = f"Basis: {pr['basis']}." if pr["basis"] else ""
        else:
            price, short, basis = self._t("cloud_noprice"), "", ""
        similar = self._t("cloud_similar", items=", ".join(d["similar_items"])) if d.get("similar_items") else ""
        sources = self._t("cloud_sources", hosts=", ".join(d["sources"])) if d.get("sources") else ""
        text = self._t("cloud_ok", name=name, description=d["description"], uses=d["typical_uses"],
                       price=price, price_short=short, basis=basis, similar=similar, sources=sources)
        return Narration(text, InfoState.TENTATIVE, "cloud")

    def quality(self, q: QualityReport) -> Narration:
        key = "q_dark" if "dark" in q.issues else "q_blur" if "blurry" in q.issues else "q_bad"
        return Narration(self._t(key), InfoState.UNKNOWN, "quality")

    def simple(self, key: str) -> Narration:
        return Narration(self._t(key), InfoState.UNKNOWN, key)
