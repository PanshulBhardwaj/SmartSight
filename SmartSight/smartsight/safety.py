"""Safety rules. SmartSight never says food/medicine is 'safe', never says 'allergen-free', never gives doses from AI inference."""
from __future__ import annotations

import re

CHECK_LABEL = "Check the label or ask a pharmacist."
AI_REMOVED_NOTE = "Some AI details were left out because they should not be relied on. Check the label or ask a pharmacist."
EXPIRY_NOTE = "I may misread dates. Please check the expiry date yourself or ask someone to confirm."
ALLERGEN_NOTE = "I only read allergen text that is printed. If allergies matter, check the whole label yourself."

# sentences matching any of these are removed from AI-sourced text
_FORBIDDEN = [
    r"\b(is|are|be|it's|looks|appears|seems)\s+(completely\s+|totally\s+|perfectly\s+|generally\s+|probably\s+)?safe\b",
    r"\bsafe\s+(to|for)\s+(eat|drink|consume|take|use|ingest|children|kids|babies|pregnan)",
    r"\b(allergen|allergy|nut|gluten|dairy|lactose)[\s-]*free\b",
    r"\bnon[\s-]?toxic\b",
    r"\b(dose|dosage|posology)\b",
    r"\btake\s+\d+",
    r"\b\d+(\.\d+)?\s?(mg|mcg|µg|ml|g|tablets?|pills?|capsules?|drops?)\b[^.]*\b(daily|per day|twice|once|every|times|a day|hours?)\b",
    r"\b(cures?|treats?|prevents?)\s+\w+",
    r"\bno\s+(side effects|allergens)\b",
]
_FORBIDDEN_RE = [re.compile(p, re.I) for p in _FORBIDDEN]
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def sanitize_ai_text(text: str) -> tuple[str, bool]:
    """Drop sentences that make safety, allergen, dosage or treatment claims. Returns (clean_text, something_removed)."""
    if not text:
        return "", False
    kept, removed = [], False
    for sent in _SENTENCE.split(text.strip()):
        if any(rx.search(sent) for rx in _FORBIDDEN_RE):
            removed = True
        else:
            kept.append(sent)
    return " ".join(kept).strip(), removed


def violates(text: str) -> bool:
    """True if a finished card/announcement text contains a forbidden claim (used by tests as a guard)."""
    # claims of safety / allergen-freeness / non-toxicity, or a dose with a quantity. Merely *mentioning* "dose" (e.g. "I cannot give doses") is fine.
    return any(rx.search(text) for rx in _FORBIDDEN_RE[:4] + [_FORBIDDEN_RE[6]])
