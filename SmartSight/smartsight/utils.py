"""Shared types and helpers (no heavy dependencies)."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union


class InfoState(Enum):
    """How well the evidence supports a statement. No fake numeric probabilities."""
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {"confirmed": "Confirmed", "tentative": "Tentative", "unknown": "Unreadable or unknown"}[self.value]

    @property
    def indicator(self) -> str:
        """Text + symbol so state is never conveyed by colour alone."""
        return {"confirmed": "[OK] Confirmed", "tentative": "[?] Tentative",
                "unknown": "[X] Unreadable or unknown"}[self.value]


@dataclass(frozen=True)
class OcrLine:
    text: str
    conf: float = 1.0
    box: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2

    @property
    def height(self) -> int:
        return (self.box[3] - self.box[1]) if self.box else 0


TextSource = Union[str, Sequence[OcrLine]]


def as_lines(source: TextSource) -> list[OcrLine]:
    if isinstance(source, str):
        return [OcrLine(t) for t in source.splitlines() if t.strip()]
    return [l for l in source if l.text.strip()]


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


def _under_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + (f" {_ONES[o]}" if o else "")


def _under_1000(n: int) -> str:
    h, r = divmod(n, 100)
    parts = []
    if h:
        parts.append(f"{_ONES[h]} hundred")
    if r:
        if h:
            parts.append("and")
        parts.append(_under_100(r))
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """Indian numbering: thousand, lakh, crore."""
    if n == 0:
        return "zero"
    parts = []
    for div, name in ((10_000_000, "crore"), (100_000, "lakh"), (1000, "thousand")):
        q, n = divmod(n, div)
        if q:
            parts.append(f"{number_to_words(q)} {name}")
    if n:
        parts.append(_under_1000(n))
    return " ".join(parts)


def amount_to_words(amount: Decimal) -> str:
    rupees = int(amount)
    paise = int(((amount - rupees) * 100).to_integral_value())
    s = f"{number_to_words(rupees)} {'rupee' if rupees == 1 else 'rupees'}"
    if paise:
        s += f" and {number_to_words(paise)} paise"
    return s
