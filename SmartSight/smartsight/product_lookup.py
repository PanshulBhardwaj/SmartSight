"""Product records: local user-defined store and an OPT-IN Open Food Facts lookup.

Only the barcode number is ever sent over the network, never images, and only when
`online_lookup` is enabled. Results are cached locally.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class ProductRecord:
    barcode: str
    name: str
    brand: str = ""
    category: str = ""
    purpose: str = ""
    source: str = "local"   # local | openfoodfacts


@dataclass
class LookupResult:
    status: str             # found | not_found | network_error | rate_limited | invalid_response | cached
    record: Optional[ProductRecord] = None


class LocalProductStore:
    """JSON list of user-defined products, e.g. [{"barcode": "...", "name": "...", "brand": "..."}]."""
    def __init__(self, path: Optional[Path] = None):
        self._by_code: dict[str, ProductRecord] = {}
        if path and Path(path).exists():
            try:
                for d in json.loads(Path(path).read_text(encoding="utf-8")):
                    self._by_code[str(d["barcode"])] = ProductRecord(
                        str(d["barcode"]), d["name"], d.get("brand", ""), d.get("category", ""),
                        d.get("purpose", ""), "local")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                log.warning("Could not load local products from %s: %s", path, exc)

    def get(self, barcode: str) -> Optional[ProductRecord]:
        return self._by_code.get(str(barcode))

    def add(self, record: ProductRecord) -> None:
        self._by_code[record.barcode] = record


def _default_fetch(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SmartSight/0.1 (assistive prototype)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class OpenFoodFactsLookup:
    URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json?fields=product_name,brands,categories"

    def __init__(self, cache_path: Optional[Path] = None, fetcher: Optional[Callable[[str, float], bytes]] = None,
                 timeout: float = 5.0, backoff_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self._cache_path = Path(cache_path) if cache_path else None
        self._fetch = fetcher or _default_fetch
        self._timeout, self._backoff, self._clock = timeout, backoff_seconds, clock
        self._blocked_until = 0.0
        self._cache: dict[str, dict] = {}
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}

    def fetch(self, barcode: str) -> LookupResult:
        code = "".join(ch for ch in str(barcode) if ch.isdigit())
        if not 6 <= len(code) <= 14:
            return LookupResult("invalid_response")
        if code in self._cache:
            return LookupResult("cached", ProductRecord(**self._cache[code]))
        if self._clock() < self._blocked_until:
            return LookupResult("rate_limited")
        try:
            raw = self._fetch(self.URL.format(code=code), self._timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                self._blocked_until = self._clock() + self._backoff
                return LookupResult("rate_limited")
            if exc.code == 404:
                return LookupResult("not_found")
            return LookupResult("network_error")
        except (urllib.error.URLError, TimeoutError, OSError):
            return LookupResult("network_error")
        try:
            data = json.loads(raw)
            if data.get("status") != 1:
                return LookupResult("not_found")
            p = data["product"]
            name = (p.get("product_name") or "").strip()
            if not name:
                return LookupResult("not_found")
            rec = ProductRecord(code, name, (p.get("brands") or "").split(",")[0].strip(),
                                (p.get("categories") or "").split(",")[0].strip(), "", "openfoodfacts")
        except (ValueError, KeyError, TypeError, AttributeError):
            return LookupResult("invalid_response")
        self._cache[code] = rec.__dict__.copy()
        self._save()
        return LookupResult("found", rec)

    def _save(self) -> None:
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except OSError as exc:
            log.warning("Could not write lookup cache: %s", exc)
