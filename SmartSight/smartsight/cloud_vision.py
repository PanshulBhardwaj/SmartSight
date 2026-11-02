"""OPTIONAL online AI vision (Anthropic Messages API) for items the local models cannot identify.

PRIVACY: a downscaled JPEG of ONE frame is sent only when the user presses "Ask AI" AND has enabled
cloud AI AND given consent in settings. Nothing is saved locally. The API key is read from the
ANTHROPIC_API_KEY environment variable, never from source code.
Results are always TENTATIVE: a model's description and price estimate are not verified facts, and an
online price is never the printed price on the item.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You help a blind or low-vision user understand an everyday item shown in a photo. "
    "Describe ONLY what is visible. Never invent a brand, model or variant: give a brand only if it is "
    "legibly printed. If you cannot tell what the item is, say so (can_identify=false). "
    "Do not identify or describe people. No medical advice and no claims that food is safe. "
    "Give an APPROXIMATE typical retail price range in Indian rupees for this kind of item (or the specific product "
    "if clearly identified), with a short basis, or null if you cannot estimate. "
    "Reply with ONLY one JSON object with keys: can_identify (bool), item (short name), brand (string or null), "
    "category (string), description (one or two plain sentences), typical_uses (one short sentence), "
    'similar_items (up to 3 short names of similar products), '
    'approx_price_inr ({"low": number, "high": number, "basis": string} or null).'
)


WEB_ADDENDUM = (" You have a web search tool: use it to check typical CURRENT retail prices in India (in rupees) and to "
                "find similar products, and base approx_price_inr on what you find. Still return only the one JSON object.")


def frame_key(frame) -> str:
    """Cheap signature so an unchanged scene is not sent twice."""
    import hashlib

    import cv2
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)) // 16
    return hashlib.md5(small.tobytes()).hexdigest()


@dataclass
class CloudResult:
    status: str            # ok | no_key | unauthorized | rate_limited | network_error | invalid_response | error
    data: Optional[dict] = None
    message: str = ""


def encode_frame(frame, max_side: int = 1024, quality: int = 80) -> bytes:
    import cv2
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode frame")
    return buf.tobytes()


def _default_transport(url: str, headers: dict, body: bytes, timeout: float):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _json_from_text(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(text[a:b + 1])
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < 10_000_000 else None


def normalize(d: dict) -> Optional[dict]:
    """Validate and clean the model's JSON. Returns None if unusable."""
    if not isinstance(d, dict) or "can_identify" not in d:
        return None
    out = {"can_identify": bool(d.get("can_identify")),
           "item": str(d.get("item") or "").strip(), "brand": (str(d["brand"]).strip() or None) if d.get("brand") else None,
           "category": str(d.get("category") or "").strip(), "description": str(d.get("description") or "").strip(),
           "typical_uses": str(d.get("typical_uses") or "").strip(), "price": None,
           "similar_items": [str(x).strip() for x in (d.get("similar_items") or []) if str(x).strip()][:3]
           if isinstance(d.get("similar_items"), list) else [], "sources": []}
    p = d.get("approx_price_inr")
    if isinstance(p, dict):
        lo, hi = _num(p.get("low")), _num(p.get("high"))
        if lo and hi:
            lo, hi = min(lo, hi), max(lo, hi)
            if hi <= lo * 20:   # a 20x range is not an estimate
                out["price"] = {"low": round(lo), "high": round(hi), "basis": str(p.get("basis") or "").strip()}
    if out["can_identify"] and not out["item"]:
        out["can_identify"] = False
    return out


class CloudVision:
    """Online / local AI description. Provider is chosen by the model name in config (cloud_model):
         claude-...            -> Anthropic (paid)
         gemini-...            -> Google Gemini API (has a FREE tier; key in GEMINI_API_KEY)
         ollama:<model>        -> Ollama on this PC (FREE, offline, private; no key; no web search)
    """
    ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: Optional[str], model: str = "claude-sonnet-5-5", web_search: bool = False,
                 transport: Optional[Callable] = None, timeout: float = 40.0):
        self.model, self.web_search = model, web_search
        self._api_key = api_key
        self._transport, self.timeout = transport or _default_transport, timeout

    # -- provider --------------------------------------------------------
    @property
    def provider(self) -> str:
        m = (self.model or "").strip().lower()
        return "ollama" if m.startswith("ollama:") else "gemini" if m.startswith("gemini") else "anthropic"

    @property
    def api_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        if self.provider == "gemini":
            import os
            return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return None

    @api_key.setter
    def api_key(self, v) -> None:
        self._api_key = v

    @property
    def configured(self) -> bool:
        return self.provider == "ollama" or bool(self.api_key)

    @property
    def URL(self) -> str:                      # kept for backwards compatibility
        return self.ANTHROPIC_URL

    # -- request builders ------------------------------------------------
    def _prompt(self, hints) -> str:
        return ("What is this item, what is it used for, what are similar items, and roughly what does it cost?"
                + (f" A local detector guessed: {', '.join(hints)} (unverified, may be wrong)." if hints else ""))

    def _build(self, jpeg: bytes, hints):
        system = SYSTEM_PROMPT + (WEB_ADDENDUM if self.web_search and self.provider != "ollama" else "")
        b64 = base64.b64encode(jpeg).decode()
        if self.provider == "gemini":
            body = {"system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                                                            {"text": self._prompt(hints)}]}],
                    "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.2}}
            if self.web_search:
                body["tools"] = [{"google_search": {}}]          # grounding with Google Search
            headers = {"content-type": "application/json", "x-goog-api-key": self.api_key or ""}
            return self.GEMINI_URL.format(model=self.model.strip()), headers, body
        if self.provider == "ollama":
            import os
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
            if not host.startswith("http"):
                host = "http://" + host
            body = {"model": self.model.split(":", 1)[1], "stream": False, "options": {"temperature": 0.2},
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": self._prompt(hints), "images": [b64]}]}
            return host + "/api/chat", {"content-type": "application/json"}, body
        body = {"model": self.model, "max_tokens": 1200, "system": system,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": self._prompt(hints)}]}]}
        if self.web_search:
            body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
        return self.ANTHROPIC_URL, {"content-type": "application/json", "x-api-key": self.api_key or "",
                                    "anthropic-version": "2023-06-01"}, body

    # -- response parsers: -> (texts, source_domains) ------------------------
    def _parse(self, raw: bytes):
        d = json.loads(raw)
        if self.provider == "gemini":
            cand = d["candidates"][0]
            texts = [p["text"] for p in cand["content"]["parts"] if isinstance(p, dict) and p.get("text")]
            chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
            from urllib.parse import urlparse
            hosts = []
            for c in chunks:
                title = (c.get("web") or {}).get("title", "")          # Gemini redirects through vertexaisearch; the title is the site
                host = title.lower().removeprefix("www.") if "." in title and " " not in title else ""
                if host and host not in hosts:
                    hosts.append(host)
            return texts, hosts[:3]
        if self.provider == "ollama":
            return [d["message"]["content"]], []
        blocks = d["content"]
        return [b.get("text", "") for b in blocks if b.get("type") == "text"], _sources(blocks)

    def describe(self, jpeg: bytes, hints=None) -> CloudResult:
        if not self.configured:
            return CloudResult("no_key", message="no API key set (GEMINI_API_KEY for Gemini, ANTHROPIC_API_KEY for Claude)")
        url, headers, body = self._build(jpeg, hints)
        try:
            status, raw = self._transport(url, headers, json.dumps(body).encode(), self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            hint = " Is Ollama running? Start it and run: ollama pull <model>" if self.provider == "ollama" else ""
            return CloudResult("network_error", message=str(exc) + hint)
        if status in (401, 403) or (status == 400 and b"API key" in (raw or b"")):
            return CloudResult("unauthorized")
        if status == 429:
            return CloudResult("rate_limited")
        if status != 200:
            return CloudResult("error", message=f"HTTP {status}")
        try:
            texts, sources = self._parse(raw)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return CloudResult("invalid_response")
        for t in reversed(texts):
            data = normalize(_json_from_text(t) or {})
            if data:
                data["sources"] = sources
                return CloudResult("ok", data)
        return CloudResult("invalid_response")


def _sources(blocks) -> list:
    """Domains the web search touched (for 'prices came from ...'). Defensive: formats may vary."""
    from urllib.parse import urlparse
    out: list = []
    for b in blocks if isinstance(blocks, list) else []:
        if isinstance(b, dict) and b.get("type") == "web_search_tool_result" and isinstance(b.get("content"), list):
            for item in b["content"]:
                url = item.get("url") if isinstance(item, dict) else None
                host = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
                if host and host not in out:
                    out.append(host)
    return out[:3]
