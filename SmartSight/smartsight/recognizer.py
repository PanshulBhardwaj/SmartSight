"""Tier 2: open-vocabulary (zero-shot) recognition. The model is swappable via the Embedder protocol.

Score = softmax over the whole vocabulary of (100 x cosine similarity). It is an UNCALIBRATED model score,
not a probability of being correct.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from .taxonomy import Vocabulary

log = logging.getLogger(__name__)
TEMPLATES = ("a photo of a {}.", "a close-up photo of a {}.", "a {} held in a hand.")


@dataclass(frozen=True)
class Candidate:
    label: str
    score: float
    category: str
    subcategory: str


class Embedder(Protocol):
    def encode_images(self, images_bgr: Sequence[np.ndarray]) -> np.ndarray: ...
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class OpenClipEmbedder:
    """OpenCLIP backend (pip install -r requirements-recognition.txt). Weights download on first use."""
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
        import open_clip
        import torch
        self._torch, self._device = torch, device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.key = f"{model_name}:{pretrained}"

    def encode_images(self, images_bgr):
        from PIL import Image
        batch = self._torch.stack([self.preprocess(Image.fromarray(im[:, :, ::-1])) for im in images_bgr]).to(self._device)
        with self._torch.no_grad():
            f = self.model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts):
        tok = self.tokenizer(list(texts)).to(self._device)
        with self._torch.no_grad():
            f = self.model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)


class ClipRecognizer:
    def __init__(self, embedder, vocab: Vocabulary, cache_dir: Optional[Path] = None, cache_key: str = "", scale: float = 100.0,
                 templates=TEMPLATES, name: str = "zero-shot (OpenCLIP)"):
        self.embedder, self.vocab, self.scale, self.templates, self.name = embedder, vocab, scale, templates, name
        self._cache_dir, self._cache_key = cache_dir, cache_key or getattr(embedder, "key", "embedder")
        self._text: Optional[np.ndarray] = None

    def _text_matrix(self) -> np.ndarray:
        if self._text is not None:
            return self._text
        h = hashlib.md5((self._cache_key + self.vocab.content_hash() + "|".join(self.templates)).encode()).hexdigest()[:16]
        path = Path(self._cache_dir) / f"clip_text_{h}.npy" if self._cache_dir else None
        if path and path.exists():
            m = np.load(path)
            if m.shape[0] == len(self.vocab.entries):
                self._text = m
                return m
        rows, spans, prompts = [], [], []
        for e in self.vocab.entries:
            ps = [t.format(n) for n in (e.name, *e.aliases) for t in self.templates]
            spans.append((len(prompts), len(prompts) + len(ps)))
            prompts += ps
        feats = self.embedder.encode_texts(prompts)
        for a, b in spans:
            v = feats[a:b].mean(axis=0)
            rows.append(v / (np.linalg.norm(v) or 1.0))
        self._text = np.stack(rows).astype(np.float32)
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self._text)
            except OSError as exc:
                log.warning("could not cache text embeddings: %s", exc)
        return self._text

    def classify(self, crop, top_k: int = 3) -> list[Candidate]:
        if not self.vocab.entries or crop is None or getattr(crop, "size", 0) == 0:
            return []
        img = self.embedder.encode_images([crop])[0]
        logits = self.scale * (self._text_matrix() @ img)
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        out = []
        for i in np.argsort(-p)[:top_k]:
            e = self.vocab.entries[int(i)]
            out.append(Candidate(e.name, float(p[i]), e.category, e.subcategory))
        return out


class NullRecognizer:
    name = "none"
    def classify(self, crop, top_k: int = 3):
        return []


def make_recognizer(settings, vocab: Vocabulary):
    """-> (recognizer | None, status_message). Never raises."""
    if not settings.zero_shot_enabled:
        return None, "zero-shot recognition disabled in config"
    if not vocab.entries:
        return None, f"zero-shot recognition unavailable: vocabulary file not found ({settings.vocabulary_path})"
    try:
        device = "cpu"
        if settings.gpu:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        emb = OpenClipEmbedder(settings.zero_shot_model, settings.zero_shot_pretrained, device)
        rec = ClipRecognizer(emb, vocab, settings.path("vocabulary_path").parent / "cache")
        return rec, f"zero-shot OpenCLIP {settings.zero_shot_model} on {device}"
    except ImportError:
        return None, "zero-shot recognition unavailable (install: pip install -r requirements-recognition.txt); using detector labels only"
    except Exception as exc:  # noqa: BLE001  (download failure, bad model name...)
        return None, f"zero-shot recognition failed to load ({exc}); using detector labels only"


def crop_box(frame, box, margin: float = 0.1):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    c = frame[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
    return c.copy() if c.size else None


def center_box(frame, frac: float = 0.6):
    h, w = frame.shape[:2]
    return (int(w * (1 - frac) / 2), int(h * (1 - frac) / 2), int(w * (1 + frac) / 2), int(h * (1 + frac) / 2))
