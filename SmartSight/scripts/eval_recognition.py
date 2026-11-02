"""Measure recognition accuracy on YOUR labelled images.

Folder layout:   IMAGES/<true label>/*.jpg      e.g.  images/water bottle/1.jpg, images/stapler/3.png
<true label> may be any vocabulary name/alias (data/vocabulary.json) or a detector class.

  python scripts/eval_recognition.py images --tier fused       # detector + zero-shot (+fusion)  [default]
  python scripts/eval_recognition.py images --tier zeroshot    # Tier 2 only, on the image centre
  python scripts/eval_recognition.py images --tier detector    # YOLO only

Reports top-1 accuracy overall and per category. Predictions are matched through the vocabulary, so aliases count as correct.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def same_object(vocab, a: str, b: str) -> bool:
    ea, eb = vocab.find(a), vocab.find(b)
    if ea is not None and eb is not None:
        return ea.name == eb.name
    return a.strip().lower() == b.strip().lower()


def evaluate(samples, predict, vocab) -> dict:
    """samples: [(image, true_label)]; predict(image) -> (label, category) or None. Pure function (unit-tested)."""
    per_cat = defaultdict(lambda: [0, 0])
    wrong, total, correct = [], 0, 0
    for img, truth in samples:
        entry = vocab.find(truth)
        cat = entry.category if entry else "Other"
        pred = predict(img)
        ok = bool(pred) and same_object(vocab, pred[0], truth)
        per_cat[cat][1] += 1; per_cat[cat][0] += ok
        total += 1; correct += ok
        if not ok:
            wrong.append((truth, pred[0] if pred else None))
    return {"top1": correct / total if total else 0.0, "n": total,
            "per_category": {c: {"top1": v[0] / v[1], "n": v[1]} for c, v in sorted(per_cat.items())}, "wrong": wrong}


def load_samples(folder: Path):
    import cv2
    out = []
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                img = cv2.imread(str(f))
                if img is not None:
                    out.append((img, d.name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--tier", choices=["fused", "zeroshot", "detector"], default="fused")
    a = ap.parse_args()
    from config.settings import Settings
    from smartsight.fusion import fuse
    from smartsight.recognizer import center_box, crop_box, make_recognizer
    from smartsight.taxonomy import Taxonomy, Vocabulary
    s = Settings.load()
    vocab, tax = Vocabulary.from_file(s.path("vocabulary_path")), Taxonomy.from_file(s.path("taxonomy_path"))
    det = rec = None
    if a.tier in ("fused", "detector"):
        from smartsight.detector import YoloDetector
        try:
            det = YoloDetector(s.model_path, s.gpu, s.detect_imgsz)
        except Exception as exc:  # noqa: BLE001
            print("Detector unavailable:", exc); return 2
    if a.tier in ("fused", "zeroshot"):
        rec, msg = make_recognizer(s, vocab)
        print(msg)
        if rec is None:
            return 2

    def predict(img):
        dets = det.detect(img, s.confidence_threshold) if det else []
        box = dets[0].box if dets else center_box(img)
        crop = crop_box(img, box)
        zs = rec.classify(crop, 3) if rec else []
        if a.tier == "zeroshot":
            return (zs[0].label, zs[0].category) if zs else None
        m = tax.map_detector(dets[0].label) if dets else None
        if a.tier == "detector":
            return (m.name, m.category) if m else None
        r = fuse(m, dets[0].confidence if dets else 0.0, zs, s.zero_shot_threshold)
        return (r.label, r.category)

    samples = load_samples(a.folder)
    if not samples:
        print("No images found. Expected IMAGES/<label>/*.jpg"); return 2
    rep = evaluate(samples, predict, vocab)
    print(f"\nTier: {a.tier}   images: {rep['n']}   overall top-1: {rep['top1']:.1%}\n")
    for c, v in rep["per_category"].items():
        print(f"  {c:28} {v['top1']:6.1%}   (n={v['n']})")
    if rep["wrong"]:
        print("\nExamples of mistakes (true -> predicted):")
        for t, p in rep["wrong"][:15]:
            print(f"  {t} -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
