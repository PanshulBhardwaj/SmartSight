"""SYNTHETIC comparison: SmartSpeak (stabilise + announce-once) vs a baseline that announces every detection.

This simulates a noisy detector (random misses and one-frame false positives) over scripted scenes. It shows
the *mechanism* reduces repetition; it is NOT a measurement of real-world camera behaviour.
Run: python scripts/narration_benchmark.py
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smartsight.narration import ObjectAnnouncer, ObjectStabilizer  # noqa: E402

HZ = 3.0
CYCLES = 200
# ground-truth appearances: (label, first_cycle, last_cycle)
TRUTH = [("bottle", 0, 80), ("cup", 40, 120), ("bottle", 150, 190)]
MISS_P, FALSE_P = 0.15, 0.03


def scene(rng):
    frames = []
    for c in range(CYCLES):
        labels = [l for l, a, b in TRUTH if a <= c <= b and rng.random() > MISS_P]
        if rng.random() < FALSE_P:
            labels.append("remote")      # one-frame false positive
        frames.append(labels)
    return frames


def run(seed):
    rng = random.Random(seed)
    frames = scene(rng)
    # baseline: speak every detection of every frame
    base = [(c, l) for c, labels in enumerate(frames) for l in labels]
    # SmartSpeak
    t = [0.0]
    stab, ann = ObjectStabilizer(4, 1), ObjectAnnouncer(3.0, lambda: t[0])
    smart = []
    for c, labels in enumerate(frames):
        t[0] = c / HZ
        for l in ann.update(stab.update(labels)):
            smart.append((c, l))

    def stats(utts):
        missed = sum(1 for l, a, b in TRUTH if not any(ul == l and a <= uc <= b + 4 for uc, ul in utts))
        false = sum(1 for uc, ul in utts if not any(ul == l and a <= uc <= b + 4 for l, a, b in TRUTH))
        repeats = len(utts) - false - (len(TRUTH) - missed)
        return len(utts), missed, false, max(0, repeats)
    return stats(base), stats(smart)


if __name__ == "__main__":
    runs = [run(s) for s in range(50)]
    names = ["utterances", "missed appearances (of 3)", "false announcements", "repeat announcements"]
    print(f"Synthetic noisy-detector simulation, {len(runs)} seeds, {CYCLES} cycles at {HZ} Hz each\n")
    print(f"{'metric':32}{'baseline (speak every detection)':>36}{'SmartSpeak':>14}")
    for i, n in enumerate(names):
        print(f"{n:32}{statistics.mean(r[0][i] for r in runs):>36.1f}{statistics.mean(r[1][i] for r in runs):>14.1f}")
