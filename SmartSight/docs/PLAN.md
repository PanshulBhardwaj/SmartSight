# Upgrade plan (M0 audit)

## Audit findings (repo as found, 154 tests passing)
* Capture is already threaded (latest frame only) but **detection, OCR, cloud calls and commands share one worker thread**, so a slow OCR/cloud call stalls detection.
* Preview: Streamlit `st.image` refresh (laggy) or `cli.py --preview`; no tracker, no stable object IDs, no per-object cache.
* Recognition = YOLO class + OCR/barcode keyword category only. No open-vocabulary tier, taxonomy or Information Card.
* Voice input is **Vosk** (not PocketSphinx as the brief says). Online AI is already integrated and consent-gated (see DECISIONS.md).
* Settings are JSON/env; no `config.yaml`. No earcons, scene scan, find mode, framing help, veg-dot, split-column tables.

## Milestones
| M | Scope | Exit check |
|---|---|---|
| M1 | Frame metadata, IoU tracker, detection thread decoupled from command thread, inference downscale, Live Mode (OpenCV window), YAML config, metrics, benchmark | pytest + synthetic non-blocking test |
| M2 | Interfaces, vocabulary data file, zero-shot recognizer (OpenCLIP, swappable), per-track async cache, label voting, fusion with OCR/barcode, eval script | pytest + eval script runs |
| M3 | Taxonomy, label->category mapper, knowledge base, Information Card, safety rules | pytest |
| M4 | Debounced announcements with categories + hedging, 3 detail levels, position/distance, framing help, scene scan, find mode, earcons, commands, self-check, plain-language errors | pytest |
| M5 | Veg-dot classifier, split-column nutrition tables | pytest |
| M6 | Docs, CHANGELOG, requirements split, benchmark/eval numbers (what could be measured here) | full pytest |

## Honesty about verification
No webcam, GPU, YOLO weights or CLIP weights in the build sandbox. Logic is verified with fakes and synthetic video; real FPS/latency/accuracy must be measured on the target PC with `scripts/benchmark_latency.py` and `scripts/eval_recognition.py`.
