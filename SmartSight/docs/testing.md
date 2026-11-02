# Testing

## What was run (Linux sandbox, Python 3.12.3): `pytest` -> 285 passed
| Area | What is covered | Kind |
|---|---|---|
| Config | defaults, YAML/JSON/env load order, invalid/corrupt files, shipped `config.yaml` has no dead keys, privacy defaults | unit |
| Tracker, metrics, sources | stable IDs, short gaps, label voting, FPS/percentiles, synthetic video | unit |
| Pipeline (fake camera/detector/OCR/backend, **synthetic video**) | display loop never waits for a 200 ms detector; detection continues during an 800 ms OCR; commands end to end; errors contained; camera failure audible; clean shutdown | integration |
| Recognition | taxonomy (all 17 categories, all 80 detector classes), vocabulary lookup, zero-shot ranking + disk cache (**fake embedder, not real OpenCLIP**), fusion rules, async/once-per-track/cached manager, retry cap, people never analysed, centre fallback, eval-script maths | unit |
| Information cards | every one of the 464 vocabulary entries builds a card with category and sources and **no safety/allergen/dose claim**; medicine always has the pharmacist caution; food fields only when printed; AI text sanitised; person card minimal | unit/property |
| Blind-user UX | position/distance/framing, find-mode state machine and cues, announcement debouncer (fake clock), earcons, Level 1/2/3 wording, hedging, scene scan, help; pipeline behaviours: more, category, read text, find, speed/verbosity/help/quit, persistence, framing + low-light hints, self-check, plain-language errors | unit + integration |
| Known gaps | veg-dot: 600 random distractor images -> 0 false positives, 300 synthetic marks (random size, blur, noise, JPEG) -> 300 correct; split-column tables incl. two columns, wide header line, %RDA column, no header | synthetic |
| Text/price/food | price vs date/weight/barcode/nutrition; ingredient parsing; MISSING vs UNREADABLE | unit |
| Real OCR | Tesseract on rendered images (not PaddleOCR / EasyOCR, not real packaging) | integration |
| Online AI | request shape, errors, rate limit, cache, consent gate, automatic look-up only when unsure, fallbacks | fake HTTP; **live API never called** |
| UI | Streamlit `AppTest`: all controls present, clicks survive with no camera | headless |

## Synthetic measurements (pipeline overhead only, NOT your hardware)
`python scripts/benchmark_latency.py --synthetic` (1280x720 @ 30 fps source, *simulated* detector, headless):
| Simulated detector | Display FPS | capture->screen p95 | Detector | First spoken label |
|---|---|---|---|---|
| 60 ms | 29.5 to 29.9 | 30.5 to 30.9 ms | 60.3 ms | 0.46 to 0.48 s |
| 150 ms | 29.9 | 30.5 ms | 150.3 ms | 0.69 s |
This shows the display loop does not wait for inference. It says nothing about real YOLO/OCR/CLIP speed or imshow cost.
`scripts/narration_benchmark.py` (synthetic noisy detector): 177.3 utterances -> 3.2, false announcements 6.3 -> 0.0, missed appearances 0 -> 0.

## NOT verified (needs your PC)
Real webcam, YOLO/OpenCLIP inference and accuracy, PaddleOCR/EasyOCR, pyttsx3 audio, `winsound` earcons, imshow/hotkeys, Vosk with a microphone,
the live online AI and web search, ONNX export, the veg-dot and table parsers on real packaging photos, NVDA behaviour of the dashboard.

## Manual checklist (record results)
| # | Test | Expected | Result |
|---|---|---|---|
| 1 | `python live.py` with the screen turned off or eyes closed: start, hear the self-check, identify 5 objects, ask "more", "read text", "scan", "find my ..." using only keys / voice | Whole flow works by audio and keyboard alone | |
| 2 | NVDA on `streamlit run app.py`: tab through headings, buttons, sidebar; trigger Identify, More, Stop | Every control has a spoken label; no colour-only status; preview off = no interrupting updates | |
| 3 | Benchmark on this PC | display >= 24 FPS, p95 < 150 ms, first label < 1.5 s (record real values and settings used) | |
| 4 | Common objects (cup, bottle, phone, laptop, book) | Category + name + hedged confidence; announced once; no flicker | |
| 5 | 20 uncommon items (stapler, tiffin box, multimeter...) with Tier 2 installed | Named or hedged "I think / I'm not sure"; eval script numbers recorded | |
| 6 | Packaged food with veg dot and a two-column nutrition table | "Vegetarian mark" / "non-vegetarian mark" with confidence; "Protein 7 grams per serving" | |
| 7 | Medicine strip | Pharmacist caution; no purpose or dose spoken unless printed | |
| 8 | Appliance (mixer, iron, microwave) | Operation, main controls, caution | |
| 9 | Person in view | "Person, on your left ..." only; nothing sent online | |
| 10 | Low light, blur, object cut off, off-centre | Spoken hint once, not repeatedly | |
| 11 | Find mode: move toward a cup | Beeps rise and speed up; "Found your cup, centred" | |
| 12 | Unplug camera / block model download / no internet | Plain-language spoken error with the fix; no crash | |
| 13 | Stop while speaking / quit | Speech cut off; camera light off | |
| 14 | Online AI without consent / without key | Nothing sent; spoken explanation | |
