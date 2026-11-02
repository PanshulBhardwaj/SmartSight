# Changelog

## 0.2.0 - upgrade (M0-M6)
### Added
* **Live Mode** (`python live.py`): native OpenCV window, real hotkeys, FPS / latency HUD, typed phrases in the console.
* **Smooth pipeline:** capture / detection / command threads decoupled; inference downscaling; adaptive frame skipping; frame metadata; `SyntheticSource`; metrics (FPS, p50/p95).
* **Tracker + temporal smoothing:** IoU tracker with stable IDs and confidence-weighted label voting.
* **Tiered recognition:** open-vocabulary Tier 2 (OpenCLIP, optional) on stable tracks only, cached per track, async; fusion with OCR text and barcode; 464-item vocabulary data file; `scripts/eval_recognition.py`.
* **Categories and Information Cards:** 17-category taxonomy, all 80 detector classes mapped, knowledge base (category defaults + 51 specific items), colour attribute, safety rules and AI-text sanitiser.
* **Audio-first UX:** category + hedged confidence in every announcement, 3 detail levels (auto / "more" / "read text"), position and distance cues, framing and low-light help, scene scan, find mode with pitch/speed beeps, earcons, new commands (more, read text, scan, find, category, faster, slower, verbosity, help, quit), startup self-check, plain-language errors with the fix, settings saved between runs.
* **Known gaps closed:** veg / non-veg mark classifier (circle or triangle in a square); split-column nutrition tables ("Protein 7 grams per serving").
* `config.yaml` as the single tunable config; `docs/PLAN.md`, `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`.
### Changed
* Vosk replaces PocketSphinx for optional voice (no C compiler needed). Online AI now runs only when Tier 1+2 are unsure or on request, and only after one-time consent.
* People are no longer ignored by default: reported as "person" with a position only.
* Requirements split into core and optional files.
### Known limitations
See README "Status".

## 0.1.0 - first version
Webcam + YOLO + OCR (Paddle / EasyOCR / Tesseract), price reader, food label parser, barcode, speech queue, Streamlit dashboard, CLI, Vosk voice, online AI.
