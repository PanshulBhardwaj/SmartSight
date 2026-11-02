# SmartSight AI

A camera assistant for blind and low-vision people. Point a camera at something and hear, by voice, **what it is, its category,
and useful information about it**: names, uses, printed price, ingredients, expiry, veg / non-veg mark, how appliances are operated, cautions.
It works with the keyboard and audio only; the screen is optional.

## Status (read this first)
Everything below was tested with unit tests, synthetic video and synthetic images in a Linux sandbox (**285 tests pass**).
**Not yet verified on a real webcam, GPU, real YOLO / OpenCLIP weights, PaddleOCR, Windows TTS/earcons, a real microphone, or the live online AI.**
Run the benchmark and eval scripts on your PC to get real numbers (see "Measure it").

## Windows setup (Python 3.12)
```powershell
cd "C:\path\to\SmartSight"            # the folder that contains requirements.txt
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt       # core (large: installs PyTorch)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"      # one-time model download
pip install -r requirements-dev.txt && pytest                     # optional: run the tests
```
Optional extras (each fails gracefully if missing; SmartSight says so by voice and in the log):
```powershell
pip install -r requirements-recognition.txt   # Tier 2: recognise almost any object (OpenCLIP; downloads weights on first use)
pip install -r requirements-paddle.txt        # preferred OCR; EasyOCR is used if this fails
pip install -r requirements-voice.txt         # offline voice commands (Vosk). Then download the model, see below
pip install -r requirements-barcode.txt       # barcode decoding (pyzbar)
pip install -r requirements-onnx.txt          # faster CPU inference after scripts\export_onnx.py
```
After a working install run `pip freeze > requirements.lock.txt` so you can rebuild the same environment later.

Voice model (offline, ~40 MB):
```powershell
Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile vosk.zip
Expand-Archive vosk.zip -DestinationPath models; Remove-Item vosk.zip
$env:SMARTSIGHT_VOICE_INPUT = "true"
```

## Run
```powershell
python live.py          # LIVE MODE (primary): smooth OpenCV window, hotkeys, works eyes-closed with audio + keyboard
streamlit run app.py    # dashboard for sighted helpers (secondary; screen-reader friendly)
python cli.py           # console mode
```
First run asks for a one-time **"I agree"** before any photo can be sent to the online AI, and an optional API key
(`$env:ANTHROPIC_API_KEY = "..."` or paste it when asked; kept in memory only). Without consent SmartSight is fully offline.

### Keys in Live Mode (also typed phrases or voice)
`i` what is this - `m` more (full Information Card) - `t` read text (all text, in order) - `r` repeat - `d` describe - `s` scene scan (what is around me) -
`f` find (then type `find my cup`; beeps get higher/faster as it nears the centre) - `c` category - `+` / `-` faster / slower speech - `v` verbosity -
`p` price - `g` ingredients - `e` expiry - `a` scan again - `o` ask online AI - `space` stop speaking - `h` help - `q` quit.
Phrases: "what is this", "more", "read text", "repeat", "describe", "scan", "find my cup", "category", "faster", "slower", "verbosity", "help", "quit",
"read the price", "read the ingredients", "read the expiry date".

## What it does
1. **Tier 1, fast:** YOLO detection + IoU tracker (stable IDs, label voting so labels do not flicker).
2. **Tier 2, open vocabulary (optional):** OpenCLIP zero-shot over a 464-item everyday-object vocabulary (`data/vocabulary.json`), run once per stable object on a background worker and cached per track; fused with OCR text and barcode data.
3. **Tier 3, descriptive AI (consent-gated):** only on request or when Tier 1+2 are unsure; adds description, similar items and an approximate online price (never the printed price). AI text is stripped of safety / allergen / dosage claims.
4. **Categories:** 17 categories (Food and Drink, Packaged Goods, Medicine and Health, Machine and Appliance, Electronics, Tool, Vehicle, Furniture, Clothing and Accessories, Plant, Animal, Person, Document and Text, Currency, Household Item, Stationery, Other). Every announcement states the category. Ingredient/nutrition text never decides the category.
5. **Information Cards:** name, category, confidence, summary, uses, attributes (colour, brand, size, material), text read, cautions, sources, plus category fields (ingredients, allergens as printed, nutrition, expiry, veg mark; how appliances are operated and their main controls).
6. **Audio-first:** hedged wording below the confidence threshold, position and distance cues, framing and low-light help, scene scan, find mode, earcons, self-check and plain-language errors spoken at start-up.
7. **Text:** price reader (₹, MRP vs sale), food label parser, split-column nutrition tables, veg / non-veg mark.

**Safety:** medicines always say "check the label or ask a pharmacist"; expiry and allergen readings add a "check it yourself" caution; SmartSight never says food or medicine is "safe", never says "allergen-free", never gives doses from AI inference, and never identifies people (only "a person" and where). It is not a replacement for a white cane or guide dog.

## Configure
Defaults and comments: `config.yaml`. Your changes are saved to `data/user_config.yaml`; any value can be overridden with `SMARTSIGHT_<NAME>` (for example `$env:SMARTSIGHT_DETECT_IMGSZ = "320"`).

## Measure it (real numbers on your PC)
```powershell
python scripts\benchmark_latency.py --seconds 15          # display FPS, capture->screen p50/p95, YOLO / OCR / zero-shot ms, time to first spoken label
python scripts\eval_recognition.py images --tier fused   # top-1 accuracy per category on YOUR labelled images (images\<label>\*.jpg)
```
Targets: display >= 24 FPS, capture->screen p95 < 150 ms, first label < 1.5 s. If a target is missed: lower `detect_imgsz` (320), `infer_max_side`, `detection_fps`, export to ONNX (`scripts\export_onnx.py`), or use a GPU (`gpu: true`).

## Troubleshooting
| Problem | Fix |
|---|---|
| `requirements.txt` not found | You are in the wrong folder (often one level too high after unzipping). `dir` should show `app.py` and `requirements.txt` |
| Camera not found | Close Teams/Zoom/browser tabs, check Windows privacy > Camera, try another index in the dashboard |
| "Open vocabulary recognition is not installed" | `pip install -r requirements-recognition.txt` (needs internet once for weights) |
| PaddleOCR fails | Ignore; EasyOCR is used automatically |
| No speech | Check Windows voices; without pyttsx3 text is printed |
| No beeps | Earcons use Windows `winsound`; disable with `SMARTSIGHT_EARCONS=false` |
| Voice unavailable | `pip install -r requirements-voice.txt`, download the Vosk model, `$env:SMARTSIGHT_VOICE_INPUT="true"` |
| Laggy | Use Live Mode (not the browser); `SMARTSIGHT_DETECT_IMGSZ=320`; run the benchmark to see which tier is slow |

More: `docs/ARCHITECTURE.md` (diagram, how to add a recognizer / category / language), `docs/DECISIONS.md`, `docs/PLAN.md`, `docs/testing.md`, `CHANGELOG.md`.
