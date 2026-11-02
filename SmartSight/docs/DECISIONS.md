# Decisions and assumptions (no questions were asked; these were assumed)

1. **Cloud mode vs "off by default".** The brief says no frame leaves the device unless cloud mode is enabled in config. An earlier instruction from the same user was that the online AI must be integrated, not an optional extra. Reconciliation: cloud is *configured on* (`cloud_ai: true`) but **nothing is sent until the user gives one-time consent** (`cloud_consent`, saved in config). Without consent the app is fully offline. When on, the screen shows "Online AI: ON" and startup speech says photos may be sent. Tier 3 now runs only on request or when Tier 1+2 confidence is low (previously it ran on every new object).
2. **Voice input is Vosk**, not PocketSphinx (PocketSphinx needs a C compiler on Windows 3.12, which violates rule 7). Vosk stays optional.
3. **Config:** `config.yaml` (shipped, commented, read-only defaults) -> `data/user_config.yaml` (saved UI changes) -> `SMARTSIGHT_*` env vars. Legacy `data/settings.json` still read. PyYAML is pure Python (no compiler).
4. **Own IoU tracker** instead of Ultralytics tracking (avoids extra dependencies like `lap`, works with any detector, trivially testable).
5. **Zero-shot = OpenCLIP, optional** (needs PyTorch, no compiler, but large). Interface allows YOLO-World or a custom model. Text embeddings of the vocabulary are cached on disk. Confidence is the softmax score over the vocabulary: **an uncalibrated model score, not a probability**. It is spoken as "percent sure" as requested; `speak_percent: false` switches to words.
6. **Local vision-language model: not implemented.** Tier 3 is the existing cloud description only; the `Describer` interface leaves room for a local model.
7. **Live Mode = OpenCV window** (also gives real hotkeys). Streamlit remains a secondary view; `streamlit-webrtc` not used (extra dependency, browser latency).
8. **Distance cues** are a heuristic from bounding-box area ("about arm's length"), not metric depth.
9. **People:** only "a person" plus position; no identification; automatic cloud look-ups are skipped when a person is in view.
10. **Knowledge base** is category/sub-category level plus ~45 specific entries (JSON, easy to extend). Anything not covered is described generically and flagged with its source.
