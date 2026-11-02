"""Shared start-up helpers for live.py / cli.py: one-time consent, API key prompt, optional voice listener."""
from __future__ import annotations

import getpass
import os


def first_run_setup(settings) -> None:
    """One-time consent (saved in config) and optional API key for the online AI. Nothing is sent before consent."""
    if settings.cloud_ai and not settings.cloud_consent:
        print("\nSmartSight can use an AI (with web search) to describe objects it cannot identify locally.")
        print("Unless you use a local model (cloud_model: ollama:...), it SENDS CAMERA PHOTOS to the provider's service when needed")
        print("(Google Gemini or Anthropic, depending on cloud_model). Photos are not saved on this computer.")
        print("Note: Google's FREE tier may use what you send to improve its products. Use ollama:<model> to keep photos on this PC.")
        if input("Type 'yes' to allow this (anything else keeps SmartSight fully offline): ").strip().lower() == "yes":
            settings.cloud_consent = True
            settings.save()
    m = settings.cloud_model.strip().lower()
    env_name = "GEMINI_API_KEY" if m.startswith("gemini") else "ANTHROPIC_API_KEY"
    needs_key = not m.startswith("ollama:") and not os.environ.get(env_name) and not (m.startswith("gemini") and os.environ.get("GOOGLE_API_KEY"))
    if settings.cloud_ai and settings.cloud_consent and needs_key:
        key = getpass.getpass(f"Paste your API key for {env_name} (Enter to skip; kept for this session only): ").strip()
        if key:
            os.environ[env_name] = key


def start_voice(pipe):
    """Start the offline voice listener if enabled. Returns the listener or None. Never raises."""
    if not pipe.s.voice_input:
        return None
    try:
        from .voice_commands import VoiceListener
        extra = []
        for name in getattr(pipe, "find_phrases", lambda: [])():
            extra.append(f"find my {name}")
        v = VoiceListener(pipe.request, str(pipe.s.path("vosk_model_path")), lambda: pipe.speech.speaking, extra_phrases=extra)
        v.start()
        print("  voice: listening")
        return v
    except Exception as exc:  # noqa: BLE001
        print(f"  voice: unavailable ({exc}). Fix: pip install vosk sounddevice, and download the model (see README).")
        return None
