"""Voice command matching (simple rules) and an optional OFFLINE (Vosk) microphone listener."""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
from enum import Enum
from typing import Callable, Optional

log = logging.getLogger(__name__)


class Command(Enum):
    IDENTIFY = "identify"
    DESCRIBE = "describe"
    PRICE = "price"
    INGREDIENTS = "ingredients"
    FULL_INGREDIENTS = "full_ingredients"
    EXPIRY = "expiry"
    LABEL = "label"
    REPEAT = "repeat"
    STOP = "stop"
    SCAN_AGAIN = "scan_again"
    ASK_AI = "ask_ai"
    AUTO_LOOKUP = "auto_lookup"   # internal: never produced by phrase matching
    MORE = "more"
    READ_TEXT = "read_text"
    SCENE = "scene"
    FIND = "find"
    CATEGORY = "category"
    FASTER = "faster"
    SLOWER = "slower"
    VERBOSITY = "verbosity"
    HELP = "help"
    QUIT = "quit"


# Order matters: more specific rules first.
_RULES = [
    (Command.STOP, r"\b(stop|quiet|silence|be quiet|shut up|enough)\b"),
    (Command.REPEAT, r"\b(repeat|say that again|say it again|what did you say)\b"),
    (Command.SCAN_AGAIN, r"\b(scan again|rescan|scan once more|try again|look again)\b"),
    (Command.ASK_AI, r"\b(ask (the )?ai|search online|search the web|find (it |this )?online|look (it |this )?up|similar items?|online price|price online)\b"),
    (Command.HELP, r"\b(help|what can i say|list commands|commands)\b"),
    (Command.FASTER, r"\b(faster|speak faster|speed up|talk faster)\b"),
    (Command.SLOWER, r"\b(slower|speak slower|slow down|talk slower)\b"),
    (Command.VERBOSITY, r"\b(verbosity|detail level|change detail)\b"),
    (Command.MORE, r"^(more|tell me more|more details?|more information|give me more|go on)$"),
    (Command.READ_TEXT, r"\b(read (all )?(the )?text|read everything|all the text)\b"),
    (Command.SCENE, r"\b(scan|what'?s around me|what is around me|around me|scene|look around)\b"),
    (Command.CATEGORY, r"\b(category|what kind|what type)\b"),
    (Command.QUIT, r"\b(quit|exit|goodbye)\b"),
    (Command.FULL_INGREDIENTS, r"\b(full|complete|entire|all|whole)\b.*\bingredients?\b|\bingredients?\b.*\b(in full|completely)\b"),
    (Command.INGREDIENTS, r"\b(ingredients?|what is it made of|what's in it|nutrition|nutritional)\b"),
    (Command.PRICE, r"\b(price|how much|cost|mrp|rupees?)\b"),
    (Command.EXPIRY, r"\b(expiry|expire[sd]?|expiration|best before|use by|shelf life)\b"),
    (Command.LABEL, r"\b(read (the |this )?(label|text|packet|package)|label)\b"),
    (Command.DESCRIBE, r"\b(describe|tell me about|more about)\b"),
    (Command.IDENTIFY, r"\b(what is (this|that|it)|what's (this|that)|identify|what am i holding)\b"),
]


def match_command(text: str) -> Optional[Command]:
    t = re.sub(r"[^\w\s']", " ", (text or "").lower()).strip()
    if not t:
        return None
    for cmd, pattern in _RULES:
        if re.search(pattern, t):
            return cmd
    return None


GRAMMAR = ["what is this", "what is that", "read the price", "how much is it", "read the ingredients",
           "read the full ingredients", "read the expiry date", "describe this product", "read the label",
           "repeat that", "stop speaking", "stop", "scan again", "search online", "ask ai", "more", "read text",
           "scan", "what is around me", "category", "faster", "slower", "verbosity", "help", "quit"]


_FIND = re.compile(r"^find (?:my |the |a |an |some )?(?P<what>.+)$")


def parse_command(text: str):
    """-> (Command | None, argument | None). 'find my cup' -> (FIND, 'cup')."""
    t = re.sub(r"[^\w\s']", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    m = _FIND.match(t)
    if m and "online" not in t:
        return Command.FIND, m.group("what").strip()
    return match_command(t), None


class VoiceListener:
    """Offline microphone commands using Vosk (no audio leaves the device, no compiler needed on Windows).

    Recognition is restricted to GRAMMAR, which is far more accurate than free dictation. While SmartSight is
    speaking, only "stop" is acted on, so it can be interrupted but does not obey its own voice.
    Needs: pip install vosk sounddevice, plus a model folder (see README). UNTESTED with a real microphone."""

    def __init__(self, on_command: Callable[[Command], None], model_dir: str = "models/vosk-model-small-en-us-0.15",
                 is_speaking: Callable[[], bool] = lambda: False, recognizer=None, device=None, extra_phrases=()):
        self._on_command, self._model_dir, self._is_speaking = on_command, model_dir, is_speaking
        self._grammar = list(GRAMMAR) + list(extra_phrases)
        self._rec, self._device = recognizer, device
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=50)
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _build(self) -> None:
        from pathlib import Path
        from vosk import KaldiRecognizer, Model, SetLogLevel
        if not Path(self._model_dir).is_dir():
            raise FileNotFoundError(f"Vosk model folder not found: {self._model_dir}")
        SetLogLevel(-1)
        self._rec = KaldiRecognizer(Model(str(self._model_dir)), 16000, json.dumps(self._grammar + ["[unk]"]))

    def feed(self, data: bytes) -> Optional[Command]:
        """Process one audio chunk (16 kHz mono int16). Returns the command acted on, if any."""
        if not self._rec.AcceptWaveform(data):
            return None
        text = json.loads(self._rec.Result()).get("text", "").strip()
        cmd, arg = parse_command(text) if text and text != "[unk]" else (None, None)
        log.info("voice heard %r -> %s %s", text, cmd, arg or "")
        if cmd is None or (self._is_speaking() and cmd != Command.STOP):
            return None
        self._on_command(cmd) if arg is None else self._on_command(cmd, arg)
        return cmd

    def start(self) -> None:
        import sounddevice as sd
        if self._rec is None:
            self._build()
        self._stop.clear()
        self._stream = sd.RawInputStream(samplerate=16000, blocksize=4000, dtype="int16", channels=1,
                                         device=self._device, callback=self._audio_cb)
        self._stream.start()
        self._thread = threading.Thread(target=self._loop, name="voice", daemon=True)
        self._thread.start()

    def _audio_cb(self, indata, frames, time_info, status) -> None:
        try:
            self._q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.feed(data)
            except Exception:  # noqa: BLE001
                log.exception("voice processing failed")

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        if self._thread:
            self._thread.join(timeout=2)
