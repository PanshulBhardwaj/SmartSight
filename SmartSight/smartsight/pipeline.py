"""Orchestration: camera -> quality -> detection (Tier 1) -> tracking -> async open-vocabulary recognition (Tier 2)
-> fusion -> debounced announcements, plus on-demand analysis (OCR, barcode, Information Cards, online AI = Tier 3).

Two threads: `detect` (never waits for OCR / cloud / commands) and `commands`. Tier 2 runs on its own worker.
All components are injected, so the pipeline is testable with fakes. Any of detector / ocr / barcode / recognizer may be None.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from .announce import Announcer
from .camera import CameraError
from .card import CardBuilder, InformationCard
from .colour import dominant_colour
from .earcons import Earcons
from .errors import explain
from .food_label_parser import parse_food_label
from .frame_quality import assess_frame, sharpness
from .fusion import Recognition, fuse
from .knowledge import KnowledgeBase
from .metrics import Percentiles
from .narration import Narration, NarrationEngine
from .nutrition_table import parse_table
from .vegmark import classify_veg_mark
from .price_reader import extract_prices
from .product_identifier import ProductIdentifier, ProductInfo
from .product_lookup import LocalProductStore
from .recognition_manager import RecognitionManager
from .recognizer import center_box, crop_box
from .selfcheck import run_selfcheck
from .spatial import FindMode, area_fraction, framing_hint, position_phrase
from .debounce import AnnouncementDebouncer
from .speech_engine import Priority, SpeechController
from .taxonomy import Taxonomy, Vocabulary
from .tracker import IouTracker
from .utils import InfoState, OcrLine
from .voice_commands import Command

log = logging.getLogger(__name__)
VERBOSITIES = ("concise", "standard", "detailed")


@dataclass
class Result:
    command: Command
    text: str
    state: InfoState
    ocr_lines: list = field(default_factory=list)
    product: Optional[ProductInfo] = None
    card: Optional[InformationCard] = None
    timestamp: float = field(default_factory=time.time)


def _load(fn, path, default):
    try:
        return fn(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load %s (%s); continuing with built-in minimal data", path, exc)
        return default


class Pipeline:
    def __init__(self, settings, camera, detector, ocr, barcode, store: LocalProductStore, speech: SpeechController,
                 lookup=None, identifier: Optional[ProductIdentifier] = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, cloud=None, recognizer=None, taxonomy=None, vocab=None,
                 async_tiers: bool = True, earcons: Optional[Earcons] = None, persist: bool = False,
                 component_status: Optional[dict] = None, knowledge: Optional[KnowledgeBase] = None):
        self.s, self.camera, self.detector, self.ocr, self.barcode = settings, camera, detector, ocr, barcode
        self.store, self.speech, self.lookup, self.cloud = store, speech, lookup, cloud
        self._clock, self._sleep, self.persist = clock, sleep, persist
        self.component_status = component_status or {}
        self.identifier = identifier or ProductIdentifier(min_conf=settings.ocr_min_confidence)
        self.narr = NarrationEngine(settings.verbosity)
        self.talk = Announcer(settings)
        self.vocab = vocab if vocab is not None else _load(Vocabulary.from_file, settings.path("vocabulary_path"), Vocabulary.empty())
        self.taxonomy = taxonomy if taxonomy is not None else _load(Taxonomy.from_file, settings.path("taxonomy_path"), Taxonomy.minimal())
        kb = knowledge if knowledge is not None else KnowledgeBase.from_file(settings.path("knowledge_path"))
        self.cards = CardBuilder(kb, self.vocab, settings.ocr_min_confidence)
        self.recog = RecognitionManager(recognizer, self.taxonomy, self.vocab, settings, async_tiers)
        self.tracker = IouTracker(settings.tracker_iou, 0.5, settings.tracker_max_misses, settings.vote_window)
        self.debounce = AnnouncementDebouncer(settings.stable_frames, settings.announce_cooldown, 1.5, clock)
        self.earcons = earcons or Earcons(None, False)
        self.find: Optional[FindMode] = None
        self._lock = threading.RLock()
        self._tracks_view: list = []
        self._recs: dict = {}
        self._announced: dict = {}
        self._frame_wh = (1, 1)
        self._last_seq = -2
        self.latency = Percentiles()          # capture -> detection result (seconds)
        self.detect_ms = Percentiles()
        self._cmds: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cmd_thread: Optional[threading.Thread] = None
        self._arg: Optional[str] = None
        self.quit_event = threading.Event()
        self._dets: list = []
        self.last_result: Optional[Result] = None
        self.last_card: Optional[InformationCard] = None
        self._last_card_time = -1e9
        self._last_rec: Optional[Recognition] = None
        self._last_announced_label: Optional[str] = None
        self.camera_status = "stopped"
        self.inference_status = "idle"
        self.metrics: dict[str, float] = {}
        self._last_error_at = -1e9
        self._cloud_cache: dict = {}
        self._cloud_times: deque = deque()
        self._last_cloud = -1e9
        self._auto_pending = False
        self._bad_frames = 0
        self._last_quality_hint = -1e9
        self._hint_since: Optional[tuple] = None
        self._last_hint = -1e9
        self._last_beep = -1e9

    # ================= lifecycle =================
    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        try:
            self.camera.start()
        except CameraError as exc:
            self.camera_status = f"error: {exc}"
            self.earcons.play("error")
            self.speech.say(explain(exc), Priority.CRITICAL)
            log.error("Camera start failed: %s", exc)
            return False
        self.camera_status = "running"
        self.tracker.reset(); self.debounce.reset(); self.recog.reset()
        self._stop.clear()
        # Two independent threads: detection never waits for OCR / cloud / user commands, and vice versa.
        self._thread = threading.Thread(target=self._det_loop, name="detect", daemon=True)
        self._cmd_thread = threading.Thread(target=self._cmd_loop, name="commands", daemon=True)
        self._thread.start(); self._cmd_thread.start()
        self.earcons.play("ready")
        if self.component_status or hasattr(self.recog, "recognizer"):
            sc = run_selfcheck(self, self.component_status)
            print("Self check:\n" + sc.printed)
            log.info("self check: %s", sc.spoken)
            self.speech.say(sc.spoken if self.component_status else self.narr.simple("started").text, Priority.USER)
        return True

    def stop(self) -> None:
        """Stop threads and release the camera. Safe to call repeatedly or while speaking."""
        self._stop.set()
        for t in (self._thread, self._cmd_thread):
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=5)
        self.camera.stop()
        self.camera_status = "stopped"
        self.inference_status = "idle"
        self.find = None
        self.speech.stop()
        self.speech.say(self.narr.simple("stopped").text, Priority.USER)

    def close(self) -> None:
        self.stop()
        self.recog.shutdown()
        self.speech.wait_idle(3)
        self.speech.shutdown()

    def set_verbosity(self, v: str) -> None:
        self.s.verbosity = v
        self.narr.verbosity = v

    def find_phrases(self) -> list:
        return [e.name for e in self.vocab.entries]

    def _save(self) -> None:
        if self.persist:
            try:
                self.s.save()
            except Exception:  # noqa: BLE001
                log.warning("could not save settings", exc_info=True)

    # ================= requests =================
    def request(self, cmd: Command, arg: Optional[str] = None) -> None:
        """Thread-safe. STOP and REPEAT take effect immediately, even if a scan is in progress."""
        if cmd == Command.STOP:
            self.speech.stop()
        elif cmd == Command.REPEAT:
            if not self.speech.repeat():
                self.speech.say("There is nothing to repeat yet.", Priority.USER)
        else:
            self._cmds.put(cmd if arg is None else (cmd, arg))

    def _cmd_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._cmds.get(timeout=0.1)
            except queue.Empty:
                continue
            self.process_command(item)

    def _det_loop(self) -> None:
        """Adaptive: as fast as inference allows up to detection_fps; unprocessed camera frames are skipped, never queued."""
        interval = 1.0 / max(self.s.detection_fps, 0.1)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            did = self.step()
            if getattr(self.camera, "disconnected", False):
                self.camera_status = f"error: {getattr(self.camera, 'error', 'disconnected')}"
                self.earcons.play("error")
                self.speech.say(explain(CameraError("camera disconnected")), Priority.CRITICAL)
                break
            wait = interval - (time.perf_counter() - t0)
            self._stop.wait(max(0.005, wait) if did else 0.01)

    # ================= detection cycle =================
    def _grab(self):
        cam = self.camera
        if hasattr(cam, "read_meta"):
            return cam.read_meta(copy=False)
        f = cam.read()
        return None if f is None else (f, time.perf_counter(), -1)

    def _downscale(self, frame):
        h, w = frame.shape[:2]
        k = min(1.0, self.s.infer_max_side / max(h, w))
        if k >= 1.0:
            return frame, 1.0
        import cv2
        return cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), k

    def _primary_score(self, box, w: int, h: int) -> float:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        central = 1 - min(1.0, (((cx - w / 2) / (w / 2)) ** 2 + ((cy - h / 2) / (h / 2)) ** 2) ** 0.5)
        return 0.5 * min(1.0, area_fraction(box, w, h) * 4) + 0.5 * central

    def step(self) -> bool:
        """One detection cycle on the newest unseen frame. Never raises. Returns True if a frame was processed."""
        try:
            m = self._grab()
            if m is None:
                return False
            frame, ts, seq = m
            if seq == self._last_seq and seq != -1:
                return False
            self._last_seq = seq
            if self.detector is None:
                self.inference_status = "detector unavailable"
                return False
            small, k = self._downscale(frame)
            q = assess_frame(small, self.s.min_sharpness, self.s.min_brightness)
            if "invalid" in q.issues:
                return False
            t0 = time.perf_counter()
            dets = [d for d in self.detector.detect(small, self.s.confidence_threshold) if d.label not in self.s.ignore_classes]
            if k < 1.0:
                from .detector import Detection
                dets = [Detection(d.label, d.confidence, tuple(int(v / k) for v in d.box)) for d in dets]
            ms = (time.perf_counter() - t0) * 1000
            self.metrics["detect_ms"] = ms
            self.detect_ms.add(ms)
            self.inference_status = "running"
            h, w = frame.shape[:2]
            self._frame_wh = (w, h)
            with self._lock:
                self._dets = dets
                tracks = self.tracker.update(dets)
            self.recog.update(frame, tracks)                      # Tier 2: async, stable tracks only, cached per track id
            if "zeroshot_ms" not in self.metrics and self.recog.zs_ms.summary()["n"]:
                pass
            if self.recog.zs_ms.summary()["n"]:
                self.metrics["zeroshot_ms"] = self.recog.zs_ms.summary()["mean"]
            recs = {t.id: self.recog.recognition(t) for t in tracks}
            with self._lock:
                self._recs = recs
                self._tracks_view = [(t.id, recs[t.id].label, recs[t.id].confidence, t.box) for t in tracks]
            self.latency.add(time.perf_counter() - ts)
            self._after_detection(frame, tracks, recs, q, w, h)
            return True
        except Exception as exc:  # noqa: BLE001
            self.inference_status = f"error: {exc}"
            log.exception("detection step failed")
            if self._clock() - self._last_error_at > 30:
                self._last_error_at = self._clock()
                self.earcons.play("error")
                self.speech.say(explain(exc, "looking at the camera image"), Priority.CRITICAL)
            return False

    def _after_detection(self, frame, tracks, recs, q, w: int, h: int) -> None:
        now = self._clock()
        if not q.ok:                                              # too dark/blurry: stay quiet rather than announce unreliable detections
            self._bad_frames += 1
            if self._bad_frames >= 6 and self.s.auto_announce and now - self._last_quality_hint > 10:
                self._last_quality_hint = now
                self.speech.say(self.narr.quality(q).text.replace("blurry.", "blurry. Hold steady."), Priority.AUTO)
            return
        self._bad_frames = 0
        live_ids = [t.id for t in self.tracker.tracks]
        gone = self.debounce.retain(live_ids)
        for g in gone:
            self._announced.pop(g, None)
        if gone:
            self.earcons.play("lost")
        if self.find is not None:
            self._run_find(recs, tracks, w, h)
            return
        stable = [t for t in tracks if t.streak >= self.s.stable_frames]
        stable.sort(key=lambda t: -self._primary_score(t.box, w, h))
        if self.s.auto_announce:
            for t in stable:
                rec = recs[t.id]
                if self.debounce.update(t.id, rec.label, t.streak):
                    update = t.id in self._announced
                    self._announced[t.id] = rec.label
                    pos = position_phrase(t.box, w, h)
                    self.speech.say(self.talk.level1(rec, pos, self._hazard(rec), update=update), Priority.AUTO)
                    self.earcons.play("locked")
                    self._last_announced_label, self._last_rec = rec.label, rec
                    break
            self._framing_help(stable, recs, w, h, now)
        vague = any(recs[t.id].generic or recs[t.id].confidence < self.s.cloud_low_confidence for t in stable)
        person = any(r.category == "Person" for r in recs.values())
        newly = any(t.id not in self._announced or self._announced.get(t.id) == recs[t.id].label for t in stable)
        if stable and vague and newly and not person and self.s.cloud_auto and not self._auto_pending and self.cloud_ready()[0]:
            self._auto_pending = True
            self._cmds.put(Command.AUTO_LOOKUP)

    def _hazard(self, rec: Recognition) -> bool:
        e = self.vocab.find(rec.label)
        item = self.cards.kb.item(rec.label) or (self.cards.kb.item(e.name) if e else None) or {}
        return bool(item.get("hazard"))

    def _framing_help(self, stable, recs, w: int, h: int, now: float) -> None:
        """Gentle, rate-limited advice when the main object stays badly framed."""
        if not stable:
            self._hint_since = None
            return
        t = stable[0]
        hint = framing_hint(t.box, w, h) if recs[t.id].category != "Person" else None
        if hint is None:
            self._hint_since = None
            return
        if self._hint_since is None or self._hint_since[0] != t.id:
            self._hint_since = (t.id, now)
            return
        if now - self._hint_since[1] >= 3.0 and now - self._last_hint >= 12.0:
            self._last_hint = now
            self.speech.say(hint, Priority.AUTO)

    def _run_find(self, recs, tracks, w: int, h: int) -> None:
        items = [(recs[t.id], t.box) for t in tracks]
        u = self.find.update(items, w, h)
        if u.earcon:
            self.earcons.play(u.earcon)
        if u.speech:
            self.speech.say(u.speech, Priority.USER)
        now = self._clock()
        if u.beep and now - self._last_beep >= u.beep[1]:
            self._last_beep = now
            self.earcons.beep(u.beep[0], 50)

    def overlay_tracks(self) -> list:
        """(id, label, confidence, box) for the display loop; never blocks on inference."""
        with self._lock:
            return list(self._tracks_view)

    # ================= online AI (Tier 3) =================
    def ensure_cloud(self):
        """Online-AI client (key from ANTHROPIC_API_KEY or set in the session)."""
        if self.cloud is None:
            import os
            from .cloud_vision import CloudVision
            self.cloud = CloudVision(os.environ.get("ANTHROPIC_API_KEY"), self.s.cloud_model, self.s.cloud_web_search)
        self.cloud.web_search = self.s.cloud_web_search
        self.cloud.model = self.s.cloud_model
        return self.cloud

    def cloud_ready(self) -> tuple[bool, str]:
        """(ready, reason). Nothing is ever sent unless this is ready."""
        if not self.s.cloud_ai:
            return False, "cloud_off"
        if not self.s.cloud_consent:
            return False, "cloud_consent"
        if not self.ensure_cloud().configured:
            return False, "cloud_nokey"
        return True, ""

    def cloud_status(self) -> str:
        ok, why = self.cloud_ready()
        return {"": "ON: camera photos may be sent to the online AI when it is needed to identify something", "cloud_off": "paused",
                "cloud_consent": "waiting for your permission", "cloud_nokey": "no API key"}["" if ok else why]

    def _cloud_call(self, frame, labels, user_initiated: bool):
        """Rate-limited, cached call. Returns CloudResult or None when skipped."""
        from .cloud_vision import encode_frame, frame_key
        key = frame_key(frame) + "|" + ",".join(sorted(labels))
        now = self._clock()
        hit = self._cloud_cache.get(key)
        if hit and now - hit[0] < 300:
            return hit[1]
        if not user_initiated:
            while self._cloud_times and now - self._cloud_times[0] > 60:
                self._cloud_times.popleft()
            if now - self._last_cloud < self.s.cloud_min_interval or len(self._cloud_times) >= self.s.cloud_max_per_minute:
                return None
        self._last_cloud = now
        self._cloud_times.append(now)
        res = self.cloud.describe(encode_frame(frame), list(labels))
        log.info("cloud AI status: %s", res.status)
        if res.status == "ok":
            self._cloud_cache[key] = (now, res)
            if len(self._cloud_cache) > 20:
                self._cloud_cache.pop(next(iter(self._cloud_cache)))
        return res

    def _ask_ai(self) -> Result:
        ok, why = self.cloud_ready()
        if not ok:
            return self._emit(Command.ASK_AI, self.narr.simple(why))
        frame = self._best_frame()
        q = assess_frame(frame, self.s.min_sharpness, self.s.min_brightness)
        if not q.ok:
            return self._emit(Command.ASK_AI, self.narr.quality(q))
        self.speech.say(self.narr.simple("cloud_wait").text, Priority.USER)
        res = self._cloud_call(frame, [d.label for d in self._dets], True)
        return self._emit(Command.ASK_AI, self.narr.cloud(res))

    def _auto_lookup(self) -> Optional[Result]:
        """Background look-up when Tier 1+2 are unsure. Quiet unless the AI actually identifies something."""
        self._auto_pending = False
        if not (self.s.cloud_auto and self.cloud_ready()[0]):
            return None
        frame = self.camera.read()
        if frame is None or not assess_frame(frame, self.s.min_sharpness, self.s.min_brightness).ok:
            return None
        res = self._cloud_call(frame, [d.label for d in self._dets], False)
        if res is None or res.status != "ok" or not res.data["can_identify"]:
            return None
        n = self.narr.cloud(res)
        self.speech.say(n.text, Priority.AUTO)
        self.last_result = Result(Command.AUTO_LOOKUP, n.text, n.state)
        return self.last_result

    # ================= on-demand commands =================
    def process_command(self, cmd, arg: Optional[str] = None) -> Optional[Result]:
        if isinstance(cmd, tuple):
            cmd, arg = cmd
        self._arg = arg
        try:
            return self._process(cmd)
        except Exception as exc:  # noqa: BLE001
            log.exception("command %s failed", cmd)
            self.earcons.play("error")
            self.speech.say(explain(exc, "doing that"), Priority.CRITICAL)
            self.inference_status = f"error: {exc}"
            return None

    def _say_user(self, text: str, state=InfoState.TENTATIVE, cmd=None, **kw) -> Result:
        self.speech.say(text, Priority.USER)
        self.last_result = Result(cmd or Command.HELP, text, state, **kw)
        return self.last_result

    def _process(self, cmd: Command) -> Optional[Result]:
        arg = self._arg
        # ---- simple commands -------------------------------------------------
        if cmd == Command.ASK_AI:
            return self._ask_ai()
        if cmd == Command.AUTO_LOOKUP:
            return self._auto_lookup()
        if cmd == Command.HELP:
            return self._say_user(self.talk.help(), InfoState.CONFIRMED, cmd)
        if cmd == Command.QUIT:
            self.speech.say("Goodbye.", Priority.USER)
            self.quit_event.set()
            return None
        if cmd in (Command.FASTER, Command.SLOWER):
            new = max(80, min(300, self.s.speech_rate + (25 if cmd == Command.FASTER else -25)))
            self.s.speech_rate = new
            if hasattr(self.speech.backend, "rate"):
                self.speech.backend.rate = new
            self._save()
            return self._say_user(f"Speech speed {new} words per minute.", InfoState.CONFIRMED, cmd)
        if cmd == Command.VERBOSITY:
            nxt = VERBOSITIES[(VERBOSITIES.index(self.s.verbosity) + 1) % 3]
            self.set_verbosity(nxt)
            self._save()
            return self._say_user(f"Verbosity {nxt}.", InfoState.CONFIRMED, cmd)
        if cmd == Command.FIND:
            return self._start_find(arg)
        if cmd == Command.SCENE:
            return self._scene()
        if cmd == Command.SCAN_AGAIN:
            self.tracker.reset(); self.debounce.reset(); self.recog.reset(); self._announced.clear(); self.find = None
            if self.ocr:
                self.ocr.clear_cache()
            if self.barcode:
                self.barcode.reset()
            self.speech.say(self.narr.simple("scan_again").text, Priority.USER)
            cmd = Command.IDENTIFY
        # ---- commands that analyse the current frame --------------------------------
        frame = self._best_frame()
        q = assess_frame(frame, self.s.min_sharpness, self.s.min_brightness)
        if not q.ok:
            return self._emit(cmd, self.narr.quality(q))
        if cmd in (Command.IDENTIFY, Command.DESCRIBE, Command.MORE, Command.CATEGORY):
            return self._identify_family(cmd, frame)
        dets = self._fresh_detections(frame)
        lines = self._read_text(frame, dets)
        if self.ocr is None:
            return self._emit(cmd, Narration("Text reading is not available on this computer.", InfoState.UNKNOWN))
        if cmd == Command.PRICE:
            n = self.narr.price(extract_prices(lines, min_confidence=self.s.price_min_confidence))
        elif cmd in (Command.INGREDIENTS, Command.FULL_INGREDIENTS):
            n = self.narr.ingredients(parse_food_label(lines), full=cmd == Command.FULL_INGREDIENTS, min_conf=self.s.ocr_min_confidence)
        elif cmd == Command.EXPIRY:
            n = self.narr.expiry(parse_food_label(lines))
        elif cmd == Command.LABEL:
            n = self.narr.label(lines, self.s.ocr_min_confidence)
        elif cmd == Command.READ_TEXT:                       # Level 3: everything, in reading order, in interruptible chunks
            chunks = self.talk.text_chunks(lines, self.s.ocr_min_confidence)
            if not chunks:
                return self._emit(cmd, self.narr.simple("label_none"), lines)
            for c in chunks:
                self.speech.say(c, Priority.USER)
            self.last_result = Result(cmd, " ".join(chunks), InfoState.TENTATIVE, list(lines))
            return self.last_result
        else:
            return None
        return self._emit(cmd, n, lines)

    def _emit(self, cmd, n: Narration, lines=(), product=None, card=None) -> Result:
        self.speech.say(n.text, Priority.USER)
        self.last_result = Result(cmd, n.text, n.state, list(lines), product, card)
        return self.last_result

    # ---- IDENTIFY / DESCRIBE / MORE / CATEGORY -----------------------------------------
    def _analyse(self, frame):
        """Fresh analysis of the current frame -> (rec, card, lines, product, dets, ai_candidate_ok)."""
        h, w = frame.shape[:2]
        dets = self._fresh_detections(frame)
        lines = self._read_text(frame, dets)
        record = self._barcode_record(frame)
        product = self.identifier.identify([d.label for d in dets], lines, record)
        ocr_ev = None
        if product.category:
            e = self.vocab.find(product.category)
            if e:
                ocr_ev = (e, 0.85)
        bar_ev = None
        if product.level == "specific" and not product.conflict:
            e = self.vocab.find(product.name or "")
            bar_ev = (product.name, e.category if e else "Packaged Goods", e.subcategory if e else "Packaged Snack",
                      0.95 if product.state == InfoState.CONFIRMED else 0.7)
        with self._lock:
            live = [(t, self._recs.get(t.id)) for t in self.tracker.tracks if t.misses == 0]
        thr = self.s.zero_shot_threshold
        if live:
            t, _ = max(live, key=lambda x: self._primary_score(x[0].box, w, h))
            self.recog.set_evidence(t.id, ocr_ev, bar_ev)
            rec, box = self.recog.recognition(t), tuple(t.box)
        else:
            if dets:
                d = dets[0]
                crop, box = crop_box(frame, d.box), d.box
                rec = fuse(self.taxonomy.map_detector(d.label), d.confidence, self.recog.classify_now(crop), thr, ocr_ev, bar_ev)
            else:
                box = center_box(frame)
                rec = fuse(None, 0.0, self.recog.classify_now(crop_box(frame, box, 0.0)), thr, ocr_ev, bar_ev)
        crop = crop_box(frame, box, 0.0)
        veg, rows = None, None
        if rec.category in ("Food and Drink", "Packaged Goods"):           # label-bearing categories only
            vm = classify_veg_mark(crop_box(frame, box, 0.1))
            veg = (vm.label, vm.confidence) if vm.label else None
            rows = parse_table(lines).phrases or None                      # split-column nutrition tables
        card = self.cards.build(rec, lines, record, None, dominant_colour(crop), veg_mark=veg, nutrition_rows=rows,
                                position=position_phrase(box, w, h))
        return rec, card, lines, product

    def _identify_family(self, cmd: Command, frame) -> Result:
        fresh = self.last_card is not None and self._clock() - self._last_card_time < 45 and cmd in (Command.MORE, Command.CATEGORY)
        if fresh:
            card, rec, lines, product = self.last_card, self._last_rec, [], None
        else:
            rec, card, lines, product = self._analyse(frame)
            self.last_card, self._last_rec, self._last_card_time = card, rec, self._clock()
        if cmd == Command.CATEGORY:
            return self._emit(cmd, Narration(self.talk.category(rec), rec.confidence >= 0.6 and InfoState.TENTATIVE or InfoState.UNKNOWN), lines, product, card)
        use_ai = False
        if cmd in (Command.IDENTIFY, Command.DESCRIBE) and rec.category != "Person" and not rec.specific_product:
            ready, _ = self.cloud_ready()
            use_ai = ready and (cmd == Command.DESCRIBE or rec.generic or rec.confidence < self.s.cloud_low_confidence)
        position = card.position
        l1 = self.talk.level1(rec, position, self._hazard(rec))
        if use_ai:
            self.speech.say(self.narr.simple("cloud_wait").text, Priority.USER)
            res = self._cloud_call(frame, [d.label for d in self._dets], True)
            if res.status == "ok" and res.data["can_identify"]:
                card = self.cards.build(rec, lines, None, res.data, card.attributes.get("colour"), position=position)
                self.last_card = card
                if cmd == Command.DESCRIBE:
                    return self._emit(cmd, Narration(self.talk.level2(card), InfoState.TENTATIVE), lines, product, card)
                self.speech.say(l1, Priority.USER)
                return self._emit(cmd, self.narr.cloud(res), lines, product, card)
            online = self.narr.cloud(res)                     # unreachable / unsure: say why, then what the local models found
            local = self.talk.level2(card) if cmd == Command.DESCRIBE else l1
            return self._emit(cmd, Narration(f"{online.text} {local}", InfoState.TENTATIVE), lines, product, card)
        text = self.talk.level2(card) if cmd in (Command.DESCRIBE, Command.MORE) else l1
        state = InfoState.CONFIRMED if rec.specific_product and rec.confidence >= 0.9 else (
            InfoState.UNKNOWN if rec.label == "unknown object" else InfoState.TENTATIVE)
        return self._emit(cmd, Narration(text, state), lines, product, card)

    # ---- find / scene -------------------------------------------------------------------
    def _start_find(self, target: Optional[str]) -> Result:
        if not target:
            if self.find is not None:
                self.find = None
                return self._say_user("Find mode off.", InfoState.CONFIRMED, Command.FIND)
            target = self._last_announced_label
            if not target:
                return self._say_user("Say what to find, for example: find my cup.", InfoState.UNKNOWN, Command.FIND)
        entry = self.vocab.find(target)
        norm = (entry.name if entry else target).lower()

        def matcher(rec) -> bool:
            if entry is not None:
                e = self.vocab.find(rec.label)
                if e is not None and e.name == entry.name:
                    return True
                if any((self.vocab.find(a) or entry.__class__("", "", "")).name == entry.name for a, _ in rec.alternatives[:1]):
                    return True
            return norm in rec.label.lower() or rec.label.lower() in norm

        self.find = FindMode(target, matcher, self._clock)
        return self._say_user(f"Looking for your {target}. Move the camera slowly. I will beep faster and higher as it gets closer "
                              "to the centre. Say find again to stop.", InfoState.TENTATIVE, Command.FIND)

    def _scene(self) -> Result:
        w, h = self._frame_wh
        with self._lock:
            items = [(self._recs[t[0]], t[3]) for t in self._tracks_view if t[0] in self._recs]
        if not items:
            c = self.recog.center_recognition()
            if c is not None:
                items = [(c, center_box(__import__("numpy").zeros((h, w, 3), "uint8")))]
        return self._say_user(self.talk.scene(items, w), InfoState.TENTATIVE if items else InfoState.UNKNOWN, Command.SCENE)

    # ================= helpers =================
    def _best_frame(self):
        """Sample a few frames and keep the sharpest (cheap protection against motion blur)."""
        best, best_s = None, -1.0
        for i in range(max(1, self.s.capture_samples)):
            f = self.camera.read()
            if f is not None:
                try:
                    sc = sharpness(f)
                except Exception:  # noqa: BLE001
                    sc = 0.0
                if sc > best_s:
                    best, best_s = f, sc
            if i < self.s.capture_samples - 1:
                self._sleep(0.1)
        return best

    def _fresh_detections(self, frame) -> list:
        if self.detector is None or frame is None:
            return []
        try:
            return [d for d in self.detector.detect(frame, self.s.confidence_threshold) if d.label not in self.s.ignore_classes]
        except Exception:  # noqa: BLE001
            log.exception("detector failed during scan")
            return []

    def _read_text(self, frame, dets) -> list[OcrLine]:
        if self.ocr is None:
            return []
        t0 = time.perf_counter()
        lines: list[OcrLine] = []
        if dets:   # try the largest object's region first (label region heuristic), then the full frame
            crop = crop_box(frame, dets[0].box, 0.1)
            if crop is not None:
                lines = self.ocr.read(crop)
        if len(lines) < 2:
            lines = self.ocr.read(frame)
        self.metrics["ocr_ms"] = (time.perf_counter() - t0) * 1000
        return lines

    def _barcode_record(self, frame):
        if self.barcode is None:
            return None
        for code in self.barcode.read(frame):
            rec = self.store.get(code)
            if rec is None and self.s.online_lookup and self.lookup is not None:
                rec = self.lookup.fetch(code).record
            if rec:
                return rec
        return None

    # ================= UI helpers =================
    def preview_frame(self):
        """Downscaled frame with boxes (smaller = less lag in the UI)."""
        frame = self.camera.read()
        if frame is None:
            return None
        try:
            import cv2
            from .detector import Detection, draw_detections
            h, w = frame.shape[:2]
            k = min(1.0, self.s.preview_width / w)
            if k < 1.0:
                frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            dets = [Detection(label, c, tuple(int(v * k) for v in box)) for _i, label, c, box in self.overlay_tracks()]
            return draw_detections(frame, dets)
        except Exception:  # noqa: BLE001
            return frame

    def snapshot(self) -> dict:
        r = self.last_result
        return {
            "camera": self.camera_status, "inference": self.inference_status,
            "objects": [f"{label} ({c:.2f})" for _i, label, c, _b in self.overlay_tracks()],
            "last_text": r.text if r else "", "state": r.state if r else None,
            "transcript": list(self.speech.history)[-5:], "metrics": dict(self.metrics),
            "cloud": self.cloud_status(), "card": self.last_card, "find": self.find.target if self.find else None,
        }
