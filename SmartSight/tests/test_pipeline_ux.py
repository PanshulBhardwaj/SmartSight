import time
import numpy as np
from config.settings import Settings
from smartsight.detector import Detection
from smartsight.earcons import Earcons, RecordingBackend
from smartsight.pipeline import Pipeline
from smartsight.product_lookup import LocalProductStore
from smartsight.recognizer import Candidate
from smartsight.selfcheck import run_selfcheck
from smartsight.speech_engine import Priority, SpeechController
from smartsight.utils import InfoState, OcrLine
from smartsight.voice_commands import Command as C
from tests.fakes import FakeBackend
from tests.test_pipeline import FakeCamera, FakeOcr, GOOD, NoBarcode


class Det:
    """Scriptable detector: .dets can be changed between steps."""
    def __init__(self, dets=None, conf=0.9): self.dets = dets if dets is not None else [("bottle", (60, 40, 140, 200))]; self.conf = conf
    def detect(self, frame, conf=0.5): return [Detection(l, self.conf, b) for l, b in self.dets]


class ZS:
    name = "fake-zs"
    def __init__(self, cands): self.cands, self.calls = cands, 0
    def classify(self, crop, top_k=3): self.calls += 1; return self.cands


def make(lines=(), det=None, recognizer=None, cam=None, tmp=None, **flags):
    s = Settings.load(path="/none", env={}); s.capture_samples = 1; s.stable_frames = 2; s.cloud_ai = False
    for k, v in flags.items(): setattr(s, k, v)
    b = FakeBackend(); eb = RecordingBackend()
    camera = cam or FakeCamera(); camera.started = True
    p = Pipeline(s, camera, det or Det(), FakeOcr(lines), NoBarcode(), LocalProductStore(), SpeechController(b, dedupe_seconds=0),
                 sleep=lambda t: None, recognizer=recognizer, async_tiers=False, earcons=Earcons(eb))
    return p, b, eb


def run(p, n=6):
    for _ in range(n): p.step()
    p.speech.wait_idle(2)


def test_identify_then_more_reads_the_full_card():
    p, b, _ = make(["HERBAL SHAMPOO", "Net Wt 200 ml"])
    p.process_command(C.IDENTIFY); p.process_command(C.MORE); p.speech.wait_idle(2)
    l1, l2 = b.spoken[-2], b.spoken[-1]
    assert l1.startswith("Packaged goods, shampoo bottle") and "Say more for details." in l1
    assert l2.startswith("Shampoo bottle. Packaged goods, personal care product.") and "Used for: washing hair" in l2
    assert "Size: 200 ml" in l2 and "Text I can read: HERBAL SHAMPOO" in l2


def test_category_command():
    p, b, _ = make(["HERBAL SHAMPOO"])
    p.process_command(C.CATEGORY); p.speech.wait_idle(2)
    assert b.spoken[-1] == "Category: packaged goods, personal care product."


def test_read_text_speaks_everything_in_order_in_chunks():
    lines = [OcrLine("second line", 0.9, (0, 80, 50, 100)), OcrLine("first line", 0.9, (0, 10, 50, 30))]
    p, b, _ = make(det=Det([]))
    p.ocr.lines = lines
    r = p.process_command(C.READ_TEXT); p.speech.wait_idle(2)
    assert b.spoken[0].startswith("Text I can read: first line") and "second line" in " ".join(b.spoken) and r.ocr_lines


def test_nothing_recognised_says_not_sure_and_asks_to_move_closer():
    p, b, _ = make(det=Det([]))
    r = p.process_command(C.IDENTIFY); p.speech.wait_idle(2)
    assert r.state == InfoState.UNKNOWN and "not sure what this is" in b.spoken[-1] and "move it closer" in b.spoken[-1]


def test_level1_is_hedged_for_low_confidence():
    p, b, _ = make(det=Det([("cup", (60, 40, 140, 200))], conf=0.45))
    run(p)
    assert b.spoken[0].startswith("I think this is a cup, in household item, 45 percent sure.")


def test_announcement_includes_category_position_and_earcon():
    p, b, e = make(det=Det([("apple", (0, 60, 60, 160))]))
    run(p); time.sleep(0.1)
    assert b.spoken[0].startswith("Food and drink, apple, 90 percent sure.") and "Far to your left" in b.spoken[0]
    assert (1000, 70) in e.played                                           # 'locked' earcon


def test_lost_earcon_and_reannounce_after_return():
    d = Det([("apple", (100, 60, 220, 200))])
    p, b, e = make(det=d, tracker_max_misses=2, announce_cooldown=0.0)
    run(p)
    d.dets = []; run(p, 6); time.sleep(0.1)
    assert (500, 110) in e.played                                           # 'lost' earcon
    d.dets = [("apple", (100, 60, 220, 200))]
    time.sleep(1.6); run(p, 6)                                              # past the 1.5 s global gap
    assert len([t for t in b.spoken if t.startswith("Food and drink, apple")]) == 2


def test_person_gets_position_only_and_is_never_sent_to_tier2_or_cloud():
    zs = ZS([Candidate("doll", 0.9, "Household Item", "Decor and Miscellaneous")])
    p, b, _ = make(det=Det([("person", (0, 40, 120, 220))]), recognizer=zs)
    run(p)
    assert b.spoken[0].startswith("Person, ") and "percent" not in b.spoken[0] and zs.calls == 0


def test_tier2_refines_the_spoken_label():
    zs = ZS([Candidate("shampoo bottle", 0.8, "Packaged Goods", "Personal Care Product")])
    p, b, _ = make(recognizer=zs)
    run(p)
    assert b.spoken[0].startswith("Packaged goods, shampoo bottle") and zs.calls == 1


def test_scene_scan_lists_left_to_right_grouped_by_category():
    d = Det([("cell phone", (240, 60, 300, 160)), ("bottle", (10, 60, 70, 160)), ("cup", (120, 60, 180, 160))])
    p, b, _ = make(det=d)
    run(p)
    p.process_command(C.SCENE); p.speech.wait_idle(2)
    t = b.spoken[-1]
    assert t.startswith("I can see 3 objects, listed from left to right.")
    assert t.index("Household item: a bottle") < t.index("Electronics: a mobile phone")
    p2, b2, _ = make(det=Det([]))
    p2.process_command(C.SCENE); p2.speech.wait_idle(2)
    assert "do not see any objects" in b2.spoken[-1]


def test_find_mode_speaks_beeps_and_stays_quiet_otherwise():
    d = Det([("cup", (10, 60, 60, 160)), ("bottle", (240, 60, 300, 160))])
    p, b, e = make(det=d)
    p.process_command((C.FIND, "my cup")) if False else p.process_command((C.FIND, "cup"))
    p.speech.wait_idle(2)
    assert b.spoken[-1].startswith("Looking for your cup")
    run(p)
    assert any(t.startswith("I can see your cup, on your left") or t.startswith("I can see your cup, far to your left") for t in b.spoken)
    assert not any(t.startswith("Household item") for t in b.spoken)         # no normal chatter while finding
    d.dets = [("cup", (130, 60, 190, 160))]
    run(p)
    assert any("Found your cup, centred" in t for t in b.spoken) and (1500, 160) in e.played
    p.process_command(C.FIND); p.speech.wait_idle(2)
    assert b.spoken[-1] == "Find mode off." and p.find is None


def test_find_without_target_or_history():
    p, b, _ = make(det=Det([]))
    p.process_command(C.FIND); p.speech.wait_idle(2)
    assert "find my cup" in b.spoken[-1]


def test_speed_verbosity_help_quit(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.USER_CONFIG_PATH", tmp_path / "u.yaml")
    p, b, _ = make(det=Det([])); p.persist = True
    p.speech.backend.rate = 170
    p.process_command(C.FASTER); p.process_command(C.SLOWER); p.process_command(C.SLOWER); p.speech.wait_idle(2)
    assert b.spoken[-3:] == ["Speech speed 195 words per minute.", "Speech speed 170 words per minute.", "Speech speed 145 words per minute."]
    assert [p.process_command(C.VERBOSITY).text for _ in range(3)] == ["Verbosity detailed.", "Verbosity concise.", "Verbosity standard."]
    saved = Settings.load(tmp_path / "u.yaml", env={})
    assert saved.speech_rate == 145 and saved.verbosity == "standard"          # settings persist between runs
    assert "what is this" in p.process_command(C.HELP).text
    assert not p.quit_event.is_set(); p.process_command(C.QUIT); assert p.quit_event.is_set()
    p.speech.wait_idle(2); assert b.spoken[-1] == "Goodbye."


def test_rate_is_clamped():
    p, b, _ = make(det=Det([]))
    for _ in range(30): p.process_command(C.SLOWER)
    assert p.s.speech_rate == 80
    for _ in range(30): p.process_command(C.FASTER)
    assert p.s.speech_rate == 300


def test_framing_help_when_object_stays_badly_placed():
    clock = [0.0]
    s = Settings.load(path="/none", env={}); s.capture_samples = 1; s.stable_frames = 2; s.cloud_ai = False
    b = FakeBackend(); cam = FakeCamera(); cam.started = True
    p = Pipeline(s, cam, Det([("cup", (0, 60, 90, 200))]), FakeOcr([]), NoBarcode(), LocalProductStore(),
                 SpeechController(b, dedupe_seconds=0), sleep=lambda t: None, clock=lambda: clock[0], async_tiers=False)
    for _ in range(3): p.step()
    clock[0] = 4.0
    for _ in range(3): p.step()
    p.speech.wait_idle(2)
    hint = [t for t in b.spoken if "cut off" in t]
    assert hint and "left edge" in hint[0] and "Move the camera" in hint[0]
    clock[0] = 5.0
    for _ in range(3): p.step()
    p.speech.wait_idle(2); assert len([t for t in b.spoken if "cut off" in t]) == 1     # rate limited


def test_low_light_is_announced_once_not_continuously():
    dark = np.zeros((240, 320, 3), np.uint8)
    p, b, _ = make(cam=FakeCamera(dark))
    run(p, 30)
    assert len([t for t in b.spoken if "too dark" in t]) == 1


def test_selfcheck_reports_selected_engines_in_plain_language():
    p, b, _ = make(det=Det([]), cloud_ai=True)
    sc = run_selfcheck(p, {"ocr": "ok (easyocr)", "speech": "Pyttsx3Backend", "zero_shot": "x"})
    assert "Text reading uses easyocr" in sc.spoken and "not installed, so I will name only common objects" in sc.spoken
    assert "waiting for your permission" in sc.spoken and sc.spoken.startswith("Self check.")
    p2, b2, _ = make(cloud_ai=True, cloud_consent=True)
    assert "no API key" in run_selfcheck(p2, {}).spoken or "Online AI is on" in run_selfcheck(p2, {}).spoken


def test_start_speaks_self_check_and_error_has_fix():
    p, b, _ = make(det=Det([])); p.component_status = {"ocr": "ok (easyocr)", "speech": "Pyttsx3Backend"}
    p.camera.started = False
    assert p.start(); p.speech.wait_idle(2)
    assert b.spoken[0].startswith("Self check.") ; p.close()
    from smartsight.camera import CameraError
    bad = FakeCamera(fail=True)
    p2, b2, _ = make(cam=bad); bad.started = False
    assert p2.start() is False; p2.speech.wait_idle(2)
    assert "Close other programs that use it" in b2.spoken[0]
