"""Integration test of the full pipeline with FAKE camera, detector and OCR (no hardware)."""
import time
import numpy as np
from config.settings import Settings
from smartsight.camera import CameraError
from smartsight.detector import Detection
from smartsight.pipeline import Pipeline
from smartsight.product_lookup import LocalProductStore
from smartsight.speech_engine import Priority, SpeechController
from smartsight.utils import InfoState, OcrLine
from smartsight.voice_commands import Command as C
from tests.fakes import FakeBackend

rng = np.random.default_rng(1)
GOOD = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


class FakeCamera:
    def __init__(self, frame=GOOD, fail=False):
        self.frame, self.fail, self.started, self.stopped = frame, fail, False, 0
        self.disconnected, self.error, self.index = False, None, 0

    def start(self):
        if self.fail: raise CameraError("no camera")
        self.started = True

    def read(self): return None if not self.started else self.frame
    def stop(self): self.stopped += 1; self.started = False


class FakeDetector:
    def __init__(self, labels=("bottle",), boom=False): self.labels, self.boom = labels, boom
    def detect(self, frame, conf=0.5):
        if self.boom: raise RuntimeError("inference exploded")
        return [Detection(l, 0.9, (10, 10, 200, 200)) for l in self.labels]


class FakeOcr:
    def __init__(self, lines): self.lines, self.cleared = [OcrLine(t, 0.9, (0, 0, 10, 20)) for t in lines], 0
    def read(self, frame): return self.lines
    def clear_cache(self): self.cleared += 1


class NoBarcode:
    def read(self, f): return []
    def reset(self): pass


def make(lines=(), detector=None, camera=None, **kw):
    s = Settings.load(path="/none", env={}); s.capture_samples = 1; s.stable_frames = 2
    for k, v in kw.items(): setattr(s, k, v)
    b = FakeBackend()
    speech = SpeechController(b, dedupe_seconds=60)
    cam = camera or FakeCamera(); cam.started = True
    p = Pipeline(s, cam, detector if detector is not False else None, FakeOcr(lines), NoBarcode(),
                 LocalProductStore(), speech, sleep=lambda _t: None)
    return p, b, cam


def test_price_command_end_to_end():
    p, b, _ = make(["PEPPY CRISPS", "Net Wt 120 g", "MRP Rs. 120/-"], FakeDetector())
    r = p.process_command(C.PRICE)
    p.speech.wait_idle(2)
    assert r.state == InfoState.CONFIRMED and "one hundred and twenty rupees" in b.spoken[-1]


def test_price_unreadable_says_so():
    p, b, _ = make(["PEPPY", "Net Wt 120 g"], FakeDetector())
    r = p.process_command(C.PRICE); p.speech.wait_idle(2)
    assert r.state == InfoState.UNKNOWN and "cannot confirm" in b.spoken[-1]


def test_identify_shampoo_bottle():
    p, b, _ = make(["HERBAL SHAMPOO", "Ingredients: water, salt"], FakeDetector())
    r = p.process_command(C.IDENTIFY); p.speech.wait_idle(2)
    assert r.product.category == "shampoo" and "shampoo bottle" in b.spoken[-1]


def test_ingredients_and_expiry():
    p, b, _ = make(["Ingredients: Potato, Palm Oil, Salt.", "Best before 6 months from manufacture"], FakeDetector())
    p.process_command(C.INGREDIENTS); p.process_command(C.EXPIRY); p.speech.wait_idle(2)
    assert "Potato, Palm Oil, Salt" in b.spoken[0] and "6 months from manufacture" in b.spoken[1]


def test_dark_frame_asks_for_light_instead_of_guessing():
    dark = np.zeros((240, 320, 3), dtype=np.uint8)
    p, b, _ = make(["MRP Rs. 5"], FakeDetector(), camera=FakeCamera(dark))
    r = p.process_command(C.PRICE); p.speech.wait_idle(2)
    assert r.state == InfoState.UNKNOWN and "too dark" in b.spoken[-1]


def test_works_without_detector_and_ocr_none():
    p, b, _ = make(["MRP Rs. 10"], detector=False)
    assert p.process_command(C.PRICE).state == InfoState.CONFIRMED


def test_stationary_object_announced_once():
    p, b, _ = make(detector=FakeDetector(("bottle",)))
    for _ in range(40): p.step()
    p.speech.wait_idle(2)
    assert len(b.spoken) == 1 and b.spoken[0].startswith("Household item, bottle, 90 percent sure.")
    assert "Say more for details." in b.spoken[0]


def test_ignored_class_not_announced():
    p, b, _ = make(detector=FakeDetector(("person",)), ignore_classes=["person"])
    for _ in range(10): p.step()
    p.speech.wait_idle(1)
    assert b.spoken == []


def test_inference_error_is_contained_and_spoken_once():
    p, b, _ = make(detector=FakeDetector(boom=True))
    for _ in range(5): p.step()
    p.speech.wait_idle(2)
    assert "error" in p.inference_status and len(b.spoken) == 1


def test_camera_failure_gives_audible_error():
    p, b, _ = make(camera=FakeCamera(fail=True), detector=FakeDetector())
    assert p.start() is False
    p.speech.wait_idle(2)
    assert "could not use the camera" in b.spoken[0] and p.camera_status.startswith("error")


def test_start_stop_releases_camera_and_thread():
    p, b, cam = make(detector=FakeDetector())
    cam.started = False
    assert p.start() and p._thread.is_alive()
    time.sleep(0.2)
    p.stop()
    assert cam.stopped >= 1 and not p._thread.is_alive() and p.camera_status == "stopped"
    p.close()


def test_stop_command_is_immediate_even_while_speaking():
    p, b, _ = make(detector=FakeDetector())
    b.delay = 2
    p.speech.say("a very long answer", Priority.USER)
    time.sleep(0.1)
    p.request(C.STOP)
    assert p.speech.wait_idle(2)


def test_scan_again_resets_state():
    p, b, _ = make(["HERBAL SHAMPOO"], FakeDetector())
    p.process_command(C.SCAN_AGAIN); p.speech.wait_idle(2)
    assert p.ocr.cleared == 1 and "Scanning again" in b.spoken[0]


def test_command_exception_contained():
    p, b, _ = make(["x"], FakeDetector())
    p.ocr.read = lambda f: (_ for _ in ()).throw(RuntimeError("ocr died"))
    assert p.process_command(C.PRICE) is None
    p.speech.wait_idle(2)
    assert "Text reading is not available" in b.spoken[-1]      # plain-language error with the fix
