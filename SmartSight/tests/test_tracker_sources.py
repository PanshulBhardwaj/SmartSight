import threading
import time
import numpy as np
from smartsight.detector import Detection
from smartsight.metrics import FpsMeter, Percentiles
from smartsight.sources import SyntheticSource
from smartsight.tracker import IouTracker, iou
from tests.fakes import FakeClock


def D(label, box, c=0.9): return Detection(label, c, box)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_stable_id_while_object_moves_slightly():
    t = IouTracker()
    ids = set()
    for i in range(10):
        tr = t.update([D("cup", (100 + i * 4, 100, 200 + i * 4, 220))])
        ids.add(tr[0].id)
    assert len(ids) == 1 and t.tracks[0].streak == 10


def test_two_objects_get_two_ids_and_swap_free():
    t = IouTracker()
    a = t.update([D("cup", (0, 0, 100, 100)), D("bottle", (300, 0, 400, 100))])
    b = t.update([D("bottle", (302, 0, 402, 100)), D("cup", (2, 0, 102, 100))])
    ids = {x.label: x.id for x in a}
    assert {x.label: x.id for x in b} == ids


def test_track_survives_short_gap_then_expires():
    t = IouTracker(max_misses=2)
    tid = t.update([D("cup", (0, 0, 100, 100))])[0].id
    t.update([]); t.update([])
    assert t.update([D("cup", (0, 0, 100, 100))])[0].id == tid          # coasted through a 2-frame gap
    t.update([]); t.update([]); t.update([])
    assert t.tracks == []


def test_label_voting_removes_flicker():
    t = IouTracker(vote_window=8)
    box = (0, 0, 100, 100)
    for lab in ["cup", "cup", "mug", "cup", "cup", "mug", "cup"]:
        tr = t.update([D(lab, box)])[0]
    label, share = tr.vote()
    assert label == "cup" and share > 0.6


def test_percentiles_and_fps():
    p = Percentiles()
    for x in range(1, 101): p.add(x)
    assert p.p(50) in (50, 51) and p.p(95) in (95, 96)
    clock = FakeClock(); f = FpsMeter(1.0, clock)
    for _ in range(31):
        f.tick(); clock.advance(1 / 30)
    assert 28 < f.fps < 32


def test_synthetic_source_metadata_and_rate():
    src = SyntheticSource(320, 240, fps=60); src.start()
    time.sleep(0.5)
    f, ts, seq = src.read_meta()
    assert f.shape == (240, 320, 3) and seq > 15 and ts <= time.perf_counter()
    src.stop(); assert not src.running


def test_display_loop_never_waits_for_slow_inference():
    """Inference takes 200 ms; the display loop reading frames must still run at video rate."""
    from config.settings import Settings
    from smartsight.pipeline import Pipeline
    from smartsight.product_lookup import LocalProductStore
    from smartsight.speech_engine import SpeechController
    from tests.fakes import FakeBackend

    class Slow:
        def detect(self, frame, conf=0.5):
            time.sleep(0.2)
            return [Detection("bottle", 0.9, (10, 10, 100, 100))]

    class NoOcr:
        def read(self, f): return []
        def clear_cache(self): pass

    class NoBar:
        def read(self, f): return []
        def reset(self): pass

    s = Settings.load(path="/none", env={}); s.detection_fps = 30
    src = SyntheticSource(320, 240, fps=60)
    p = Pipeline(s, src, Slow(), NoOcr(), NoBar(), LocalProductStore(), SpeechController(FakeBackend()), sleep=lambda t: None)
    assert p.start()
    frames, last, t_end = 0, -1, time.perf_counter() + 1.0
    while time.perf_counter() < t_end:                  # the "display loop"
        m = src.read_meta(copy=False)
        if m and m[2] != last:
            last = m[2]; frames += 1
        time.sleep(0.002)
    infer_done = p.detect_ms.summary()["n"]
    p.close()
    assert frames >= 40                                  # ~60 fps source observed continuously
    assert 2 <= infer_done <= 7                          # while only ~5 slow inferences ran (frames skipped, never queued)
    assert abs(p.latency.summary()["n"] - infer_done) <= 1


def test_commands_do_not_block_detection():
    from config.settings import Settings
    from smartsight.pipeline import Pipeline
    from smartsight.product_lookup import LocalProductStore
    from smartsight.speech_engine import SpeechController
    from smartsight.voice_commands import Command as C
    from tests.fakes import FakeBackend

    class Fast:
        def detect(self, frame, conf=0.5): return [Detection("bottle", 0.9, (10, 10, 100, 100))]

    class SlowOcr:
        def read(self, f): time.sleep(0.8); return []
        def clear_cache(self): pass

    class NoBar:
        def read(self, f): return []
        def reset(self): pass

    s = Settings.load(path="/none", env={}); s.detection_fps = 30; s.capture_samples = 1
    src = SyntheticSource(320, 240, fps=60)
    p = Pipeline(s, src, Fast(), SlowOcr(), NoBar(), LocalProductStore(), SpeechController(FakeBackend()), sleep=lambda t: None)
    p.start(); time.sleep(0.3)
    before = p.detect_ms.summary()["n"]
    p.request(C.LABEL)                                   # triggers an 0.8 s OCR in the command thread
    time.sleep(0.6)
    during = p.detect_ms.summary()["n"] - before
    p.close()
    assert during >= 5                                   # detection kept running while OCR was busy
