import numpy as np
from smartsight.frame_quality import assess_frame

rng = np.random.default_rng(0)


def test_good_frame():
    f = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert assess_frame(f).ok


def test_dark_frame():
    f = (rng.integers(0, 20, (240, 320, 3))).astype(np.uint8)
    assert "dark" in assess_frame(f).issues


def test_blurry_frame():
    f = np.full((240, 320, 3), 140, dtype=np.uint8)
    assert "blurry" in assess_frame(f).issues


def test_invalid_frames():
    for bad in (None, np.array([]), "x", np.zeros((3,))):
        r = assess_frame(bad)
        assert not r.ok and r.issues == ["invalid"]
