from smartsight.camera import Camera, CameraError
import pytest


def test_missing_camera_raises_clean_error_and_stop_is_safe():
    cam = Camera(index=99)
    with pytest.raises(CameraError):
        cam.start()
    cam.stop()          # must not raise
    assert cam.read() is None and not cam.running
