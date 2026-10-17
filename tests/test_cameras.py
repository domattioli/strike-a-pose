import numpy as np

from strike_a_pose.datagen.cameras import RING8_NAMES, camera_ring, project, visibility
from strike_a_pose.datagen.config import CameraRingConfig


def _ring(count=8):
    return camera_ring(CameraRingConfig(count=count), 256, 192)


def test_ring_layout():
    cams = _ring()
    assert [c.name for c in cams] == list(RING8_NAMES)
    centres = [-c.R.T @ c.t for c in cams]
    assert np.allclose(centres[0], [0, 0, 3.5])
    assert centres[1][0] < 0 and centres[1][2] > 0  # FrontRight_45: subject's right (-X)
    assert np.allclose(centres[2], [-3.5, 0, 0])
    for c in cams:
        assert np.allclose(c.R @ c.R.T, np.eye(3)) and np.isclose(np.linalg.det(c.R), 1)
    assert [c.name for c in _ring(4)] == ["Cam0_0", "Cam1_90", "Cam2_180", "Cam3_270"]


def test_projection_conventions():
    cam = _ring()[0]
    xc, uv = project(cam, np.array([[0, 0, 0], [0, 0.5, 0], [0.5, 0, 0]]))
    assert np.allclose(uv[0], [96, 128]) and np.isclose(xc[0, 2], 3.5)
    assert uv[1, 1] < 128  # +Y is up in the image
    assert uv[2, 0] > 96  # subject's left (+X) appears on image right from the front


def test_visibility():
    cam = _ring()[0]
    pts = np.array([[0, 0, 0], [10, 0, 0], [0, 0, 5]])
    xc, uv = project(cam, pts)
    assert visibility(xc, uv, 256, 192).tolist() == [True, False, False]
