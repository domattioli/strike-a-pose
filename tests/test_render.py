import numpy as np

from strike_a_pose.datagen.cameras import Camera
from strike_a_pose.datagen.render import render

H, W = 40, 50
CAM = Camera("c", np.array([[10.0, 0, 25], [0, 10.0, 20], [0, 0, 1]]), np.eye(3), np.zeros(3),
             W, H, 0.0)  # fmt: skip


def _px_tri(pts_uv, z=1.0):
    """World verts whose projection (focal 10, centre (25, 20)) is pts_uv at depth z."""
    p = np.asarray(pts_uv, float) - [25, 20]
    return np.column_stack([p[:, 0] * z / 10, p[:, 1] * z / 10, np.full(len(p), z)])


def test_empty():
    rgb, mask = render(np.zeros((0, 3)), np.zeros((0, 3), int), CAM)
    assert rgb.sum() == 0 and mask.sum() == 0


def test_square_coverage():
    v = _px_tri([[10, 10], [30, 10], [30, 30], [10, 30]])
    rgb, mask = render(v, np.array([[0, 1, 2], [0, 2, 3]]), CAM)
    assert (mask > 0).sum() == 400  # pixel centres 10.5..29.5 -> 20x20
    assert mask[10:30, 10:30].all() and rgb[mask == 0].sum() == 0
    assert rgb[15, 15, 0] == rgb[15, 15, 1] == rgb[15, 15, 2]
    assert round(255 * 0.15) <= rgb[15, 15, 0] <= round(255 * 0.8)


def test_shading_on_axis():
    v = _px_tri([[15, 10], [35, 10], [25, 30]])
    v[:, 0] -= v[:, 0].mean()
    v[:, 1] -= v[:, 1].mean()  # centroid on the optical axis, facing the camera
    rgb, _ = render(v, np.array([[0, 1, 2]]), CAM)
    assert rgb[20, 25, 0] == round(255 * 0.8)


def test_occlusion_and_determinism():
    near = _px_tri([[5, 5], [25, 5], [5, 25]], z=1.0)
    far_ = _px_tri([[5, 5], [25, 5], [5, 25]], z=2.0)
    tilted = far_.copy()
    tilted[1, 2] = 3.0
    v = np.concatenate([tilted, near])
    rgb, mask = render(v, np.array([[0, 1, 2], [3, 4, 5]]), CAM)
    rgb2, _ = render(v, np.array([[3, 4, 5], [0, 1, 2]]), CAM)
    assert np.array_equal(rgb, rgb2)  # nearer face wins regardless of order
    near_only, _ = render(near, np.array([[0, 1, 2]]), CAM)
    assert rgb[8, 8, 0] == near_only[8, 8, 0]  # near face visible over the far one
    rgb3, mask3 = render(v, np.array([[0, 1, 2], [3, 4, 5]]), CAM)
    assert np.array_equal(rgb, rgb3) and np.array_equal(mask, mask3)


def test_behind_camera_skipped():
    v = _px_tri([[5, 5], [25, 5], [5, 25]], z=1.0) * np.array([1, 1, -1])
    _, mask = render(v, np.array([[0, 1, 2]]), CAM)
    assert mask.sum() == 0
