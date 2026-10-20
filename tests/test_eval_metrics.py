"""Metric functions: hand-computed cases on NumPy and torch (spec 003 FR-001..004, SC-001)."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from strike_a_pose.data import PoseImageDataset
from strike_a_pose.eval.metrics import (
    IOU_THRESHOLD,
    image_mse,
    keypoint_error_2d,
    mpjpe,
    project_points,
    psnr,
    silhouette_iou,
)

FIXTURE = Path(__file__).parents[1] / "tests" / "fixtures" / "smpl_fixture"
BACKENDS = [np.asarray, lambda x: torch.as_tensor(np.asarray(x))]


def _np(y):
    return y.numpy() if isinstance(y, torch.Tensor) else y


@pytest.fixture(params=BACKENDS, ids=["numpy", "torch"])
def as_(request):
    return request.param


def _check_type(y, like, b):
    assert type(y) is type(like)
    assert tuple(y.shape) == (b,)


def test_image_mse_and_psnr_hand_computed(as_):
    # sample 0: errors (0.5, 0, 0, 0) -> mse 0.25/4 = 0.0625; sample 1: exact
    pred = as_(np.array([[[0.5, 0.0], [0.0, 0.0]], [[1.0, 0.2], [0.3, 0.4]]]))
    tgt = as_(np.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.2], [0.3, 0.4]]]))
    m = image_mse(pred, tgt)
    _check_type(m, pred, 2)
    np.testing.assert_allclose(_np(m), [0.0625, 0.0])
    p = psnr(pred, tgt)
    _check_type(p, pred, 2)
    # 10 log10(1/0.0625) = 12.0412 dB; mse 0 -> floor 1e-10 -> 100 dB
    np.testing.assert_allclose(_np(p), [10 * math.log10(16), 100.0], rtol=1e-12)
    np.testing.assert_allclose(_np(psnr(pred, tgt, max_val=2.0))[0], 10 * math.log10(64))


def test_image_mse_any_trailing_shape_and_mismatch(as_):
    x = as_(np.zeros((3, 1, 4, 5)))
    y = as_(np.ones((3, 1, 4, 5)))
    np.testing.assert_allclose(_np(image_mse(x, y)), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        image_mse(x, as_(np.ones((3, 20))))


def test_silhouette_iou_hand_computed(as_):
    pred = as_(np.array([[0.9, 0.9, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.075, 0.08, 0.5, 0.0]]))
    mask = as_(np.array([[1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 1]], dtype=bool))
    iou = silhouette_iou(pred, mask)
    _check_type(iou, pred, 3)
    # s0: inter 1, union 3; s1: empty/empty -> 1; s2: 0.075 is not > thr -> {1,2} vs {1,2,3}
    np.testing.assert_allclose(_np(iou), [1 / 3, 1.0, 2 / 3])
    np.testing.assert_allclose(_np(silhouette_iou(pred, mask, threshold=0.85))[2], 0.0)


def test_mpjpe_hand_computed_and_root_relative(as_):
    tgt = np.zeros((2, 3, 3))
    tgt[:, 1] = [1.0, 0.0, 0.0]
    tgt[:, 2] = [0.0, 1.0, 0.0]
    pred = tgt.copy()
    pred[0, 2] += [3.0, 4.0, 0.0]  # 3-4-5: one joint off by 5 -> mean 5/3
    pred[1] += [10.0, -2.0, 7.0]  # pure translation -> 0
    m = mpjpe(as_(pred), as_(tgt))
    _check_type(m, as_(pred), 2)
    np.testing.assert_allclose(_np(m), [5 / 3, 0.0], atol=1e-12)
    # root choice matters: rooting at the moved joint shifts the other two by 5 each
    np.testing.assert_allclose(_np(mpjpe(as_(pred), as_(tgt), root=2))[0], 10 / 3)
    with pytest.raises(ValueError):
        mpjpe(as_(pred[:, :, :2]), as_(tgt[:, :, :2]).reshape(2, 6))


def test_keypoint_error_2d_visibility(as_):
    tgt = np.zeros((3, 3, 2))
    pred = tgt.copy()
    pred[:, 0] = [3.0, 4.0]  # joint 0 off by 5 px
    pred[:, 1] = [0.0, 2.0]  # joint 1 off by 2 px
    vis = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=bool)
    e = keypoint_error_2d(as_(pred), as_(tgt), as_(vis))
    _check_type(e, as_(pred), 3)
    e = _np(e)
    np.testing.assert_allclose(e[:2], [7 / 3, 1.0])
    assert np.isnan(e[2])
    np.testing.assert_allclose(_np(keypoint_error_2d(as_(pred), as_(tgt))), [7 / 3] * 3)


def test_project_points_hand_computed(as_):
    K = np.array([[[100.0, 0, 50], [0, 200.0, 60], [0, 0, 1]]])
    R = np.eye(3)[None]
    t = np.array([[0.0, 0.0, 2.0]])
    x = np.array([[[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]]])  # z_c = 2 and 4
    uv = project_points(as_(x), as_(K), as_(R), as_(t))
    assert type(uv) is type(as_(x)) and tuple(uv.shape) == (1, 2, 2)
    np.testing.assert_allclose(_np(uv)[0], [[50.0, 60.0], [75.0, 10.0]])


def test_fixture_projection_and_threshold():
    """True labels re-project to the labelled 2D keypoints; true images threshold to masks."""
    ds = PoseImageDataset(FIXTURE)
    items = [ds[i] for i in range(len(ds))]
    j = torch.stack([s["joints_3d_world"] for s in items]).double()
    cam = {k: torch.stack([s["camera"][k] for s in items]).double() for k in ("K", "R", "t")}
    kp = torch.stack([s["keypoints_2d"] for s in items]).double()
    err = keypoint_error_2d(project_points(j, cam["K"], cam["R"], cam["t"]), kp)
    assert float(err.max()) < 1.0
    img = torch.stack([s["image"] for s in items])
    mask = torch.stack([s["mask"] for s in items])
    assert torch.equal(img > IOU_THRESHOLD, mask)
    assert torch.equal(silhouette_iou(img, mask), torch.ones(len(ds)))
    assert float(image_mse(img, img).max()) == 0.0


def test_float32_inputs_keep_dtype():
    x = torch.rand(2, 5, 3)
    assert mpjpe(x, x).dtype == torch.float32
    assert image_mse(x.numpy(), x.numpy()).dtype == np.float32
    assert silhouette_iou(x, x > 0.5).dtype == torch.float32
