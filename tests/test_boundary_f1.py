"""Gradient loss and boundary F1 (spec 020 FR-002)."""

import numpy as np
import pytest
import torch

from strike_a_pose.eval import BOUNDARY_TOL_PX, boundary_f1, gradient_loss


def test_gradient_loss_zero_for_constant_offset():
    x = torch.rand(2, 1, 8, 6, generator=torch.Generator().manual_seed(0))
    assert abs(float(gradient_loss(x + 0.3, x))) < 1e-7
    assert float(gradient_loss(x, x)) == 0.0


def test_gradient_loss_value():
    x = torch.zeros(1, 1, 3, 3)
    y = x.clone()
    y[0, 0, 1, 1] = 1.0
    yg = y.clone().requires_grad_(True)
    loss = gradient_loss(yg, x)
    assert float(loss) == pytest.approx(4 / 6)
    loss.backward()
    assert torch.isfinite(yg.grad).all()
    assert float(yg.grad[0, 0, 1, 1]) != 0.0


def test_boundary_tol_constant():
    assert BOUNDARY_TOL_PX == 1


def test_boundary_f1_identical_and_empty():
    m = torch.zeros(2, 10, 10, dtype=torch.bool)
    m[0, 3:7, 3:7] = True
    assert boundary_f1(m.float(), m).tolist() == [1.0, 1.0]
    assert boundary_f1(torch.zeros(2, 10, 10), m).tolist() == [0.0, 1.0]


def test_boundary_f1_shift_tolerance():
    m = torch.zeros(1, 12, 12, dtype=torch.bool)
    m[0, 3:7, 3:7] = True
    p1 = torch.zeros(1, 12, 12)
    p1[0, 3:7, 4:8] = 1.0
    p2 = torch.zeros(1, 12, 12)
    p2[0, 3:7, 5:9] = 1.0
    assert float(boundary_f1(p1, m, tol_px=1)[0]) == 1.0
    assert float(boundary_f1(p2, m, tol_px=1)[0]) < 1.0
    assert float(boundary_f1(p2, m, tol_px=2)[0]) == 1.0


def _np_contour(b: np.ndarray) -> np.ndarray:
    h, w = b.shape
    pad = np.pad(b, 1, constant_values=True)
    er = np.ones_like(b)
    for dy in range(3):
        for dx in range(3):
            er &= pad[dy : dy + h, dx : dx + w]
    return b & ~er


def _np_near(c: np.ndarray, tol: int) -> np.ndarray:
    h, w = c.shape
    out = np.zeros_like(c)
    for y, x in zip(*np.nonzero(c), strict=True):
        out[max(0, y - tol) : y + tol + 1, max(0, x - tol) : x + tol + 1] = True
    return out


def test_boundary_f1_missing_limb_lowers_recall():
    m = np.zeros((16, 16), dtype=bool)
    m[2:6, 2:6] = True  # 4x4 square
    m[4, 7:12] = True  # 1x5 arm
    pred = np.zeros((16, 16), dtype=bool)
    pred[2:6, 2:6] = True
    cp, cm = _np_contour(pred), _np_contour(m)
    prec = (cp & _np_near(cm, 1)).sum() / cp.sum()
    rec = (cm & _np_near(cp, 1)).sum() / cm.sum()
    want = 2 * prec * rec / (prec + rec)
    assert prec == 1.0 and rec < 1.0 and 0.5 < want < 1.0
    got = boundary_f1(torch.from_numpy(pred[None]).float(), torch.from_numpy(m[None]))
    assert float(got[0]) == pytest.approx(want)


def test_boundary_f1_flat_input_and_threshold():
    m = torch.zeros(2, 10, 10, dtype=torch.bool)
    m[0, 3:7, 3:7] = True
    flat_m = m.reshape(2, -1)
    assert boundary_f1(m.float().reshape(2, -1), flat_m, image_size=(10, 10)).tolist() == [1, 1]
    low = (m.float() * 0.07).reshape(2, -1)
    assert boundary_f1(low, flat_m, image_size=(10, 10)).tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="image_size"):
        boundary_f1(low, flat_m)


def test_boundary_f1_numpy_in_numpy_out():
    m = np.zeros((2, 10, 10), dtype=bool)
    m[0, 3:7, 3:7] = True
    out = boundary_f1(m.astype(np.float32), m)
    assert isinstance(out, np.ndarray) and out.shape == (2,)
    assert out.tolist() == [1.0, 1.0]
