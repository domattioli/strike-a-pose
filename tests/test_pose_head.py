"""Pose head, target statistics and loss (spec 018 FR-002)."""

import pytest
import torch

from strike_a_pose.models import PoseHead, PoseTargetStats
from strike_a_pose.models.pose_head import pose_loss, root_relative


def test_root_relative_zeroes_the_root():
    j = torch.arange(2 * 24 * 3, dtype=torch.float32).reshape(2, 24, 3)
    r = root_relative(j)
    assert r.shape == (2, 72)
    assert (r[:, :3] == 0).all()
    assert r[0, 3:6].tolist() == [3.0, 3.0, 3.0]


def test_head_shapes_and_parameter_count():
    torch.manual_seed(0)
    h = PoseHead(64)
    assert h(torch.zeros(5, 64)).shape == (5, 72)
    assert h.parameter_count() == 100936
    assert h.hparams() == {"latent_dim": 64, "hidden": [256, 256], "d_out": 72}
    PoseHead(**h.hparams())


def _y():
    y = torch.randn(100, 72, generator=torch.Generator().manual_seed(0))
    y[:, :3] = 0.0
    y[:, 3] = 0.25
    return y


def test_stats_standardize_round_trip_and_inactive():
    y = _y()
    s = PoseTargetStats.fit(y)
    assert s.active.sum() == 68
    assert not s.active[:4].any()
    assert (s.scale[:4] == 1).all()
    assert torch.allclose(s.unstandardize(s.standardize(y)), y, atol=1e-6)
    z = s.standardize(y)[:, 4:]
    assert z.mean(0).abs().max() < 1e-5
    assert (z.std(0, unbiased=False) - 1).abs().max() < 1e-4
    assert s.unstandardize(torch.full((1, 72), 7.0))[0, 3] == pytest.approx(0.25)


def test_stats_dict_round_trip(tmp_path):
    s = PoseTargetStats.fit(_y())
    t = PoseTargetStats.from_dict(s.to_dict())
    assert torch.equal(s.mean, t.mean)
    assert torch.equal(s.scale, t.scale)
    assert torch.equal(s.active, t.active)
    torch.save(s.to_dict(), tmp_path / "s.pt")
    back = torch.load(tmp_path / "s.pt", weights_only=True)
    assert torch.equal(back["mean"], s.mean)


def test_stats_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        PoseTargetStats.fit(torch.empty(0, 72))


def test_pose_loss_ignores_inactive():
    active = torch.ones(72, dtype=torch.bool)
    active[:3] = False
    pred = torch.zeros(4, 72, requires_grad=True)
    t = torch.ones(4, 72)
    t[:, :3] = 100.0
    loss = pose_loss(pred, t, active)
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert (pred.grad[:, :3] == 0).all()
    assert (pred.grad[:, 3:] != 0).all()
