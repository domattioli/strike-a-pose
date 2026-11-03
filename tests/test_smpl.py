import numpy as np
import pytest

from strike_a_pose.datagen.assets import MissingAssetError
from strike_a_pose.datagen.smpl import SMPLModel, rodrigues

from .conftest import TPOSE


def test_rodrigues():
    assert np.allclose(rodrigues(np.zeros(3)), np.eye(3))
    r = rodrigues(np.array([0, 0, np.pi / 2]))
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0])
    rs = rodrigues(np.random.default_rng(0).normal(size=(5, 3)))
    assert np.allclose(rs @ rs.transpose(0, 2, 1), np.eye(3))
    assert np.allclose(np.linalg.det(rs), 1)


def test_zero_pose_is_template(synthetic_smpl_npz):
    m = SMPLModel.load(synthetic_smpl_npz)
    v, j = m.forward(np.zeros(72), np.zeros(10))
    assert np.allclose(v, m.v_template)
    assert np.allclose(j, TPOSE)


def test_hip_rotation_moves_leg_only(synthetic_smpl_npz):
    m = SMPLModel.load(synthetic_smpl_npz)
    pose = np.zeros(72)
    pose[3:6] = [np.pi / 2, 0, 0]  # left_hip 90 deg about X
    v, j = m.forward(pose, np.zeros(10))
    r = rodrigues(np.array([np.pi / 2, 0, 0]))
    for k in (4, 7, 10):  # left knee, ankle, foot follow the hip
        assert np.allclose(j[k], TPOSE[1] + r @ (TPOSE[k] - TPOSE[1]))
    for k in (0, 2, 5, 15, 21):
        assert np.allclose(j[k], TPOSE[k])
    knee = v[8 * 4 : 8 * 4 + 8]
    assert np.allclose(knee, TPOSE[1] + (m.v_template[32:40] - TPOSE[1]) @ r.T)


def test_missing_and_bad_model(tmp_path):
    with pytest.raises(MissingAssetError, match="docs/data-assets.md"):
        SMPLModel.load(tmp_path / "nope.npz")
    bad = tmp_path / "bad.npz"
    np.savez(bad, v_template=np.zeros((3, 3)))
    with pytest.raises(MissingAssetError):
        SMPLModel.load(bad)
