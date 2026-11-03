import numpy as np
import pytest

from strike_a_pose.datagen.amass import enumerate_candidates, sample_poses
from strike_a_pose.datagen.assets import MissingAssetError


def test_candidates(synthetic_amass):
    c = enumerate_candidates(synthetic_amass, ["SubB", "SubA"], 10)
    files = [f for f, _ in c]
    assert files[0].startswith("SubB/") and not any("shape" in f for f in files)
    per_file = {f: [fr for g, fr in c if g == f] for f in dict.fromkeys(files)}
    assert per_file["SubB/subj0/seq0_poses.npz"] == list(range(0, 95, 10))
    assert len(c) == 2 * (10 + 7)


def test_sampling(synthetic_amass):
    a = sample_poses(synthetic_amass, ["SubA", "SubB"], 10, 5, seed=7)
    b = sample_poses(synthetic_amass, ["SubA", "SubB"], 10, 5, seed=7)
    c = sample_poses(synthetic_amass, ["SubA", "SubB"], 10, 5, seed=8)
    assert a == b and a != c
    assert [r.index for r in a] == list(range(5))
    for r in a:
        p = np.array(r.pose)
        assert p.shape == (72,) and np.all(p[:3] == 0) and np.all(p[66:] == 0)
        src = np.load(synthetic_amass / r.source_file)["poses"][r.frame]
        assert np.allclose(p[3:66], src[3:66]) and np.allclose(r.source_root_orient, src[:3])


def test_errors(synthetic_amass, tmp_path):
    with pytest.raises(MissingAssetError):
        sample_poses(synthetic_amass, ["Nope"], 10, 1, 0)
    with pytest.raises(MissingAssetError):
        sample_poses(tmp_path / "none", ["SubA"], 10, 1, 0)
    with pytest.raises(ValueError):
        sample_poses(synthetic_amass, ["SubA"], 10, 1000, 0)
