import numpy as np
import pytest

from strike_a_pose.datagen.config import config_from_dict
from strike_a_pose.datagen.generate import generate_dataset
from strike_a_pose.datagen.io import load_manifest, load_sample, validate_dataset
from strike_a_pose.datagen.procedural import (
    RANGES_DEG,
    RANGES_VERSION,
    euler_xyz_to_matrix,
    matrix_to_axis_angle,
    sample_procedural_poses,
)


def _aa_to_matrix(v):
    t = np.linalg.norm(v)
    if t < 1e-12:
        return np.eye(3)
    k = v / t
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * kx + (1 - np.cos(t)) * kx @ kx


def test_ranges_table_well_formed():
    assert sorted(RANGES_DEG) == list(range(24))
    for axes in RANGES_DEG.values():
        assert len(axes) == 3 and all(lo <= 0 <= hi and isinstance(lo, int) for lo, hi in axes)
    assert RANGES_DEG[0] == ((0, 0), (0, 0), (0, 0))
    assert RANGES_DEG[18] == ((-60, 110), (-145, 0), (-35, 5))  # spot-check vs predecessor


def test_ranges_respected_sparsified_and_pose_matches_euler():
    poses = sample_procedural_poses(300, seed=7)
    frac = []
    for p in poses:
        deg = np.array(p.euler_deg)
        aa = np.array(p.pose).reshape(24, 3)
        assert (deg == np.round(deg)).all() and not deg[0].any()
        for j in range(24):
            for a in range(3):
                lo, hi = RANGES_DEG[j][a]
                v = deg[j, a]
                assert v == 0 or lo <= v < hi
            r = euler_xyz_to_matrix(np.radians(deg[j]))
            assert np.allclose(_aa_to_matrix(aa[j]), r, atol=1e-6)
        frac.append(np.count_nonzero(deg[1:]) / 69)
    # k uniform in [5, 69) kept -> mean kept fraction ~ 0.53; hands (6) and the
    # 4 zero-width axes are always 0, so the max nonzero fraction is <= 59/69
    assert 0.35 < np.mean(frac) < 0.65 and min(frac) < 0.2 and max(frac) > 0.75


def test_matches_scipy_from_euler_xyz():
    rot = pytest.importorskip("scipy.spatial.transform").Rotation
    rng = np.random.default_rng(0)
    for ang in rng.uniform(-170, 170, (200, 3)):
        ref = rot.from_euler("xyz", ang, degrees=True)
        assert np.allclose(euler_xyz_to_matrix(np.radians(ang)), ref.as_matrix(), atol=1e-9)
        assert np.allclose(
            matrix_to_axis_angle(euler_xyz_to_matrix(np.radians(ang))), ref.as_rotvec(), atol=1e-6
        )


def test_axis_angle_near_pi():
    r = euler_xyz_to_matrix(np.radians([179.9999999, 0, 0]))
    assert np.allclose(_aa_to_matrix(matrix_to_axis_angle(r)), r, atol=1e-6)


def test_deterministic():
    a, b = sample_procedural_poses(5, seed=3), sample_procedural_poses(5, seed=3)
    assert a == b and a != sample_procedural_poses(5, seed=4)


@pytest.fixture
def proc_cfg(synthetic_smpl_npz, tmp_path):
    return config_from_dict({
        "dataset_name": "proc", "output_root": str(tmp_path / "out"),
        "smpl_model_path": str(synthetic_smpl_npz), "amass_root": str(tmp_path / "no_amass"),
        "num_poses": 2, "image": {"height": 64, "width": 48},
        "camera_ring": {"count": 8, "distance_m": 3.5, "height_m": 0.0, "focal_px": 80.0},
    })  # fmt: skip


def test_generate_procedural_without_amass(proc_cfg, tmp_path):
    out = generate_dataset(proc_cfg, seed=0)
    assert validate_dataset(out) == []
    m = load_manifest(out)
    src = m["inputs"]["pose_source"]
    assert src["type"] == "procedural" and src["ranges_version"] == RANGES_VERSION
    assert len(src["ranges_deg"]) == 24 and "amass" not in m["inputs"]
    lab = load_sample(out, m["samples"][0]["id"])["labels"]
    assert lab["pose_source"]["type"] == "procedural"
    again = generate_dataset(proc_cfg, seed=0, output_root=tmp_path / "b")
    assert (out / "manifest.json").read_bytes() == (again / "manifest.json").read_bytes()
