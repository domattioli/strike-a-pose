"""Synthetic stand-ins for licensed assets. NOT real SMPL/AMASS data."""

from pathlib import Path

import numpy as np
import pytest

from strike_a_pose.datagen.config import config_from_dict
from strike_a_pose.datagen.skeleton import SMPL24_PARENTS

# Rough T-pose joint positions (metres, +Y up, facing +Z, pelvis at origin).
TPOSE = np.array([
    [0, 0, 0], [0.09, -0.08, 0], [-0.09, -0.08, 0], [0, 0.1, 0], [0.1, -0.45, 0],
    [-0.1, -0.45, 0], [0, 0.23, 0], [0.1, -0.85, 0], [-0.1, -0.85, 0], [0, 0.28, 0],
    [0.1, -0.9, 0.1], [-0.1, -0.9, 0.1], [0, 0.45, 0], [0.08, 0.38, 0], [-0.08, 0.38, 0],
    [0, 0.6, 0], [0.18, 0.4, 0], [-0.18, 0.4, 0], [0.45, 0.4, 0], [-0.45, 0.4, 0],
    [0.7, 0.4, 0], [-0.7, 0.4, 0], [0.78, 0.4, 0], [-0.78, 0.4, 0],
], dtype=np.float64)  # fmt: skip

_CUBE_V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
_CUBE_F = np.array([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
    [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
])  # fmt: skip


def make_synthetic_smpl(path: Path, half: float = 0.03) -> Path:
    v, f, w = [], [], np.zeros((24 * 8, 24))
    for j in range(24):
        v.append(TPOSE[j] + half * _CUBE_V)
        f.append(_CUBE_F + 8 * j)
        w[8 * j : 8 * j + 8, j] = 1.0
    v = np.concatenate(v)
    jreg = w.T / 8.0
    kt = np.stack([np.array(SMPL24_PARENTS, dtype=np.int64), np.arange(24)])
    kt[0, 0] = 2**32 - 1
    np.savez(
        path, v_template=v, shapedirs=np.zeros((len(v), 3, 10)),
        posedirs=np.zeros((len(v), 3, 207)), J_regressor=jreg, weights=w,
        kintree_table=kt, f=np.concatenate(f),
    )  # fmt: skip
    return path


def make_synthetic_amass(root: Path) -> Path:
    rng = np.random.default_rng(1234)
    for sub in ("SubA", "SubB"):
        for k, frames in enumerate((95, 61)):
            d = root / sub / f"subj{k}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"seq{k}_poses.npz", poses=rng.normal(0, 0.3, (frames, 156)))
        np.savez(root / sub / "shape.npz", betas=np.zeros(16))
    return root


@pytest.fixture
def synthetic_smpl_npz(tmp_path: Path) -> Path:
    return make_synthetic_smpl(tmp_path / "SMPL_SYNTH.npz")


@pytest.fixture
def synthetic_amass(tmp_path: Path) -> Path:
    return make_synthetic_amass(tmp_path / "amass")


@pytest.fixture
def small_cfg(synthetic_smpl_npz: Path, synthetic_amass: Path, tmp_path: Path):
    return config_from_dict({
        "dataset_name": "synth", "output_root": str(tmp_path / "out"),
        "smpl_model_path": str(synthetic_smpl_npz), "amass_root": str(synthetic_amass),
        "amass_subsets": ["SubA", "SubB"], "num_poses": 2, "frame_stride": 10,
        "image": {"height": 64, "width": 48},
        "camera_ring": {"count": 8, "distance_m": 3.5, "height_m": 0.0, "focal_px": 80.0},
    })  # fmt: skip
