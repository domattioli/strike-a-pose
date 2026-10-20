"""Procedural pose source (spec 001 FR-002, Clarifications Session 2026-10-20).

Faithful port of the predecessor's sampler (body-model-visualizer
``generate_random_slider_vals`` + ``refine_random_sliders``):
1. per non-root joint and axis, an integer angle uniform in [-A, B) degrees;
2. sparsify: draw k uniform in [0, 69) until k >= 5, keep k randomly chosen of
   the 69 values, zero the rest;
3. Euler 'xyz' (scipy lower-case = extrinsic, R = Rz @ Ry @ Rx) -> axis-angle.
Root orientation is zero; body shape is mean shape (betas = 0).
Bump RANGES_VERSION whenever the table or rule changes (recorded in manifest).
"""

from dataclasses import dataclass

import numpy as np

from .skeleton import SMPL24_JOINT_NAMES

RANGES_VERSION = "predecessor-v1"
SPARSIFY_MIN_KEEP = 5  # keep k >= 5 of the 69 values, zero the rest

# SMPL joint index -> ((-A, B) for x, y, z): integer degrees uniform in [-A, B).
# Source: predecessor body-model-visualizer generate_random_slider_vals
# (its body_pose index i == SMPL joint i + 1).
RANGES_DEG: dict[int, tuple[tuple[int, int], ...]] = {
    0: ((0, 0), (0, 0), (0, 0)),  # pelvis (root orientation stays zero)
    1: ((-130, 80), (-20, 20), (-10, 10)),  # L_hip
    2: ((-130, 80), (-20, 20), (-10, 10)),  # R_hip
    3: ((-50, 100), (-75, 75), (-15, 15)),  # spine1
    4: ((0, 130), (-20, 20), (0, 0)),  # L_knee
    5: ((0, 130), (-20, 20), (0, 0)),  # R_knee
    6: ((-20, 20), (-15, 15), (-20, 20)),  # spine2
    7: ((-30, 70), (-30, 30), (-25, 30)),  # L_ankle
    8: ((-30, 70), (-30, 30), (-25, 30)),  # R_ankle
    9: ((-35, 25), (-50, 50), (-45, 45)),  # spine3
    10: ((-40, 40), (0, 0), (-20, 20)),  # L_foot
    11: ((-40, 40), (0, 0), (-20, 20)),  # R_foot
    12: ((-45, 35), (-50, 50), (-20, 20)),  # neck
    13: ((-10, 10), (-10, 10), (-30, 30)),  # L_collar
    14: ((-10, 10), (-10, 10), (-30, 30)),  # R_collar
    15: ((-30, 30), (-30, 30), (-30, 30)),  # head
    16: ((-20, 20), (-40, 20), (-50, 50)),  # L_shoulder
    17: ((-20, 20), (-40, 20), (-50, 50)),  # R_shoulder
    18: ((-60, 110), (-145, 0), (-35, 5)),  # L_elbow
    19: ((-60, 110), (0, 145), (-35, 5)),  # R_elbow
    20: ((-50, 50), (-35, 35), (-50, 50)),  # L_wrist
    21: ((-50, 50), (-35, 35), (-50, 50)),  # R_wrist
    22: ((-25, 25), (-15, 15), (-90, 45)),  # L_hand
    23: ((-25, 25), (-15, 15), (-90, 45)),  # R_hand
}


def ranges_table() -> list[dict]:
    """Manifest-friendly copy of the ranges (joint name -> per-axis degrees)."""
    return [
        {"joint": SMPL24_JOINT_NAMES[j], "index": j,
         "x_deg": list(r[0]), "y_deg": list(r[1]), "z_deg": list(r[2])}
        for j, r in sorted(RANGES_DEG.items())
    ]  # fmt: skip


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_xyz_to_matrix(ang_rad: np.ndarray) -> np.ndarray:
    """Extrinsic x, then y, then z (scipy ``from_euler('xyz')``)."""
    return _rz(ang_rad[2]) @ _ry(ang_rad[1]) @ _rx(ang_rad[0])


def matrix_to_axis_angle(r: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos))
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi - theta < 1e-6:  # near 180 deg: axis from the symmetric part
        m = (r + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(m)))
        axis = m[:, k] / np.sqrt(m[k, k])
        return axis * theta
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return w / (2.0 * np.sin(theta)) * theta


@dataclass(frozen=True)
class ProceduralPose:
    index: int
    euler_deg: tuple[tuple[float, float, float], ...]  # 24 x (x, y, z) sampled angles
    pose: tuple[float, ...]  # 72 axis-angle values


def sample_procedural_poses(n: int, seed: int) -> list[ProceduralPose]:
    rng = np.random.default_rng(seed)
    lo = np.array([[a[0] for a in RANGES_DEG[j]] for j in range(1, 24)], dtype=np.int64)
    hi = np.array([[a[1] for a in RANGES_DEG[j]] for j in range(1, 24)], dtype=np.int64)
    nvals = lo.size  # 69
    out = []
    for i in range(n):
        # integer degrees uniform in [lo, hi); zero-width ranges give 0
        deg = rng.integers(lo, np.maximum(hi, lo + 1)).astype(np.float64)
        k = 0
        while k < SPARSIFY_MIN_KEEP:
            k = int(rng.integers(0, nvals))
        keep = rng.choice(nvals, size=k, replace=False)
        flat = np.zeros(nvals)
        flat[keep] = deg.reshape(-1)[keep]
        deg = np.vstack([np.zeros((1, 3)), flat.reshape(23, 3)])
        aa = np.stack([matrix_to_axis_angle(euler_xyz_to_matrix(np.radians(d))) for d in deg])
        out.append(
            ProceduralPose(
                index=i,
                euler_deg=tuple(tuple(float(v) for v in d) for d in deg),
                pose=tuple(aa.reshape(-1).tolist()),
            )
        )
    return out
