"""Procedural pose source (spec 001 FR-002, Clarifications Session 2026-10-20).

Each non-root body joint gets three angles drawn independently and uniformly
from a hardcoded range (degrees) about its SMPL *local* x, y, z axes, composed
as intrinsic X-then-Y-then-Z Euler (R = Rx @ Ry @ Rz) and converted to
axis-angle. SMPL rest frame: +Y up, body faces +Z, +X is the subject's left.

Sign conventions (checked by rendering; see research R5):
- hips/knees/ankles/spine/neck/head x: + bends forward-down (knee + = flexion,
  hip - = thigh forward, spine + = forward bend).
- left limbs z: + = abduction (leg out to the left / left arm up); right limbs mirrored.
- shoulder y: left - / right + = arm forward. Rest pose is a T-pose, so the
  shoulder z range is centred on "arm down" (left about -65 deg).
- elbow y: left - / right + = flexion (forearm forward).
Root (0) and hand (22, 23) joints stay zero; betas are mean shape.

Ranges are ad hoc but chosen to stay inside normal human range of motion.
Bump RANGES_VERSION whenever the table changes; it is recorded in the manifest.
"""

from dataclasses import dataclass

import numpy as np

from .skeleton import SMPL24_JOINT_NAMES

RANGES_VERSION = "proc-v1"

_Z = (0.0, 0.0)
# joint index -> ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)) in degrees
RANGES_DEG: dict[int, tuple[tuple[float, float], ...]] = {
    0: (_Z, _Z, _Z),  # pelvis (root orientation stays zero)
    1: ((-90, 20), (-25, 25), (-10, 40)),  # L_hip
    2: ((-90, 20), (-25, 25), (-40, 10)),  # R_hip
    3: ((-10, 30), (-15, 15), (-15, 15)),  # spine1
    4: ((0, 120), _Z, _Z),  # L_knee
    5: ((0, 120), _Z, _Z),  # R_knee
    6: ((-5, 15), (-10, 10), (-10, 10)),  # spine2
    7: ((-20, 30), (-10, 10), (-10, 10)),  # L_ankle
    8: ((-20, 30), (-10, 10), (-10, 10)),  # R_ankle
    9: ((-5, 15), (-10, 10), (-10, 10)),  # spine3
    10: ((-10, 10), _Z, _Z),  # L_foot (toes)
    11: ((-10, 10), _Z, _Z),  # R_foot (toes)
    12: ((-20, 30), (-30, 30), (-15, 15)),  # neck
    13: (_Z, (-10, 10), (-10, 15)),  # L_collar
    14: (_Z, (-10, 10), (-15, 10)),  # R_collar
    15: ((-20, 20), (-30, 30), (-15, 15)),  # head
    16: ((-30, 30), (-80, 20), (-80, 30)),  # L_shoulder
    17: ((-30, 30), (-20, 80), (-30, 80)),  # R_shoulder
    18: (_Z, (-130, 0), _Z),  # L_elbow
    19: (_Z, (0, 130), _Z),  # R_elbow
    20: ((-20, 20), (-10, 10), (-30, 30)),  # L_wrist
    21: ((-20, 20), (-10, 10), (-30, 30)),  # R_wrist
    22: (_Z, _Z, _Z),  # L_hand (hands/fingers zero)
    23: (_Z, _Z, _Z),  # R_hand
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
    return _rx(ang_rad[0]) @ _ry(ang_rad[1]) @ _rz(ang_rad[2])


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
    lo = np.array([[a[0] for a in RANGES_DEG[j]] for j in range(24)], dtype=np.float64)
    hi = np.array([[a[1] for a in RANGES_DEG[j]] for j in range(24)], dtype=np.float64)
    out = []
    for i in range(n):
        deg = rng.uniform(lo, hi)  # (24, 3); zero-width ranges give exactly lo
        aa = np.stack([matrix_to_axis_angle(euler_xyz_to_matrix(np.radians(d))) for d in deg])
        aa[0] = 0.0
        aa[22:] = 0.0
        out.append(
            ProceduralPose(
                index=i,
                euler_deg=tuple(tuple(float(v) for v in d) for d in deg),
                pose=tuple(aa.reshape(-1).tolist()),
            )
        )
    return out
