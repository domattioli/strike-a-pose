"""Interim license-free capsule body (spec 013): 28 rigid capsules on SMPL's rest skeleton.

Writes an SMPL-format ``.npz`` (the seven arrays ``SMPLModel.load`` reads) that the unchanged
generator uses as its body. The only SMPL-derived input is the 24 rest-joint positions of the
neutral mean body, recovered at build time from the committed fixture labels by undoing the
kinematic chain (surrogate ruling Q3 (b): the fixture stays the single source). Mesh, radii,
skinning, regressor and shape directions are authored here. Results on this body are
infrastructure tests, never SMPL results.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .assets import MissingAssetError
from .skeleton import NUM_JOINTS, SMPL24_PARENTS
from .smpl import rodrigues

CAPSULE_VERSION = "capsule_v1"
# SHA-256 of the v1 file built from the committed fixture (locked environment, both CI legs).
CAPSULE_V1_SHA256 = "bc1f3eb7275ce0fabf6fdab3fb80e36dc2bcd29eb96a6590b009dac5d0b4116a"
DEFAULT_FIXTURE = Path("tests/fixtures/smpl_fixture")
DEFAULT_OUT = Path("assets/synthetic/capsule_v1.npz")

N_SEG = 16  # vertices per ring
N_CAP = 4  # latitude rings per hemi-ellipsoidal cap (plus a pole)
MAX_RING_STEP_M = 0.03  # cylinder rings at most this far apart (rasterizer cost, spec 013)
DECIMALS = 6  # every float written is rounded to 1e-6 (byte-stable across numpy builds)
NUM_SHAPE = 10  # shapedirs components (every config uses num_betas: 10); 3-9 are zero
SIZE_PER_BETA = 0.04  # component 0: scale about the world origin
THICKNESS_PER_BETA = 0.10  # component 1: limb radii
TORSO_WIDTH_PER_BETA = 0.08  # component 2: pelvis / spine width (r1)
TORSO_SKIN = (0, 3, 6)  # pelvis and spine segments (component 2)
NO_THICKNESS_SKIN = (0, 3, 6, 9, 12, 15)  # trunk, neck and head keep their radius (component 1)

_NPZ_KEYS = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f")


def _seg(skin, a, b, r1, r2=None):
    """A capsule: skin joint, endpoints (joint, rest-frame offset m), section semi-axes (m)."""
    return (skin, a, b, r1, r1 if r2 is None else r2)


def _limbs(side: int) -> list:
    """Arm / leg / hand / foot segments of one side (side = +1 left, -1 right)."""
    left = side > 0
    sh, el, wr, ha = (16, 18, 20, 22) if left else (17, 19, 21, 23)
    co = 13 if left else 14
    hip, kn, an, ft = (1, 4, 7, 10) if left else (2, 5, 8, 11)
    z = (0.0, 0.0, 0.0)
    return [
        _seg(9, (9, z), (co, z), 0.08),
        _seg(co, (co, z), (sh, z), 0.06),
        _seg(sh, (sh, z), (el, z), 0.05),
        _seg(el, (el, z), (wr, z), 0.04),
        _seg(wr, (wr, z), (ha, z), 0.018, 0.04),  # palm: thin vertically, wide front-back
        _seg(ha, (ha, z), (ha, (side * 0.08, -0.005, 0.0)), 0.015, 0.04),  # fingers
        _seg(0, (0, z), (hip, z), 0.085),
        _seg(hip, (hip, z), (kn, z), 0.075),
        _seg(kn, (kn, z), (an, z), 0.05),
        _seg(an, (an, (0.0, -0.06, -0.06)), (ft, (0.0, -0.02, 0.0)), 0.04, 0.03),  # heel -> ball
        _seg(ft, (ft, (0.0, -0.02, 0.0)), (ft, (0.0, -0.03, 0.08)), 0.04, 0.025),  # toes
    ]


_Z = (0.0, 0.0, 0.0)
SEGMENTS: tuple = tuple(
    [
        _seg(0, (0, _Z), (3, _Z), 0.13, 0.10),
        _seg(3, (3, _Z), (6, _Z), 0.13, 0.10),
        _seg(6, (6, _Z), (9, _Z), 0.135, 0.10),
        _seg(9, (9, _Z), (12, _Z), 0.06),
        _seg(12, (12, _Z), (15, _Z), 0.055),
        _seg(15, (15, (0.0, 0.07, 0.02)), (15, (0.0, 0.08, 0.02)), 0.095),  # head
    ]
    + _limbs(+1)
    + _limbs(-1)
)


def rest_skeleton(fixture_dir: Path | str = DEFAULT_FIXTURE) -> tuple[np.ndarray, np.ndarray]:
    """Rest joints (24, 3) recovered from the fixture labels, and the per-pose recoveries.

    Per pose: ``j[0] = J[0]``, ``j[i] = j[p] + G_p^T (J[i] - J[p])`` with ``G`` the global joint
    rotations from the label's ``pose``. Result: mean over poses (ascending index), rounded to
    1e-6 m. Every view of a pose must carry the same joints and pose.
    """
    samples = Path(fixture_dir) / "samples"
    files = sorted(samples.glob("*.json")) if samples.is_dir() else []
    if not files:
        raise MissingAssetError(samples, "fixture labels (capsule body skeleton source)")
    by_pose: dict[int, tuple[list, list]] = {}
    for f in files:
        lab = json.loads(f.read_text())
        key = (lab["joints_3d_world"], lab["pose"])
        seen = by_pose.setdefault(int(lab["pose_index"]), key)
        if seen != key:
            raise ValueError(f"{f.name}: joints / pose differ between views of one pose")
    recoveries = []
    for _, (joints, pose) in sorted(by_pose.items()):
        jw = np.asarray(joints, dtype=np.float64)
        rots = rodrigues(np.asarray(pose, dtype=np.float64).reshape(NUM_JOINTS, 3))
        g = np.zeros((NUM_JOINTS, 3, 3))
        rest = np.zeros((NUM_JOINTS, 3))
        for i, p in enumerate(SMPL24_PARENTS):
            if p < 0:
                g[i], rest[i] = rots[i], jw[i]
            else:
                g[i] = g[p] @ rots[i]
                rest[i] = rest[p] + g[p].T @ (jw[i] - jw[p])
        recoveries.append(rest)
    rec = np.stack(recoveries)
    return np.round(rec.mean(0), DECIMALS) + 0.0, rec


def section_frame(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(e1, e2) for an axis u: e2 = world +Z made perpendicular to u (+Y if |u_z| >= 0.9)."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = ref - (ref @ u) * u
    e2 /= np.linalg.norm(e2)
    return np.cross(e2, u), e2


def capsule_mesh(
    a: np.ndarray, b: np.ndarray, r1: float, r2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Closed capsule a -> b: elliptical rings (r1 along e1, r2 along e2), caps of axial radius
    max(r1, r2) centred on a and b. Vertex 0 and the last are the poles."""
    d = b - a
    length = float(np.linalg.norm(d))
    u = d / length
    e1, e2 = section_frame(u)
    ru = max(r1, r2)
    psi = 2.0 * np.pi * np.arange(N_SEG) / N_SEG
    ring = np.cos(psi)[:, None] * r1 * e1 + np.sin(psi)[:, None] * r2 * e2  # (N_SEG, 3)
    lat = -np.pi / 2 + (np.pi / 2) * np.arange(1, N_CAP + 1) / N_CAP  # pole side -> equator
    rings = [a + np.cos(t) * ring + np.sin(t) * ru * u for t in lat]
    n_mid = max(int(np.ceil(length / MAX_RING_STEP_M)) - 1, 0)
    rings += [a + (length * k / (n_mid + 1)) * u + ring for k in range(1, n_mid + 1)]
    rings += [b + np.cos(t) * ring - np.sin(t) * ru * u for t in lat[::-1]]
    verts = np.concatenate([[a - ru * u], *rings, [b + ru * u]])
    nr, s = len(rings), np.arange(N_SEG)
    s1 = (s + 1) % N_SEG
    faces = [np.stack([np.zeros(N_SEG, int), 1 + s1, 1 + s], 1)]
    for r in range(nr - 1):
        o0, o1 = 1 + r * N_SEG, 1 + (r + 1) * N_SEG
        faces.append(np.stack([o0 + s, o0 + s1, o1 + s1], 1))
        faces.append(np.stack([o0 + s, o1 + s1, o1 + s], 1))
    top, o = len(verts) - 1, 1 + (nr - 1) * N_SEG
    faces.append(np.stack([np.full(N_SEG, top), o + s, o + s1], 1))
    return verts, np.concatenate(faces)


def _point(rest: np.ndarray, spec: tuple) -> np.ndarray:
    joint, offset = spec
    return rest[joint] + np.asarray(offset, dtype=np.float64)


def build_capsule_body(rest: np.ndarray) -> dict[str, np.ndarray]:
    """The seven SMPL-format arrays (rounded, fixed dtypes) of the v1 body on ``rest`` (24, 3)."""
    verts, faces, weights, shape = [], [], [], []
    n = 0
    for skin, sa, sb, r1, r2 in SEGMENTS:
        a, b = _point(rest, sa), _point(rest, sb)
        v, f = capsule_mesh(a, b, r1, r2)
        u = (b - a) / np.linalg.norm(b - a)
        e1, _ = section_frame(u)
        t = np.clip((v - a) @ u, 0.0, float(np.linalg.norm(b - a)))
        radial = v - (a + t[:, None] * u)  # from the closest axis point
        sd = np.zeros((len(v), 3, NUM_SHAPE))
        sd[:, :, 0] = SIZE_PER_BETA * v
        if skin not in NO_THICKNESS_SKIN:
            sd[:, :, 1] = THICKNESS_PER_BETA * radial
        if skin in TORSO_SKIN:
            sd[:, :, 2] = TORSO_WIDTH_PER_BETA * (radial @ e1)[:, None] * e1
        w = np.zeros((len(v), NUM_JOINTS))
        w[:, skin] = 1.0
        verts.append(v)
        faces.append(f + n)
        weights.append(w)
        shape.append(sd)
        n += len(v)
    # 24 joint markers: in no face, weight 1 on their own joint, the regressor's one-hot rows.
    sd = np.zeros((NUM_JOINTS, 3, NUM_SHAPE))
    sd[:, :, 0] = SIZE_PER_BETA * rest
    verts.append(rest)
    weights.append(np.eye(NUM_JOINTS))
    shape.append(sd)
    v_all = np.concatenate(verts)
    jreg = np.zeros((NUM_JOINTS, len(v_all)))
    jreg[np.arange(NUM_JOINTS), n + np.arange(NUM_JOINTS)] = 1.0
    kt = np.stack([np.array(SMPL24_PARENTS, dtype=np.int64), np.arange(NUM_JOINTS)])
    kt[0, 0] = 2**32 - 1

    def rnd(x):
        return np.round(np.asarray(x, dtype=np.float64), DECIMALS) + 0.0  # -0.0 -> +0.0

    return {
        "v_template": rnd(v_all),
        "shapedirs": rnd(np.concatenate(shape)),
        "posedirs": np.zeros((len(v_all), 3, 207), dtype=np.float32),
        "J_regressor": jreg,
        "weights": np.concatenate(weights),
        "kintree_table": kt,
        "f": np.concatenate(faces).astype(np.int64),
    }


def capsule_body_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Uncompressed ``.npz`` bytes, keys in a fixed order (zip entries dated 1980: stable)."""
    buf = io.BytesIO()
    np.savez(buf, **{k: arrays[k] for k in _NPZ_KEYS})
    return buf.getvalue()


def build_capsule_bytes(fixture_dir: Path | str = DEFAULT_FIXTURE) -> tuple[bytes, str]:
    """(file bytes, SHA-256 hex) of the v1 body built from ``fixture_dir``."""
    data = capsule_body_bytes(build_capsule_body(rest_skeleton(fixture_dir)[0]))
    return data, hashlib.sha256(data).hexdigest()
