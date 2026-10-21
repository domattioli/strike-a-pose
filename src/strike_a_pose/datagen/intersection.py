"""Limb–torso intersection count for posed bodies (spec 014 FR-004, FR-005; NumPy only).

Arm vertices (dominant skinning weight on joints 16–23: upper arms, forearms, hands, fingers)
are tested against the torso (dominant joints 0, 3, 6, 9: pelvis, spine1–3). Each torso segment
is the elliptic cylinder (axis y) inscribed in the axis-aligned bounding box of its rest-pose
vertices, every half-extent shrunk by ``margin_m``; posed arm vertices are mapped into the
segment's rest frame by the inverse of the best-fit rigid motion (Kabsch) of the segment's own
vertices, rest -> posed. Rationale and limits: spec 014 FR-004 and clarification 4.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

TORSO_JOINTS = (0, 3, 6, 9)  # pelvis, spine1, spine2, spine3
ARM_JOINTS = tuple(range(16, 24))  # shoulders (upper arm) .. hands (fingers), both sides
MARGIN_M = 0.01  # a vertex must lie this far inside (ignores grazing contact, skinning blend)
MAX_INTERSECTING_FRACTION = 0.05  # operator 2026-10-21: v2a's split gate


class LBSBody(Protocol):
    weights: np.ndarray

    def forward(self, pose: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit proper rotation R and centroids with dst ≈ (src - c_src) @ R.T + c_dst."""
    c_src, c_dst = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - c_src).T @ (dst - c_dst))
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, c_src, c_dst


@dataclass(frozen=True)
class _Segment:
    joint: int
    index: np.ndarray  # vertex indices dominated by the joint
    rest: np.ndarray  # their rest-pose positions
    centre: np.ndarray  # box centre (rest frame)
    half: np.ndarray  # shrunk half-extents (x, y, z)


class TorsoVolume:
    """The torso approximation of one rest-pose body; ``count`` tests a posed copy of it."""

    def __init__(self, rest_verts: np.ndarray, weights: np.ndarray, margin_m: float = MARGIN_M):
        rest_verts = np.asarray(rest_verts, dtype=np.float64)
        dominant = np.asarray(weights).argmax(axis=1)
        self.margin_m = float(margin_m)
        self.arm_index = np.flatnonzero(np.isin(dominant, ARM_JOINTS))
        self.segments: list[_Segment] = []
        for j in TORSO_JOINTS:
            idx = np.flatnonzero(dominant == j)
            if len(idx) < 3:
                raise ValueError(f"torso segment {j}: {len(idx)} vertices (need >= 3)")
            pts = rest_verts[idx]
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            half = (hi - lo) / 2.0 - self.margin_m
            if (half <= 0).any():
                raise ValueError(f"torso segment {j}: half-extent <= margin {self.margin_m} m")
            self.segments.append(_Segment(j, idx, pts, (lo + hi) / 2.0, half))
        if len(self.arm_index) == 0:
            raise ValueError("no arm vertices (dominant joints 16-23)")

    def inside(self, posed_verts: np.ndarray) -> np.ndarray:
        """Boolean per arm vertex: inside any torso segment of this posed body."""
        posed_verts = np.asarray(posed_verts, dtype=np.float64)
        arm = posed_verts[self.arm_index]
        hit = np.zeros(len(arm), dtype=bool)
        for s in self.segments:
            r, c_rest, c_posed = kabsch(s.rest, posed_verts[s.index])
            q = (arm - c_posed) @ r + c_rest - s.centre  # posed -> segment rest frame
            hx, hy, hz = s.half
            hit |= ((q[:, 0] / hx) ** 2 + (q[:, 2] / hz) ** 2 <= 1.0) & (np.abs(q[:, 1]) <= hy)
        return hit

    def count(self, posed_verts: np.ndarray) -> int:
        return int(self.inside(posed_verts).sum())


def count_arm_torso_intersections(
    rest_verts: np.ndarray, posed_verts: np.ndarray, weights: np.ndarray, margin_m: float = MARGIN_M
) -> int:
    """Number of arm vertices of ``posed_verts`` inside the torso (see the module docstring)."""
    return TorsoVolume(rest_verts, weights, margin_m).count(posed_verts)


@dataclass(frozen=True)
class Survey:
    counts: tuple[int, ...]  # arm vertices inside the torso, per pose
    rest_counts: tuple[int, ...]  # the same at the rest pose of each pose's shape (expected 0)
    arm_vertices: int
    torso_vertices: dict[int, int]  # per torso joint

    @property
    def intersecting(self) -> int:
        return sum(c > 0 for c in self.counts)

    @property
    def fraction(self) -> float:
        return self.intersecting / len(self.counts) if self.counts else 0.0


def survey(
    body: LBSBody,
    poses: Sequence[np.ndarray],
    betas: Sequence[np.ndarray],
    margin_m: float = MARGIN_M,
) -> Survey:
    """Pose each pose with its betas and count its arm vertices inside the torso."""
    if len(poses) != len(betas) or not poses:
        raise ValueError("need at least one pose and one betas vector per pose")
    counts, rest_counts = [], []
    key = vol = rest_count = None
    for pose, b in zip(poses, betas, strict=True):
        b = np.asarray(b, dtype=np.float64)
        if b.tobytes() != key:  # one rest body per shape (all poses share it without body_shape)
            key = b.tobytes()
            rest, _ = body.forward(np.zeros(72), b)
            vol = TorsoVolume(rest, body.weights, margin_m)
            rest_count = vol.count(rest)
        verts, _ = body.forward(np.asarray(pose, dtype=np.float64), b)
        counts.append(vol.count(verts))
        rest_counts.append(rest_count)
    torso = {s.joint: len(s.index) for s in vol.segments}
    return Survey(tuple(counts), tuple(rest_counts), len(vol.arm_index), torso)
