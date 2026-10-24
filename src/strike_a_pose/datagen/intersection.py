"""Limb–torso intersection count for posed bodies (spec 014 FR-004, FR-005; NumPy only).

Arm vertices (dominant skinning weight on joints 16–23: upper arms, forearms, hands, fingers)
are tested against the torso (dominant joints 0, 3, 6, 9: pelvis, spine1–3). Each torso segment
is the elliptic cylinder (axis y) inscribed in the axis-aligned bounding box of its rest-pose
vertices, every half-extent shrunk by ``margin_m``; posed arm vertices are mapped into the
segment's rest frame by the inverse of the best-fit rigid motion (Kabsch) of the segment's own
vertices, rest -> posed. Rationale and limits: spec 014 FR-004 and clarification 4.

Recalibration (spec 017 FR-001, FR-002): 014's box reached past the shoulders on real SMPL (80 %
of v1 poses flagged by armpit skin), so spine3's lateral half-extent is capped at the rest
shoulder half-span minus ``SHOULDER_CLEARANCE_M``, and a pose intersects only when at least
``ceil(MIN_INSIDE_FRACTION x arm vertices)`` arm vertices are inside.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

TORSO_JOINTS = (0, 3, 6, 9)  # pelvis, spine1, spine2, spine3
ARM_JOINTS = tuple(range(16, 24))  # shoulders (upper arm) .. hands (fingers), both sides
MARGIN_M = 0.01  # a vertex must lie this far inside (ignores grazing contact, skinning blend)
MAX_INTERSECTING_FRACTION = 0.05  # operator 2026-10-21: v2a's split gate
SPINE3 = 9
SHOULDER_JOINTS = (16, 17)
SHOULDER_CLEARANCE_M = 0.05  # ~ SMPL upper-arm radius; spec 017 FR-001
MIN_INSIDE_FRACTION = 0.004  # spec 017 FR-002


def min_inside_vertices(arm_vertices: int) -> int:
    """Arm vertices that must be inside for a pose to count as intersecting (spec 017 FR-002)."""
    return max(1, math.ceil(MIN_INSIDE_FRACTION * arm_vertices))


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

    def __init__(
        self,
        rest_verts: np.ndarray,
        rest_joints: np.ndarray,
        weights: np.ndarray,
        margin_m: float = MARGIN_M,
        shoulder_clearance_m: float = SHOULDER_CLEARANCE_M,
    ):
        rest_verts = np.asarray(rest_verts, dtype=np.float64)
        rest_joints = np.asarray(rest_joints, dtype=np.float64)
        dominant = np.asarray(weights).argmax(axis=1)
        self.margin_m = float(margin_m)
        self.shoulder_clearance_m = float(shoulder_clearance_m)
        self.spine3_half_x = 0.0
        self.arm_index = np.flatnonzero(np.isin(dominant, ARM_JOINTS))
        self.segments: list[_Segment] = []
        for j in TORSO_JOINTS:
            idx = np.flatnonzero(dominant == j)
            if len(idx) < 3:
                raise ValueError(f"torso segment {j}: {len(idx)} vertices (need >= 3)")
            pts = rest_verts[idx]
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            half = (hi - lo) / 2.0 - self.margin_m
            if j == SPINE3:  # spec 017 FR-001: never reach past the shoulders
                s_l, s_r = SHOULDER_JOINTS
                cap = abs(rest_joints[s_l, 0] - rest_joints[s_r, 0]) / 2.0
                half[0] = min(half[0], cap - self.shoulder_clearance_m)
                self.spine3_half_x = float(half[0])
            if (half <= 0).any():
                raise ValueError(f"torso segment {j}: half-extent <= margin {self.margin_m} m")
            self.segments.append(_Segment(j, idx, pts, (lo + hi) / 2.0, half))
        if len(self.arm_index) == 0:
            raise ValueError("no arm vertices (dominant joints 16-23)")
        self.min_inside = min_inside_vertices(len(self.arm_index))

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
    rest_verts: np.ndarray,
    rest_joints: np.ndarray,
    posed_verts: np.ndarray,
    weights: np.ndarray,
    margin_m: float = MARGIN_M,
    shoulder_clearance_m: float = SHOULDER_CLEARANCE_M,
) -> int:
    """Raw number of arm vertices of ``posed_verts`` inside the torso (module docstring)."""
    vol = TorsoVolume(rest_verts, rest_joints, weights, margin_m, shoulder_clearance_m)
    return vol.count(posed_verts)


@dataclass(frozen=True)
class Survey:
    counts: tuple[int, ...]  # arm vertices inside the torso, per pose
    rest_counts: tuple[int, ...]  # the same at the rest pose of each pose's shape (expected 0)
    arm_vertices: int
    torso_vertices: dict[int, int]  # per torso joint
    min_inside: int  # a pose intersects at >= this many arm vertices inside (spec 017 FR-002)
    spine3_half_x: float  # capped lateral half-extent of the spine3 segment, m

    @property
    def intersecting(self) -> int:
        return sum(c >= self.min_inside for c in self.counts)

    @property
    def fraction(self) -> float:
        return self.intersecting / len(self.counts) if self.counts else 0.0


def survey(
    body: LBSBody,
    poses: Sequence[np.ndarray],
    betas: Sequence[np.ndarray],
    margin_m: float = MARGIN_M,
    shoulder_clearance_m: float = SHOULDER_CLEARANCE_M,
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
            rest, rest_j = body.forward(np.zeros(72), b)
            vol = TorsoVolume(rest, rest_j, body.weights, margin_m, shoulder_clearance_m)
            rest_count = vol.count(rest)
        verts, _ = body.forward(np.asarray(pose, dtype=np.float64), b)
        counts.append(vol.count(verts))
        rest_counts.append(rest_count)
    torso = {s.joint: len(s.index) for s in vol.segments}
    return Survey(
        tuple(counts), tuple(rest_counts), len(vol.arm_index), torso, vol.min_inside,
        vol.spine3_half_x,
    )  # fmt: skip
