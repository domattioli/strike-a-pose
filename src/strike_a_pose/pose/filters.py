"""Pose filters: joint-angle limits and a capsule self-intersection proxy (FR-001, research R3).

A drawn pose is accepted only when it passes two tests. The caller redraws a rejected pose and
counts the rejections (FR-001, data-model PoseFilter result).

Joint angles. The rotation angle of a joint is the norm of its axis-angle vector. A pose fails
when any of the 21 joints of ``pose_body`` turns by more than its limit. The limits come from the
table ``pose.limits_deg`` through ``joint_limits_rad`` of pose/limits.py, the function the
``limits`` pose source uses too, so the filter and the sampler apply one limit to each joint (the
two feet take the ankle limit there). The root orientation has no limit, because turning the whole
body changes no joint angle and no distance between body parts.

Self-intersection. Each of the 22 body joints owns one capsule, the part of the body that moves
with that joint: the capsule of ``left_hip`` is the left thigh, ``left_knee`` the left shin,
``left_shoulder`` the left upper arm, ``left_elbow`` the left forearm, ``left_wrist`` the left
hand, ``spine1`` to ``spine3`` the torso, and so on. A capsule is a line segment with a radius. Its
axis runs from the joint to its child joint (the pelvis capsule ends at ``spine1``, the ``spine3``
capsule at ``neck``). The five joints without a child (head, both wrists, both feet) end at a tip,
which the filter finds from the vertices of the part: it follows the direction from the joint to
the centre of the part and stops one radius short of the farthest vertex, because the rounded cap
supplies that last radius. The radius is the median distance of the part's vertices to the axis
segment in the canonical pose (research R3). Two capsules overlap by the sum of their radii minus
the distance between their axes. A pose fails when a tested pair overlaps by more than
``pose.capsule_overlap_cm`` (1 cm by default).

A pair is not tested in three cases, each from research R3:

* Both capsules belong to the trunk chain (pelvis, spine1, spine2, spine3, neck, head, both
  collars). Torso capsules are wide against the joint spacing and overlap by 5 to 20 cm in the
  canonical pose (the T011 amendment), so a test of the trunk against itself would reject every
  pose. The test applies to pairs that involve a limb capsule.
* The two capsules share a joint, so they touch by construction (thigh and shin, upper arm and
  forearm, collar and upper arm, and so on).
* One capsule is the pelvis or a spine segment and the other is a thigh or an upper arm. Those
  limbs start inside the torso.

Reference shape. The filter judges a pose, not a body. The radii and the proportions of the
skeleton come from the canonical-pose mesh and the zero-pose joints of one reference shape, the
``betas`` given to the constructor (the mean shape, zeros, in ``from_config``). A pose is therefore
accepted or rejected the same way for every body, and the filter is built once per run. The
posed capsules come from forward kinematics with axis-angle rotations, as in SMPL (Loper et al.,
"SMPL: A Skinned Multi-Person Linear Model", ACM Transactions on Graphics 34(6), 2015), which every
body model of this package follows: the constructor checks that the canonical joints of the model
agree with it. Research R3 requires the canonical pose to pass the filter. The constructor logs a
warning, and does not raise, when the canonical pose of a model fails, because only the stand-in
can be tested without licensed files.

Public sources. The method and its limits are research R3. Capsules, also called swept spheres or
line-swept spheres, as collision proxies, and the closest points of two line segments: C. Ericson,
"Real-Time Collision Detection" (Morgan Kaufmann, 2004), chapters 4 and 5. Rotations from
axis-angle vectors: Rodrigues' rotation formula
(https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula). The limit values are rounded
range-of-motion figures (research R3).

Arithmetic. Sums and products are written as elementwise operations in a fixed order, with no
BLAS or einsum call, as in body/standin.py, so a decision does not depend on the thread count.
"""

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from strike_a_pose.body.base import (
    BODY_POSE_SIZE,
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENTS,
    ROOT_POSE_SIZE,
    BodyModel,
)
from strike_a_pose.pose.limits import joint_limits_rad

__all__ = ["PoseFilter", "PoseFilterResult", "segment_distance"]

logger = logging.getLogger(__name__)

# A rotation exactly at its limit passes. This slack, in radians, keeps rounding in the norm of an
# axis-angle vector from rejecting such a rotation.
_ANGLE_TOLERANCE_RAD = 1e-9

# The largest distance, in metres, between a joint of the model and the same joint from the filter's
# own forward kinematics. Float32 body models agree to a few micrometres, a wrong convention by
# centimetres.
_KINEMATICS_TOLERANCE_M = 1e-4

# A segment shorter than 1e-9 m (squared length below this value) is treated as a single point.
_POINT_LENGTH_SQUARED = 1e-18

# Lines whose direction vectors are this parallel (relative to the product of their squared
# lengths) have no single closest-point pair, and only the segment ends decide the distance.
_PARALLEL_TOLERANCE = 1e-9

# The joint at which the capsule of each joint ends: its child joint along the limb or the spine,
# or None for a joint without a child, whose capsule ends at a tip found from the vertices.
_CAPSULE_END_NAMES: dict[str, str | None] = {
    "pelvis": "spine1",
    "left_hip": "left_knee",
    "right_hip": "right_knee",
    "spine1": "spine2",
    "left_knee": "left_ankle",
    "right_knee": "right_ankle",
    "spine2": "spine3",
    "left_ankle": "left_foot",
    "right_ankle": "right_foot",
    "spine3": "neck",
    "left_foot": None,
    "right_foot": None,
    "neck": "head",
    "left_collar": "left_shoulder",
    "right_collar": "right_shoulder",
    "head": None,
    "left_shoulder": "left_elbow",
    "right_shoulder": "right_elbow",
    "left_elbow": "left_wrist",
    "right_elbow": "right_wrist",
    "left_wrist": None,
    "right_wrist": None,
}
_CAPSULE_END_JOINTS: tuple[int | None, ...] = tuple(
    None if _CAPSULE_END_NAMES[name] is None else JOINT_INDEX[_CAPSULE_END_NAMES[name]]
    for name in JOINT_NAMES
)


def _joint_set(*names: str) -> frozenset[int]:
    return frozenset(JOINT_INDEX[name] for name in names)


# The trunk chain: every pair of these capsules is exempt (research R3, T011 amendment).
_TRUNK_CHAIN = _joint_set(
    "pelvis", "spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar"
)
# The pelvis and the three spine segments, which the thighs and the upper arms start inside.
_TORSO_CORE = _joint_set("pelvis", "spine1", "spine2", "spine3")
_THIGHS_AND_UPPER_ARMS = _joint_set("left_hip", "right_hip", "left_shoulder", "right_shoulder")


def _is_exempt(first: int, second: int) -> bool:
    """Return True for a pair of capsules that the self-intersection test skips."""
    if first in _TRUNK_CHAIN and second in _TRUNK_CHAIN:
        return True
    if _CAPSULE_END_JOINTS[first] == second or _CAPSULE_END_JOINTS[second] == first:
        return True  # the two capsules share a joint
    return (first in _TORSO_CORE and second in _THIGHS_AND_UPPER_ARMS) or (
        second in _TORSO_CORE and first in _THIGHS_AND_UPPER_ARMS
    )


def _tree_levels() -> tuple[tuple[NDArray[np.intp], NDArray[np.intp]], ...]:
    """Group the joints below the root by depth in the kinematic tree.

    Each group is a pair (joints, their parents). The parents of one group all lie in earlier
    groups, so one group at a time can be posed with array operations.
    """
    depth = [0] * NUM_JOINTS
    for joint in range(1, NUM_JOINTS):
        depth[joint] = depth[PARENTS[joint]] + 1
    levels = []
    for level in range(1, max(depth) + 1):
        joints = np.array([joint for joint in range(NUM_JOINTS) if depth[joint] == level])
        parents = np.array([PARENTS[joint] for joint in joints])
        levels.append((joints.astype(np.intp), parents.astype(np.intp)))
    return tuple(levels)


_TREE_LEVELS = _tree_levels()


def _dot(first: NDArray[np.float64], second: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dot product over the last axis, adding the three products in a fixed order."""
    from_x = first[..., 0] * second[..., 0]
    from_y = first[..., 1] * second[..., 1]
    from_z = first[..., 2] * second[..., 2]
    return (from_x + from_y) + from_z


def _length(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Euclidean length over the last axis."""
    return np.sqrt(_dot(vectors, vectors))


def _matrix_product(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply matrices of shape (..., 3, 3), adding the three products in a fixed order."""
    products = left[..., :, :, None] * right[..., None, :, :]
    return (products[..., 0, :] + products[..., 1, :]) + products[..., 2, :]


def _matrix_vector(matrix: NDArray[np.float64], vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply matrices (..., 3, 3) to vectors (..., 3), adding the products in a fixed order."""
    from_first = matrix[..., :, 0] * vector[..., 0:1]
    from_second = matrix[..., :, 1] * vector[..., 1:2]
    from_third = matrix[..., :, 2] * vector[..., 2:3]
    return (from_first + from_second) + from_third


def _axis_angle_to_matrices(axis_angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert axis-angle vectors of shape (..., 3) to rotation matrices of shape (..., 3, 3).

    Rodrigues' rotation formula, R = I + sin(t)/t K + (1 - cos(t))/t^2 K K, where K is the
    cross-product matrix of the vector and t its length. The scale factors are written so that
    they stay accurate for tiny angles, and a zero vector gives the exact identity.
    """
    x, y, z = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    angle_squared = x * x + y * y + z * z
    positive = angle_squared > 0.0
    angle = np.where(positive, np.sqrt(angle_squared), 1.0)
    sine_scale = np.where(positive, np.sin(angle) / angle, 1.0)  # sin(t) / t
    half_angle = 0.5 * angle
    # (1 - cos(t)) / t^2 equals 0.5 * (sin(t / 2) / (t / 2))^2 and has no cancellation.
    cosine_scale = np.where(positive, 0.5 * (np.sin(half_angle) / half_angle) ** 2, 0.5)
    matrices = np.empty(axis_angle.shape[:-1] + (3, 3), dtype=np.float64)
    matrices[..., 0, 0] = 1.0 - cosine_scale * (y * y + z * z)
    matrices[..., 0, 1] = cosine_scale * (x * y) - sine_scale * z
    matrices[..., 0, 2] = cosine_scale * (x * z) + sine_scale * y
    matrices[..., 1, 0] = cosine_scale * (x * y) + sine_scale * z
    matrices[..., 1, 1] = 1.0 - cosine_scale * (x * x + z * z)
    matrices[..., 1, 2] = cosine_scale * (y * z) - sine_scale * x
    matrices[..., 2, 0] = cosine_scale * (x * z) - sine_scale * y
    matrices[..., 2, 1] = cosine_scale * (y * z) + sine_scale * x
    matrices[..., 2, 2] = 1.0 - cosine_scale * (x * x + y * y)
    return matrices


def _forward_kinematics(
    rest_offsets: NDArray[np.float64],
    pose_root: NDArray[np.float64],
    pose_body: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the global rotation and the position of every joint, relative to the pelvis.

    ``rest_offsets`` (22, 3) holds each joint's zero-pose position minus its parent's; the row of
    the root is unused. The shapes of the results are (22, 3, 3) and (22, 3). The root rotates
    about the pelvis, which stays at the origin. A joint's global rotation is its parent's global
    rotation times its own local rotation, and its position is its parent's position plus the rest
    offset turned by the parent's global rotation.
    """
    axis_angle = np.concatenate([pose_root.reshape(1, 3), pose_body.reshape(NUM_JOINTS - 1, 3)])
    local_rotations = _axis_angle_to_matrices(axis_angle)
    global_rotations = np.empty((NUM_JOINTS, 3, 3), dtype=np.float64)
    positions = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    global_rotations[0] = local_rotations[0]
    for joints, parents in _TREE_LEVELS:
        parent_rotations = global_rotations[parents]
        global_rotations[joints] = _matrix_product(parent_rotations, local_rotations[joints])
        positions[joints] = positions[parents] + _matrix_vector(
            parent_rotations, rest_offsets[joints]
        )
    return global_rotations, positions


def _point_segment_distance(
    points: NDArray[np.float64], starts: NDArray[np.float64], ends: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the distance from each point to the segment that runs from its start to its end."""
    direction = ends - starts
    length_squared = _dot(direction, direction)
    has_length = length_squared > _POINT_LENGTH_SQUARED
    fraction = np.where(
        has_length,
        _dot(points - starts, direction) / np.where(has_length, length_squared, 1.0),
        0.0,
    )
    nearest = starts + direction * np.clip(fraction, 0.0, 1.0)[..., None]
    return _length(points - nearest)


def segment_distance(
    first_start: ArrayLike, first_end: ArrayLike, second_start: ArrayLike, second_end: ArrayLike
) -> NDArray[np.float64]:
    """Return the smallest distance between two line segments, for arrays of shape (..., 3).

    The leading dimensions broadcast against each other and the result has the broadcast shape. A
    segment whose ends coincide is a point. The minimum over the two segments is reached either at
    an end of one segment, or at the closest points of the two infinite lines when both lie inside
    their segments, so the result is the smallest of the four end-to-segment distances and, when
    it applies, the distance between those line points.
    """
    first_start, first_end, second_start, second_end = np.broadcast_arrays(
        *(
            np.asarray(points, dtype=np.float64)
            for points in (first_start, first_end, second_start, second_end)
        )
    )
    candidates = [
        _point_segment_distance(first_start, second_start, second_end),
        _point_segment_distance(first_end, second_start, second_end),
        _point_segment_distance(second_start, first_start, first_end),
        _point_segment_distance(second_end, first_start, first_end),
    ]

    # Closest points of the two infinite lines: the fractions along the first and along the second
    # segment that minimise the distance between the two points they name. The points count only
    # when both fractions lie strictly inside their segments.
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    offset = first_start - second_start
    first_squared = _dot(first_direction, first_direction)
    second_squared = _dot(second_direction, second_direction)
    cross_term = _dot(first_direction, second_direction)
    first_offset = _dot(first_direction, offset)
    second_offset = _dot(second_direction, offset)
    determinant = first_squared * second_squared - cross_term * cross_term
    well_posed = determinant > _PARALLEL_TOLERANCE * first_squared * second_squared
    safe_determinant = np.where(well_posed, determinant, 1.0)
    along_first = (cross_term * second_offset - first_offset * second_squared) / safe_determinant
    along_second = (first_squared * second_offset - cross_term * first_offset) / safe_determinant
    inside = (
        well_posed
        & (along_first > 0.0)
        & (along_first < 1.0)
        & (along_second > 0.0)
        & (along_second < 1.0)
    )
    first_point = first_start + first_direction * along_first[..., None]
    second_point = second_start + second_direction * along_second[..., None]
    candidates.append(np.where(inside, _length(first_point - second_point), np.inf))
    return np.minimum.reduce(candidates)


@dataclass(frozen=True)
class PoseFilterResult:
    """The outcome of the two pose tests (data-model: PoseFilter result).

    ``joint_angle_ok`` is True when every joint turns by no more than its limit.
    ``self_intersection_ok`` is True when no tested pair of capsules overlaps by more than the
    tolerance. ``accepted`` is True when both are. Both tests always run, so a rejection reports
    every reason.
    """

    joint_angle_ok: bool
    self_intersection_ok: bool
    accepted: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", self.joint_angle_ok and self.self_intersection_ok)


@dataclass(frozen=True, eq=False)
class _CapsuleSet:
    """The skeleton and the capsules of the reference shape, in the frame of the pelvis."""

    rest_offsets: NDArray[np.float64]  # (22, 3) zero-pose joint minus its parent's
    end_offsets: NDArray[np.float64]  # (22, 3) capsule end minus joint, in the joint's rest frame
    radii: NDArray[np.float64]  # (22,) capsule radii in metres


def _pose_arrays(pose: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a pose into float64 arrays of shape (3,) and (63,), and check them."""
    try:
        pose_root, pose_body = pose
    except (TypeError, ValueError) as error:
        raise ValueError("a pose must be a pair (pose_root, pose_body)") from error
    root = np.asarray(pose_root, dtype=np.float64)
    body = np.asarray(pose_body, dtype=np.float64)
    if root.shape != (ROOT_POSE_SIZE,):
        raise ValueError(f"pose_root must have shape ({ROOT_POSE_SIZE},), got {root.shape}")
    if body.shape != (BODY_POSE_SIZE,):
        raise ValueError(f"pose_body must have shape ({BODY_POSE_SIZE},), got {body.shape}")
    if not (np.isfinite(root).all() and np.isfinite(body).all()):
        raise ValueError("pose values must be finite")
    return root, body


def _build_capsules(body: BodyModel, betas: NDArray[np.float64]) -> _CapsuleSet:
    """Measure the capsules of ``body`` for the shape ``betas`` in its canonical pose."""
    if tuple(body.joint_names) != JOINT_NAMES:
        raise ValueError("the joint names of the body model differ from body/base.py JOINT_NAMES")
    zero_root = np.zeros(ROOT_POSE_SIZE, dtype=np.float64)
    zero_body = np.zeros(BODY_POSE_SIZE, dtype=np.float64)
    canonical_root, canonical_body = body.canonical()
    canonical_root = np.asarray(canonical_root, dtype=np.float64)
    canonical_body = np.asarray(canonical_body, dtype=np.float64)

    rest_joints = np.asarray(body.joints(betas, zero_root, zero_body), dtype=np.float64)
    canonical_joints = np.asarray(
        body.joints(betas, canonical_root, canonical_body), dtype=np.float64
    )
    canonical_vertices = np.asarray(
        body.vertices(betas, canonical_root, canonical_body), dtype=np.float64
    )
    part_ids = np.asarray(body.part_ids)

    # Everything is measured from the pelvis, so a model that moves the whole body (for example to
    # stand on the floor) gives the same capsules.
    canonical_vertices = canonical_vertices - canonical_joints[0]
    canonical_joints = canonical_joints - canonical_joints[0]
    rest_joints = rest_joints - rest_joints[0]
    rest_offsets = rest_joints.copy()
    for joint in range(1, NUM_JOINTS):
        rest_offsets[joint] = rest_joints[joint] - rest_joints[PARENTS[joint]]
    rest_offsets[0] = 0.0

    canonical_rotations, kinematic_joints = _forward_kinematics(
        rest_offsets, canonical_root, canonical_body
    )
    worst = float(np.max(_length(kinematic_joints - canonical_joints)))
    if worst > _KINEMATICS_TOLERANCE_M:
        raise ValueError(
            "the canonical joints of the body model differ from forward kinematics of its "
            f"zero-pose joints by up to {worst:.6f} m; the pose filter cannot pose its capsules"
        )

    end_offsets = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    radii = np.zeros(NUM_JOINTS, dtype=np.float64)
    for joint in range(NUM_JOINTS):
        part = canonical_vertices[part_ids == joint]
        if part.shape[0] == 0:
            raise ValueError(f"the part of joint '{JOINT_NAMES[joint]}' has no vertices")
        start = canonical_joints[joint]
        end_joint = _CAPSULE_END_JOINTS[joint]
        if end_joint is None:
            end = _tip(part, start, canonical_joints[joint] - canonical_joints[PARENTS[joint]])
        else:
            end = canonical_joints[end_joint]
        radii[joint] = float(np.median(_point_segment_distance(part, start, end)))
        # Store the end in the joint's own rest frame, so that posing needs one rotation per joint.
        to_rest_frame = np.swapaxes(canonical_rotations[joint], -1, -2)
        end_offsets[joint] = _matrix_vector(to_rest_frame, end - start)
    return _CapsuleSet(rest_offsets=rest_offsets, end_offsets=end_offsets, radii=radii)


def _tip(
    part: NDArray[np.float64], start: NDArray[np.float64], bone: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the end of the capsule of a joint that has no child joint.

    The axis leaves the joint towards the centre of the part (or along ``bone``, the direction
    from the parent joint, when the centre sits on the joint). Its length reaches the farthest
    vertex along that direction minus the radius of the capsule, because the cap adds that radius.
    """
    towards_centre = part.mean(axis=0) - start
    if _length(towards_centre) > 1e-9:
        direction = towards_centre / _length(towards_centre)
    elif _length(bone) > 1e-9:
        direction = bone / _length(bone)
    else:
        raise ValueError("cannot place a capsule on a joint at the same point as its parent")
    reach = max(float(np.max(_dot(part - start, direction))), 0.0)
    # The radius is measured against the full reach first, then the axis stops one radius short.
    radius = float(np.median(_point_segment_distance(part, start, start + direction * reach)))
    return start + direction * max(reach - radius, 0.0)


class PoseFilter:
    """Accept or reject a pose by its joint angles and by the overlap of its body capsules.

    ``body`` is any body model of this package. ``limits_deg`` is the table ``pose.limits_deg``
    (degrees per joint group), ``capsule_overlap_cm`` is ``pose.capsule_overlap_cm``, and
    ``betas`` is the reference shape for the capsules (see the module docstring). The object does
    not change after construction, so one filter serves a whole run. The usual call is
    ``PoseFilter.from_config(config, body).accept(pose).accepted``.

    Raises ValueError for a limit table that ``joint_limits_rad`` refuses, a negative or
    non-finite tolerance, a body model whose joints do not follow forward kinematics, or a body
    part without vertices.
    """

    def __init__(
        self,
        body: BodyModel,
        limits_deg: Mapping[str, float],
        capsule_overlap_cm: float,
        betas: ArrayLike,
    ) -> None:
        self._limits_rad = joint_limits_rad(limits_deg)
        tolerance_cm = float(capsule_overlap_cm)
        if not math.isfinite(tolerance_cm) or tolerance_cm < 0.0:
            raise ValueError(
                "capsule_overlap_cm must be a finite number of at least 0; "
                f"got {capsule_overlap_cm!r}"
            )
        self._tolerance_m = tolerance_cm / 100.0
        reference_betas = np.asarray(betas, dtype=np.float64)
        if reference_betas.ndim != 1:
            raise ValueError(f"betas must be one-dimensional, got shape {reference_betas.shape}")
        self._capsules = _build_capsules(body, reference_betas)
        pairs = [
            (first, second)
            for first, second in combinations(range(NUM_JOINTS), 2)
            if not _is_exempt(first, second)
        ]
        self._first_capsule = np.array([first for first, _ in pairs], dtype=np.intp)
        self._second_capsule = np.array([second for _, second in pairs], dtype=np.intp)
        self._radius_sums = (
            self._capsules.radii[self._first_capsule] + self._capsules.radii[self._second_capsule]
        )
        self._warn_if_canonical_pose_fails(body)

    def _warn_if_canonical_pose_fails(self, body: BodyModel) -> None:
        """Log a warning when the canonical pose of the body fails this filter.

        Research R3 requires the canonical pose to pass. A model that breaks the rule would have
        most poses near its canonical pose rejected, and a run would stop on pose.max_rejections.
        The stand-in is tested; the SMPL-X and SMPL models cannot be tested without their files.
        """
        canonical = body.canonical()
        result = self.accept(canonical)
        if result.accepted:
            return
        reasons = [f"{name} over its limit" for name in self.over_limit_joints(canonical)]
        reasons += [
            f"{first} and {second} overlap" for first, second in self.overlapping_pairs(canonical)
        ]
        logger.warning(
            "the canonical pose of the body model fails the pose filter (%s); poses near it will "
            "be rejected. Check the body model, or raise pose.capsule_overlap_cm.",
            "; ".join(reasons),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], body: BodyModel) -> "PoseFilter":
        """Build the filter from a resolved configuration (config.py ``resolve_config``).

        It reads ``pose.limits_deg`` and ``pose.capsule_overlap_cm``, and takes the mean shape,
        ``body.n_betas`` zeros, as the reference shape.
        """
        return cls(
            body,
            config["pose"]["limits_deg"],
            config["pose"]["capsule_overlap_cm"],
            np.zeros(config["body"]["n_betas"], dtype=np.float64),
        )

    @property
    def capsule_radii_m(self) -> NDArray[np.float64]:
        """The capsule radius of each of the 22 joints in metres, in joint order. A new array."""
        return self._capsules.radii.copy()

    @property
    def checked_pairs(self) -> tuple[tuple[str, str], ...]:
        """The pairs of capsules that the self-intersection test covers, as joint-name pairs."""
        return tuple(
            (JOINT_NAMES[first], JOINT_NAMES[second])
            for first, second in zip(self._first_capsule, self._second_capsule, strict=True)
        )

    def capsule_axes(
        self, pose: tuple[ArrayLike, ArrayLike]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the start and the end of the axis of every capsule in a pose.

        Both results have shape (22, 3), in joint order, in metres, relative to the pelvis. The
        start of capsule ``j`` is joint ``j``.
        """
        pose_root, pose_body = _pose_arrays(pose)
        return self._posed_axes(pose_root, pose_body)

    def over_limit_joints(self, pose: tuple[ArrayLike, ArrayLike]) -> list[str]:
        """Return the names of the joints that turn by more than their limit, in joint order."""
        _, pose_body = _pose_arrays(pose)
        return self._joints_over_limit(pose_body)

    def overlapping_pairs(self, pose: tuple[ArrayLike, ArrayLike]) -> list[tuple[str, str]]:
        """Return the tested capsule pairs that overlap by more than the tolerance.

        Each pair holds the names of the two joints that own the capsules, in joint order. The
        capsule of ``left_hip`` is the left thigh, of ``left_knee`` the left shin, and so on (see
        the module docstring).
        """
        pose_root, pose_body = _pose_arrays(pose)
        too_deep = self._overlap_depths(pose_root, pose_body) > self._tolerance_m
        return [
            (JOINT_NAMES[first], JOINT_NAMES[second])
            for first, second in zip(
                self._first_capsule[too_deep], self._second_capsule[too_deep], strict=True
            )
        ]

    def accept(self, pose: tuple[ArrayLike, ArrayLike]) -> PoseFilterResult:
        """Test a pose given as ``(pose_root, pose_body)`` with angles in radians.

        That is the pair that ``BodyPose`` and the pose sources return. Raises ValueError when the
        arrays do not have shapes (3,) and (63,) or hold a value that is not finite.
        """
        pose_root, pose_body = _pose_arrays(pose)
        joint_angle_ok = not self._joints_over_limit(pose_body)
        deepest = self._overlap_depths(pose_root, pose_body)
        self_intersection_ok = not bool(np.any(deepest > self._tolerance_m))
        return PoseFilterResult(joint_angle_ok, self_intersection_ok)

    def _joints_over_limit(self, pose_body: NDArray[np.float64]) -> list[str]:
        angles = _length(pose_body.reshape(NUM_JOINTS - 1, 3))
        over = angles > self._limits_rad + _ANGLE_TOLERANCE_RAD
        return [JOINT_NAMES[joint + 1] for joint in np.flatnonzero(over)]

    def _overlap_depths(
        self, pose_root: NDArray[np.float64], pose_body: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return, for each tested pair, the sum of the two radii minus the distance of the axes."""
        starts, ends = self._posed_axes(pose_root, pose_body)
        distances = segment_distance(
            starts[self._first_capsule],
            ends[self._first_capsule],
            starts[self._second_capsule],
            ends[self._second_capsule],
        )
        return self._radius_sums - distances

    def _posed_axes(
        self, pose_root: NDArray[np.float64], pose_body: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the start and the end of every capsule axis; both move with the owning joint."""
        capsules = self._capsules
        rotations, starts = _forward_kinematics(capsules.rest_offsets, pose_root, pose_body)
        return starts, starts + _matrix_vector(rotations, capsules.end_offsets)
