"""Procedural capsule mannequin: a pure-NumPy stand-in body model for tests (research R4).

The mannequin lets every stage (pose filters, cameras, rendering, measurement, training,
calibration) run through the BodyModel interface of ``base.py`` with no licensed asset and no GPU
(constitution Principle VI). It has the 22 SMPL-X body joints, ten shape coefficients, and 4,268
vertices, and it is deterministic: the same inputs always give the same numbers.

Geometry. Each of the 22 joints owns exactly one capsule (rigid skinning, so ``part_ids`` is the
owning joint). The capsule of a joint starts at the joint and ends at one child joint, or at a tip
point for the five joints without a child (head, both wrists, both feet). A capsule is a straight
axis from the start to the end with a rounded cap at each end. Limbs, neck, head, hands, and feet
are circular. The four torso capsules (pelvis, spine1, spine2, spine3) are elliptical, so a frontal
view shows the torso width and a side view shows the torso depth, and more views carry more
information about the shape. The torso caps are flattened so the stacked torso capsules keep a
visible waist. Axes and units follow ``base.py``: metres, y up, the body faces +z, left is +x. In
the canonical pose (all rotations zero, both arms 30 degrees away from the torso) the soles rest
on y = 0 and the top of the head is at the standing height.

Shape coefficients. Each coefficient moves one parameter linearly (``shape_parameters`` returns
all of them). A coefficient outside the range -5 to 5 is clipped to it, so every finite input
gives a valid body.

    index  parameter                                  nominal   change per unit coefficient
    0      standing height                            1.75 m    4 percent
    1      torso width (chest)                        0.34 m    8 percent
    2      torso depth (chest)                        0.235 m   8 percent
    3      hip width (distance of the hip joints)     0.19 m    8 percent
    4      thigh radius                               0.085 m   8 percent
    5      arm radius                                 0.045 m   8 percent
    6      shoulder width (distance of the joints)    0.36 m    6 percent
    7      head radius                                0.085 m   5 percent
    8      leg-length ratio (hip height / height)     0.52      4 percent
    9      waist indent (share by which the waist     0.12      0.03 added
           is narrower than the chest)

Two rules keep every body valid: the hip joints are at least a thigh radius plus 6 mm from the
midline, so the thighs never overlap in the canonical pose, and the shoulder joints stay outside
the upper chest.

Public sources. The capsule design is research R4. The joint order and parent table are those of
SMPL-X (see ``base.py``). Posing is forward kinematics from axis-angle rotations as in SMPL (Loper
et al., "SMPL: A Skinned Multi-Person Linear Model", ACM Transactions on Graphics 34(6), 2015), with
each rotation matrix from Rodrigues' rotation formula. The proportions are rounded adult
body-segment fractions of the standing height (Drillis and Contini, "Body Segment Parameters",
1966, as tabulated in Winter, "Biomechanics and Motor Control of Human Movement").

Measurements. With the definitions of research R6, the canonical mannequin has a height equal to
``height_m`` and a thigh circumference equal to the perimeter of a regular polygon with
``radial_segments`` sides and circumradius ``thigh_radius_m``; for the default 16 sides that is
0.64 percent below 2 pi r.

Arithmetic. Rotations and sums are written as elementwise operations in a fixed order, with no
BLAS or einsum call, so that a result does not depend on the batch size or the thread count.
"""

import math
import operator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from strike_a_pose.body.base import (
    BODY_POSE_SIZE,
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENTS,
    ROOT_POSE_SIZE,
    BodyPose,
)

__all__ = [
    "BETA_LIMIT",
    "DEFAULT_CAP_RINGS",
    "DEFAULT_RADIAL_SEGMENTS",
    "NUM_BETAS",
    "StandInBody",
    "StandInShape",
    "shape_parameters",
]

# Number of shape coefficients, and the largest magnitude of a coefficient that is used.
NUM_BETAS: int = 10
BETA_LIMIT: float = 5.0

# Default mesh resolution: sides of the polygon around a capsule axis, and rings on each rounded
# cap. A capsule then has 2 + 2 * 6 * 16 = 194 vertices and the mannequin has 22 * 194 = 4,268.
DEFAULT_RADIAL_SEGMENTS: int = 16
DEFAULT_CAP_RINGS: int = 6

# Rounded adult proportions as fractions of the standing height.
_ANKLE_HEIGHT_FRACTION = 0.05
_KNEE_SHARE = 0.52  # knee height above the ankle, as a share of the ankle-to-hip height
_PELVIS_ABOVE_HIP_FRACTION = 0.046
_HEAD_JOINT_BELOW_TOP_FRACTION = 0.1257
_COLLAR_BELOW_NECK_FRACTION = 0.012
_SHOULDER_BELOW_COLLAR_FRACTION = 0.010
_UPPER_ARM_FRACTION = 0.186
_FOREARM_FRACTION = 0.146
_HAND_FRACTION = 0.108
_FOOT_FORWARD_FRACTION = 0.075  # ankle to the ball of the foot
_TOE_FRACTION = 0.045  # ball of the foot to the toe tip
_FOOT_RADIUS_FRACTION = 0.02

# Vertical shares of the span from the pelvis joint to the head joint: spine1, spine2, spine3, and
# neck (the head takes the rest). The torso therefore gets shorter when the legs get longer.
_TORSO_SHARES = (11.0 / 54.0, 13.0 / 54.0, 6.0 / 54.0, 14.0 / 54.0)

# Torso cross-section of the upper chest as a fraction of the chest half width and half depth.
_UPPER_CHEST_SCALE = 0.92

# Limb radii as fractions of the thigh, arm, or head radius.
_SHIN_RADIUS_SHARE = 0.55
_FOREARM_RADIUS_SHARE = 0.8
_HAND_RADIUS_SHARE = 0.6
_NECK_RADIUS_SHARE = 0.6

# Smallest gap between the two thighs, and the collar joint's share of the chest half width.
_MINIMUM_LEG_GAP_M = 0.012
_COLLAR_HALF_WIDTH_SHARE = 0.6

# In the canonical pose the arms hang straight, 30 degrees away from the torso.
_ARM_SIDEWAYS = math.sin(math.radians(30.0))
_ARM_DOWN = -math.cos(math.radians(30.0))

# The end of the capsule that each joint owns: the child joint where the capsule ends, or None
# for a joint without a child, whose capsule ends at a tip point (toes, top of head, fingertips).
_SEGMENT_END_NAMES: dict[str, str | None] = {
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
_SEGMENT_END_JOINTS: tuple[int | None, ...] = tuple(
    None if _SEGMENT_END_NAMES[name] is None else JOINT_INDEX[_SEGMENT_END_NAMES[name]]
    for name in JOINT_NAMES
)

# How far a rounded cap reaches beyond the end point, as a share of the mean cross-section radius.
# 1 is a hemisphere. The torso caps are flatter so that stacked torso capsules keep a waist.
_TORSO_CAP_SCALE = {"pelvis": 0.5, "spine1": 0.5, "spine2": 0.5, "spine3": 0.35}
_CAP_SCALE = np.array([_TORSO_CAP_SCALE.get(name, 1.0) for name in JOINT_NAMES], dtype=np.float64)
_CAP_SCALE.flags.writeable = False

# Direction that fixes the first cross-section axis of each capsule: the x axis, except for the
# collar capsules, which run along x themselves and so use the z axis.
_REFERENCE_AXES = np.array(
    [[0.0, 0.0, 1.0] if name.endswith("_collar") else [1.0, 0.0, 0.0] for name in JOINT_NAMES],
    dtype=np.float64,
)
_REFERENCE_AXES.flags.writeable = False


@dataclass(frozen=True, eq=False)
class StandInShape:
    """Mannequin proportions for shape coefficients; each field has the batch shape of the input."""

    height_m: NDArray[np.float64]  # standing height; the vertex extent along y when canonical
    torso_width_m: NDArray[np.float64]  # full chest width
    torso_depth_m: NDArray[np.float64]  # full chest depth
    hip_width_m: NDArray[np.float64]  # distance between the hip joints, before the leg-gap rule
    thigh_radius_m: NDArray[np.float64]  # radius of the thigh capsule
    arm_radius_m: NDArray[np.float64]  # radius of the upper-arm capsule
    shoulder_width_m: NDArray[np.float64]  # distance between the shoulder joints
    head_radius_m: NDArray[np.float64]  # radius of the head capsule
    leg_ratio: NDArray[np.float64]  # height of the hip joints divided by the standing height
    waist_indent: NDArray[np.float64]  # share by which the waist is narrower than the chest


def _as_float_array(value: ArrayLike, name: str, size: int) -> NDArray[np.float64]:
    """Convert to a float64 array whose last dimension has the given size."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have a last dimension of {size}, got shape {array.shape}")
    return array


def shape_parameters(betas: ArrayLike) -> StandInShape:
    """Return the mannequin proportions for shape coefficients of shape (..., 10).

    The fields have the batch shape of ``betas``. A coefficient outside -5 to 5 is clipped.
    """
    clipped = np.clip(_as_float_array(betas, "betas", NUM_BETAS), -BETA_LIMIT, BETA_LIMIT)
    return StandInShape(
        height_m=1.75 * (1.0 + 0.04 * clipped[..., 0]),
        torso_width_m=0.34 * (1.0 + 0.08 * clipped[..., 1]),
        torso_depth_m=0.235 * (1.0 + 0.08 * clipped[..., 2]),
        hip_width_m=0.19 * (1.0 + 0.08 * clipped[..., 3]),
        thigh_radius_m=0.085 * (1.0 + 0.08 * clipped[..., 4]),
        arm_radius_m=0.045 * (1.0 + 0.08 * clipped[..., 5]),
        shoulder_width_m=0.36 * (1.0 + 0.06 * clipped[..., 6]),
        head_radius_m=0.085 * (1.0 + 0.05 * clipped[..., 7]),
        leg_ratio=0.52 * (1.0 + 0.04 * clipped[..., 8]),
        waist_indent=0.12 + 0.03 * clipped[..., 9],
    )


def _point(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Join three coordinate arrays of shape (count,) into points of shape (count, 3)."""
    return np.stack([x, y, z], axis=-1)


def _rest_joints(shape: StandInShape) -> NDArray[np.float64]:
    """Return the joint positions of the zero-pose mannequin, shape (count, 22, 3).

    Every field of ``shape`` has shape (count,).
    """
    height = shape.height_m
    zero = np.zeros_like(height)

    # Heights of the joints above the floor.
    hip_height = shape.leg_ratio * height
    ankle_height = _ANKLE_HEIGHT_FRACTION * height
    knee_height = ankle_height + _KNEE_SHARE * (hip_height - ankle_height)
    pelvis_height = hip_height + _PELVIS_ABOVE_HIP_FRACTION * height
    head_height = (1.0 - _HEAD_JOINT_BELOW_TOP_FRACTION) * height
    span = head_height - pelvis_height
    spine1_height = pelvis_height + _TORSO_SHARES[0] * span
    spine2_height = spine1_height + _TORSO_SHARES[1] * span
    spine3_height = spine2_height + _TORSO_SHARES[2] * span
    neck_height = spine3_height + _TORSO_SHARES[3] * span
    collar_height = neck_height - _COLLAR_BELOW_NECK_FRACTION * height
    shoulder_height = collar_height - _SHOULDER_BELOW_COLLAR_FRACTION * height

    # Sideways offsets of the joint pairs; the left joint of a pair is on +x. The hips keep the
    # thighs apart, and the shoulders stay outside the upper chest.
    hip_half = np.maximum(0.5 * shape.hip_width_m, shape.thigh_radius_m + 0.5 * _MINIMUM_LEG_GAP_M)
    upper_chest_half = 0.5 * shape.torso_width_m * _UPPER_CHEST_SCALE
    shoulder_half = np.maximum(
        0.5 * shape.shoulder_width_m, upper_chest_half + 0.5 * shape.arm_radius_m
    )
    collar_half = _COLLAR_HALF_WIDTH_SHARE * 0.5 * shape.torso_width_m

    # The ball of each foot rests on the floor; the foot slopes down from the ankle to the ball.
    foot_radius = _FOOT_RADIUS_FRACTION * height
    foot_forward = _FOOT_FORWARD_FRACTION * height

    # The arms hang straight, 30 degrees away from the torso.
    elbow_x = shoulder_half + _UPPER_ARM_FRACTION * height * _ARM_SIDEWAYS
    elbow_y = shoulder_height + _UPPER_ARM_FRACTION * height * _ARM_DOWN
    wrist_x = elbow_x + _FOREARM_FRACTION * height * _ARM_SIDEWAYS
    wrist_y = elbow_y + _FOREARM_FRACTION * height * _ARM_DOWN

    by_name = {
        "pelvis": _point(zero, pelvis_height, zero),
        "left_hip": _point(hip_half, hip_height, zero),
        "right_hip": _point(-hip_half, hip_height, zero),
        "spine1": _point(zero, spine1_height, zero),
        "left_knee": _point(hip_half, knee_height, zero),
        "right_knee": _point(-hip_half, knee_height, zero),
        "spine2": _point(zero, spine2_height, zero),
        "left_ankle": _point(hip_half, ankle_height, zero),
        "right_ankle": _point(-hip_half, ankle_height, zero),
        "spine3": _point(zero, spine3_height, zero),
        "left_foot": _point(hip_half, foot_radius, foot_forward),
        "right_foot": _point(-hip_half, foot_radius, foot_forward),
        "neck": _point(zero, neck_height, zero),
        "left_collar": _point(collar_half, collar_height, zero),
        "right_collar": _point(-collar_half, collar_height, zero),
        "head": _point(zero, head_height, zero),
        "left_shoulder": _point(shoulder_half, shoulder_height, zero),
        "right_shoulder": _point(-shoulder_half, shoulder_height, zero),
        "left_elbow": _point(elbow_x, elbow_y, zero),
        "right_elbow": _point(-elbow_x, elbow_y, zero),
        "left_wrist": _point(wrist_x, wrist_y, zero),
        "right_wrist": _point(-wrist_x, wrist_y, zero),
    }
    return np.stack([by_name[name] for name in JOINT_NAMES], axis=1)


def _capsule_ends_and_radii(
    shape: StandInShape, rest_joints: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the end point and the two cross-section radii of the capsule of every joint.

    The results have shapes (count, 22, 3), (count, 22), and (count, 22). The first radius is
    along the first cross-section axis (x for the torso) and the second along the other axis (z).
    """
    height = shape.height_m
    thigh_radius = shape.thigh_radius_m
    arm_radius = shape.arm_radius_m
    head_radius = shape.head_radius_m
    foot_radius = _FOOT_RADIUS_FRACTION * height

    # Tip points of the five joints that have no child joint.
    toe_length = _TOE_FRACTION * height
    hand_length = _HAND_FRACTION * height
    left_foot = rest_joints[:, JOINT_INDEX["left_foot"]]
    right_foot = rest_joints[:, JOINT_INDEX["right_foot"]]
    head = rest_joints[:, JOINT_INDEX["head"]]
    left_wrist = rest_joints[:, JOINT_INDEX["left_wrist"]]
    right_wrist = rest_joints[:, JOINT_INDEX["right_wrist"]]
    tips = {
        "left_foot": _point(left_foot[:, 0], left_foot[:, 1], left_foot[:, 2] + toe_length),
        "right_foot": _point(right_foot[:, 0], right_foot[:, 1], right_foot[:, 2] + toe_length),
        "head": _point(head[:, 0], height - head_radius, head[:, 2]),
        "left_wrist": _point(
            left_wrist[:, 0] + hand_length * _ARM_SIDEWAYS,
            left_wrist[:, 1] + hand_length * _ARM_DOWN,
            left_wrist[:, 2],
        ),
        "right_wrist": _point(
            right_wrist[:, 0] - hand_length * _ARM_SIDEWAYS,
            right_wrist[:, 1] + hand_length * _ARM_DOWN,
            right_wrist[:, 2],
        ),
    }
    ends = np.stack(
        [
            tips[name] if end is None else rest_joints[:, end]
            for name, end in zip(JOINT_NAMES, _SEGMENT_END_JOINTS, strict=True)
        ],
        axis=1,
    )

    # Cross-section radii. The torso is elliptical; every other capsule is circular.
    chest_half_width = 0.5 * shape.torso_width_m
    chest_half_depth = 0.5 * shape.torso_depth_m
    waist_scale = 1.0 - shape.waist_indent
    hip_half = rest_joints[:, JOINT_INDEX["left_hip"], 0]
    shin_radius = _SHIN_RADIUS_SHARE * thigh_radius
    neck_radius = _NECK_RADIUS_SHARE * head_radius
    forearm_radius = _FOREARM_RADIUS_SHARE * arm_radius
    hand_radius = _HAND_RADIUS_SHARE * arm_radius
    radius_first = {
        "pelvis": hip_half + 0.8 * thigh_radius,
        "left_hip": thigh_radius,
        "right_hip": thigh_radius,
        "spine1": chest_half_width * waist_scale,
        "left_knee": shin_radius,
        "right_knee": shin_radius,
        "spine2": chest_half_width,
        "left_ankle": foot_radius,
        "right_ankle": foot_radius,
        "spine3": chest_half_width * _UPPER_CHEST_SCALE,
        "left_foot": foot_radius,
        "right_foot": foot_radius,
        "neck": neck_radius,
        "left_collar": arm_radius,
        "right_collar": arm_radius,
        "head": head_radius,
        "left_shoulder": arm_radius,
        "right_shoulder": arm_radius,
        "left_elbow": forearm_radius,
        "right_elbow": forearm_radius,
        "left_wrist": hand_radius,
        "right_wrist": hand_radius,
    }
    radius_second = dict(radius_first)
    radius_second["pelvis"] = chest_half_depth
    radius_second["spine1"] = chest_half_depth * waist_scale
    radius_second["spine2"] = chest_half_depth
    radius_second["spine3"] = chest_half_depth * _UPPER_CHEST_SCALE
    return (
        ends,
        np.stack([radius_first[name] for name in JOINT_NAMES], axis=1),
        np.stack([radius_second[name] for name in JOINT_NAMES], axis=1),
    )


def _axis_angle_to_matrices(axis_angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert axis-angle vectors of shape (..., 3) to rotation matrices of shape (..., 3, 3).

    Uses Rodrigues' rotation formula, R = I + sin(t)/t K + (1 - cos(t))/t^2 K K, where K is the
    cross-product matrix of the vector and t its length. The two scale factors are written so
    that they stay accurate for tiny angles, and a zero vector gives the exact identity.
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


def _dot(first: NDArray[np.float64], second: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dot product over the last axis, adding the three products in a fixed order."""
    from_x = first[..., 0] * second[..., 0]
    from_y = first[..., 1] * second[..., 1]
    from_z = first[..., 2] * second[..., 2]
    return (from_x + from_y) + from_z


def _cross(first: NDArray[np.float64], second: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cross product over the last axis."""
    return np.stack(
        [
            first[..., 1] * second[..., 2] - first[..., 2] * second[..., 1],
            first[..., 2] * second[..., 0] - first[..., 0] * second[..., 2],
            first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0],
        ],
        axis=-1,
    )


def _forward_kinematics(
    rest_joints: NDArray[np.float64],
    pose_root: NDArray[np.float64],
    pose_body: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the global rotation and the posed position of every joint.

    The shapes are (count, 22, 3, 3) and (count, 22, 3). The root rotates about the pelvis joint,
    which keeps its rest position. A joint's global rotation is its parent's global rotation times
    its own local rotation, and its position is its parent's position plus the rest offset from
    the parent turned by the parent's global rotation.
    """
    count = rest_joints.shape[0]
    local_axis_angle = np.concatenate(
        [pose_root.reshape(count, 1, 3), pose_body.reshape(count, NUM_JOINTS - 1, 3)], axis=1
    )
    local_rotations = _axis_angle_to_matrices(local_axis_angle)
    global_rotations = np.empty((count, NUM_JOINTS, 3, 3), dtype=np.float64)
    posed_joints = np.empty((count, NUM_JOINTS, 3), dtype=np.float64)
    global_rotations[:, 0] = local_rotations[:, 0]
    posed_joints[:, 0] = rest_joints[:, 0]
    for joint in range(1, NUM_JOINTS):
        parent = PARENTS[joint]
        offset = rest_joints[:, joint] - rest_joints[:, parent]
        global_rotations[:, joint] = _matrix_product(
            global_rotations[:, parent], local_rotations[:, joint]
        )
        posed_joints[:, joint] = posed_joints[:, parent] + _matrix_vector(
            global_rotations[:, parent], offset
        )
    return global_rotations, posed_joints


@dataclass(frozen=True, eq=False)
class _CapsuleTemplate:
    """Mesh layout shared by all capsules: a bottom pole, 2 * cap_rings rings, and a top pole."""

    vertex_count: int
    on_upper_half: NDArray[np.bool_]  # measured from the end point instead of the start point
    axial_scale: NDArray[np.float64]  # cosine of the angle from the pole: 1 at a pole, 0 at a rim
    first_scale: NDArray[np.float64]  # ring radius scale times cosine of the angle around the axis
    second_scale: NDArray[np.float64]  # ring radius scale times sine of the angle around the axis
    faces: NDArray[np.int64]  # (4 * radial_segments * cap_rings, 3) with outward winding


def _snap_to_zero(values: list[float]) -> list[float]:
    """Replace values that are zero up to rounding error by exact zeros."""
    return [0.0 if abs(value) < 1e-12 else value for value in values]


def _capsule_template(radial_segments: int, cap_rings: int) -> _CapsuleTemplate:
    """Build the unit capsule layout for the given resolution.

    Vertex 0 is the bottom pole, then come 2 * cap_rings rings of radial_segments vertices from
    the bottom to the top (the rim of the lower cap is ring cap_rings - 1, the rim of the upper
    cap is ring cap_rings), and the last vertex is the top pole.
    """
    ring_count = 2 * cap_rings
    vertex_count = 2 + ring_count * radial_segments

    # Angle from the pole of each cap ring, from the ring next to the pole to the rim.
    polar_angles = [0.5 * math.pi * (index + 1) / cap_rings for index in range(cap_rings)]
    axial_by_polar = _snap_to_zero([math.cos(angle) for angle in polar_angles])
    radius_by_polar = [math.sin(angle) for angle in polar_angles]
    around_angles = [2.0 * math.pi * index / radial_segments for index in range(radial_segments)]
    cosines = _snap_to_zero([math.cos(angle) for angle in around_angles])
    sines = _snap_to_zero([math.sin(angle) for angle in around_angles])

    on_upper_half = np.zeros(vertex_count, dtype=bool)
    axial_scale = np.zeros(vertex_count, dtype=np.float64)
    first_scale = np.zeros(vertex_count, dtype=np.float64)
    second_scale = np.zeros(vertex_count, dtype=np.float64)
    axial_scale[0] = 1.0  # bottom pole
    on_upper_half[-1] = True  # top pole
    axial_scale[-1] = 1.0
    for ring in range(ring_count):
        upper = ring >= cap_rings
        polar_index = ring_count - 1 - ring if upper else ring
        for side in range(radial_segments):
            vertex = 1 + ring * radial_segments + side
            on_upper_half[vertex] = upper
            axial_scale[vertex] = axial_by_polar[polar_index]
            first_scale[vertex] = radius_by_polar[polar_index] * cosines[side]
            second_scale[vertex] = radius_by_polar[polar_index] * sines[side]

    def ring_vertex(ring: int, side: int) -> int:
        return 1 + ring * radial_segments + side % radial_segments

    # The winding makes every triangle normal point out of the capsule when the two cross-section
    # axes and the capsule axis form a right-handed frame.
    top_pole = vertex_count - 1
    faces: list[tuple[int, int, int]] = []
    for side in range(radial_segments):
        faces.append((0, ring_vertex(0, side + 1), ring_vertex(0, side)))
    for ring in range(ring_count - 1):
        for side in range(radial_segments):
            first = ring_vertex(ring, side)
            second = ring_vertex(ring, side + 1)
            third = ring_vertex(ring + 1, side + 1)
            fourth = ring_vertex(ring + 1, side)
            faces.append((first, second, third))
            faces.append((first, third, fourth))
    last_ring = ring_count - 1
    for side in range(radial_segments):
        faces.append((top_pole, ring_vertex(last_ring, side), ring_vertex(last_ring, side + 1)))
    return _CapsuleTemplate(
        vertex_count=vertex_count,
        on_upper_half=on_upper_half,
        axial_scale=axial_scale,
        first_scale=first_scale,
        second_scale=second_scale,
        faces=np.array(faces, dtype=np.int64),
    )


def _capsule_offsets(
    template: _CapsuleTemplate,
    rest_joints: NDArray[np.float64],
    ends: NDArray[np.float64],
    radius_first: NDArray[np.float64],
    radius_second: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Return the vertex offsets of every capsule from its own joint in the rest pose.

    The result is one array per coordinate, each of shape (count, 22, vertices per capsule).
    """
    axis = ends - rest_joints
    length = np.sqrt(_dot(axis, axis))
    direction = axis / length[..., None]
    reference = np.broadcast_to(_REFERENCE_AXES, direction.shape)
    first_raw = reference - _dot(reference, direction)[..., None] * direction
    first_axis = first_raw / np.sqrt(_dot(first_raw, first_raw))[..., None]
    second_axis = _cross(direction, first_axis)  # the three axes form a right-handed frame
    cap_height = _CAP_SCALE * (0.5 * (radius_first + radius_second))

    cap_reach = cap_height[..., None] * template.axial_scale
    axial = np.where(template.on_upper_half, length[..., None] + cap_reach, -cap_reach)
    across_first = radius_first[..., None] * template.first_scale
    across_second = radius_second[..., None] * template.second_scale
    offsets = []
    for coordinate in range(3):
        along_axis = axial * direction[..., None, coordinate]
        along_first = across_first * first_axis[..., None, coordinate]
        along_second = across_second * second_axis[..., None, coordinate]
        offsets.append((along_axis + along_first) + along_second)
    return offsets


def _flatten_inputs(
    betas: ArrayLike, pose_root: ArrayLike, pose_body: ArrayLike
) -> tuple[tuple[int, ...], StandInShape, NDArray[np.float64], NDArray[np.float64]]:
    """Check the three inputs, broadcast their batch dimensions, and flatten them.

    Returns the batch shape, the proportions with fields of shape (count,), the root pose of
    shape (count, 3), and the body pose of shape (count, 63).
    """
    betas_array = _as_float_array(betas, "betas", NUM_BETAS)
    root_array = _as_float_array(pose_root, "pose_root", ROOT_POSE_SIZE)
    body_array = _as_float_array(pose_body, "pose_body", BODY_POSE_SIZE)
    try:
        batch_shape = np.broadcast_shapes(
            betas_array.shape[:-1], root_array.shape[:-1], body_array.shape[:-1]
        )
    except ValueError as error:
        raise ValueError(
            "the leading dimensions of betas, pose_root, and pose_body do not broadcast: "
            f"{betas_array.shape}, {root_array.shape}, {body_array.shape}"
        ) from error
    count = math.prod(batch_shape)

    def flatten(array: NDArray[np.float64], size: int) -> NDArray[np.float64]:
        return np.broadcast_to(array, batch_shape + (size,)).reshape(count, size)

    return (
        batch_shape,
        shape_parameters(flatten(betas_array, NUM_BETAS)),
        flatten(root_array, ROOT_POSE_SIZE),
        flatten(body_array, BODY_POSE_SIZE),
    )


class StandInBody:
    """The procedural capsule mannequin; it implements the ``BodyModel`` protocol of base.py.

    ``radial_segments`` is the number of sides of the polygon around each capsule axis and
    ``cap_rings`` the number of rings on each rounded cap. A capsule has
    ``2 + 2 * cap_rings * radial_segments`` vertices, the same for every shape and pose.

    The ten shape coefficients are described in the module docstring; ``shape_parameters`` returns
    the resulting proportions. While ``vertices`` runs it needs about 0.4 MB per mesh, a quarter
    of it the returned array, so a caller with many meshes calls it in chunks.
    """

    #: Number of shape coefficients that ``vertices`` and ``joints`` expect.
    n_betas: int = NUM_BETAS

    def __init__(
        self,
        radial_segments: int = DEFAULT_RADIAL_SEGMENTS,
        cap_rings: int = DEFAULT_CAP_RINGS,
    ) -> None:
        radial_segments = operator.index(radial_segments)
        cap_rings = operator.index(cap_rings)
        if radial_segments < 3:
            raise ValueError(f"radial_segments must be at least 3, got {radial_segments}")
        if cap_rings < 1:
            raise ValueError(f"cap_rings must be at least 1, got {cap_rings}")
        self._template = _capsule_template(radial_segments, cap_rings)
        capsule_vertices = self._template.vertex_count
        first_vertex = capsule_vertices * np.arange(NUM_JOINTS, dtype=np.int64)
        faces_by_capsule = self._template.faces[None, :, :] + first_vertex[:, None, None]
        self._faces = faces_by_capsule.reshape(-1, 3)
        self._part_ids = np.repeat(np.arange(NUM_JOINTS, dtype=np.int64), capsule_vertices)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The 22 body joint names in model order."""
        return JOINT_NAMES

    @property
    def faces(self) -> NDArray[np.int64]:
        """Triangle vertex indices, shape (F, 3). Each access returns a new array."""
        return self._faces.copy()

    @property
    def part_ids(self) -> NDArray[np.int64]:
        """For each vertex the index of the joint that owns it, shape (V,). A new array."""
        return self._part_ids.copy()

    @property
    def vertex_count(self) -> int:
        """Number of vertices of every mesh this model returns."""
        return int(self._part_ids.shape[0])

    def canonical(self) -> BodyPose:
        """Return the canonical pose: every rotation zero, arms 30 degrees from the torso."""
        return BodyPose(
            pose_root=np.zeros(ROOT_POSE_SIZE, dtype=np.float64),
            pose_body=np.zeros(BODY_POSE_SIZE, dtype=np.float64),
        )

    def joints(
        self, betas: ArrayLike, pose_root: ArrayLike, pose_body: ArrayLike
    ) -> NDArray[np.float64]:
        """Return the posed joint positions in metres, shape (..., 22, 3)."""
        batch_shape, shape, flat_root, flat_body = _flatten_inputs(betas, pose_root, pose_body)
        _, posed_joints = _forward_kinematics(_rest_joints(shape), flat_root, flat_body)
        return posed_joints.reshape(batch_shape + (NUM_JOINTS, 3))

    def vertices(
        self, betas: ArrayLike, pose_root: ArrayLike, pose_body: ArrayLike
    ) -> NDArray[np.float64]:
        """Return the posed mesh vertices in metres, shape (..., V, 3) with V = vertex_count."""
        batch_shape, shape, flat_root, flat_body = _flatten_inputs(betas, pose_root, pose_body)
        rest_joints = _rest_joints(shape)
        ends, radius_first, radius_second = _capsule_ends_and_radii(shape, rest_joints)
        offsets = _capsule_offsets(self._template, rest_joints, ends, radius_first, radius_second)
        global_rotations, posed_joints = _forward_kinematics(rest_joints, flat_root, flat_body)

        # Turn each capsule with its joint and move it to the posed joint, one coordinate at a time.
        count = rest_joints.shape[0]
        posed = np.empty((count, NUM_JOINTS, self._template.vertex_count, 3), dtype=np.float64)
        for coordinate in range(3):
            rotation_row = global_rotations[:, :, None, coordinate, :]
            from_first = rotation_row[..., 0] * offsets[0]
            from_second = rotation_row[..., 1] * offsets[1]
            from_third = rotation_row[..., 2] * offsets[2]
            turned = (from_first + from_second) + from_third
            posed[..., coordinate] = turned + posed_joints[:, :, None, coordinate]
        return posed.reshape(batch_shape + (self.vertex_count, 3))
