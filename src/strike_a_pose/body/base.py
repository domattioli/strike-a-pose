"""Body-model interface: the BodyModel protocol, the SMPL-X body joint table, and the bone list.

Every body model of this package (the procedural stand-in, SMPL-X, and SMPL) returns meshes and
joints through the one interface below, so pose filters, rendering, and measurement never depend on
which model produced a body (plan.md, Project Structure; research R4 and R5).

Public source of the joint table: the 22 body joints of SMPL-X, in the joint order and with the
parent table of the open-source ``smplx`` package (https://pypi.org/project/smplx, research R5),
described in Pavlakos et al., "Expressive Body Capture: 3D Hands, Face, and Body from a Single
Image" (CVPR 2019, https://smpl-x.is.tue.mpg.de/). The order of joints 1 to 21 is the order of the
AMASS ``pose_body`` values (research R2), so a motion-capture pose needs no reordering.

Conventions shared by every implementation:

* Lengths are in metres. The vertical axis is y (up), the body faces +z, and the subject's left
  side is +x, as in SMPL-X.
* ``betas`` are the shape coefficients. ``pose_root`` is the root orientation (3 axis-angle values
  in radians, applied about the pelvis joint). ``pose_body`` holds the axis-angle rotation of
  joints 1 to 21 relative to their parents (63 values, in the order of ``JOINT_NAMES``).
* Every argument may carry leading batch dimensions, and the leading dimensions broadcast against
  each other. The result carries the broadcast batch shape in front of its own shape.
* A part is the set of vertices that move with one joint (for SMPL-X, the argmax of the skinning
  weights). ``part_ids[v]`` is the index in ``JOINT_NAMES`` of the joint that owns vertex ``v``, so
  the part names of research R6 (pelvis, spine1, left_hip, and so on) are joint names.
* The canonical pose is the one fixed reference pose in which measurements are taken (FR-004).
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BODY_POSE_SIZE",
    "BONES",
    "JOINT_INDEX",
    "JOINT_NAMES",
    "NUM_JOINTS",
    "PARENTS",
    "ROOT_POSE_SIZE",
    "BodyModel",
    "BodyPose",
    "canonical_mesh",
    "joint_subtree",
]

# The 22 body joints of SMPL-X in model order. Joint 0 is the root (the pelvis).
JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

NUM_JOINTS: int = len(JOINT_NAMES)

# Number of values in pose_root (one axis-angle rotation) and in pose_body (joints 1 to 21).
ROOT_POSE_SIZE: int = 3
BODY_POSE_SIZE: int = 3 * (NUM_JOINTS - 1)

# Parent joint index of every joint; the root has parent -1. A parent always has a lower index
# than its children, so one pass in index order visits every parent before its children.
PARENTS: tuple[int, ...] = (
    -1,  # pelvis
    0,  # left_hip
    0,  # right_hip
    0,  # spine1
    1,  # left_knee
    2,  # right_knee
    3,  # spine2
    4,  # left_ankle
    5,  # right_ankle
    6,  # spine3
    7,  # left_foot
    8,  # right_foot
    9,  # neck
    9,  # left_collar
    9,  # right_collar
    12,  # head
    13,  # left_shoulder
    14,  # right_shoulder
    16,  # left_elbow
    17,  # right_elbow
    18,  # left_wrist
    19,  # right_wrist
)

# The bone list: one (parent joint, child joint) pair per non-root joint. Bone b joins joint
# BONES[b][0] to joint BONES[b][1], and its child joint is b + 1.
BONES: tuple[tuple[int, int], ...] = tuple(
    (PARENTS[joint], joint) for joint in range(1, NUM_JOINTS)
)

# Joint name to joint index (read-only).
JOINT_INDEX: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(JOINT_NAMES)}
)


class BodyPose(NamedTuple):
    """The pose values of one body: root orientation and body joint rotations, in radians."""

    pose_root: NDArray[np.float64]  # shape (3,), axis-angle
    pose_body: NDArray[np.float64]  # shape (63,), axis-angle of joints 1 to 21


@runtime_checkable
class BodyModel(Protocol):
    """A parametric human body that maps shape and pose values to a metric mesh and joints.

    ``isinstance(model, BodyModel)`` checks that a model has every member below.
    """

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The names of the 22 body joints in model order; equal to ``JOINT_NAMES``."""

    @property
    def faces(self) -> NDArray[np.int64]:
        """Triangle vertex indices, shape (F, 3). The topology never changes with shape or pose."""

    @property
    def part_ids(self) -> NDArray[np.int64]:
        """For each vertex, the index of the joint that owns it, shape (V,), values 0 to 21.

        A model with more joints than the 22 body joints (SMPL-X also has jaw, eye, and finger
        joints) gives such a vertex to its nearest ancestor among the 22 body joints: a finger
        joint to its wrist, the jaw and the eyes to the head.
        """

    def vertices(
        self, betas: ArrayLike, pose_root: ArrayLike, pose_body: ArrayLike
    ) -> NDArray[np.float64]:
        """Return the posed mesh vertices in metres, shape (..., V, 3).

        The vertex count V is the same for every input. ``betas`` has shape (..., n_betas),
        ``pose_root`` shape (..., 3), and ``pose_body`` shape (..., 63).
        """

    def joints(
        self, betas: ArrayLike, pose_root: ArrayLike, pose_body: ArrayLike
    ) -> NDArray[np.float64]:
        """Return the posed positions of the 22 body joints in metres, shape (..., 22, 3).

        The arguments are the same as for ``vertices``. A model with more joints returns only the
        22 body joints, in the order of ``JOINT_NAMES``.
        """

    def canonical(self) -> BodyPose:
        """Return the canonical pose: the fixed reference pose for measurements (FR-004).

        The result is ``(pose_root, pose_body)``. Pass both to ``vertices`` and ``joints`` to get
        the canonical-pose mesh of any shape; ``canonical_mesh`` does this in one call.
        """


def canonical_mesh(
    body: BodyModel, betas: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the canonical-pose vertices and joints of ``body`` for the shape coefficients."""
    pose_root, pose_body = body.canonical()
    return (
        body.vertices(betas, pose_root, pose_body),
        body.joints(betas, pose_root, pose_body),
    )


def joint_subtree(joint_index: int) -> tuple[int, ...]:
    """Return a joint and every joint below it in the kinematic tree, in index order."""
    if not 0 <= joint_index < NUM_JOINTS:
        raise ValueError(f"joint index must be in 0..{NUM_JOINTS - 1}, got {joint_index}")
    members = {joint_index}
    for joint in range(joint_index + 1, NUM_JOINTS):
        if PARENTS[joint] in members:
            members.add(joint)
    return tuple(sorted(members))
