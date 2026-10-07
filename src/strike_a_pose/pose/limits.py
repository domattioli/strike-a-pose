"""Asset-free pose sampler: each joint is drawn uniformly inside its limit (research R2 and R3).

This is the ``limits`` pose source. It needs no licensed asset, so it serves the tiny
configuration, the tests, and the fallback when the operator has no AMASS files (research R2).
For each of the 21 joints of ``pose_body`` it draws a rotation angle uniformly between zero and
the limit of the joint's group in ``pose.limits_deg`` (contracts/config.md), and a rotation axis
uniformly over the unit sphere. The joint's axis-angle vector is the angle times the axis, so its
norm, which is the rotation angle, never exceeds the limit. The filters of pose/filters.py (FR-001)
still check the whole pose: the joint limits and the self-intersection of the body.

The root orientation is zero, so the body stands upright. The rig's uniform azimuth supplies the
viewing direction, and AMASS root orientations are discarded in the same way (research R2).

Joint groups. The limit table has ten groups for the 21 joints of ``pose_body``. The two feet have
no group of their own, so each foot takes the ankle limit. This is an assumption of this module:
the table gives no range for the foot, and the foot rotates about its ankle. Every consumer of the
limit table should read ``JOINT_LIMIT_GROUP``, so that the sampler and the filters apply the same
limit to each joint.

Public sources. A direction uniform over the sphere from normalized standard normal vectors: M. E.
Muller, "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres",
Communications of the ACM 2(4), 1959. Axis-angle vectors and their rotation angle: Rodrigues'
rotation formula (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), in the convention
of body/base.py.
"""

import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from strike_a_pose.body.base import BODY_POSE_SIZE, JOINT_NAMES, ROOT_POSE_SIZE, BodyPose

__all__ = [
    "JOINT_LIMIT_GROUP",
    "LIMIT_GROUPS",
    "LimitPoseSource",
    "joint_limits_rad",
]

# The joint groups of the limit table pose.limits_deg, in the table order of contracts/config.md.
LIMIT_GROUPS: tuple[str, ...] = (
    "spine",
    "neck",
    "head",
    "collar",
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
)

# The limit group of each body joint. The root (pelvis) is not a joint of pose_body.
_GROUP_OF_JOINT: dict[str, str] = {
    "left_hip": "hip",
    "right_hip": "hip",
    "spine1": "spine",
    "left_knee": "knee",
    "right_knee": "knee",
    "spine2": "spine",
    "left_ankle": "ankle",
    "right_ankle": "ankle",
    "spine3": "spine",
    "left_foot": "ankle",  # no foot group in the table; the foot takes the ankle limit
    "right_foot": "ankle",  # no foot group in the table; the foot takes the ankle limit
    "neck": "neck",
    "left_collar": "collar",
    "right_collar": "collar",
    "head": "head",
    "left_shoulder": "shoulder",
    "right_shoulder": "shoulder",
    "left_elbow": "elbow",
    "right_elbow": "elbow",
    "left_wrist": "wrist",
    "right_wrist": "wrist",
}

# The limit group of each joint of pose_body, in pose_body order: joint j (1 to 21) is entry j - 1.
JOINT_LIMIT_GROUP: tuple[str, ...] = tuple(_GROUP_OF_JOINT[name] for name in JOINT_NAMES[1:])


def joint_limits_rad(limits_deg: Mapping[str, float]) -> NDArray[np.float64]:
    """Return the rotation limit of each joint of pose_body in radians, shape (21,).

    ``limits_deg`` must give a positive number of degrees for each group of ``LIMIT_GROUPS``, and
    for no other key. A missing group, an unknown group, or a value that is not a positive finite
    number raises ValueError, which names the key at fault.
    """
    for group in LIMIT_GROUPS:
        if group not in limits_deg:
            raise ValueError(f"pose.limits_deg has no value for the joint group '{group}'")
    for group in limits_deg:
        if group not in LIMIT_GROUPS:
            raise ValueError(f"pose.limits_deg has an unknown joint group '{group}'")
    limit_rad: dict[str, float] = {}
    for group in LIMIT_GROUPS:
        degrees = float(limits_deg[group])
        if not math.isfinite(degrees) or degrees <= 0:
            raise ValueError(
                f"pose.limits_deg.{group} must be a positive number of degrees; got {degrees!r}"
            )
        limit_rad[group] = math.radians(degrees)
    return np.array([limit_rad[group] for group in JOINT_LIMIT_GROUP], dtype=np.float64)


def _unit_axes(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    """Return ``count`` unit vectors, one per row, each uniform over the sphere."""
    axes = rng.normal(size=(count, 3))
    norms = np.linalg.norm(axes, axis=1)
    while np.any(norms == 0.0):  # a zero vector has no direction, so that row is drawn again
        redraw = norms == 0.0
        axes[redraw] = rng.normal(size=(int(np.count_nonzero(redraw)), 3))
        norms = np.linalg.norm(axes, axis=1)
    return axes / norms[:, np.newaxis]


class LimitPoseSource:
    """The ``limits`` pose source: a uniform draw inside the joint-limit table.

    ``name`` is the value of pose.source that selects this source. Two draws from generators in the
    same state give the same pose, so the source is deterministic per seed.
    """

    name = "limits"

    def __init__(self, limits_deg: Mapping[str, float]) -> None:
        self._limits_rad = joint_limits_rad(limits_deg)

    def draw(self, rng: np.random.Generator) -> BodyPose:
        """Return ``(pose_root, pose_body)``: zero root orientation and 63 joint values, radians.

        The angle of each joint is uniform between zero and its limit, and its axis is uniform
        over the sphere. The draw consumes ``rng`` in a fixed order, so the same state gives the
        same pose.
        """
        angles = rng.uniform(0.0, self._limits_rad)
        axes = _unit_axes(rng, len(self._limits_rad))
        pose_body = (angles[:, np.newaxis] * axes).reshape(BODY_POSE_SIZE)
        return BodyPose(np.zeros(ROOT_POSE_SIZE, dtype=np.float64), pose_body)
