"""Smoke tests for pose/limits.py: the joint-group table, the radian limits, and the limits draws.

Every draw comes from a generator seeded through rng_for, so the tests repeat exactly. No licensed
asset is read.
"""

import math

import numpy as np
import pytest

from strike_a_pose.body.base import BODY_POSE_SIZE, JOINT_NAMES, ROOT_POSE_SIZE, BodyPose
from strike_a_pose.pose.limits import (
    JOINT_LIMIT_GROUP,
    LIMIT_GROUPS,
    LimitPoseSource,
    joint_limits_rad,
)
from strike_a_pose.seeding import rng_for

# The ten joint groups of the limit table, in table order (contracts/config.md). Written out here,
# so that an accidental change to the table in limits.py fails a test instead of passing silently.
EXPECTED_LIMIT_GROUPS = (
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

# The default table of contracts/config.md, in degrees.
DEFAULT_LIMITS_DEG = {
    "spine": 35.0,
    "neck": 50.0,
    "head": 50.0,
    "collar": 20.0,
    "shoulder": 150.0,
    "elbow": 150.0,
    "wrist": 60.0,
    "hip": 120.0,
    "knee": 150.0,
    "ankle": 45.0,
}

# The limit group of each joint of pose_body, by joint name. The table has no group for the feet,
# so each foot takes the ankle limit (limits.py).
EXPECTED_GROUP_BY_JOINT = {
    "left_hip": "hip",
    "right_hip": "hip",
    "spine1": "spine",
    "left_knee": "knee",
    "right_knee": "knee",
    "spine2": "spine",
    "left_ankle": "ankle",
    "right_ankle": "ankle",
    "spine3": "spine",
    "left_foot": "ankle",
    "right_foot": "ankle",
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

# pose_body holds joints 1 to 21, three values each, so 21 rotation angles per pose.
NUM_BODY_JOINTS = len(JOINT_NAMES) - 1
DRAW_COUNT = 2000
# Angles are compared with this slack in radians, far above rounding error near one radian.
ANGLE_SLACK_RADIANS = 1e-12


def _draws(source: LimitPoseSource, seed: int, count: int) -> np.ndarray:
    """Return the pose_body of ``count`` draws from the generator of ``seed``, shape (count, 63)."""
    rng = rng_for(seed, 0)
    return np.stack([source.draw(rng).pose_body for _ in range(count)])


def _joint_angles(poses: np.ndarray) -> np.ndarray:
    """Return the rotation angle of each joint of each pose, shape (count, 21).

    The rotation angle of a joint is the norm of its axis-angle vector, in radians.
    """
    return np.linalg.norm(poses.reshape(len(poses), NUM_BODY_JOINTS, 3), axis=2)


def test_the_limit_groups_are_the_ten_groups_of_the_table_in_order():
    assert LIMIT_GROUPS == EXPECTED_LIMIT_GROUPS


def test_each_joint_takes_its_group_and_each_foot_takes_the_ankle_limit():
    assert set(EXPECTED_GROUP_BY_JOINT) == set(JOINT_NAMES[1:])
    assert JOINT_LIMIT_GROUP == tuple(EXPECTED_GROUP_BY_JOINT[name] for name in JOINT_NAMES[1:])


def test_the_default_table_converts_to_radians_for_each_joint():
    limits = joint_limits_rad(DEFAULT_LIMITS_DEG)
    expected = [
        math.radians(DEFAULT_LIMITS_DEG[EXPECTED_GROUP_BY_JOINT[name]]) for name in JOINT_NAMES[1:]
    ]
    assert limits.shape == (NUM_BODY_JOINTS,)
    assert limits.dtype == np.float64
    assert np.allclose(limits, expected, rtol=0.0, atol=1e-15)


def test_a_table_without_one_group_is_refused_and_the_group_is_named():
    table = {group: value for group, value in DEFAULT_LIMITS_DEG.items() if group != "hip"}
    with pytest.raises(ValueError, match="'hip'"):
        joint_limits_rad(table)


def test_a_table_with_an_unknown_group_is_refused_and_the_group_is_named():
    with pytest.raises(ValueError, match="'jaw'"):
        joint_limits_rad({**DEFAULT_LIMITS_DEG, "jaw": 10.0})


@pytest.mark.parametrize("value", [0.0, -10.0, math.nan, math.inf])
def test_a_limit_that_is_not_a_positive_finite_number_is_refused_with_its_key(value):
    table = {**DEFAULT_LIMITS_DEG, "knee": value}
    with pytest.raises(ValueError, match=r"pose\.limits_deg\.knee"):
        joint_limits_rad(table)
    with pytest.raises(ValueError, match=r"pose\.limits_deg\.knee"):
        LimitPoseSource(table)


def test_each_draw_has_a_zero_root_and_63_body_values():
    source = LimitPoseSource(DEFAULT_LIMITS_DEG)
    pose = source.draw(rng_for(14, 0))
    assert isinstance(pose, BodyPose)
    assert pose.pose_root.dtype == np.float64
    assert pose.pose_root.shape == (ROOT_POSE_SIZE,)
    assert np.array_equal(pose.pose_root, np.zeros(ROOT_POSE_SIZE))
    assert pose.pose_body.dtype == np.float64
    assert pose.pose_body.shape == (BODY_POSE_SIZE,)


def test_every_draw_keeps_each_joint_within_the_limit_of_its_group():
    poses = _draws(LimitPoseSource(DEFAULT_LIMITS_DEG), seed=11, count=DRAW_COUNT)
    limits = joint_limits_rad(DEFAULT_LIMITS_DEG)
    assert np.all(_joint_angles(poses) <= limits + ANGLE_SLACK_RADIANS)


def test_the_draws_reach_each_limit_so_the_whole_range_is_used():
    poses = _draws(LimitPoseSource(DEFAULT_LIMITS_DEG), seed=11, count=DRAW_COUNT)
    limits = joint_limits_rad(DEFAULT_LIMITS_DEG)
    assert np.all(_joint_angles(poses).max(axis=0) >= 0.9 * limits)


def test_a_table_with_one_value_per_group_limits_each_joint_by_its_own_group():
    table = {group: 5.0 * (index + 1) for index, group in enumerate(LIMIT_GROUPS)}
    poses = _draws(LimitPoseSource(table), seed=12, count=DRAW_COUNT)
    limits = joint_limits_rad(table)
    angles = _joint_angles(poses)
    assert np.all(angles <= limits + ANGLE_SLACK_RADIANS)
    assert np.all(angles.max(axis=0) >= 0.9 * limits)


def test_the_rotation_axes_are_spread_over_the_sphere():
    poses = _draws(LimitPoseSource(DEFAULT_LIMITS_DEG), seed=13, count=DRAW_COUNT)
    vectors = poses.reshape(-1, 3)
    axes = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # A direction uniform over the sphere has mean 0 and mean square 1/3 in each coordinate.
    assert np.all(np.abs(axes.mean(axis=0)) < 0.02)
    assert np.all(np.abs((axes**2).mean(axis=0) - 1.0 / 3.0) < 0.02)


def test_the_same_seed_and_path_give_the_same_poses():
    first = _draws(LimitPoseSource(DEFAULT_LIMITS_DEG), seed=15, count=100)
    second = _draws(LimitPoseSource(DEFAULT_LIMITS_DEG), seed=15, count=100)
    assert np.array_equal(first, second)


def test_another_seed_gives_other_poses():
    source = LimitPoseSource(DEFAULT_LIMITS_DEG)
    assert not np.array_equal(_draws(source, 15, 100), _draws(source, 16, 100))
