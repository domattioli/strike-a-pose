"""Smoke tests for pose/filters.py: joint limits, capsule radii, self-intersection, and geometry."""

import dataclasses
import logging

import numpy as np
import pytest

from strike_a_pose.body.base import (
    BODY_POSE_SIZE,
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    ROOT_POSE_SIZE,
    BodyPose,
)
from strike_a_pose.body.standin import StandInBody, shape_parameters
from strike_a_pose.config import load_config, resolve_config
from strike_a_pose.pose.filters import PoseFilter, PoseFilterResult, segment_distance
from strike_a_pose.pose.limits import JOINT_LIMIT_GROUP, LimitPoseSource
from strike_a_pose.seeding import rng_for

# A unit axis for turning a joint by a chosen angle. It is not along a coordinate axis, so the angle
# has to come from the norm of the whole vector.
DIAGONAL_AXIS = np.array([1.0, 2.0, 2.0]) / 3.0

TRUNK_CHAIN = frozenset(
    {"pelvis", "spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar"}
)


@pytest.fixture(scope="module")
def body() -> StandInBody:
    return StandInBody()


@pytest.fixture(scope="module")
def config() -> dict:
    """The default configuration, which holds the tiny values and the default limit table."""
    return resolve_config({"seed": 0})


@pytest.fixture(scope="module")
def pose_filter(body, config) -> PoseFilter:
    """The filter of the default configuration for the mean shape; it never changes."""
    return PoseFilter.from_config(config, body)


def make_pose(**axis_angle_deg: tuple[float, float, float]) -> BodyPose:
    """Build a pose from joint names and axis-angle vectors in degrees; other joints are zero."""
    pose_body = np.zeros(BODY_POSE_SIZE)
    for name, vector in axis_angle_deg.items():
        joint = JOINT_INDEX[name]
        pose_body[3 * (joint - 1) : 3 * joint] = np.radians(vector)
    return BodyPose(np.zeros(ROOT_POSE_SIZE), pose_body)


def ordered(first: str, second: str) -> tuple[str, str]:
    """Return two joint names in joint order, the order of ``checked_pairs``."""
    return tuple(sorted((first, second), key=JOINT_INDEX.__getitem__))


def reference_distance(first_start, first_end, second_start, second_end):
    """Smallest distance between segments, found by a different method than the module uses.

    A point moves along the first segment. Its distance to the second segment is a convex function
    of its position, so a ternary search finds the minimum. The arguments are arrays of shape
    (..., 3).
    """
    direction = second_end - second_start
    length_squared = np.sum(direction**2, axis=-1)

    def distance_to_second(points):
        fraction = np.sum((points - second_start) * direction, axis=-1) / np.maximum(
            length_squared, 1e-30
        )
        nearest = second_start + np.clip(fraction, 0.0, 1.0)[..., None] * direction
        return np.linalg.norm(points - nearest, axis=-1)

    def along_first(fraction):
        return first_start + fraction[..., None] * (first_end - first_start)

    low = np.zeros(first_start.shape[:-1])
    high = np.ones_like(low)
    for _ in range(100):
        third = (high - low) / 3.0
        left, right = low + third, high - third
        keep_left = distance_to_second(along_first(left)) < distance_to_second(along_first(right))
        high = np.where(keep_left, right, high)
        low = np.where(keep_left, low, left)
    return distance_to_second(along_first(0.5 * (low + high)))


class WrappedBody:
    """The stand-in body with one change that a test asks for.

    ``offset`` moves every vertex and joint. ``canonical_pose`` replaces the canonical pose.
    ``joint_error_m`` shifts the left wrist joint in every call whose pose is not all zero, so the
    joints stop following forward kinematics. ``part_ids`` replaces the part of every vertex.
    """

    def __init__(
        self, inner, offset=(0.0, 0.0, 0.0), canonical_pose=None, joint_error_m=0.0, part_ids=None
    ):
        self._inner = inner
        self._offset = np.asarray(offset, dtype=np.float64)
        self._canonical_pose = canonical_pose
        self._joint_error_m = joint_error_m
        self._part_ids = part_ids

    @property
    def joint_names(self):
        return self._inner.joint_names

    @property
    def faces(self):
        return self._inner.faces

    @property
    def part_ids(self):
        return self._inner.part_ids if self._part_ids is None else self._part_ids

    def canonical(self):
        return self._inner.canonical() if self._canonical_pose is None else self._canonical_pose

    def vertices(self, betas, pose_root, pose_body):
        return self._inner.vertices(betas, pose_root, pose_body) + self._offset

    def joints(self, betas, pose_root, pose_body):
        joints = self._inner.joints(betas, pose_root, pose_body) + self._offset
        if self._joint_error_m and np.any(np.asarray(pose_body) != 0.0):
            joints[..., JOINT_INDEX["left_wrist"], 0] += self._joint_error_m
        return joints


# ---- the three named behaviours of the task -------------------------------------------------


def test_canonical_pose_is_accepted(pose_filter, body):
    result = pose_filter.accept(body.canonical())
    assert result == PoseFilterResult(joint_angle_ok=True, self_intersection_ok=True)
    assert result.accepted
    assert pose_filter.over_limit_joints(body.canonical()) == []
    assert pose_filter.overlapping_pairs(body.canonical()) == []


def test_canonical_pose_is_accepted_for_extreme_reference_shapes(body, config):
    # Corners of the shape range the generator can draw (beta_clip = 3): every coefficient at the
    # lower or the upper end, in 16 mixed patterns.
    clip = config["body"]["beta_clip"]
    for pattern in range(0, 1024, 67):
        betas = np.array([clip if (pattern >> bit) & 1 else -clip for bit in range(10)])
        shape_filter = PoseFilter(
            body, config["pose"]["limits_deg"], config["pose"]["capsule_overlap_cm"], betas
        )
        assert shape_filter.accept(body.canonical()).accepted, betas


@pytest.mark.parametrize("joint", range(1, NUM_JOINTS), ids=JOINT_NAMES[1:])
def test_an_over_limit_joint_is_rejected(pose_filter, config, joint):
    name = JOINT_NAMES[joint]
    limit_deg = config["pose"]["limits_deg"][JOINT_LIMIT_GROUP[joint - 1]]

    over = make_pose(**{name: tuple((limit_deg + 1.0) * DIAGONAL_AXIS)})
    result = pose_filter.accept(over)
    assert not result.joint_angle_ok
    assert not result.accepted
    assert pose_filter.over_limit_joints(over) == [name]

    # Exactly at the limit and just under it, the joint passes its angle test.
    for angle_deg in (limit_deg, limit_deg - 1.0):
        within = make_pose(**{name: tuple(angle_deg * DIAGONAL_AXIS)})
        assert pose_filter.accept(within).joint_angle_ok
        assert pose_filter.over_limit_joints(within) == []


def test_the_angle_is_the_norm_of_the_axis_angle_vector(pose_filter):
    # The wrist limit is 60 degrees. A vector of (40, 40, 0) degrees turns the wrist by 56.6
    # degrees, and (45, 45, 0) by 63.6 degrees, although no single value reaches 60.
    assert pose_filter.over_limit_joints(make_pose(left_wrist=(40.0, 40.0, 0.0))) == []
    assert pose_filter.over_limit_joints(make_pose(left_wrist=(45.0, 45.0, 0.0))) == ["left_wrist"]


def test_a_leg_through_leg_pose_is_rejected(pose_filter):
    # The left leg swings 30 degrees across the midline, well inside the hip limit of 120 degrees,
    # so only the self-intersection test can reject it.
    pose = make_pose(left_hip=(0.0, 0.0, -30.0))
    result = pose_filter.accept(pose)
    assert result.joint_angle_ok
    assert not result.self_intersection_ok
    assert not result.accepted
    assert ("left_hip", "right_hip") in pose_filter.overlapping_pairs(pose)  # the two thighs

    mirrored = make_pose(right_hip=(0.0, 0.0, 30.0))
    assert not pose_filter.accept(mirrored).self_intersection_ok
    assert ("left_hip", "right_hip") in pose_filter.overlapping_pairs(mirrored)


def test_a_limb_through_the_torso_or_the_head_is_rejected(pose_filter):
    # The left arm swings 60 degrees across the body: the forearm passes through the torso.
    across_body = make_pose(left_shoulder=(0.0, 0.0, -60.0))
    result = pose_filter.accept(across_body)
    assert result.joint_angle_ok
    assert not result.self_intersection_ok
    assert ("pelvis", "left_elbow") in pose_filter.overlapping_pairs(across_body)

    # The left arm goes up and the forearm folds back over the head.
    hand_on_head = make_pose(left_shoulder=(0.0, 0.0, 140.0), left_elbow=(0.0, 0.0, 140.0))
    result = pose_filter.accept(hand_on_head)
    assert result.joint_angle_ok
    assert not result.self_intersection_ok
    assert ("head", "left_wrist") in pose_filter.overlapping_pairs(hand_on_head)


# ---- which pairs are tested ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "pose",
    [
        make_pose(spine1=(35.0, 0.0, 0.0), spine2=(35.0, 0.0, 0.0), spine3=(35.0, 0.0, 0.0)),
        make_pose(spine1=(-35.0, 0.0, 0.0), spine2=(-35.0, 0.0, 0.0), spine3=(-35.0, 0.0, 0.0)),
        make_pose(spine1=(0.0, 35.0, 0.0), spine2=(0.0, 35.0, 0.0), spine3=(0.0, 35.0, 0.0)),
        make_pose(neck=(50.0, 0.0, 0.0), head=(50.0, 0.0, 0.0)),
        make_pose(left_knee=(150.0, 0.0, 0.0), right_knee=(150.0, 0.0, 0.0)),
        make_pose(left_elbow=(150.0, 0.0, 0.0), right_elbow=(150.0, 0.0, 0.0)),
        make_pose(left_hip=(-90.0, 0.0, 0.0), right_hip=(-90.0, 0.0, 0.0)),
        make_pose(
            left_hip=(-90.0, 0.0, 0.0),
            right_hip=(-90.0, 0.0, 0.0),
            left_knee=(90.0, 0.0, 0.0),
            right_knee=(90.0, 0.0, 0.0),
        ),
    ],
    ids=[
        "spine bent forward",
        "spine bent backward",
        "spine twisted",
        "neck and head bent forward",
        "both knees fully bent",
        "both elbows fully bent",
        "both hips bent to a right angle",
        "sitting",
    ],
)
def test_trunk_and_joint_neighbour_capsules_are_exempt(pose_filter, pose):
    # These poses squeeze capsules that are exempt (inside the trunk chain, or sharing a joint),
    # so the self-intersection test must still pass.
    assert pose_filter.accept(pose).accepted, pose_filter.overlapping_pairs(pose)


def test_checked_pairs_follow_the_exemption_rules(pose_filter):
    checked = pose_filter.checked_pairs
    assert len(set(checked)) == len(checked)
    # 231 pairs of 22 capsules, minus 28 inside the trunk chain, minus 12 more that share a joint
    # (17 neighbour pairs, 5 of them inside the trunk chain), minus 16 pairs of a torso capsule
    # (pelvis, spine1 to spine3) with a thigh or an upper arm.
    assert len(checked) == 231 - 28 - 12 - 16

    assert not [pair for pair in checked if set(pair) <= TRUNK_CHAIN]
    for neighbours in [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("left_ankle", "left_foot"),
        ("left_collar", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
    ]:
        assert neighbours not in checked
    for torso in ("pelvis", "spine1", "spine2", "spine3"):
        for limb in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
            assert ordered(torso, limb) not in checked

    for pair in [
        ("left_hip", "right_hip"),  # thigh and thigh
        ("left_hip", "right_knee"),  # thigh and the other shin
        ("left_foot", "right_foot"),
        ("pelvis", "left_knee"),  # torso and shin
        ("spine2", "left_elbow"),  # torso and forearm
        ("neck", "left_shoulder"),  # trunk chain and upper arm
        ("head", "left_wrist"),  # head and hand
        ("left_collar", "left_elbow"),  # collar and forearm
    ]:
        assert ordered(*pair) in checked, pair


# ---- capsule geometry -----------------------------------------------------------------------


def test_capsule_radii_come_from_the_canonical_vertices(pose_filter, body, config):
    # The stand-in is built from capsules, so the radius of every limb capsule is known.
    shape = shape_parameters(np.zeros(10))
    foot_radius = 0.02 * shape.height_m
    expected = {
        "left_hip": shape.thigh_radius_m,
        "right_hip": shape.thigh_radius_m,
        "left_knee": 0.55 * shape.thigh_radius_m,
        "right_knee": 0.55 * shape.thigh_radius_m,
        "left_ankle": foot_radius,
        "right_ankle": foot_radius,
        "left_foot": foot_radius,
        "right_foot": foot_radius,
        "neck": 0.6 * shape.head_radius_m,
        "head": shape.head_radius_m,
        "left_collar": shape.arm_radius_m,
        "right_collar": shape.arm_radius_m,
        "left_shoulder": shape.arm_radius_m,
        "right_shoulder": shape.arm_radius_m,
        "left_elbow": 0.8 * shape.arm_radius_m,
        "right_elbow": 0.8 * shape.arm_radius_m,
        "left_wrist": 0.6 * shape.arm_radius_m,
        "right_wrist": 0.6 * shape.arm_radius_m,
    }
    radii = pose_filter.capsule_radii_m
    assert radii.shape == (NUM_JOINTS,)
    for name, radius in expected.items():
        assert radii[JOINT_INDEX[name]] == pytest.approx(radius, rel=1e-9), name
    # The torso capsules are elliptical, so their radius is a median between the two half-axes.
    for name in ("pelvis", "spine1", "spine2", "spine3"):
        assert 0.08 < radii[JOINT_INDEX[name]] < 0.15, name

    # A reference shape with thicker thighs and thinner arms changes exactly those radii.
    betas = np.zeros(10)
    betas[4], betas[5] = 2.0, -2.0
    other = PoseFilter(body, config["pose"]["limits_deg"], 1.0, betas)
    other_shape = shape_parameters(betas)
    assert other.capsule_radii_m[JOINT_INDEX["left_hip"]] == pytest.approx(
        other_shape.thigh_radius_m, rel=1e-9
    )
    assert other.capsule_radii_m[JOINT_INDEX["left_shoulder"]] == pytest.approx(
        other_shape.arm_radius_m, rel=1e-9
    )
    assert other.capsule_radii_m[JOINT_INDEX["head"]] == pytest.approx(radii[JOINT_INDEX["head"]])


def test_posed_capsules_follow_the_body_mesh(pose_filter, body, config):
    # In a posed body, every vertex of a limb part lies at the capsule radius from the posed axis,
    # because the stand-in is made of capsules. A random root orientation is included.
    source = LimitPoseSource(config["pose"]["limits_deg"])
    rng = rng_for(7, 1)
    betas = np.zeros(10)
    part_ids = body.part_ids
    torso = {JOINT_INDEX[name] for name in ("pelvis", "spine1", "spine2", "spine3")}
    for _ in range(12):
        _, pose_body = source.draw(rng)
        pose_root = 0.5 * rng.normal(size=3)
        joints = body.joints(betas, pose_root, pose_body)
        vertices = body.vertices(betas, pose_root, pose_body) - joints[0]
        starts, ends = pose_filter.capsule_axes((pose_root, pose_body))
        np.testing.assert_allclose(starts, joints - joints[0], atol=1e-9)
        for owner, child in [
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("pelvis", "spine1"),
            ("spine3", "neck"),
            ("left_collar", "left_shoulder"),
            ("left_elbow", "left_wrist"),
        ]:
            # The capsule of a joint ends at the joint of its child.
            np.testing.assert_allclose(
                ends[JOINT_INDEX[owner]], joints[JOINT_INDEX[child]] - joints[0], atol=1e-9
            )
        for joint in range(NUM_JOINTS):
            if joint in torso:
                continue
            offsets = vertices[part_ids == joint] - starts[joint]
            axis = ends[joint] - starts[joint]
            length_squared = axis @ axis
            fractions = np.clip((offsets @ axis) / length_squared, 0.0, 1.0)
            distances = np.linalg.norm(offsets - fractions[:, None] * axis, axis=1)
            np.testing.assert_allclose(
                distances, pose_filter.capsule_radii_m[joint], atol=1e-9, err_msg=JOINT_NAMES[joint]
            )


def test_overlaps_match_an_independent_distance_between_the_posed_axes(pose_filter, body, config):
    # For random poses, the pairs that the filter reports are exactly the checked pairs whose radii
    # sum minus the distance of the posed axes exceeds the tolerance. The distance comes from the
    # ternary search above, the radii and the axes from the filter. Two tolerances are compared.
    source = LimitPoseSource(config["pose"]["limits_deg"])
    rng = rng_for(7, 5)
    poses = [source.draw(rng) for _ in range(80)] + [make_pose(left_hip=(0.0, 0.0, -30.0))]
    radii = pose_filter.capsule_radii_m
    tight = PoseFilter(body, config["pose"]["limits_deg"], 0.0, np.zeros(10))
    pairs = pose_filter.checked_pairs
    first = np.array([JOINT_INDEX[name] for name, _ in pairs])
    second = np.array([JOINT_INDEX[name] for _, name in pairs])
    reported = 0
    for pose in poses:
        starts, ends = pose_filter.capsule_axes(pose)
        distances = reference_distance(starts[first], ends[first], starts[second], ends[second])
        depths = radii[first] + radii[second] - distances
        for shape_filter, tolerance_m in [(pose_filter, 0.01), (tight, 0.0)]:
            clear = np.abs(depths - tolerance_m) > 1e-7  # skip a pair that sits on the threshold
            expected = {pairs[i] for i in np.flatnonzero(depths > tolerance_m) if clear[i]}
            found = set(shape_filter.overlapping_pairs(pose))
            borderline = {pairs[i] for i in np.flatnonzero(~clear)}
            assert found - borderline == expected, (found ^ expected) - borderline
            reported += len(found)
    assert reported > 250  # the poses press many pairs, so the comparison has much to check


def test_decisions_do_not_depend_on_the_root_orientation(pose_filter, config):
    source = LimitPoseSource(config["pose"]["limits_deg"])
    rng = rng_for(7, 2)
    for _ in range(40):
        _, pose_body = source.draw(rng)
        upright = pose_filter.accept((np.zeros(3), pose_body))
        turned = pose_filter.accept((rng.normal(size=3), pose_body))
        assert turned == upright


def test_a_model_that_moves_the_whole_body_gives_the_same_capsules(body, config, pose_filter):
    moved = WrappedBody(body, offset=(0.3, 0.1, -0.2))
    moved_filter = PoseFilter.from_config(config, moved)
    np.testing.assert_allclose(
        moved_filter.capsule_radii_m, pose_filter.capsule_radii_m, atol=1e-12
    )
    pose = make_pose(left_hip=(0.0, 0.0, -30.0))
    assert moved_filter.overlapping_pairs(pose) == pose_filter.overlapping_pairs(pose)


def test_a_canonical_pose_other_than_zero_gives_the_same_capsules(body, config, pose_filter):
    # A body model may declare a canonical pose with turned joints. The same body then gives the
    # same capsules, and the same decisions for every pose.
    canonical = make_pose(
        left_shoulder=(0.0, 0.0, -20.0),
        right_shoulder=(0.0, 0.0, 20.0),
        left_knee=(15.0, 0.0, 0.0),
        spine1=(5.0, 0.0, 0.0),
        head=(0.0, 10.0, 0.0),
        left_wrist=(0.0, 0.0, 12.0),
        left_ankle=(10.0, 0.0, 0.0),
    )
    canonical = BodyPose(np.array([0.1, -0.2, 0.05]), canonical.pose_body)
    other = PoseFilter.from_config(config, WrappedBody(body, canonical_pose=canonical))
    np.testing.assert_allclose(other.capsule_radii_m, pose_filter.capsule_radii_m, atol=1e-9)

    source = LimitPoseSource(config["pose"]["limits_deg"])
    rng = rng_for(7, 3)
    for _ in range(10):
        pose = source.draw(rng)
        for ours, theirs in zip(
            other.capsule_axes(pose), pose_filter.capsule_axes(pose), strict=True
        ):
            np.testing.assert_allclose(ours, theirs, atol=1e-9)
        assert other.accept(pose) == pose_filter.accept(pose)


def test_a_canonical_pose_that_passes_the_filter_is_not_reported(body, config, caplog):
    with caplog.at_level(logging.WARNING, logger="strike_a_pose.pose.filters"):
        PoseFilter.from_config(config, body)
    assert not caplog.records


def test_a_canonical_pose_that_fails_the_filter_is_reported(body, config, caplog):
    # A body model that declares crossed legs as its canonical pose breaks research R3. The filter
    # still builds, and says why most poses will be rejected.
    crossed = make_pose(left_hip=(0.0, 0.0, -30.0), left_elbow=(160.0, 0.0, 0.0))
    with caplog.at_level(logging.WARNING, logger="strike_a_pose.pose.filters"):
        PoseFilter.from_config(config, WrappedBody(body, canonical_pose=crossed))
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "canonical pose" in messages[0]
    assert "left_elbow over its limit" in messages[0]
    assert "left_hip and right_hip overlap" in messages[0]


# ---- configuration ----------------------------------------------------------------------------


def test_the_overlap_tolerance_comes_from_the_configuration(body, config):
    limits = config["pose"]["limits_deg"]

    def thighs_overlap(pose, tolerance_cm):
        shape_filter = PoseFilter(body, limits, tolerance_cm, np.zeros(10))
        return ("left_hip", "right_hip") in shape_filter.overlapping_pairs(pose)

    # The canonical thighs are 2 cm apart. Turning the left leg across the midline by 3 degrees
    # presses the thighs into each other by less than 1 cm, and 6 degrees by more than 1 cm.
    slight = make_pose(left_hip=(0.0, 0.0, -3.0))
    deep = make_pose(left_hip=(0.0, 0.0, -6.0))
    assert not thighs_overlap(slight, 1.0)
    assert thighs_overlap(slight, 0.0)
    assert thighs_overlap(deep, 1.0)
    assert not thighs_overlap(deep, 100.0)
    # With a tolerance of a metre, even a leg swung 30 degrees across the midline passes.
    wide = PoseFilter(body, limits, 100.0, np.zeros(10))
    assert wide.accept(make_pose(left_hip=(0.0, 0.0, -30.0))).self_intersection_ok


def test_the_filter_reads_its_limits_from_the_configuration(body, tiny_config_path):
    default = PoseFilter.from_config(load_config(tiny_config_path), body)
    tight = PoseFilter.from_config(
        load_config(tiny_config_path, ["pose.limits_deg.elbow=90"]), body
    )
    bent = make_pose(left_elbow=(100.0, 0.0, 0.0))
    assert default.over_limit_joints(bent) == []
    assert tight.over_limit_joints(bent) == ["left_elbow"]
    assert not tight.accept(bent).accepted


def test_limits_sampler_draws_pass_the_angle_test_at_a_usable_rate(pose_filter, config):
    # The limits pose source draws every joint inside the same table, so it never fails the angle
    # test. A usable share of its draws must pass the self-intersection test too, or a body would
    # run into pose.max_rejections.
    source = LimitPoseSource(config["pose"]["limits_deg"])
    rng = rng_for(7, 4)
    accepted = 0
    longest_run = run = 0
    draws = 300
    for _ in range(draws):
        result = pose_filter.accept(source.draw(rng))
        assert result.joint_angle_ok
        accepted += result.accepted
        run = 0 if result.accepted else run + 1
        longest_run = max(longest_run, run)
    assert 0.2 < accepted / draws < 0.7
    assert longest_run < config["pose"]["max_rejections"]


# ---- results and arguments -----------------------------------------------------------------


def test_accept_is_repeatable_and_leaves_the_pose_alone(pose_filter):
    pose = make_pose(left_hip=(0.0, 0.0, -30.0), right_elbow=(20.0, 10.0, 5.0))
    root, pose_body = pose.pose_root.copy(), pose.pose_body.copy()
    first = pose_filter.accept(pose)
    assert pose_filter.accept(pose) == first
    # A plain tuple and lists give the same answer as a BodyPose.
    assert pose_filter.accept((root.tolist(), pose_body.tolist())) == first
    np.testing.assert_array_equal(pose.pose_root, root)
    np.testing.assert_array_equal(pose.pose_body, pose_body)


def test_result_holds_both_tests_and_the_combined_decision():
    for angle_ok in (True, False):
        for intersection_ok in (True, False):
            result = PoseFilterResult(angle_ok, intersection_ok)
            assert result.accepted is (angle_ok and intersection_ok)
    assert [field.name for field in dataclasses.fields(PoseFilterResult)] == [
        "joint_angle_ok",
        "self_intersection_ok",
        "accepted",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        PoseFilterResult(True, True).accepted = False


def test_both_tests_always_run(pose_filter):
    # A rejection reports every reason: this pose is over the elbow limit and crosses the legs.
    pose = make_pose(left_elbow=(160.0, 0.0, 0.0), left_hip=(0.0, 0.0, -30.0))
    assert pose_filter.accept(pose) == PoseFilterResult(False, False)


@pytest.mark.parametrize(
    "pose",
    [
        None,
        np.zeros(66),
        (np.zeros(3),),
        (np.zeros(2), np.zeros(63)),
        (np.zeros(3), np.zeros(62)),
        (np.zeros((1, 3)), np.zeros(63)),
        (np.zeros(3), np.full(63, np.nan)),
        (np.array([0.0, np.inf, 0.0]), np.zeros(63)),
    ],
    ids=[
        "none",
        "one array",
        "one element",
        "short root",
        "short body",
        "batched root",
        "not a number",
        "infinite root",
    ],
)
def test_a_malformed_pose_raises(pose_filter, pose):
    with pytest.raises(ValueError):
        pose_filter.accept(pose)
    with pytest.raises(ValueError):
        pose_filter.overlapping_pairs(pose)


def test_bad_constructor_arguments_raise(body, config):
    limits = config["pose"]["limits_deg"]
    betas = np.zeros(10)
    with pytest.raises(ValueError, match="capsule_overlap_cm"):
        PoseFilter(body, limits, -0.5, betas)
    with pytest.raises(ValueError, match="capsule_overlap_cm"):
        PoseFilter(body, limits, float("nan"), betas)
    with pytest.raises(ValueError, match="knee"):
        PoseFilter(
            body,
            {group: degrees for group, degrees in limits.items() if group != "knee"},
            1.0,
            betas,
        )
    with pytest.raises(ValueError, match="foot"):
        PoseFilter(body, {**limits, "foot": 30.0}, 1.0, betas)
    with pytest.raises(ValueError, match="one-dimensional"):
        PoseFilter(body, limits, 1.0, np.zeros((2, 10)))
    with pytest.raises(ValueError):
        PoseFilter(body, limits, 1.0, np.zeros(7))  # the stand-in takes ten shape coefficients


def test_a_body_whose_joints_break_forward_kinematics_is_refused(body, config):
    turned = make_pose(left_elbow=(30.0, 0.0, 0.0))
    broken = WrappedBody(body, canonical_pose=turned, joint_error_m=0.05)
    with pytest.raises(ValueError, match="forward kinematics"):
        PoseFilter.from_config(config, broken)


def test_a_body_part_without_vertices_is_refused(body, config):
    part_ids = body.part_ids
    part_ids[part_ids == JOINT_INDEX["left_collar"]] = JOINT_INDEX["left_shoulder"]
    with pytest.raises(ValueError, match="left_collar"):
        PoseFilter.from_config(config, WrappedBody(body, part_ids=part_ids))


# ---- segment distance -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (((-1, 0, 0), (1, 0, 0)), ((0, -1, 1), (0, 1, 1)), 1.0),  # crossing at a height of 1
        (((0, 0, 0), (4, 0, 0)), ((1, 1, 0), (3, 1, 0)), 1.0),  # parallel, side by side
        (((0, 0, 0), (1, 0, 0)), ((5, 1, 0), (6, 1, 0)), np.sqrt(17.0)),  # parallel, apart
        (((0, 0, 0), (2, 0, 0)), ((1, 0, 0), (3, 0, 0)), 0.0),  # collinear, overlapping
        (((0, 0, 0), (1, 0, 0)), ((3, 0, 0), (4, 0, 0)), 2.0),  # collinear, apart
        (((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 5, 0)), 0.0),  # sharing an end
        (((0, 0, 0), (2, 0, 0)), ((1, 1, 0), (1, 3, 0)), 1.0),  # end of one meets the middle
        (((0, 0, 0), (2, 0, 0)), ((3, 1, 0), (5, 1, 0)), np.sqrt(2.0)),  # end to end
        (((1, 1, 1), (1, 1, 1)), ((0, 0, 0), (2, 0, 0)), np.sqrt(2.0)),  # a point and a segment
        (((1, 2, 3), (1, 2, 3)), ((4, 6, 3), (4, 6, 3)), 5.0),  # two points
        (((0, 0, 0), (2, 0, 0)), ((1, 1, 1e-7), (1, 3, -1e-7)), 1.0),  # nearly crossing
    ],
)
def test_segment_distance_of_known_configurations(first, second, expected):
    for one, other in [(first, second), (second, first)]:  # either order
        for forward in (True, False):  # either direction of the first segment
            start, end = one if forward else one[::-1]
            assert segment_distance(start, end, *other) == pytest.approx(expected, abs=1e-12)


def test_segment_distance_matches_dense_sampling():
    rng = np.random.default_rng(5)
    along = np.linspace(0.0, 1.0, 201)
    for _ in range(150):
        points = rng.uniform(-1.0, 1.0, size=(4, 3))
        if rng.random() < 0.3:  # make the segments nearly parallel now and then
            points[3] = points[2] + (points[1] - points[0]) * rng.uniform(-1.5, 1.5)
            points[3] += 1e-3 * rng.normal(size=3)
        first = points[0] + along[:, None] * (points[1] - points[0])
        second = points[2] + along[:, None] * (points[3] - points[2])
        sampled = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2).min()
        exact = float(segment_distance(*points))
        assert exact <= sampled + 1e-12  # the true minimum is never above a sampled value
        # The grid is close: moving each parameter by half a step changes a distance by at most
        # (total segment length) / 400, which is below 0.0174 for segments inside this cube.
        assert sampled - exact < 0.02


def test_segment_distance_broadcasts_leading_dimensions():
    first_start = np.zeros((4, 1, 3))
    first_end = np.tile([1.0, 0.0, 0.0], (4, 1, 1))
    second_start = np.zeros((1, 5, 3))
    second_start[0, :, 1] = np.arange(5.0)
    second_end = second_start + [1.0, 0.0, 0.0]
    distances = segment_distance(first_start, first_end, second_start, second_end)
    assert distances.shape == (4, 5)
    np.testing.assert_allclose(distances, np.tile(np.arange(5.0), (4, 1)))
