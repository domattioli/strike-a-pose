"""Smoke tests for body/base.py: joint table, bones, subtrees, protocol, and canonical mesh."""

import numpy as np
import pytest

from strike_a_pose.body.base import (
    BODY_POSE_SIZE,
    BONES,
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENTS,
    ROOT_POSE_SIZE,
    BodyModel,
    BodyPose,
    canonical_mesh,
    joint_subtree,
)

# The 22 SMPL-X body joints in model order (research R5). Written out again here, so that an
# accidental change to the table in base.py fails a test instead of passing silently.
EXPECTED_JOINT_NAMES = (
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

# The parent of every joint except the pelvis, by name: the SMPL-X kinematic tree.
EXPECTED_PARENT_NAMES = {
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "head": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
}


def _is_ancestor_or_self(ancestor: int, joint: int) -> bool:
    """Return True when ``ancestor`` is ``joint`` or one of its ancestors."""
    while joint >= 0:
        if joint == ancestor:
            return True
        joint = PARENTS[joint]
    return False


class _FixedModel:
    """A minimal body model with a fixed one-triangle mesh; it records every call it receives."""

    joint_names = JOINT_NAMES
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    part_ids = np.zeros(3, dtype=np.int64)

    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    def _record(self, name, betas, pose_root, pose_body):
        self.calls.append((name, np.asarray(betas), np.asarray(pose_root), np.asarray(pose_body)))

    def vertices(self, betas, pose_root, pose_body):
        self._record("vertices", betas, pose_root, pose_body)
        return np.zeros((3, 3))

    def joints(self, betas, pose_root, pose_body):
        self._record("joints", betas, pose_root, pose_body)
        return np.zeros((NUM_JOINTS, 3))

    def canonical(self) -> BodyPose:
        return BodyPose(pose_root=np.zeros(ROOT_POSE_SIZE), pose_body=np.zeros(BODY_POSE_SIZE))


def test_joint_names_are_the_22_smplx_body_joints_in_order():
    assert NUM_JOINTS == 22
    assert JOINT_NAMES == EXPECTED_JOINT_NAMES
    assert len(set(JOINT_NAMES)) == NUM_JOINTS


def test_joint_names_match_the_installed_smplx_package():
    # The first 22 names of smplx's joint list are the body joints, in the order of base.py.
    smplx_joint_names = pytest.importorskip("smplx.joint_names").JOINT_NAMES
    assert tuple(smplx_joint_names[:NUM_JOINTS]) == JOINT_NAMES


def test_the_joints_named_by_the_measurement_rules_exist():
    # Research R5: the measurement definitions need the pelvis, spine, neck, hip, knee, and collar.
    needed = {
        "pelvis",
        "spine1",
        "spine2",
        "spine3",
        "neck",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_collar",
        "right_collar",
    }
    assert needed <= set(JOINT_NAMES)


def test_joint_index_maps_each_name_to_its_position_and_is_read_only():
    assert dict(JOINT_INDEX) == {name: index for index, name in enumerate(JOINT_NAMES)}
    assert JOINT_INDEX["pelvis"] == 0
    assert JOINT_INDEX["right_wrist"] == 21
    with pytest.raises(TypeError):
        JOINT_INDEX["pelvis"] = 5


def test_parent_table_is_the_smplx_kinematic_tree():
    assert len(PARENTS) == NUM_JOINTS
    assert PARENTS[0] == -1
    for joint in range(1, NUM_JOINTS):
        assert PARENTS[joint] < joint
        assert JOINT_NAMES[PARENTS[joint]] == EXPECTED_PARENT_NAMES[JOINT_NAMES[joint]]


def test_bones_join_each_parent_to_its_child():
    assert isinstance(BONES, tuple)
    assert len(BONES) == NUM_JOINTS - 1
    for bone, (parent, child) in enumerate(BONES):
        assert child == bone + 1
        assert parent == PARENTS[child]


def test_pose_sizes_follow_from_the_joint_count():
    assert ROOT_POSE_SIZE == 3
    assert BODY_POSE_SIZE == 3 * (NUM_JOINTS - 1) == 63


def test_joint_subtree_examples():
    assert joint_subtree(0) == tuple(range(NUM_JOINTS))
    assert joint_subtree(1) == (1, 4, 7, 10)
    assert joint_subtree(9) == (9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    assert joint_subtree(21) == (21,)


@pytest.mark.parametrize("joint", range(NUM_JOINTS))
def test_joint_subtree_is_the_joint_and_every_descendant(joint):
    expected = tuple(other for other in range(NUM_JOINTS) if _is_ancestor_or_self(joint, other))
    assert joint_subtree(joint) == expected


@pytest.mark.parametrize("joint", [-1, NUM_JOINTS])
def test_joint_subtree_rejects_an_index_outside_the_table(joint):
    with pytest.raises(ValueError):
        joint_subtree(joint)


def test_a_class_with_every_member_satisfies_the_protocol():
    assert isinstance(_FixedModel(), BodyModel)


def test_a_class_missing_a_member_does_not_satisfy_the_protocol():
    class MissingCanonical:
        joint_names = JOINT_NAMES
        faces = np.zeros((1, 3), dtype=np.int64)
        part_ids = np.zeros(3, dtype=np.int64)

        def vertices(self, betas, pose_root, pose_body):
            return np.zeros((3, 3))

        def joints(self, betas, pose_root, pose_body):
            return np.zeros((NUM_JOINTS, 3))

    assert not isinstance(MissingCanonical(), BodyModel)
    assert not isinstance(object(), BodyModel)


def test_canonical_mesh_evaluates_both_outputs_at_the_canonical_pose():
    model = _FixedModel()
    betas = np.linspace(-1.0, 1.0, 10)
    vertices, joints = canonical_mesh(model, betas)
    assert vertices.shape == (3, 3)
    assert joints.shape == (NUM_JOINTS, 3)
    assert [call[0] for call in model.calls] == ["vertices", "joints"]
    for _, call_betas, pose_root, pose_body in model.calls:
        assert np.array_equal(call_betas, betas)
        assert np.array_equal(pose_root, np.zeros(ROOT_POSE_SIZE))
        assert np.array_equal(pose_body, np.zeros(BODY_POSE_SIZE))


def test_body_pose_names_its_root_and_body_parts():
    assert BodyPose._fields == ("pose_root", "pose_body")
    pose = _FixedModel().canonical()
    assert pose.pose_root.shape == (ROOT_POSE_SIZE,)
    assert pose.pose_body.shape == (BODY_POSE_SIZE,)
    assert not pose.pose_root.any() and not pose.pose_body.any()


def test_table_constants_are_tuples():
    assert isinstance(JOINT_NAMES, tuple)
    assert isinstance(PARENTS, tuple)
    assert isinstance(BONES, tuple)
