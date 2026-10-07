"""Tests for body/smplx_body.py: joint names, pose slots, part ids, and the asset check.

The licensed SMPL-X file is not in the repository, so the one test that needs it skips. The other
tests put a fake smplx module in place of the package, and a placeholder file in place of the model.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from strike_a_pose.assets import MissingAssetError, asset_root, require_asset
from strike_a_pose.body.base import (
    BODY_POSE_SIZE,
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENTS,
    canonical_mesh,
)
from strike_a_pose.body.smplx_body import SMPLX_ASSET_KEY, SMPLX_ASSET_PATH, SmplxBody

# The SMPL-X kinematic tree has 55 joints. Joints 22 to 24 are the jaw and the two eyes, which hang
# from the head. Joints 25 to 54 are the hands: 15 joints for each hand, in five finger chains of
# three joints, which hang from the wrist.
SMPLX_JOINT_COUNT = 55
HEAD = JOINT_INDEX["head"]
LEFT_WRIST = JOINT_INDEX["left_wrist"]
RIGHT_WRIST = JOINT_INDEX["right_wrist"]
FINGERS_PER_HAND = 5
JOINTS_PER_FINGER = 3

# Two triangles over vertices 0 to 4. The fake mesh needs no more than that.
FACES = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)


def _smplx_parents() -> list[int]:
    """Return the parent of each of the 55 joints: the body parents, then the face and the hands."""
    parents = list(PARENTS) + [HEAD, HEAD, HEAD]  # jaw, left eye, right eye
    for wrist in (LEFT_WRIST, RIGHT_WRIST):
        for _ in range(FINGERS_PER_HAND):
            previous = wrist
            for _ in range(JOINTS_PER_FINGER):
                parents.append(previous)
                previous = len(parents) - 1
    return parents


def _owner(parents: list[int], joint: int) -> int:
    """Return the body joint that owns a joint: the joint itself, or its nearest body ancestor."""
    while joint >= NUM_JOINTS:
        joint = parents[joint]
    return joint


def _skinning_weights(joint_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one vertex per joint, in reverse order, and the joint that each vertex weighs most.

    Vertex v weighs its dominant joint 0.6 and the next joint 0.4, so its largest weight is on the
    dominant joint. For vertex 0 the first non-zero weight is on joint 0 instead, so a rule that
    takes the first non-zero weight gives a different answer there.
    """
    dominant = np.arange(joint_count)[::-1]
    weights = np.zeros((joint_count, joint_count))
    vertices = np.arange(joint_count)
    weights[vertices, dominant] = 0.6
    weights[vertices, (dominant + 1) % joint_count] = 0.4
    return weights, dominant


class _FakeModel:
    """A fake of a model that the smplx package builds: fixed tables and a recording forward pass.

    Joint j sits at x = j and vertex v at x = v, so the outputs show the order in which they come.
    """

    def __init__(self, parents: list[int], weights: np.ndarray, faces: np.ndarray) -> None:
        self.parents = torch.as_tensor(parents, dtype=torch.long)
        self.lbs_weights = torch.as_tensor(weights, dtype=torch.float64)
        self.faces = faces
        self.joint_count = len(parents)
        self.vertex_count = weights.shape[0]
        self.received: list[dict[str, torch.Tensor]] = []

    def __call__(self, **arguments: torch.Tensor) -> SimpleNamespace:
        self.received.append(arguments)
        count = arguments["betas"].shape[0]
        vertices = torch.zeros(count, self.vertex_count, 3, dtype=torch.float64)
        vertices[:, :, 0] = torch.arange(self.vertex_count, dtype=torch.float64)
        joints = torch.zeros(count, self.joint_count, 3, dtype=torch.float64)
        joints[:, :, 0] = torch.arange(self.joint_count, dtype=torch.float64)
        return SimpleNamespace(vertices=vertices, joints=joints)


class _FakeSmplx:
    """Stands in for the smplx package. ``create`` records its arguments and returns the model."""

    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.created: list[tuple[str, str, dict[str, object]]] = []

    def create(self, model_path: str, model_type: str = "smpl", **kwargs: object) -> _FakeModel:
        self.created.append((str(model_path), model_type, kwargs))
        return self.model


def _install_fake_smplx(monkeypatch: pytest.MonkeyPatch, model: _FakeModel) -> _FakeSmplx:
    """Make ``import smplx`` return a fake package whose ``create`` returns the model."""
    recorder = _FakeSmplx(model)
    fake_package = ModuleType("smplx")
    fake_package.create = recorder.create
    monkeypatch.setitem(sys.modules, "smplx", fake_package)
    return recorder


def _placeholder_root(tmp_path: Path) -> Path:
    """Return an asset root that holds a placeholder in place of the licensed SMPL-X file."""
    (tmp_path / "smplx").mkdir()
    (tmp_path / "smplx" / "SMPLX_NEUTRAL.npz").write_bytes(b"placeholder, not a model")
    return tmp_path


def _fake_model() -> _FakeModel:
    """Return the fake SMPL-X model: the real parent table, one vertex per joint, two triangles."""
    weights, _ = _skinning_weights(SMPLX_JOINT_COUNT)
    return _FakeModel(_smplx_parents(), weights, FACES)


def _licensed_root_or_skip() -> Path:
    """Return the asset root if the licensed SMPL-X file is under it; otherwise skip the test."""
    root = asset_root()
    try:
        require_asset(SMPLX_ASSET_PATH, SMPLX_ASSET_KEY, root)
    except MissingAssetError as error:
        pytest.skip(f"the licensed SMPL-X file is absent: {error}")
    return root


def test_no_asset_root_names_the_model_file_and_its_key():
    with pytest.raises(MissingAssetError, match="no asset root is set") as error:
        SmplxBody(None)
    assert error.value.key == SMPLX_ASSET_KEY
    assert error.value.asset.endswith("smplx/SMPLX_NEUTRAL.npz")


def test_an_empty_asset_root_names_the_model_file_and_its_key(tmp_path):
    with pytest.raises(MissingAssetError) as error:
        SmplxBody(tmp_path)
    assert error.value.key == SMPLX_ASSET_KEY
    assert Path(error.value.asset) == tmp_path / "smplx" / "SMPLX_NEUTRAL.npz"


def test_the_licensed_file_gives_a_model_whose_joints_are_the_body_joints():
    body = SmplxBody(_licensed_root_or_skip())
    assert body.joint_names == JOINT_NAMES
    vertices, joints = canonical_mesh(body, np.zeros(10))
    assert joints.shape == (NUM_JOINTS, 3)
    assert vertices.shape == (body.vertex_count, 3)
    assert body.part_ids.min() >= 0
    assert body.part_ids.max() < NUM_JOINTS


def test_the_model_file_is_read_with_the_smplx_arguments(monkeypatch, tmp_path):
    root = _placeholder_root(tmp_path)
    fake = _install_fake_smplx(monkeypatch, _fake_model())
    SmplxBody(root)
    [(path, model_type, kwargs)] = fake.created
    assert Path(path) == root / "smplx" / "SMPLX_NEUTRAL.npz"
    assert model_type == "smplx"
    assert kwargs == {
        "gender": "neutral",
        "ext": "npz",
        "num_betas": 10,
        "num_expression_coeffs": 10,
        "use_pca": False,
        "flat_hand_mean": True,
        "dtype": torch.float64,
    }


def test_the_first_22_joints_of_the_model_are_the_body_joints_in_order(monkeypatch, tmp_path):
    _install_fake_smplx(monkeypatch, _fake_model())
    body = SmplxBody(_placeholder_root(tmp_path))
    assert body.joint_names == JOINT_NAMES
    joints = body.joints(np.zeros(10), np.zeros(3), np.zeros(BODY_POSE_SIZE))
    # The fake puts joint j at x = j, so row j of the result is the model's joint j. The jaw,
    # the eyes, and the hands are not part of the result.
    assert joints.shape == (NUM_JOINTS, 3)
    assert np.array_equal(joints[:, 0], np.arange(NUM_JOINTS))
    assert joints[JOINT_INDEX["left_wrist"], 0] == LEFT_WRIST
    assert joints[JOINT_INDEX["right_wrist"], 0] == RIGHT_WRIST


def test_each_joint_rotation_goes_to_the_slot_of_its_joint(monkeypatch, tmp_path):
    fake = _install_fake_smplx(monkeypatch, _fake_model())
    body = SmplxBody(_placeholder_root(tmp_path))
    pose_root = np.array([0.4, 0.5, 0.6])
    pose_body = np.zeros(BODY_POSE_SIZE)
    knee = JOINT_INDEX["left_knee"]  # joint 4; its three values sit at slots 9 to 11
    pose_body[3 * (knee - 1) : 3 * knee] = [0.1, 0.2, 0.3]
    body.vertices(np.zeros(10), pose_root, pose_body)
    [call] = fake.model.received
    assert set(call) == {
        "betas",
        "global_orient",
        "body_pose",
        "transl",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "expression",
    }
    assert np.array_equal(call["global_orient"].numpy(), [[0.4, 0.5, 0.6]])
    assert np.array_equal(call["body_pose"].numpy(), [pose_body])
    assert call["body_pose"].numpy()[0, 9:12].tolist() == [0.1, 0.2, 0.3]


def test_a_parent_table_that_differs_from_the_body_joints_is_refused(monkeypatch, tmp_path):
    parents = _smplx_parents()
    parents[JOINT_INDEX["left_knee"]] = JOINT_INDEX["spine1"]  # the knee must hang from the hip
    weights, _ = _skinning_weights(SMPLX_JOINT_COUNT)
    _install_fake_smplx(monkeypatch, _FakeModel(parents, weights, FACES))
    with pytest.raises(ValueError, match="base.PARENTS"):
        SmplxBody(_placeholder_root(tmp_path))


def test_a_joint_whose_parent_has_a_higher_index_is_refused(monkeypatch, tmp_path):
    parents = _smplx_parents()
    parents[30] = 40  # a left-hand joint that hangs from a joint with a higher index
    weights, _ = _skinning_weights(SMPLX_JOINT_COUNT)
    _install_fake_smplx(monkeypatch, _FakeModel(parents, weights, FACES))
    with pytest.raises(ValueError, match="a parent must have a lower index"):
        SmplxBody(_placeholder_root(tmp_path))


def test_part_ids_are_the_owner_of_the_largest_skinning_weight(monkeypatch, tmp_path):
    parents = _smplx_parents()
    weights, dominant = _skinning_weights(SMPLX_JOINT_COUNT)
    _install_fake_smplx(monkeypatch, _FakeModel(parents, weights, FACES))
    body = SmplxBody(_placeholder_root(tmp_path))
    # Vertex v weighs joint 54 - v most, so joint j is the dominant joint of vertex 54 - j.
    # The jaw and the eyes pass to the head, and each hand passes to its wrist.
    expected = {
        0: 0,
        21: 21,
        22: HEAD,
        23: HEAD,
        24: HEAD,
        25: LEFT_WRIST,
        39: LEFT_WRIST,
        40: RIGHT_WRIST,
        54: RIGHT_WRIST,
    }
    for joint, owner in expected.items():
        assert body.part_ids[SMPLX_JOINT_COUNT - 1 - joint] == owner
    assert np.array_equal(body.part_ids, [_owner(parents, joint) for joint in dominant])
