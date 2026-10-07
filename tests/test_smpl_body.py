"""Tests for body/smpl_body.py: gender selection, joint names, and part ids (research R5).

The licensed SMPL files are not in the repository, so the one test that needs them skips. The other
tests put a fake smplx module in place of the package, and placeholder files in place of the models.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from strike_a_pose.assets import ROOT_PLACEHOLDER, MissingAssetError, asset_root, require_asset
from strike_a_pose.body.base import (
    BODY_POSE_SIZE,
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENTS,
    canonical_mesh,
)
from strike_a_pose.body.smpl_body import SMPL_ASSET_KEY, SMPL_GENDERS, SmplBody

# SMPL has 24 joints: the 22 body joints, then the left and right hand, which hang from the wrists.
SMPL_JOINT_COUNT = 24
LEFT_WRIST = JOINT_INDEX["left_wrist"]
RIGHT_WRIST = JOINT_INDEX["right_wrist"]

# The two hand joints have three rotation values each, which stay at zero after the 63 body values.
HAND_ROTATION_SIZE = 3 * 2

# Two triangles over vertices 0 to 4. The fake mesh needs no more than that.
FACES = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)


def _smpl_parents() -> list[int]:
    """Return the parent of each of the 24 joints: the 22 body parents, then the two hands."""
    return list(PARENTS) + [LEFT_WRIST, RIGHT_WRIST]


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
    """Return an asset root with a placeholder in place of each of the three SMPL files."""
    (tmp_path / "smpl").mkdir()
    for gender in SMPL_GENDERS:
        (tmp_path / "smpl" / f"SMPL_{gender.upper()}.pkl").write_bytes(b"placeholder, not a model")
    return tmp_path


def _fake_model() -> _FakeModel:
    """Return the fake SMPL model: the real parent table, one vertex per joint, two triangles."""
    weights, _ = _skinning_weights(SMPL_JOINT_COUNT)
    return _FakeModel(_smpl_parents(), weights, FACES)


def _licensed_root_or_skip(gender: str) -> Path:
    """Return the asset root if the licensed SMPL file of the gender is under it; otherwise skip."""
    root = asset_root()
    try:
        require_asset(f"{ROOT_PLACEHOLDER}/smpl/SMPL_{gender.upper()}.pkl", SMPL_ASSET_KEY, root)
    except MissingAssetError as error:
        pytest.skip(f"the licensed SMPL file is absent: {error}")
    return root


@pytest.mark.parametrize("gender", SMPL_GENDERS)
def test_the_licensed_file_of_each_gender_gives_a_model_with_the_body_joints(gender):
    model = SmplBody(_licensed_root_or_skip(gender), gender)
    assert model.gender == gender
    assert model.joint_names == JOINT_NAMES
    vertices, joints = canonical_mesh(model, np.zeros(10))
    assert joints.shape == (NUM_JOINTS, 3)
    assert vertices.shape == (model.vertex_count, 3)
    assert model.part_ids.min() >= 0
    assert model.part_ids.max() < NUM_JOINTS


@pytest.mark.parametrize("gender", SMPL_GENDERS)
def test_each_gender_reads_its_own_model_file(monkeypatch, tmp_path, gender):
    root = _placeholder_root(tmp_path)
    fake = _install_fake_smplx(monkeypatch, _fake_model())
    model = SmplBody(root, gender)
    [(path, model_type, kwargs)] = fake.created
    assert Path(path) == root / "smpl" / f"SMPL_{gender.upper()}.pkl"
    assert model_type == "smpl"
    assert kwargs == {"gender": gender, "num_betas": 10, "dtype": torch.float64}
    assert model.gender == gender


def test_the_default_gender_is_neutral(monkeypatch, tmp_path):
    fake = _install_fake_smplx(monkeypatch, _fake_model())
    model = SmplBody(_placeholder_root(tmp_path))
    [(path, _, kwargs)] = fake.created
    assert Path(path).name == "SMPL_NEUTRAL.pkl"
    assert kwargs["gender"] == "neutral"
    assert model.gender == "neutral"


def test_a_missing_file_for_the_gender_is_named_with_its_key(monkeypatch, tmp_path):
    (tmp_path / "smpl").mkdir()
    (tmp_path / "smpl" / "SMPL_MALE.pkl").write_bytes(b"placeholder, not a model")
    fake = _install_fake_smplx(monkeypatch, _fake_model())
    with pytest.raises(MissingAssetError) as error:
        SmplBody(tmp_path, "female")
    assert Path(error.value.asset).name == "SMPL_FEMALE.pkl"
    assert error.value.key == SMPL_ASSET_KEY
    assert fake.created == []


def test_an_unknown_gender_is_refused_before_any_file_is_read(monkeypatch, tmp_path):
    fake = _install_fake_smplx(monkeypatch, _fake_model())
    with pytest.raises(ValueError, match="gender must be one of male, female, neutral"):
        SmplBody(_placeholder_root(tmp_path), "other")
    assert fake.created == []


def test_the_22_body_joints_come_first_and_the_hand_joints_are_dropped(monkeypatch, tmp_path):
    fake = _install_fake_smplx(monkeypatch, _fake_model())
    body = SmplBody(_placeholder_root(tmp_path), "neutral")
    assert body.joint_names == JOINT_NAMES
    pose_body = np.arange(BODY_POSE_SIZE, dtype=np.float64) / 100
    joints = body.joints(np.zeros(10), np.zeros(3), pose_body)
    # The fake puts joint j at x = j. The two hand joints (22 and 23) are not part of the result.
    assert joints.shape == (NUM_JOINTS, 3)
    assert np.array_equal(joints[:, 0], np.arange(NUM_JOINTS))
    # The model receives the 63 body values, then zeros for the two hand joints.
    [call] = fake.model.received
    assert call["body_pose"].shape == (1, BODY_POSE_SIZE + HAND_ROTATION_SIZE)
    assert np.array_equal(call["body_pose"].numpy()[0, :BODY_POSE_SIZE], pose_body)
    assert not call["body_pose"].numpy()[0, BODY_POSE_SIZE:].any()


def test_part_ids_are_the_owner_of_the_largest_skinning_weight(monkeypatch, tmp_path):
    parents = _smpl_parents()
    weights, dominant = _skinning_weights(SMPL_JOINT_COUNT)
    _install_fake_smplx(monkeypatch, _FakeModel(parents, weights, FACES))
    body = SmplBody(_placeholder_root(tmp_path), "neutral")
    # Vertex v weighs joint 23 - v most, so joint j is the dominant joint of vertex 23 - j.
    # The left hand (22) passes to the left wrist and the right hand (23) to the right wrist.
    expected = {0: 0, 21: 21, 22: LEFT_WRIST, 23: RIGHT_WRIST}
    for joint, owner in expected.items():
        assert body.part_ids[SMPL_JOINT_COUNT - 1 - joint] == owner
    assert np.array_equal(body.part_ids, [_owner(parents, joint) for joint in dominant])
