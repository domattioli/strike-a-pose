"""Smoke tests for pose/source.py: registry names, the built sources, and seeded draws.

The amass source reads small synthetic .npz files written inside each test, under a temporary asset
root. No licensed asset is read, and the limits source needs no asset at all.
"""

from pathlib import Path

import numpy as np
import pytest

from strike_a_pose.assets import MissingAssetError
from strike_a_pose.body.base import BODY_POSE_SIZE, ROOT_POSE_SIZE
from strike_a_pose.config import ConfigError, load_config, resolve_config
from strike_a_pose.pose.amass import AmassPoseSource
from strike_a_pose.pose.limits import LIMIT_GROUPS, LimitPoseSource, joint_limits_rad
from strike_a_pose.pose.source import POSE_SOURCE_NAMES, PoseSource, create_pose_source
from strike_a_pose.seeding import rng_for

# The configuration seed is required by resolve_config. The tests of this module never use it.
RUN_SEED = 1
# pose_body holds the 21 body joints of body/base.py, three values each.
NUM_BODY_JOINTS = BODY_POSE_SIZE // 3
DRAW_COUNT = 200
# Joint angles are compared with this slack in radians, far above rounding error near one radian.
ANGLE_SLACK_RADIANS = 1e-12

# Frames of two synthetic AMASS files. Each file stores a non-zero root_orient, so a draw that kept
# the root orientation would fail the zero-root checks below.
ACCAD_FRAMES = np.random.default_rng(101).normal(size=(4, BODY_POSE_SIZE))
CMU_FRAMES = np.random.default_rng(102).normal(size=(4, BODY_POSE_SIZE))


def _write_motion(path: Path, frames: np.ndarray) -> None:
    """Write an SMPL-X style AMASS file: pose_body and a non-zero root_orient, one row per frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pose_body=frames, root_orient=np.full((len(frames), 3), 0.5))


@pytest.fixture
def asset_root(tmp_path: Path) -> Path:
    """A synthetic asset root with two AMASS subset folders, ACCAD and CMU. No licensed file."""
    root = tmp_path / "assets"
    _write_motion(root / "amass" / "ACCAD" / "walk.npz", ACCAD_FRAMES)
    _write_motion(root / "amass" / "CMU" / "run.npz", CMU_FRAMES)
    return root


def _resolved(
    source: str,
    asset_root: Path | None = None,
    subsets: list[str] | None = None,
    limits: dict[str, float] | None = None,
) -> dict:
    """Return the resolved configuration of one pose source, with its asset root and folders."""
    pose: dict[str, object] = {"source": source}
    if subsets is not None:
        pose["amass_subsets"] = subsets
    if limits is not None:
        pose["limits_deg"] = limits
    raw: dict[str, object] = {"seed": RUN_SEED, "pose": pose}
    if asset_root is not None:
        raw["assets"] = {"root": str(asset_root)}
    return resolve_config(raw)


def _config_for(name: str, asset_root: Path) -> dict:
    """Return a resolved configuration that selects the registered source ``name``."""
    if name == "limits":
        return _resolved("limits")
    return _resolved("amass", asset_root, subsets=["ACCAD", "CMU"])


def _draws(source: PoseSource, seed: int, count: int) -> np.ndarray:
    """Return the pose_body of ``count`` draws from the generator of ``seed``, shape (count, 63)."""
    rng = rng_for(seed, 0)
    return np.stack([source.draw(rng).pose_body for _ in range(count)])


def _within(drawn: np.ndarray, frames: np.ndarray) -> bool:
    """Return True when every drawn pose equals one of the frames exactly."""
    return all(bool(np.any(np.all(frames == vector, axis=1))) for vector in drawn)


def _covers(drawn: np.ndarray, frames: np.ndarray) -> bool:
    """Return True when every frame appears among the drawn poses."""
    return all(bool(np.any(np.all(drawn == row, axis=1))) for row in frames)


def test_the_registry_holds_the_amass_and_limits_sources():
    assert set(POSE_SOURCE_NAMES) == {"amass", "limits"}


def test_every_registered_name_is_a_value_the_configuration_accepts():
    for name in POSE_SOURCE_NAMES:
        assert _resolved(name)["pose"]["source"] == name


def test_the_limits_name_builds_the_limits_source_without_an_asset_root():
    source = create_pose_source(_resolved("limits"))
    assert isinstance(source, LimitPoseSource)
    assert isinstance(source, PoseSource)
    assert source.name == "limits"


def test_the_tiny_configuration_builds_a_limits_source_within_its_table(tiny_config_path: Path):
    config = load_config(tiny_config_path)
    assert config["pose"]["source"] == "limits"
    source = create_pose_source(config)
    assert isinstance(source, LimitPoseSource)
    limits = joint_limits_rad(config["pose"]["limits_deg"])
    poses = _draws(source, config["seed"], DRAW_COUNT)
    angles = np.linalg.norm(poses.reshape(DRAW_COUNT, NUM_BODY_JOINTS, 3), axis=2)
    assert np.all(angles <= limits + ANGLE_SLACK_RADIANS)


def test_the_configured_limit_table_is_the_one_the_limits_source_draws_within():
    table = dict.fromkeys(LIMIT_GROUPS, 5.0)
    source = create_pose_source(_resolved("limits", limits=table))
    limits = joint_limits_rad(table)
    poses = _draws(source, seed=23, count=DRAW_COUNT)
    angles = np.linalg.norm(poses.reshape(DRAW_COUNT, NUM_BODY_JOINTS, 3), axis=2)
    assert np.all(angles <= limits + ANGLE_SLACK_RADIANS)
    assert np.all(angles.max(axis=0) >= 0.9 * limits)


def test_the_amass_name_builds_the_amass_source_from_the_listed_subset_only(asset_root: Path):
    source = create_pose_source(_resolved("amass", asset_root, subsets=["ACCAD"]))
    assert isinstance(source, AmassPoseSource)
    assert isinstance(source, PoseSource)
    assert source.name == "amass"
    drawn = _draws(source, seed=24, count=DRAW_COUNT)
    assert _within(drawn, ACCAD_FRAMES)
    assert _covers(drawn, ACCAD_FRAMES)


def test_without_subsets_every_motion_file_under_the_amass_folder_is_drawn(asset_root: Path):
    source = create_pose_source(_resolved("amass", asset_root, subsets=[]))
    drawn = _draws(source, seed=25, count=400)
    assert _covers(drawn, ACCAD_FRAMES)
    assert _covers(drawn, CMU_FRAMES)


def test_a_subset_folder_that_is_absent_is_a_missing_asset_naming_the_key(asset_root: Path):
    with pytest.raises(MissingAssetError) as caught:
        create_pose_source(_resolved("amass", asset_root, subsets=["DANCE"]))
    assert caught.value.key == "pose.source"
    assert "DANCE" in str(caught.value)


def test_the_amass_name_without_an_asset_root_is_a_missing_asset():
    with pytest.raises(MissingAssetError) as caught:
        create_pose_source(_resolved("amass", subsets=["ACCAD"]))
    assert caught.value.key == "pose.source"
    assert "no asset root is set" in str(caught.value)


def test_an_unregistered_name_is_refused_with_the_pose_source_key():
    config = _resolved("limits")
    config["pose"]["source"] = "vposer"
    with pytest.raises(ConfigError) as caught:
        create_pose_source(config)
    assert caught.value.key == "pose.source"
    assert "vposer" in str(caught.value)
    for name in POSE_SOURCE_NAMES:
        assert name in str(caught.value)


@pytest.mark.parametrize("name", POSE_SOURCE_NAMES)
def test_every_source_returns_a_zero_root_and_63_body_values(asset_root: Path, name: str):
    source = create_pose_source(_config_for(name, asset_root))
    rng = rng_for(26, 0)
    for _ in range(50):
        pose = source.draw(rng)
        assert pose.pose_root.dtype == np.float64
        assert pose.pose_root.shape == (ROOT_POSE_SIZE,)
        assert np.array_equal(pose.pose_root, np.zeros(ROOT_POSE_SIZE))
        assert pose.pose_body.dtype == np.float64
        assert pose.pose_body.shape == (BODY_POSE_SIZE,)


@pytest.mark.parametrize("name", POSE_SOURCE_NAMES)
def test_each_source_repeats_its_draws_for_one_seed_and_changes_for_another(
    asset_root: Path, name: str
):
    config = _config_for(name, asset_root)
    first = _draws(create_pose_source(config), seed=27, count=50)
    assert np.array_equal(first, _draws(create_pose_source(config), seed=27, count=50))
    assert not np.array_equal(first, _draws(create_pose_source(config), seed=28, count=50))
