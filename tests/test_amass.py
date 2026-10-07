"""Smoke tests for pose/amass.py: both AMASS layouts, the root discard, seeded draws, and bad files.

Each motion file is a small synthetic .npz written inside the test. No licensed AMASS file is read.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from strike_a_pose.body.base import BODY_POSE_SIZE, ROOT_POSE_SIZE
from strike_a_pose.pose.amass import AmassDataError, AmassPoseSource
from strike_a_pose.seeding import rng_for

# An SMPL-H row holds 52 joints of 3 values. The fixtures give the root and the hands values that
# are not body values, so a draw that reads the wrong columns cannot pass the row checks.
SMPLH_POSE_SIZE = 3 * 52
ROOT_MARKER = 0.25
HAND_MARKER = 9.0


def _frames(count: int, seed: int) -> np.ndarray:
    """Return ``count`` distinct rows of 63 body values."""
    return np.random.default_rng(seed).normal(size=(count, BODY_POSE_SIZE))


def _write_smplx(path: Path, pose_body: np.ndarray, root_orient: np.ndarray) -> None:
    """Write an SMPL-X style AMASS file with the keys that AMASS files carry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pose_body=pose_body,
        root_orient=root_orient,
        trans=np.zeros((len(pose_body), 3)),
        betas=np.zeros(16),
        gender=np.array("neutral"),
    )


def _write_smpl_h(path: Path, poses: np.ndarray) -> None:
    """Write an SMPL-H style AMASS file: one poses array of 156 values per frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        poses=poses,
        trans=np.zeros((len(poses), 3)),
        betas=np.zeros(16),
        gender=np.array("neutral"),
        mocap_framerate=np.array(120.0),
    )


def _write_motion(path: Path, layout: str, frames: np.ndarray) -> None:
    """Write body frames in one layout, "smplx" or "smpl_h", with a root and hand marker value."""
    count = len(frames)
    if layout == "smplx":
        _write_smplx(path, frames, np.full((count, 3), ROOT_MARKER))
        return
    poses = np.empty((count, SMPLH_POSE_SIZE))
    poses[:, :3] = ROOT_MARKER
    poses[:, 3:66] = frames
    poses[:, 66:] = HAND_MARKER
    _write_smpl_h(path, poses)


def _draws(source: AmassPoseSource, count: int, seed: int) -> np.ndarray:
    """Return the pose_body of ``count`` draws from the generator of ``seed``, shape (count, 63)."""
    rng = rng_for(seed, 0)
    return np.stack([source.draw(rng).pose_body for _ in range(count)])


def _is_one_of(vector: np.ndarray, rows: np.ndarray) -> bool:
    """Return True when the vector equals one row of ``rows`` exactly."""
    return bool(np.any(np.all(rows == vector, axis=1)))


def _all_within(drawn: np.ndarray, rows: np.ndarray) -> bool:
    """Return True when every drawn pose is one of the rows."""
    return all(_is_one_of(vector, rows) for vector in drawn)


def _all_cover(drawn: np.ndarray, rows: np.ndarray) -> bool:
    """Return True when every row appears among the drawn poses."""
    return all(_is_one_of(row, drawn) for row in rows)


@pytest.mark.parametrize("layout", ["smplx", "smpl_h"])
def test_each_layout_returns_body_joints_and_a_zero_root(tmp_path: Path, layout: str) -> None:
    frames = _frames(6, seed=1)
    folder = tmp_path / "amass" / "ACCAD"
    _write_motion(folder / "walk.npz", layout, frames)
    source = AmassPoseSource([folder])
    rng = rng_for(11, 0)
    for _ in range(40):
        pose = source.draw(rng)
        assert pose.pose_root.dtype == np.float64
        assert np.array_equal(pose.pose_root, np.zeros(ROOT_POSE_SIZE))
        assert pose.pose_body.shape == (BODY_POSE_SIZE,)
        assert _is_one_of(pose.pose_body, frames)


def test_only_npz_files_under_the_listed_folders_are_read(tmp_path: Path) -> None:
    amass = tmp_path / "amass"
    accad = _frames(4, seed=2)
    cmu = _frames(4, seed=3)
    deep = _frames(4, seed=4)
    unlisted = _frames(4, seed=5)
    _write_motion(amass / "ACCAD" / "session" / "first.npz", "smplx", accad)
    _write_motion(amass / "CMU" / "second.npz", "smpl_h", cmu)
    _write_motion(amass / "CMU" / "deeper" / "third" / "third.npz", "smplx", deep)
    _write_motion(amass / "Unlisted" / "other.npz", "smplx", unlisted)
    (amass / "CMU" / "notes.txt").write_text("not a motion file")

    listed = AmassPoseSource([amass / "ACCAD", amass / "CMU"])
    drawn = _draws(listed, 300, seed=6)
    assert _all_within(drawn, np.vstack([accad, cmu, deep]))
    for frames in (accad, cmu, deep):
        assert _all_cover(drawn, frames)

    whole = AmassPoseSource([amass])
    assert _all_cover(_draws(whole, 400, seed=6), unlisted)


def test_the_same_seed_draws_the_same_frames(tmp_path: Path) -> None:
    folder = tmp_path / "amass" / "CMU"
    _write_motion(folder / "walk.npz", "smplx", _frames(8, seed=7))
    first = _draws(AmassPoseSource([folder]), 25, seed=9)
    assert np.array_equal(first, _draws(AmassPoseSource([folder]), 25, seed=9))
    assert not np.array_equal(first, _draws(AmassPoseSource([folder]), 25, seed=10))


def test_each_draw_returns_its_own_copy(tmp_path: Path) -> None:
    folder = tmp_path / "amass" / "CMU"
    _write_motion(folder / "pose.npz", "smplx", _frames(1, seed=8))
    source = AmassPoseSource([folder])
    rng = rng_for(12, 0)
    first = source.draw(rng).pose_body
    expected = first.copy()
    first[:] = 0.0
    assert np.array_equal(source.draw(rng).pose_body, expected)


def test_the_keys_of_each_layout_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="strike_a_pose.pose.amass")
    folder = tmp_path / "amass" / "CMU"
    _write_motion(folder / "a.npz", "smplx", _frames(2, seed=13))
    _write_motion(folder / "b.npz", "smpl_h", _frames(2, seed=14))
    AmassPoseSource([folder])
    assert "root_orient" in caplog.text
    assert "pose_body" in caplog.text
    assert "poses" in caplog.text


def test_a_file_with_neither_layout_names_the_file_and_its_keys(tmp_path: Path) -> None:
    folder = tmp_path / "amass" / "CMU"
    folder.mkdir(parents=True)
    np.savez(folder / "shape.npz", betas=np.zeros(10), gender=np.array("neutral"))
    with pytest.raises(AmassDataError) as caught:
        AmassPoseSource([folder])
    assert "shape.npz" in str(caught.value)
    assert "betas" in str(caught.value)


def test_a_pose_array_of_the_wrong_width_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "amass" / "CMU"
    _write_smplx(folder / "short.npz", np.zeros((3, 60)), np.zeros((3, 3)))
    with pytest.raises(AmassDataError, match=r"'pose_body' must have shape \(frames, 63\)"):
        AmassPoseSource([folder])


def test_a_non_finite_pose_value_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "amass" / "CMU"
    frames = _frames(3, seed=15)
    frames[1, 4] = np.nan
    _write_motion(folder / "bad.npz", "smplx", frames)
    with pytest.raises(AmassDataError, match="not finite"):
        AmassPoseSource([folder])


def test_a_folder_without_npz_files_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "amass" / "ACCAD"
    folder.mkdir(parents=True)
    (folder / "readme.txt").write_text("no motion")
    with pytest.raises(AmassDataError, match="no AMASS .npz file under"):
        AmassPoseSource([folder])


def test_files_without_frames_are_skipped_and_an_all_empty_folder_is_refused(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "amass" / "CMU"
    _write_motion(folder / "a_empty.npz", "smplx", np.zeros((0, BODY_POSE_SIZE)))
    frames = _frames(3, seed=16)
    _write_motion(folder / "b.npz", "smpl_h", frames)
    drawn = _draws(AmassPoseSource([folder]), 60, seed=17)
    assert _all_within(drawn, frames)
    assert _all_cover(drawn, frames)

    empty = tmp_path / "amass" / "Empty"
    _write_motion(empty / "only.npz", "smpl_h", np.zeros((0, BODY_POSE_SIZE)))
    with pytest.raises(AmassDataError, match="no pose frame"):
        AmassPoseSource([empty])
