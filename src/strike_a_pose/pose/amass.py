"""AMASS pose source: seeded draws of body poses from the motion-capture files under the asset root.

This is the ``amass`` pose source of pose/source.py (research R2). Every ``.npz`` file under the
folders it is given is an AMASS motion, and each frame of each file is one candidate pose. Two
layouts are read. The SMPL-X layout holds ``pose_body`` (N by 63) and ``root_orient`` (N by 3). The
SMPL-H layout holds ``poses`` (N by 156): values 0 to 2 are the root orientation, values 3 to 65 are
the 21 body joints, and the rest are hand joints. A file that holds both layouts is read as SMPL-X.
The key names are assumed (research R2), so the loader logs the keys it finds.

The root orientation is discarded, so ``pose_root`` is zero (research R2). AMASS stores it in its
mocap world frame, and the rig's uniform azimuth supplies the viewing direction. Hand, jaw, and eye
values are never read, so the body has no hand pose. The joints of ``pose_body`` are in the joint
order of body/base.py, which is the AMASS order (research R2).

Public source: AMASS, Mahmood, Ghorbani, Troje, Pons-Moll, and Black, "AMASS: Archive of Motion
Capture as Surface Shapes", ICCV 2019 (https://amass.is.tue.mpg.de/).
"""

import logging
import zipfile
from collections import Counter
from itertools import accumulate
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from strike_a_pose.body.base import BODY_POSE_SIZE, ROOT_POSE_SIZE, BodyPose

__all__ = ["AmassDataError", "AmassPoseSource"]

logger = logging.getLogger(__name__)

# The keys that identify each layout (research R2). The SMPL-X layout needs both of its keys.
_SMPLX_KEYS = frozenset({"pose_body", "root_orient"})
_SMPLH_KEY = "poses"

# An SMPL-H pose holds 52 joints of 3 values: the root, the 21 body joints, and 30 hand joints. The
# body joints are values 3 to 65 of each row, which is the pose_body of body/base.py.
_SMPLH_POSE_SIZE = 3 * 52
_SMPLH_BODY_START = ROOT_POSE_SIZE
_SMPLH_BODY_STOP = ROOT_POSE_SIZE + BODY_POSE_SIZE


class AmassDataError(ValueError):
    """An AMASS folder or file cannot supply body poses. The message names the file or folder."""


class AmassPoseSource:
    """The ``amass`` pose source: a seeded draw of one AMASS frame, with a zero root orientation.

    ``name`` is the value of pose.source that selects this source. The constructor reads every
    motion file once and keeps the frames in memory as float64, so construction is the slow step.
    A draw picks one frame uniformly over all frames of all files, so a longer motion is drawn more
    often. Two draws from generators in the same state give the same pose.
    """

    name = "amass"

    def __init__(self, folders: list[Path]) -> None:
        names = ", ".join(str(folder) for folder in folders)
        files = _motion_files(folders)
        if not files:
            raise AmassDataError(f"no AMASS .npz file under {names}")
        frame_arrays: list[NDArray[np.float64]] = []
        layouts: Counter[tuple[str, tuple[str, ...]]] = Counter()
        for path in files:
            layout, keys, frames = _read_file(path)
            layouts[(layout, keys)] += 1
            if len(frames) == 0:
                logger.debug("AMASS file %s holds no frames and is skipped", path)
            else:
                frame_arrays.append(frames)
        for (layout, keys), count in sorted(layouts.items()):
            logger.info("AMASS layout %s, keys %s, in %d file(s)", layout, ", ".join(keys), count)
        if not frame_arrays:
            raise AmassDataError(f"no pose frame in the AMASS files under {names}")
        # bounds[i] is the index of the first frame of file i; the last bound is the frame count.
        # The frames stay in one array per file, so no concatenated copy is made.
        bounds = list(accumulate((len(frames) for frames in frame_arrays), initial=0))
        self._frames = frame_arrays
        self._starts = np.array(bounds[:-1], dtype=np.int64)
        self._count = bounds[-1]
        logger.info("AMASS: %d frame(s) in %d file(s)", self._count, len(frame_arrays))

    def draw(self, rng: np.random.Generator) -> BodyPose:
        """Return ``(pose_root, pose_body)``: a zero root orientation and one drawn frame of joints.

        One integer draw from ``rng`` picks the frame, uniformly over all frames. ``pose_body`` is a
        copy, so the caller may change it without changing later draws.
        """
        index = int(rng.integers(self._count))
        file_index = int(np.searchsorted(self._starts, index, side="right")) - 1
        frame = self._frames[file_index][index - int(self._starts[file_index])]
        return BodyPose(np.zeros(ROOT_POSE_SIZE, dtype=np.float64), frame.copy())


def _motion_files(folders: list[Path]) -> list[Path]:
    """Return every ``.npz`` file under the folders, each file once, in sorted path order."""
    found = {path for folder in folders for path in folder.rglob("*.npz") if path.is_file()}
    return sorted(found)


def _read_file(path: Path) -> tuple[str, tuple[str, ...], NDArray[np.float64]]:
    """Return the layout, the sorted keys, and the body frames (N by 63, float64) of one file.

    The SMPL-X layout is used when the file has both of its keys, otherwise the SMPL-H layout. The
    root_orient array is never loaded, so the root orientation is discarded.
    """
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise AmassDataError(f"cannot read the AMASS file {path}: {error}") from error
    with archive:
        keys = tuple(sorted(archive.files))
        if _SMPLX_KEYS <= frozenset(keys):
            values = _as_matrix(path, "pose_body", archive["pose_body"], BODY_POSE_SIZE)
            return "SMPL-X", keys, _check_finite(path, "the 'pose_body' values", values)
        if _SMPLH_KEY in keys:
            poses = _as_matrix(path, _SMPLH_KEY, archive[_SMPLH_KEY], _SMPLH_POSE_SIZE)
            body = poses[:, _SMPLH_BODY_START:_SMPLH_BODY_STOP].copy()
            return "SMPL-H", keys, _check_finite(path, "the body joints of 'poses'", body)
    raise AmassDataError(
        f"{path} has neither layout the amass source reads (pose_body with root_orient, or "
        f"poses); its keys are {', '.join(keys)}"
    )


def _as_matrix(path: Path, key: str, values: NDArray, columns: int) -> NDArray[np.float64]:
    """Check that a stored array is real with ``columns`` columns; return it as float64."""
    if values.ndim != 2 or values.shape[1] != columns:
        raise AmassDataError(
            f"{path}: '{key}' must have shape (frames, {columns}), but it has shape {values.shape}"
        )
    if values.dtype.kind not in "fiu":
        raise AmassDataError(f"{path}: '{key}' must hold real numbers, not {values.dtype}")
    return np.asarray(values, dtype=np.float64)


def _check_finite(path: Path, what: str, frames: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the frames, or raise AmassDataError when one of their values is not finite."""
    if not np.all(np.isfinite(frames)):
        raise AmassDataError(f"{path}: {what} are not finite")
    return frames
