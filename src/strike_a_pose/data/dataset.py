"""Torch dataset over the data shards: view subsets, placement noise per cell, camera encodings.

A ``ShapeDataset`` serves the bodies of one split, or of any range of body ids, from the shard
files under ``<out>/data`` (contracts/artifacts.md). A sample holds the first k cameras of the
body's rig: the silhouettes, the camera encodings of the placements given to the model, and the
true shape coefficients and measurements. ``collate_samples`` stacks samples into the
``ViewBatch`` of ``model/vae.py`` and keeps the targets next to it. Public sources:

* Wu and Goodman, "Multimodal Generative Models for Scalable Weakly-Supervised Learning" (NeurIPS
  2018, https://arxiv.org/abs/1802.05335), for training one model on subsets of its inputs. The
  training mode applies it to the cameras of a rig: the number of views is drawn per sample
  (research R7).
* Rodrigues' rotation formula (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula) for
  the placement-noise rotation about an axis, and Zhou, Barnes, Lu, Yang, and Li, "On the
  Continuity of Rotation Representations in Neural Networks" (CVPR 2019,
  https://arxiv.org/abs/1812.07035) for the 6D rotation in the camera encoding (research R7).
  Both are implemented in ``camera.py``, which this module calls.
* The PyTorch documentation of datasets and of the ``collate_fn`` of a data loader
  (https://pytorch.org/docs/stable/data.html).

Views and placement noise (data-model.md, CameraPlacement and ExperimentCondition). A sample with
k views holds cameras 0 to k - 1 of the rig, the same views that the k-view cell of the evaluation
uses. The camera encoding of a view is ``camera.encode_camera(R_given, t_true)``, where
``R_given = camera.given_rotation(R_true, noise_axis, angle)`` turns the true rotation by ``angle``
degrees about the unit axis that the shard stores for this body and this view (FR-005). The
translation is the true one, and the silhouette is the one rendered from the true placement, so
placement noise changes the rotation part of the camera encoding and nothing else. A dataset
chooses the number of views and the angles in one of two ways.

* A cell (``ShapeDataset.for_cell``): a fixed number of views and one angle for every view, as in
  a cell of ``evaluate.views`` by ``evaluate.noise_deg``. Nothing is random. The same cell serves
  the same samples every time, and the angle 0 serves the true rotation exactly.
* Training (``ShapeDataset.for_training``): the number of views is drawn uniformly from
  ``train.views_train`` and the angle of each view uniformly from ``train.noise_train_deg``, a
  separate draw per view, about the stored axis. Every cell of the evaluation then lies inside the
  training distribution (FR-009).

Random draws (research R9). The draws of a training sample come from the stream
``rng_for(seed, STAGES.index("train"), epoch, body_id, 0)``. The generator is read in a fixed
order: one integer, the position of the view count in ``train.views_train``, then one uniform angle
for each of the four camera slots, of which the first k are used. The stream depends on the seed,
the epoch, and the body id alone. A sample is therefore the same whatever order the samples are
read in, whatever the batch size or the number of loader workers, whichever other bodies the
dataset holds, and whether or not the run was resumed in between. The trainer calls ``set_epoch``
at the start of each epoch, which gives the epoch new draws. A dataset that never sets its epoch
stays at epoch 0 and serves the same samples on every pass, which is what the loss-monitoring slice
of research R9 needs.

Flagged bodies. A body that carries any flag (empty_mask, out_of_frame, or slice_nan; data-model.md,
Body flags) is never served: it is excluded from training and counted as skipped in calibration
and evaluation. ``skipped_body_ids`` lists the flagged bodies of the requested range, so that a
stage can report how many it skipped.

Shards. A dataset reads, at construction, the shards that hold the requested bodies and no other.
The silhouettes stay bit-packed in memory, one bit per pixel as in the file, and a sample unpacks
its own views only. A missing shard, or a shard that does not hold the body ids that
``data.shard_size`` implies, is an error that names the file.

Batches. ``collate_samples`` pads each sample to the four view slots of a rig, where slot i holds
camera i, and builds the boolean view mask of the ``ViewBatch``. Absent slots hold zeros, which the
model never reads.
"""

import logging
import operator
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from strike_a_pose.camera import CAMERA_ENCODING_DIM, encode_camera, given_rotation
from strike_a_pose.config import ConfigError
from strike_a_pose.data.manifest import CAMERA_COUNT, ShardArrays, read_shard, shard_path
from strike_a_pose.data.splits import DataSplit
from strike_a_pose.model.vae import ViewBatch
from strike_a_pose.runrecord import STAGES
from strike_a_pose.seeding import rng_for

__all__ = [
    "ShapeBatch",
    "ShapeDataset",
    "ShapeSample",
    "ViewPlan",
    "collate_samples",
]

logger = logging.getLogger(__name__)

# The first element of the stream key of a training draw after the seed (research R9).
_STAGE_ID = STAGES.index("train")
# The last element of that key: which draw of a sample the stream makes. Part 0 is the number of
# views and the noise angles.
_VIEW_DRAW_STREAM = 0


class ViewPlan(NamedTuple):
    """The views of one sample: its body, how many cameras it uses, and the noise angle of each.

    ``noise_deg`` holds one angle in degrees for each of cameras 0 to ``n_views - 1``. The camera
    encoding of view v turns the true rotation by ``noise_deg[v]`` about the axis that the shard
    stores for this body and this view.
    """

    body_id: int
    n_views: int
    noise_deg: tuple[float, ...]


class ShapeSample(NamedTuple):
    """One body with its first k views, as tensors.

    ``silhouettes`` has shape (k, S, S) for ``camera.image_size`` S, float32, with values 0 and 1.
    ``cameras`` has shape (k, ``CAMERA_ENCODING_DIM``), float32: the encoding of ``R_given`` and
    ``t_true`` of each view. ``betas`` holds the true shape coefficients, float32, as the model
    reads them. ``measurements`` holds the five true measurements in centimeters as the shard
    stores them, float64. ``body_id`` is the id of the body in the manifest.
    """

    body_id: int
    silhouettes: Tensor
    cameras: Tensor
    betas: Tensor
    measurements: Tensor


class ShapeBatch(NamedTuple):
    """A batch of bodies: the model input and the targets that belong to it.

    ``views`` is the ``ViewBatch`` of ``model/vae.py`` with V = 4 view slots, where slot i holds
    camera i of the rig: silhouettes (B, V, S, S) float32, cameras (B, V, 9) float32, and the
    boolean view mask (B, V). ``betas`` has shape (B, n_betas), float32, and is the target of
    ``ShapeVAE.loss``. ``measurements`` has shape (B, 5), float64, in centimeters. ``body_id`` has
    shape (B,), int64.
    """

    views: ViewBatch
    betas: Tensor
    measurements: Tensor
    body_id: Tensor


@dataclass(frozen=True, eq=False)
class _Bodies:
    """The arrays a dataset keeps, one row per served body in ascending body id order."""

    body_id: NDArray[np.int64]  # (bodies,)
    masks: NDArray[np.uint8]  # (bodies, cameras, packed bytes of one mask), bit-packed
    R_true: NDArray[np.float64]  # (bodies, cameras, 3, 3)
    t_true: NDArray[np.float64]  # (bodies, cameras, 3)
    noise_axis: NDArray[np.float64]  # (bodies, cameras, 3)
    betas: NDArray[np.float32]  # (bodies, 10)
    measurements: NDArray[np.float64]  # (bodies, 5)
    skipped: NDArray[np.int64]  # flagged body ids of the request, left out of the rows above


class ShapeDataset(Dataset[ShapeSample]):
    """The unflagged bodies of a range of body ids, each with its first k views and noise angles.

    ``config`` is a resolved configuration (it supplies ``seed``, ``camera.image_size``, and the
    ``data`` section). ``data_dir`` is the data folder of a run, ``<out>/data``. ``bodies`` is a
    split name (``train``, ``cal``, or ``test``) or a contiguous ``range`` of body ids, such as
    ``DataSplit.sampler_range`` for the training sampler or ``DataSplit.monitor_range`` for the
    loss-monitoring slice. A split name selects the whole split, so a training stage passes the
    sampler range to keep the monitoring slice out of its gradient steps.

    ``view_counts`` lists the numbers of views a sample may have, each from 1 to 4. ``noise_deg``
    is the pair ``[low, high]`` of the placement-noise angle in degrees. A sample takes its number
    of views uniformly from ``view_counts`` and the angle of each view uniformly from the range.
    With one view count and ``low == high`` nothing is drawn: use ``for_cell`` for that case and
    ``for_training`` for the training distribution of the configuration.

    The dataset has one element per unflagged body, in ascending body id order. Element ``i`` is a
    ``ShapeSample``, and ``view_plan(i)`` says how many views it has and which angles it carries.
    Use ``collate_samples`` as the ``collate_fn`` of a ``DataLoader``.

    Raises ConfigError (exit code 2) for a data section that gives no valid split, ValueError for
    a bad selection, view count, or angle range or for a shard that does not hold the expected
    bodies, and FileNotFoundError for a missing shard.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        data_dir: str | os.PathLike[str],
        bodies: str | range,
        *,
        view_counts: Sequence[int],
        noise_deg: Sequence[float],
    ) -> None:
        super().__init__()
        self._view_counts = _checked_view_counts(view_counts)
        self._noise_range = _checked_noise_range(noise_deg)
        # With one view count and one angle there is nothing to draw, so no stream is read.
        self._random = len(self._view_counts) > 1 or self._noise_range[0] < self._noise_range[1]
        self._seed = int(config["seed"])
        self._epoch = 0
        self._side = int(config["camera"]["image_size"])

        split = _data_split(config)
        selected = _body_range(bodies, split)
        loaded = _read_bodies(
            Path(data_dir), selected, int(config["data"]["shard_size"]), split.n_bodies, self._side
        )
        self._body_ids = tuple(loaded.body_id.tolist())
        self._skipped = tuple(loaded.skipped.tolist())
        self._masks = loaded.masks
        self._R_true = loaded.R_true
        self._t_true = loaded.t_true
        self._noise_axis = loaded.noise_axis
        self._betas = loaded.betas
        self._measurements = loaded.measurements
        logger.info(
            "dataset over %d requested bodies: %d served, %d flagged and skipped",
            len(selected),
            len(self._body_ids),
            len(self._skipped),
        )

    @classmethod
    def for_training(
        cls,
        config: Mapping[str, Any],
        data_dir: str | os.PathLike[str],
        bodies: str | range,
    ) -> "ShapeDataset":
        """Build the dataset of the training distribution: ``train.views_train`` and noise range.

        The number of views is drawn per sample and the angle per view, as the module docstring
        describes (FR-009).
        """
        train = config["train"]
        return cls(
            config,
            data_dir,
            bodies,
            view_counts=train["views_train"],
            noise_deg=train["noise_train_deg"],
        )

    @classmethod
    def for_cell(
        cls,
        config: Mapping[str, Any],
        data_dir: str | os.PathLike[str],
        bodies: str | range,
        *,
        views: int,
        noise_deg: float,
    ) -> "ShapeDataset":
        """Build the dataset of one cell: cameras 0 to ``views - 1``, each noised by ``noise_deg``.

        ``noise_deg`` is the rotation angle in degrees, applied to every view about that view's
        stored axis. Nothing is drawn, so the samples are the same on every pass.
        """
        return cls(
            config,
            data_dir,
            bodies,
            view_counts=[views],
            noise_deg=[noise_deg, noise_deg],
        )

    @property
    def body_ids(self) -> tuple[int, ...]:
        """The ids of the served bodies, ascending: element ``i`` is the body ``body_ids[i]``."""
        return self._body_ids

    @property
    def skipped_body_ids(self) -> tuple[int, ...]:
        """The flagged bodies of the requested range that the dataset leaves out, ascending."""
        return self._skipped

    @property
    def epoch(self) -> int:
        """The epoch whose draws a training dataset serves; 0 until ``set_epoch`` is called."""
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        """Serve the draws of this epoch from now on (a cell dataset serves the same at any epoch).

        Call it in the process that reads the samples, at the start of each epoch, and again after
        a resume with the epoch that continues. With loader worker processes, keep
        ``persistent_workers`` off (the default), so that every epoch starts workers that see the
        new epoch.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer; got {epoch!r}")
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._body_ids)

    def view_plan(self, index: int) -> ViewPlan:
        """Return the number of views and the noise angle of each view of element ``index``.

        This is the draw of a training dataset (at the current epoch) or the fixed plan of a cell.
        ``__getitem__`` builds its sample from this plan.
        """
        position = self._position(index)
        body_id = self._body_ids[position]
        low, high = self._noise_range
        if not self._random:
            n_views = self._view_counts[0]
            return ViewPlan(body_id, n_views, (low,) * n_views)
        rng = rng_for(self._seed, _STAGE_ID, self._epoch, body_id, _VIEW_DRAW_STREAM)
        n_views = self._view_counts[int(rng.integers(len(self._view_counts)))]
        angles = rng.uniform(low, high, size=CAMERA_COUNT)
        return ViewPlan(body_id, n_views, tuple(float(angle) for angle in angles[:n_views]))

    def __getitem__(self, index: int) -> ShapeSample:
        position = self._position(index)
        plan = self.view_plan(position)
        n_views = plan.n_views
        pixels = self._side * self._side
        silhouettes = np.unpackbits(self._masks[position, :n_views], axis=-1, count=pixels)
        rotation = given_rotation(
            self._R_true[position, :n_views], self._noise_axis[position, :n_views], plan.noise_deg
        )
        cameras = encode_camera(rotation, self._t_true[position, :n_views])
        # The rows of the stored arrays are copied, so a caller that edits a sample in place
        # cannot change the dataset.
        return ShapeSample(
            body_id=plan.body_id,
            silhouettes=torch.from_numpy(
                silhouettes.reshape(n_views, self._side, self._side).astype(np.float32)
            ),
            cameras=torch.from_numpy(cameras.astype(np.float32)),
            betas=torch.from_numpy(self._betas[position].copy()),
            measurements=torch.from_numpy(self._measurements[position].copy()),
        )

    def _position(self, index: int) -> int:
        """Return the position of an element index; a negative index counts from the end."""
        try:
            position = operator.index(index)
        except TypeError:
            raise TypeError(
                f"dataset indices must be integers, not {type(index).__name__}"
            ) from None
        size = len(self._body_ids)
        if position < 0:
            position += size
        if not 0 <= position < size:
            raise IndexError(f"index {index} is out of range for a dataset of {size} bodies")
        return position


def collate_samples(samples: Sequence[ShapeSample]) -> ShapeBatch:
    """Stack samples into a ``ShapeBatch`` of four view slots, with the view mask of each body.

    Sample b with k views fills slots 0 to k - 1 of row b and sets those entries of the mask. Every
    other slot holds zeros in the silhouettes and the cameras and is false in the mask, so a batch
    of bodies with different numbers of views is one rectangular ``ViewBatch``. Raises ValueError
    for no samples, for a sample with no view or with more views than a rig has, and for samples
    of different image sizes.
    """
    if len(samples) == 0:
        raise ValueError("cannot collate an empty list of samples")
    side = samples[0].silhouettes.shape[-1]
    count = len(samples)
    silhouettes = torch.zeros(count, CAMERA_COUNT, side, side, dtype=torch.float32)
    cameras = torch.zeros(count, CAMERA_COUNT, CAMERA_ENCODING_DIM, dtype=torch.float32)
    view_mask = torch.zeros(count, CAMERA_COUNT, dtype=torch.bool)
    for row, sample in enumerate(samples):
        n_views = sample.silhouettes.shape[0]
        if not 1 <= n_views <= CAMERA_COUNT:
            raise ValueError(
                f"the sample of body {sample.body_id} has {n_views} views, but a sample needs "
                f"from 1 to {CAMERA_COUNT}"
            )
        if tuple(sample.silhouettes.shape[1:]) != (side, side):
            raise ValueError(
                f"the sample of body {sample.body_id} has silhouettes of shape "
                f"{tuple(sample.silhouettes.shape[1:])}, but the first sample has {side} by {side}"
            )
        silhouettes[row, :n_views] = sample.silhouettes
        cameras[row, :n_views] = sample.cameras
        view_mask[row, :n_views] = True
    return ShapeBatch(
        views=ViewBatch(silhouettes, cameras, view_mask),
        betas=torch.stack([sample.betas for sample in samples]),
        measurements=torch.stack([sample.measurements for sample in samples]),
        body_id=torch.tensor([sample.body_id for sample in samples], dtype=torch.int64),
    )


def _checked_view_counts(view_counts: Sequence[int]) -> tuple[int, ...]:
    """Return the view counts as a tuple of ints, after checking each is from 1 to the rig size."""
    counts = tuple(view_counts)
    if not counts:
        raise ValueError("view_counts must hold at least one number of views")
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise ValueError(f"every view count must be an integer; got {count!r}")
        if not 1 <= count <= CAMERA_COUNT:
            raise ValueError(
                f"every view count must be from 1 to {CAMERA_COUNT} (the cameras of a rig); "
                f"got {count}"
            )
    return tuple(int(count) for count in counts)


def _checked_noise_range(noise_deg: Sequence[float]) -> tuple[float, float]:
    """Return the noise range as a pair of floats, after checking 0 <= low <= high and finite."""
    try:
        low, high = (float(angle) for angle in noise_deg)
    except (TypeError, ValueError):
        raise ValueError(
            f"noise_deg must be a pair [low, high] of angles in degrees; got {noise_deg!r}"
        ) from None
    if not (0.0 <= low <= high < float("inf")):
        raise ValueError(
            f"noise_deg must be a pair [low, high] of finite angles with 0 <= low <= high; "
            f"got [{low}, {high}]"
        )
    return low, high


def _data_split(config: Mapping[str, Any]) -> DataSplit:
    """Return the data split of the configuration, or raise ConfigError for sizes that give none."""
    data = config["data"]
    try:
        return DataSplit(data["n_train"], data["n_cal"], data["n_test"])
    except ValueError as error:
        raise ConfigError(f"configuration key 'data.n_train': {error}", "data.n_train") from error


def _body_range(bodies: str | range, split: DataSplit) -> range:
    """Return the body ids that ``bodies`` selects, checked against the generated bodies."""
    if not isinstance(bodies, range):
        try:
            return split.range_of(bodies)
        except ValueError:
            raise ValueError(
                f"bodies must be a range of body ids or one of the split names train, cal, test; "
                f"got {bodies!r}"
            ) from None
    if bodies.step != 1 or len(bodies) == 0:
        raise ValueError(f"bodies must be a non-empty range with step 1; got {bodies}")
    if bodies.start < 0 or bodies.stop > split.n_bodies:
        raise ValueError(
            f"bodies {bodies.start} to {bodies.stop - 1} lie outside the {split.n_bodies} "
            "generated bodies (data.n_train + data.n_cal + data.n_test)"
        )
    return bodies


def _read_checked_shard(
    data_dir: Path, index: int, first: int, stop: int, shard_size: int, side: int
) -> ShardArrays:
    """Read shard ``index`` and check that it holds exactly body ids ``first`` up to ``stop``."""
    path = shard_path(data_dir, index)
    if not path.is_file():
        raise FileNotFoundError(
            f"shard {index} of the data folder is missing: {path} does not exist; "
            "run the generate stage first"
        )
    shard = read_shard(path, side)
    if not np.array_equal(shard.body_id, np.arange(first, stop, dtype=np.int64)):
        found = f"{shard.body_id[0]} to {shard.body_id[-1]}" if len(shard.body_id) else "none"
        raise ValueError(
            f"{path} holds {len(shard.body_id)} bodies with body ids {found}, but data.shard_size "
            f"{shard_size} puts body ids {first} to {stop - 1} in shard {index}; the data was "
            "generated with another data.shard_size or other split sizes"
        )
    return shard


def _read_bodies(
    data_dir: Path, selected: range, shard_size: int, n_bodies: int, side: int
) -> _Bodies:
    """Read the shards of the selected body ids and keep the unflagged bodies among them.

    Shard ``s`` holds body ids ``s * shard_size`` up to the next multiple, or up to ``n_bodies``
    for the last shard. The silhouettes are packed to one bit per pixel again, which is the form
    the dataset keeps them in.
    """
    names = ("body_id", "masks", "R_true", "t_true", "noise_axis", "betas", "measurements")
    kept: dict[str, list[NDArray[Any]]] = {name: [] for name in names}
    skipped: list[NDArray[np.int64]] = []
    for index in range(selected.start // shard_size, (selected.stop - 1) // shard_size + 1):
        first = index * shard_size
        stop = min(first + shard_size, n_bodies)
        shard = _read_checked_shard(data_dir, index, first, stop, shard_size, side)
        rows = slice(max(selected.start, first) - first, min(selected.stop, stop) - first)
        unflagged = shard.flags[rows] == 0
        skipped.append(shard.body_id[rows][~unflagged])
        masks = shard.masks[rows][unflagged]
        kept["body_id"].append(shard.body_id[rows][unflagged])
        kept["masks"].append(
            np.packbits(masks.reshape(len(masks), CAMERA_COUNT, side * side), axis=-1)
        )
        kept["R_true"].append(shard.R_true[rows][unflagged])
        kept["t_true"].append(shard.t_true[rows][unflagged])
        kept["noise_axis"].append(shard.noise_axis[rows][unflagged])
        kept["betas"].append(shard.betas[rows][unflagged].astype(np.float32))
        kept["measurements"].append(shard.measurements[rows][unflagged])
    joined = {name: np.concatenate(parts) for name, parts in kept.items()}
    return _Bodies(skipped=np.concatenate(skipped), **joined)
