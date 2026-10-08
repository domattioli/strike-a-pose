"""Smoke tests for data/dataset.py: view subsets, noise per cell, training draws, flags, collate."""

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

from strike_a_pose.camera import (
    CAMERA_ENCODING_DIM,
    TRANSLATION_SCALE_M,
    encode_camera,
    given_rotation,
    intrinsics,
    rotation_angle_deg,
    sample_rig,
)
from strike_a_pose.config import ConfigError, load_config
from strike_a_pose.data.dataset import (
    ShapeBatch,
    ShapeDataset,
    ShapeSample,
    ViewPlan,
    collate_samples,
)
from strike_a_pose.data.generate import data_directory, generate_dataset
from strike_a_pose.data.manifest import (
    FLAG_BITS,
    MANIFEST_NAME,
    read_manifest,
    read_shard,
    shard_path,
    write_shard,
)
from strike_a_pose.data.splits import DataSplit, Split
from strike_a_pose.model.vae import ShapeVAE, ViewBatch

# Four view slots, one per camera of a rig, and 32 px silhouettes.
SLOTS = 4
IMAGE_SIZE = 32

# Eighteen bodies in three shards of seven, the last one holding four: train 0 to 9, calibration 10
# to 13, test 14 to 17. Shard 0 holds bodies 0 to 6, shard 1 bodies 7 to 13, and shard 2 bodies 14
# to 17. With these sizes the loss-monitoring slice is body 9, and the sampler range is 0 to 8.
SYNTHETIC_OVERRIDES: dict[str, object] = {
    "data.n_train": 10,
    "data.n_cal": 4,
    "data.n_test": 4,
    "data.shard_size": 7,
    "data.min_unflagged": 1,
    "calibrate.min_cal": 2,
    "camera.image_size": IMAGE_SIZE,
    "camera.focal_px": IMAGE_SIZE,
}
SYNTHETIC_BODIES = 18

# The flags of the synthetic bodies: one flag each for bodies 1, 7, and 8, and two flags for body
# 16. Every other body is clean.
SYNTHETIC_FLAGS: dict[int, int] = {
    1: FLAG_BITS["empty_mask"],
    7: FLAG_BITS["out_of_frame"],
    8: FLAG_BITS["slice_nan"],
    16: FLAG_BITS["empty_mask"] | FLAG_BITS["out_of_frame"],
}
FLAGGED_BODIES = (1, 7, 8, 16)
UNFLAGGED_TRAIN = (0, 2, 3, 4, 5, 6, 9)
UNFLAGGED_CAL = (10, 11, 12, 13)
UNFLAGGED_TEST = (14, 15, 17)

# Extra overrides for the generated dataset: sixteen bodies in four shards of five, the last one
# holding one, with a floor of 0 unflagged bodies. With focal_px 48 on 32 px images, about half of
# the bodies reach the image border in some view, so the data holds flagged and clean bodies
# (tests/test_generate.py).
GENERATED_OVERRIDES: dict[str, object] = {
    "data.n_train": 8,
    "data.n_cal": 4,
    "data.n_test": 4,
    "data.shard_size": 5,
    "data.min_unflagged": 0,
    "calibrate.min_cal": 2,
    "camera.focal_px": 48,
}


def build_config(tiny_config_path: Path, *override_sets: dict[str, object]) -> dict[str, Any]:
    """Return tiny.yaml with the given dotted-key overrides applied; a later set wins a key."""
    merged: dict[str, object] = {}
    for overrides in override_sets:
        merged.update(overrides)
    return load_config(
        tiny_config_path, [f"{key}={json.dumps(value)}" for key, value in merged.items()]
    )


@dataclass(frozen=True, eq=False)
class SyntheticData:
    """Shard files written from arrays that the tests know, so every sample can be checked."""

    config: dict[str, Any]
    data_dir: Path
    masks: np.ndarray  # (bodies, 4, side, side) uint8, values 0 and 1
    R_true: np.ndarray  # (bodies, 4, 3, 3)
    t_true: np.ndarray  # (bodies, 4, 3)
    noise_axis: np.ndarray  # (bodies, 4, 3)
    betas: np.ndarray  # (bodies, 10)
    measurements: np.ndarray  # (bodies, 5)
    flags: np.ndarray  # (bodies,)


def write_synthetic_data(root: Path, config: dict[str, Any]) -> SyntheticData:
    """Write the shards of config's bodies from random masks and real camera rigs.

    The flags are those of SYNTHETIC_FLAGS, set for the bodies the configuration has.
    """
    data = config["data"]
    total = data["n_train"] + data["n_cal"] + data["n_test"]
    side = config["camera"]["image_size"]
    rng = np.random.default_rng(20261008)
    rigs = [sample_rig(rng, config["camera"], (0.0, 0.9, 0.0)) for _ in range(total)]
    flags = np.zeros(total, dtype=np.int64)
    for body, bits in SYNTHETIC_FLAGS.items():
        if body < total:
            flags[body] = bits
    synthetic = SyntheticData(
        config=config,
        data_dir=root / "data",
        masks=rng.integers(0, 2, size=(total, SLOTS, side, side), dtype=np.uint8),
        R_true=np.stack([rig.R_true for rig in rigs]),
        t_true=np.stack([rig.t_true for rig in rigs]),
        noise_axis=np.stack([rig.noise_axis for rig in rigs]),
        betas=rng.standard_normal((total, 10)),
        measurements=rng.uniform(20.0, 120.0, size=(total, 5)),
        flags=flags,
    )
    shard_size = data["shard_size"]
    for index in range(-(-total // shard_size)):
        rows = slice(index * shard_size, min((index + 1) * shard_size, total))
        count = rows.stop - rows.start
        write_shard(
            shard_path(synthetic.data_dir, index),
            body_id=np.arange(rows.start, rows.stop, dtype=np.int64),
            masks=synthetic.masks[rows],
            K=intrinsics(side, config["camera"]["focal_px"]),
            R_true=synthetic.R_true[rows],
            t_true=synthetic.t_true[rows],
            noise_axis=synthetic.noise_axis[rows],
            betas=synthetic.betas[rows],
            pose_root=np.zeros((count, 3)),
            pose_body=np.zeros((count, 63)),
            measurements=synthetic.measurements[rows],
            flags=flags[rows],
        )
    return synthetic


@pytest.fixture
def synthetic(tmp_path, tiny_config_path) -> SyntheticData:
    """Eighteen synthetic bodies in three shards, four of them flagged."""
    return write_synthetic_data(tmp_path, build_config(tiny_config_path, SYNTHETIC_OVERRIDES))


class GeneratedData(NamedTuple):
    """The data folder that generate_dataset wrote, and the configuration that wrote it."""

    config: dict[str, Any]
    data_dir: Path


# The dataset is generated by the first test that asks for it, and then shared. Tests only read it.
_GENERATED: dict[str, GeneratedData] = {}


@pytest.fixture
def generated(tmp_path_factory, tiny_config_path, small_config) -> GeneratedData:
    """A dataset from the generate stage: the small_config overrides plus GENERATED_OVERRIDES."""
    if "mixed" not in _GENERATED:
        config = build_config(tiny_config_path, small_config, GENERATED_OVERRIDES)
        out = tmp_path_factory.mktemp("generated")
        generate_dataset(config, out)
        _GENERATED["mixed"] = GeneratedData(config, data_directory(out))
    return _GENERATED["mixed"]


def rotation_from_6d(code: np.ndarray) -> np.ndarray:
    """Return the rotation matrices of 6D codes (first column, second column) of shape (..., 6).

    The columns are made orthonormal by Gram-Schmidt, and the third column is their cross product.
    """
    first = code[..., :3] / np.linalg.norm(code[..., :3], axis=-1, keepdims=True)
    second = code[..., 3:] - np.sum(first * code[..., 3:], axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack([first, second, np.cross(first, second)], axis=-1)


def assert_noise_about_stored_axes(
    data: SyntheticData, body: int, cameras: torch.Tensor, angles: tuple[float, ...]
) -> None:
    """Check that view v of a camera encoding is the true placement turned by angles[v].

    The turn is about the axis that the shard stores for this body and view, and the translation
    is the true one divided by 5 m. The check recovers the rotation from the 6D code and compares
    its angle and axis with the stored ones, so it does not repeat the code under test.
    """
    views = len(angles)
    code = cameras.numpy().astype(np.float64)
    assert code.shape == (views, CAMERA_ENCODING_DIM)
    given = rotation_from_6d(code[:, :6])
    relative = given @ np.swapaxes(data.R_true[body, :views], -1, -2)  # R_given R_true^T = R_noise
    np.testing.assert_allclose(rotation_angle_deg(relative), angles, atol=1e-3)
    skew = np.stack(
        [
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ],
        axis=-1,
    )
    for view, angle in enumerate(angles):
        if angle >= 1.0:  # a smaller angle leaves too short an axis for the float32 code
            axis = skew[view] / np.linalg.norm(skew[view])
            np.testing.assert_allclose(axis, data.noise_axis[body, view], atol=1e-4)
    expected_shift = (data.t_true[body, :views] / TRANSLATION_SCALE_M).astype(np.float32)
    np.testing.assert_array_equal(cameras.numpy()[:, 6:], expected_shift)


def all_samples(dataset: ShapeDataset) -> list[ShapeSample]:
    """Return every element of the dataset, in order."""
    return [dataset[position] for position in range(len(dataset))]


def assert_batches_equal(first: ShapeBatch, second: ShapeBatch) -> None:
    """Assert that every tensor of two batches is identical."""
    for one, other in zip(
        (*first.views, first.betas, first.measurements, first.body_id),
        (*second.views, second.betas, second.measurements, second.body_id),
        strict=True,
    ):
        assert torch.equal(one, other)


# ---- shapes for 1 to 4 views and the content of a sample -------------------------------------


@pytest.mark.parametrize("views", [1, 2, 3, 4])
def test_a_cell_sample_holds_the_first_k_cameras_of_the_body(synthetic, views):
    dataset = ShapeDataset.for_cell(
        synthetic.config, synthetic.data_dir, "train", views=views, noise_deg=0.0
    )
    assert dataset.body_ids == UNFLAGGED_TRAIN
    for position, body in enumerate(UNFLAGGED_TRAIN):
        sample = dataset[position]
        assert isinstance(sample, ShapeSample)
        assert sample.body_id == body
        assert sample.silhouettes.shape == (views, IMAGE_SIZE, IMAGE_SIZE)
        assert sample.silhouettes.dtype == torch.float32
        np.testing.assert_array_equal(sample.silhouettes.numpy(), synthetic.masks[body, :views])
        assert sample.cameras.shape == (views, CAMERA_ENCODING_DIM)
        assert sample.cameras.dtype == torch.float32
        # With 0 degrees of noise the encoding carries the true rotation exactly.
        expected = encode_camera(synthetic.R_true[body, :views], synthetic.t_true[body, :views])
        np.testing.assert_array_equal(sample.cameras.numpy(), expected.astype(np.float32))
        assert sample.betas.shape == (10,)
        assert sample.betas.dtype == torch.float32
        np.testing.assert_array_equal(
            sample.betas.numpy(), synthetic.betas[body].astype(np.float32)
        )
        assert sample.measurements.shape == (5,)
        assert sample.measurements.dtype == torch.float64
        np.testing.assert_array_equal(sample.measurements.numpy(), synthetic.measurements[body])
        assert dataset.view_plan(position) == ViewPlan(body, views, (0.0,) * views)


@pytest.mark.parametrize("views", [1, 2, 3, 4])
def test_collate_pads_to_four_slots_and_marks_the_present_views(synthetic, views):
    dataset = ShapeDataset.for_cell(
        synthetic.config, synthetic.data_dir, "train", views=views, noise_deg=0.0
    )
    samples = all_samples(dataset)
    count = len(samples)
    batch = collate_samples(samples)

    assert isinstance(batch, ShapeBatch)
    assert type(batch.views) is ViewBatch
    silhouettes, cameras, view_mask = batch.views
    assert silhouettes.shape == (count, SLOTS, IMAGE_SIZE, IMAGE_SIZE)
    assert silhouettes.dtype == torch.float32
    assert cameras.shape == (count, SLOTS, CAMERA_ENCODING_DIM)
    assert cameras.dtype == torch.float32
    assert view_mask.shape == (count, SLOTS)
    assert view_mask.dtype == torch.bool
    assert view_mask.tolist() == [[True] * views + [False] * (SLOTS - views)] * count

    # Present slots hold the sample's views in camera order, and absent slots hold exact zeros.
    for row, sample in enumerate(samples):
        assert torch.equal(silhouettes[row, :views], sample.silhouettes)
        assert torch.equal(cameras[row, :views], sample.cameras)
    assert bool((silhouettes[~view_mask] == 0).all())
    assert bool((cameras[~view_mask] == 0).all())

    assert batch.betas.shape == (count, 10)
    assert batch.betas.dtype == torch.float32
    assert batch.measurements.shape == (count, 5)
    assert batch.measurements.dtype == torch.float64
    assert batch.body_id.dtype == torch.int64
    assert batch.body_id.tolist() == list(UNFLAGGED_TRAIN)
    assert torch.equal(batch.betas, torch.stack([sample.betas for sample in samples]))


def test_a_batch_of_training_samples_mixes_view_counts_in_one_view_batch(synthetic):
    dataset = ShapeDataset.for_training(synthetic.config, synthetic.data_dir, "train")
    batch = collate_samples(all_samples(dataset))
    counts = [dataset.view_plan(position).n_views for position in range(len(dataset))]
    assert len(set(counts)) > 1  # the seeded draws of these seven bodies are not all alike
    # Every row is a prefix of present slots: camera 0 first, no gaps.
    prefix = torch.arange(SLOTS).unsqueeze(0) < torch.tensor(counts).unsqueeze(1)
    assert torch.equal(batch.views.view_mask, prefix)
    assert bool((batch.views.silhouettes[~prefix] == 0).all())


def test_a_collated_batch_is_what_the_vae_consumes(generated):
    dataset = ShapeDataset.for_training(generated.config, generated.data_dir, "train")
    batch = collate_samples(all_samples(dataset))
    torch.manual_seed(0)
    model = ShapeVAE.from_config(generated.config)

    terms = model.loss(batch.views, batch.betas, generator=torch.Generator().manual_seed(1))
    for part in (terms.total, terms.nll_joint, terms.nll_single, terms.kl_joint, terms.kl_single):
        assert bool(torch.isfinite(part))
    experts = model.encode(batch.views)
    assert experts.mu_v.shape == (len(dataset), SLOTS, model.latent_dim)
    assert bool((experts.mu_v[~batch.views.view_mask] == 0).all())
    posterior = model(batch.views)
    assert posterior.mu.shape == (len(dataset), model.latent_dim)
    samples = model.sample_betas(batch.views, 3, torch.Generator().manual_seed(2))
    assert samples.shape == (len(dataset), 3, model.n_betas)


@pytest.mark.parametrize("image_size", [15, 31])
def test_silhouettes_survive_the_bit_packing_for_sizes_that_are_not_a_multiple_of_eight(
    tmp_path, tiny_config_path, image_size
):
    # 15 by 15 = 225 pixels and 31 by 31 = 961 pixels leave padding bits in the last byte of a mask.
    config = build_config(
        tiny_config_path,
        SYNTHETIC_OVERRIDES,
        {"camera.image_size": image_size, "camera.focal_px": image_size},
    )
    data = write_synthetic_data(tmp_path, config)
    dataset = ShapeDataset.for_cell(
        config, data.data_dir, range(SYNTHETIC_BODIES), views=4, noise_deg=0.0
    )
    assert len(dataset) == SYNTHETIC_BODIES - len(FLAGGED_BODIES)
    for position, body in enumerate(dataset.body_ids):
        silhouettes = dataset[position].silhouettes
        assert silhouettes.shape == (4, image_size, image_size)
        np.testing.assert_array_equal(silhouettes.numpy(), data.masks[body])


# ---- placement noise per cell ----------------------------------------------------------------


def test_cell_noise_changes_the_rotation_of_the_camera_encoding_and_nothing_else(synthetic):
    config, data_dir = synthetic.config, synthetic.data_dir
    batches = {}
    for noise in (0.0, 2.0, 5.0):
        dataset = ShapeDataset.for_cell(
            config, data_dir, range(SYNTHETIC_BODIES), views=4, noise_deg=noise
        )
        batches[noise] = collate_samples(all_samples(dataset))
    base = batches[0.0]
    for noise in (2.0, 5.0):
        other = batches[noise]
        assert torch.equal(other.views.silhouettes, base.views.silhouettes)
        assert torch.equal(other.views.view_mask, base.views.view_mask)
        assert torch.equal(other.betas, base.betas)
        assert torch.equal(other.measurements, base.measurements)
        assert torch.equal(other.body_id, base.body_id)
        # The translation part of the encoding is the true one at every noise level.
        assert torch.equal(other.views.cameras[..., 6:], base.views.cameras[..., 6:])
        # The rotation part moves in every view of every body.
        moved = (other.views.cameras[..., :6] != base.views.cameras[..., :6]).any(dim=-1)
        assert bool(moved.all())


@pytest.mark.parametrize("noise", [0.0, 2.0, 5.0])
def test_the_noise_turns_each_view_about_its_stored_axis_by_the_cell_angle(synthetic, noise):
    dataset = ShapeDataset.for_cell(
        synthetic.config, synthetic.data_dir, range(SYNTHETIC_BODIES), views=4, noise_deg=noise
    )
    # The stored axes differ between the views of a body, so each view is checked against its own.
    for body in dataset.body_ids:
        assert len({tuple(axis) for axis in synthetic.noise_axis[body].round(6)}) == SLOTS
    for position, body in enumerate(dataset.body_ids):
        assert_noise_about_stored_axes(synthetic, body, dataset[position].cameras, (noise,) * SLOTS)


@pytest.mark.parametrize("noise", [0.0, 2.0, 5.0])
def test_a_k_view_cell_is_the_first_k_slots_of_the_four_view_cell(synthetic, noise):
    # Prediction encodes the four views of a body once and fuses the first k of them for the k-view
    # cell, so the k-view batch must be exactly the first k slots of the four-view batch.
    config, data_dir = synthetic.config, synthetic.data_dir
    four = collate_samples(
        all_samples(ShapeDataset.for_cell(config, data_dir, "cal", views=4, noise_deg=noise))
    )
    for views in (1, 2, 3):
        few = collate_samples(
            all_samples(
                ShapeDataset.for_cell(config, data_dir, "cal", views=views, noise_deg=noise)
            )
        )
        assert torch.equal(few.views.silhouettes[:, :views], four.views.silhouettes[:, :views])
        assert torch.equal(few.views.cameras[:, :views], four.views.cameras[:, :views])
        assert torch.equal(
            few.views.view_mask, four.views.view_mask & (torch.arange(SLOTS) < views)
        )
        assert torch.equal(few.betas, four.betas)
        assert torch.equal(few.measurements, four.measurements)
        assert torch.equal(few.body_id, four.body_id)


def test_a_cell_serves_the_same_samples_on_every_pass_and_at_every_epoch(synthetic):
    dataset = ShapeDataset.for_cell(
        synthetic.config, synthetic.data_dir, "train", views=3, noise_deg=2.0
    )
    first = collate_samples(all_samples(dataset))
    dataset.set_epoch(7)
    assert dataset.epoch == 7
    assert_batches_equal(first, collate_samples(all_samples(dataset)))
    assert all(dataset.view_plan(i).noise_deg == (2.0, 2.0, 2.0) for i in range(len(dataset)))


# ---- flagged bodies --------------------------------------------------------------------------


def test_flagged_bodies_are_excluded_and_counted_as_skipped(synthetic):
    config, data_dir = synthetic.config, synthetic.data_dir
    assert tuple(np.flatnonzero(synthetic.flags).tolist()) == FLAGGED_BODIES
    everything = ShapeDataset.for_cell(
        config, data_dir, range(SYNTHETIC_BODIES), views=2, noise_deg=0.0
    )
    clean = tuple(body for body in range(SYNTHETIC_BODIES) if body not in FLAGGED_BODIES)
    assert everything.body_ids == clean
    assert everything.skipped_body_ids == FLAGGED_BODIES
    assert len(everything) == SYNTHETIC_BODIES - len(FLAGGED_BODIES)

    # Each element is the body its id says, including bodies that follow a flagged one in a shard.
    for position, body in enumerate(everything.body_ids):
        sample = everything[position]
        assert sample.body_id == body
        np.testing.assert_array_equal(sample.silhouettes.numpy(), synthetic.masks[body, :2])
        np.testing.assert_array_equal(
            sample.betas.numpy(), synthetic.betas[body].astype(np.float32)
        )
        np.testing.assert_array_equal(sample.measurements.numpy(), synthetic.measurements[body])

    for split, served, skipped in (
        ("train", UNFLAGGED_TRAIN, (1, 7, 8)),
        ("cal", UNFLAGGED_CAL, ()),
        ("test", UNFLAGGED_TEST, (16,)),
    ):
        dataset = ShapeDataset.for_cell(config, data_dir, split, views=1, noise_deg=0.0)
        assert dataset.body_ids == served
        assert dataset.skipped_body_ids == skipped


def test_a_range_of_flagged_bodies_gives_an_empty_dataset(synthetic):
    dataset = ShapeDataset.for_cell(
        synthetic.config, synthetic.data_dir, range(7, 9), views=1, noise_deg=0.0
    )
    assert len(dataset) == 0
    assert dataset.body_ids == ()
    assert dataset.skipped_body_ids == (7, 8)
    with pytest.raises(IndexError):
        dataset[0]


def test_flagged_bodies_of_the_generated_data_are_excluded_and_the_rest_match_the_shards(generated):
    config, data_dir = generated.config, generated.data_dir
    rows = read_manifest(data_dir / MANIFEST_NAME)
    flagged = tuple(row.body_id for row in rows if row.flags)
    clean = tuple(row.body_id for row in rows if not row.flags)
    assert flagged and clean  # the data holds both kinds of body, so the check has power

    dataset = ShapeDataset.for_cell(config, data_dir, range(len(rows)), views=4, noise_deg=0.0)
    assert dataset.body_ids == clean
    assert dataset.skipped_body_ids == flagged

    shard_size = config["data"]["shard_size"]
    side = config["camera"]["image_size"]
    for position, body in enumerate(clean):
        shard = read_shard(shard_path(data_dir, body // shard_size), side)
        local = body % shard_size
        sample = dataset[position]
        np.testing.assert_array_equal(sample.silhouettes.numpy(), shard.masks[local])
        expected = encode_camera(shard.R_true[local], shard.t_true[local]).astype(np.float32)
        np.testing.assert_array_equal(sample.cameras.numpy(), expected)
        np.testing.assert_array_equal(sample.betas.numpy(), shard.betas[local].astype(np.float32))
        np.testing.assert_array_equal(sample.measurements.numpy(), shard.measurements[local])


# ---- which bodies a dataset serves -------------------------------------------------------------


def test_a_split_name_or_a_range_selects_the_bodies(synthetic):
    config, data_dir = synthetic.config, synthetic.data_dir
    split = DataSplit(10, 4, 4)
    sampler = ShapeDataset.for_training(config, data_dir, split.sampler_range)
    monitor = ShapeDataset.for_training(config, data_dir, split.monitor_range)
    assert sampler.body_ids == (0, 2, 3, 4, 5, 6)
    assert sampler.skipped_body_ids == (1, 7, 8)
    assert monitor.body_ids == (9,)
    assert set(sampler.body_ids).isdisjoint(monitor.body_ids)

    # A range may start inside a shard and end in the next one.
    crossing = ShapeDataset.for_cell(config, data_dir, range(4, 9), views=1, noise_deg=0.0)
    assert crossing.body_ids == (4, 5, 6)
    assert crossing.skipped_body_ids == (7, 8)

    # A member of the Split enumeration is a split name too.
    calibration = ShapeDataset.for_training(config, data_dir, Split.CALIBRATION)
    assert calibration.body_ids == UNFLAGGED_CAL


def test_a_dataset_reads_only_the_shards_that_hold_its_bodies(synthetic, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(synthetic.data_dir, copy)
    shard_path(copy, 1).unlink()
    # Shard 0 alone holds bodies 0 to 6, and shard 2 alone, the short last one, bodies 14 to 17.
    first = ShapeDataset.for_training(synthetic.config, copy, range(0, 7))
    assert first.body_ids == (0, 2, 3, 4, 5, 6)
    last = ShapeDataset.for_training(synthetic.config, copy, "test")
    assert last.body_ids == UNFLAGGED_TEST
    with pytest.raises(FileNotFoundError, match="shard_0001.npz") as refusal:
        ShapeDataset.for_training(synthetic.config, copy, "train")  # bodies 7 to 9 need shard 1
    assert "shard 1" in str(refusal.value)


def test_shards_that_do_not_match_the_shard_size_are_refused(synthetic, tiny_config_path):
    other = build_config(tiny_config_path, SYNTHETIC_OVERRIDES, {"data.shard_size": 5})
    with pytest.raises(ValueError, match="data.shard_size 5 puts body ids 0 to 4 in shard 0"):
        ShapeDataset.for_training(other, synthetic.data_dir, "train")


def test_a_data_section_that_gives_no_split_is_a_configuration_error(synthetic, tiny_config_path):
    config = build_config(tiny_config_path, SYNTHETIC_OVERRIDES, {"data.n_train": 1})
    with pytest.raises(ConfigError, match="n_train must be at least 2") as refusal:
        ShapeDataset.for_training(config, synthetic.data_dir, "train")
    assert refusal.value.key == "data.n_train"


@pytest.mark.parametrize(
    ("bodies", "message"),
    [
        ("validation", "split names train, cal, test"),
        (range(10, 19), "outside the 18 generated bodies"),
        (range(-1, 3), "outside the 18 generated bodies"),
        (range(0, 0), "non-empty range with step 1"),
        (range(0, 10, 2), "non-empty range with step 1"),
    ],
)
def test_a_selection_that_names_no_bodies_of_the_data_is_refused(synthetic, bodies, message):
    with pytest.raises(ValueError, match=message):
        ShapeDataset.for_training(synthetic.config, synthetic.data_dir, bodies)


@pytest.mark.parametrize("views", [0, 5, -1, 2.5, True, "2"])
def test_a_cell_needs_a_view_count_from_one_to_four(synthetic, views):
    with pytest.raises(ValueError, match="view count"):
        ShapeDataset.for_cell(
            synthetic.config, synthetic.data_dir, "train", views=views, noise_deg=0.0
        )


@pytest.mark.parametrize("noise", [-1.0, float("nan"), float("inf"), "five"])
def test_a_cell_needs_a_finite_non_negative_noise_angle(synthetic, noise):
    with pytest.raises(ValueError, match="noise_deg"):
        ShapeDataset.for_cell(
            synthetic.config, synthetic.data_dir, "train", views=2, noise_deg=noise
        )


@pytest.mark.parametrize(
    ("view_counts", "noise_deg"),
    [
        ([], [0.0, 5.0]),
        ([1, 2], [5.0, 0.0]),
        ([1, 2], [0.0]),
        ([1, 2], [-1.0, 5.0]),
        ([1, 9], [0, 5]),
    ],
)
def test_the_training_distribution_is_checked(synthetic, view_counts, noise_deg):
    with pytest.raises(ValueError):
        ShapeDataset(
            synthetic.config,
            synthetic.data_dir,
            "train",
            view_counts=view_counts,
            noise_deg=noise_deg,
        )


def test_indexing_follows_the_sequence_protocol(synthetic):
    dataset = ShapeDataset.for_cell(
        synthetic.config, synthetic.data_dir, "train", views=2, noise_deg=0.0
    )
    assert len(dataset) == len(UNFLAGGED_TRAIN) == 7
    assert dataset[-1].body_id == dataset[6].body_id == 9
    assert dataset[np.int64(2)].body_id == dataset[torch.tensor(2)].body_id == 3
    # Iteration relies on IndexError at the end, as the old sequence protocol defines.
    assert [sample.body_id for sample in dataset] == list(UNFLAGGED_TRAIN)
    for bad in (7, -8):
        with pytest.raises(IndexError, match="out of range"):
            dataset[bad]
    for bad in ("0", 1.5, None):
        with pytest.raises(TypeError, match="integers"):
            dataset[bad]


def test_editing_a_sample_in_place_does_not_change_the_dataset(synthetic):
    dataset = ShapeDataset.for_cell(
        synthetic.config, synthetic.data_dir, "train", views=2, noise_deg=2.0
    )
    reference = dataset[0]
    edited = dataset[0]
    for tensor in (edited.silhouettes, edited.cameras, edited.betas, edited.measurements):
        tensor.zero_()
    again = dataset[0]
    for kept, fresh in zip(reference[1:], again[1:], strict=True):
        assert torch.equal(kept, fresh)
        assert bool((fresh != 0).any())


# ---- training draws --------------------------------------------------------------------------


def test_training_draws_depend_on_the_seed_the_epoch_and_the_body_alone(
    synthetic, tiny_config_path
):
    config, data_dir = synthetic.config, synthetic.data_dir
    dataset = ShapeDataset.for_training(config, data_dir, "train")
    plans = [dataset.view_plan(position) for position in range(len(dataset))]
    assert dataset.epoch == 0

    # A second dataset, asked in the opposite order, draws the same plans.
    twin = ShapeDataset.for_training(config, data_dir, "train")
    backwards = [twin.view_plan(position) for position in reversed(range(len(twin)))]
    assert backwards[::-1] == plans

    # A dataset over other bodies gives a body the plan it has here.
    subset = ShapeDataset.for_training(config, data_dir, range(4, 8))
    assert subset.body_ids == (4, 5, 6)
    for position, body in enumerate(subset.body_ids):
        assert subset.view_plan(position) == plans[dataset.body_ids.index(body)]

    # Another seed gives other draws.
    other_seed = build_config(tiny_config_path, SYNTHETIC_OVERRIDES, {"seed": 2})
    reseeded = ShapeDataset.for_training(other_seed, data_dir, "train")
    assert [reseeded.view_plan(i) for i in range(len(reseeded))] != plans

    # Another epoch gives other draws, and going back to the first epoch gives the first draws.
    dataset.set_epoch(1)
    assert dataset.epoch == 1
    assert [dataset.view_plan(i) for i in range(len(dataset))] != plans
    dataset.set_epoch(0)
    assert [dataset.view_plan(i) for i in range(len(dataset))] == plans


def test_training_draws_cover_the_configured_view_counts_and_noise_range(synthetic):
    dataset = ShapeDataset.for_training(synthetic.config, synthetic.data_dir, "train")
    epochs = 80
    counts: Counter[int] = Counter()
    angles: list[float] = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        for position in range(len(dataset)):
            plan = dataset.view_plan(position)
            assert plan.body_id == dataset.body_ids[position]
            assert len(plan.noise_deg) == plan.n_views
            # The views of one sample are drawn independently, so their angles differ.
            assert len(set(plan.noise_deg)) == plan.n_views
            counts[plan.n_views] += 1
            angles.extend(plan.noise_deg)
    draws = epochs * len(dataset)

    # The tiny configuration draws views from [1, 2, 3, 4] and angles from [0, 5] degrees.
    assert set(counts) == {1, 2, 3, 4}
    for views in (1, 2, 3, 4):
        assert abs(counts[views] / draws - 0.25) < 0.07
    assert 0.0 <= min(angles) < 0.3
    assert 4.7 < max(angles) <= 5.0
    assert abs(float(np.mean(angles)) - 2.5) < 0.15


def test_a_training_configuration_can_narrow_the_views_and_the_noise(synthetic, tiny_config_path):
    narrow = build_config(
        tiny_config_path,
        SYNTHETIC_OVERRIDES,
        {"train.views_train": [2, 3], "train.noise_train_deg": [1.0, 1.0]},
    )
    dataset = ShapeDataset.for_training(narrow, synthetic.data_dir, "train")
    seen = set()
    for epoch in range(10):
        dataset.set_epoch(epoch)
        for position in range(len(dataset)):
            plan = dataset.view_plan(position)
            assert plan.n_views in (2, 3)
            assert plan.noise_deg == (1.0,) * plan.n_views
            seen.add(plan.n_views)
    assert seen == {2, 3}

    fixed = build_config(
        tiny_config_path,
        SYNTHETIC_OVERRIDES,
        {"train.views_train": [3], "train.noise_train_deg": [2.0, 2.0]},
    )
    constant = ShapeDataset.for_training(fixed, synthetic.data_dir, "train")
    constant.set_epoch(5)
    assert {constant.view_plan(i) for i in range(len(constant))} == {
        ViewPlan(body, 3, (2.0, 2.0, 2.0)) for body in constant.body_ids
    }


def test_a_training_sample_rotates_about_the_stored_axes_by_the_drawn_angles(synthetic):
    dataset = ShapeDataset.for_training(synthetic.config, synthetic.data_dir, "train")
    checked_views = set()
    for epoch in range(6):
        dataset.set_epoch(epoch)
        for position in range(len(dataset)):
            plan = dataset.view_plan(position)
            sample = dataset[position]
            body = plan.body_id
            assert sample.body_id == body
            assert sample.silhouettes.shape[0] == plan.n_views
            np.testing.assert_array_equal(
                sample.silhouettes.numpy(), synthetic.masks[body, : plan.n_views]
            )
            assert_noise_about_stored_axes(synthetic, body, sample.cameras, plan.noise_deg)
            # The encoding is the one the camera module builds for the drawn angles.
            expected = encode_camera(
                given_rotation(
                    synthetic.R_true[body, : plan.n_views],
                    synthetic.noise_axis[body, : plan.n_views],
                    plan.noise_deg,
                ),
                synthetic.t_true[body, : plan.n_views],
            )
            np.testing.assert_array_equal(sample.cameras.numpy(), expected.astype(np.float32))
            checked_views.add(plan.n_views)
    assert checked_views == {1, 2, 3, 4}


def run_epoch(dataset: ShapeDataset, epoch: int) -> list[ShapeBatch]:
    """Read one epoch through a DataLoader with a seeded random sampler and the collate function."""
    dataset.set_epoch(epoch)
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(7))
    loader = DataLoader(dataset, batch_size=3, sampler=sampler, collate_fn=collate_samples)
    return list(loader)


def test_an_epoch_through_a_seeded_data_loader_is_reproducible(synthetic):
    dataset = ShapeDataset.for_training(synthetic.config, synthetic.data_dir, "train")
    first = run_epoch(dataset, 0)
    again = run_epoch(dataset, 0)
    assert [len(batch.body_id) for batch in first] == [3, 3, 1]
    for one, other in zip(first, again, strict=True):
        assert_batches_equal(one, other)
    served = sorted(torch.cat([batch.body_id for batch in first]).tolist())
    assert served == list(UNFLAGGED_TRAIN)

    # Another epoch reads the bodies in the same order, with new view counts and angles.
    later = run_epoch(dataset, 1)
    assert [batch.body_id.tolist() for batch in later] == [
        batch.body_id.tolist() for batch in first
    ]
    assert any(
        not torch.equal(one.views.cameras, other.views.cameras)
        for one, other in zip(first, later, strict=True)
    )
    # A new dataset that starts at that epoch, as after a resume, reads the same batches.
    resumed = ShapeDataset.for_training(synthetic.config, synthetic.data_dir, "train")
    for one, other in zip(later, run_epoch(resumed, 1), strict=True):
        assert_batches_equal(one, other)


@pytest.mark.parametrize("epoch", [-1, 1.5, "1", True, None])
def test_set_epoch_needs_a_non_negative_integer(synthetic, epoch):
    dataset = ShapeDataset.for_training(synthetic.config, synthetic.data_dir, "train")
    with pytest.raises(ValueError, match="epoch"):
        dataset.set_epoch(epoch)
    assert dataset.epoch == 0


# ---- collate ---------------------------------------------------------------------------------


def test_collate_refuses_an_empty_list_and_malformed_samples(synthetic):
    with pytest.raises(ValueError, match="empty"):
        collate_samples([])
    dataset = ShapeDataset.for_cell(
        synthetic.config, synthetic.data_dir, "train", views=4, noise_deg=0.0
    )
    sample = dataset[0]
    too_many = sample._replace(
        silhouettes=torch.zeros(5, IMAGE_SIZE, IMAGE_SIZE), cameras=torch.zeros(5, 9)
    )
    with pytest.raises(ValueError, match="has 5 views"):
        collate_samples([too_many])
    none = sample._replace(
        silhouettes=torch.zeros(0, IMAGE_SIZE, IMAGE_SIZE), cameras=torch.zeros(0, 9)
    )
    with pytest.raises(ValueError, match="has 0 views"):
        collate_samples([none])
    larger = sample._replace(
        silhouettes=torch.zeros(4, IMAGE_SIZE + 1, IMAGE_SIZE + 1), cameras=torch.zeros(4, 9)
    )
    with pytest.raises(ValueError, match="silhouettes of shape"):
        collate_samples([sample, larger])
