"""Smoke tests for data/generate.py: determinism, resume, the rejection rate, flags, and exit 4."""

import dataclasses
import gc
import json
import logging
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch

from strike_a_pose.assets import MissingAssetError
from strike_a_pose.body.base import NUM_JOINTS, PARENTS, canonical_mesh
from strike_a_pose.body.smplx_body import SmplxBody
from strike_a_pose.body.standin import StandInBody
from strike_a_pose.camera import azimuths_are_separated, intrinsics, project_points, sample_rig
from strike_a_pose.checkpoint import (
    DONE_MARKER_NAME,
    StageStatus,
    TimeBudget,
    check_stage,
    read_done_marker,
)
from strike_a_pose.config import ConfigError, config_hash, load_config
from strike_a_pose.data import generate as generate_module
from strike_a_pose.data.generate import (
    DATA_DIRECTORY_NAME,
    STAGE_NAME,
    STAGE_REFUSED_EXIT_CODE,
    SUMMARY_NAME,
    GenerationRefusedError,
    GenerationResult,
    create_body_model,
    data_directory,
    generate_dataset,
)
from strike_a_pose.data.manifest import (
    FLAG_BITS,
    MANIFEST_COLUMNS,
    MANIFEST_NAME,
    read_manifest,
    read_shard,
    shard_path,
)
from strike_a_pose.data.splits import DataSplit
from strike_a_pose.measure import measure_mesh
from strike_a_pose.pose.filters import PoseFilter
from strike_a_pose.pose.source import create_pose_source
from strike_a_pose.render import render_silhouette, silhouette_from_mask
from strike_a_pose.runrecord import STAGES, RunRecord, start_run_record
from strike_a_pose.seeding import rng_for

# Extra overrides on top of the small_config of contracts/config.md for the tests that run the stage
# many times: ten bodies in four shards of at most three. The last shard is short, and two shards
# hold bodies of two splits. Bodies 0 to 3 train, 4 to 6 calibrate, and 7 to 9 test.
SHRUNK_OVERRIDES: dict[str, object] = {
    "data.n_train": 4,
    "data.n_cal": 3,
    "data.n_test": 3,
    "data.shard_size": 3,
    "data.min_unflagged": 2,
    "calibrate.min_cal": 2,
}
SHRUNK_BODIES = 10
SHRUNK_SHARDS = 4

# Thirty-two bodies in two shards of sixteen, for the tests that need many bodies but not 128. The
# floor of unflagged bodies is 0, so that a camera setting that flags bodies cannot stop the stage.
MIXED_OVERRIDES: dict[str, object] = {
    "data.n_train": 16,
    "data.n_cal": 8,
    "data.n_test": 8,
    "data.shard_size": 16,
    "data.min_unflagged": 0,
    "calibrate.min_cal": 2,
}
MIXED_BODIES = 32
MIXED_SHARDS = 2

# The shape of the dataset that the small_config overrides give: four shards of 32 bodies.
SMALL_BODIES = 128
SMALL_SHARD_SIZE = 32

# The stream parts of generate.py (module docstring): shape coefficients, pose, camera rig.
BETAS_PART, POSE_PART, RIG_PART = 0, 1, 2
GENERATE_STAGE_ID = STAGES.index("generate")


class ReferenceRun(NamedTuple):
    """One generation with the small_config overrides, shared by the tests that only read it."""

    config: dict[str, Any]
    out: Path
    result: GenerationResult


class FakeClock:
    """A clock that moves only when a test moves it, so a time budget never waits for real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# The reference run is made by the first test that asks for it, and then shared. Tests never write
# into it, so the share saves one generation of 128 bodies for each test that reads it.
_REFERENCE_RUNS: dict[str, ReferenceRun] = {}


def build_config(tiny_config_path: Path, *override_sets: dict[str, object]) -> dict[str, Any]:
    """Return tiny.yaml with the given dotted-key overrides applied; a later set wins a key."""
    merged: dict[str, object] = {}
    for overrides in override_sets:
        merged.update(overrides)
    return load_config(
        tiny_config_path, [f"{key}={json.dumps(value)}" for key, value in merged.items()]
    )


def fixed_record(config: dict[str, Any]) -> RunRecord:
    """Return a run record with fixed provenance, so summary.json ignores the checkout."""
    return start_run_record(
        config,
        hardware_class="cpu",
        device_name="cpu",
        started_at=datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc),
        code_version="0.1.0+test",
    )


def generate(config: dict[str, Any], out: Path, **options: Any) -> GenerationResult:
    """Run the stage with fixed provenance."""
    return generate_dataset(config, out, run_record=fixed_record(config), **options)


@pytest.fixture
def reference(tmp_path_factory, tiny_config_path, small_config) -> ReferenceRun:
    """One generation with the small_config overrides, made by the first test that asks for it."""
    if "small" not in _REFERENCE_RUNS:
        config = build_config(tiny_config_path, small_config)
        out = tmp_path_factory.mktemp("reference")
        _REFERENCE_RUNS["small"] = ReferenceRun(config, out, generate(config, out))
    return _REFERENCE_RUNS["small"]


@pytest.fixture
def shrunk_config(tiny_config_path, small_config) -> dict[str, Any]:
    """The small configuration cut down to ten bodies in four shards."""
    return build_config(tiny_config_path, small_config, SHRUNK_OVERRIDES)


@pytest.fixture
def written_shards(monkeypatch) -> list[str]:
    """The file name of every shard npz that the stage writes, in order."""
    names: list[str] = []
    real_write_shard = generate_module.write_shard

    def recording_write_shard(destination, **arrays):
        names.append(Path(destination).name)
        return real_write_shard(destination, **arrays)

    monkeypatch.setattr(generate_module, "write_shard", recording_write_shard)
    return names


def shard_files(data: Path) -> dict[str, Path]:
    """Return the files of the shard folder by name."""
    return {path.name: path for path in sorted((data / "shards").iterdir())}


def shard_name(index: int, extension: str) -> str:
    """Return the file name of one shard file, such as shard_0002.npz."""
    return f"shard_{index:04d}.{extension}"


def all_shard_names(shards: int) -> list[str]:
    """Return the names of the csv and npz file of each of the first ``shards`` shards."""
    return [shard_name(index, extension) for index in range(shards) for extension in ("csv", "npz")]


def read_summary(data: Path) -> dict[str, Any]:
    """Read summary.json of a data folder."""
    return json.loads((data / SUMMARY_NAME).read_text(encoding="utf-8"))


def snapshot(data: Path) -> dict[str, bytes]:
    """Return the bytes of every shard file and of manifest.csv, by name."""
    files = {name: path.read_bytes() for name, path in shard_files(data).items()}
    files[MANIFEST_NAME] = (data / MANIFEST_NAME).read_bytes()
    return files


def assert_same_data(first: Path, second: Path) -> None:
    """Assert that two data folders hold the same shards and the same manifest, byte for byte."""
    first_files, second_files = snapshot(first), snapshot(second)
    assert list(first_files) == list(second_files)
    different = [name for name in first_files if first_files[name] != second_files[name]]
    assert not different, f"{different} differ between {first} and {second}"


def slow_shards(patch: pytest.MonkeyPatch, clock: FakeClock, seconds: float) -> None:
    """Make every shard take ``seconds`` of the fake clock, which a time budget then counts."""
    real_write_shard = generate_module.write_shard

    def slow_write_shard(destination, **arrays):
        result = real_write_shard(destination, **arrays)
        clock.now += seconds
        return result

    patch.setattr(generate_module, "write_shard", slow_write_shard)


# ---- layout and determinism ----------------------------------------------------------------


def test_the_output_directory_follows_the_artifact_contract(reference):
    data = data_directory(reference.out)
    assert data == reference.out / DATA_DIRECTORY_NAME
    assert sorted(path.name for path in data.iterdir()) == [
        DONE_MARKER_NAME,
        MANIFEST_NAME,
        "shards",
        SUMMARY_NAME,
    ]
    assert list(shard_files(data)) == all_shard_names(4)
    result = reference.result
    assert result.completed
    assert (result.shards_total, result.shards_generated, result.shards_reused) == (4, 4, 0)

    # manifest.csv is the header once, then the shard tables in shard order.
    rows = read_manifest(data / MANIFEST_NAME)
    split = DataSplit(64, 32, 32)
    assert [row.body_id for row in rows] == list(range(SMALL_BODIES))
    assert [row.shard for row in rows] == [body // SMALL_SHARD_SIZE for body in range(SMALL_BODIES)]
    assert [row.split for row in rows] == [split.split_of(i).value for i in range(SMALL_BODIES)]
    assert {row.pose_source for row in rows} == {"limits"}
    lines = (data / MANIFEST_NAME).read_text(encoding="utf-8").splitlines(keepends=True)
    expected = [",".join(MANIFEST_COLUMNS) + "\n"]
    for index in range(4):
        table = shard_path(data, index).with_suffix(".csv")
        expected += table.read_text(encoding="utf-8").splitlines(keepends=True)[1:]
    assert lines == expected

    # summary.json holds the fields of the contract, and DONE.json the stage provenance.
    summary = read_summary(data)
    assert list(summary) == [
        "n_bodies",
        "n_flagged",
        "n_unflagged",
        "min_unflagged",
        "n_rejections",
        "n_draws",
        "rejection_rate",
        "per_shard",
        "config_hash",
        "seed",
        "code_version",
        "hardware_class",
    ]
    assert summary["n_bodies"] == SMALL_BODIES
    flagged = sum(1 for row in rows if row.flags)
    assert sum(summary["n_unflagged"].values()) + flagged == SMALL_BODIES
    assert summary["min_unflagged"] == 16
    assert [entry["shard"] for entry in summary["per_shard"]] == [0, 1, 2, 3]
    assert {entry["n_bodies"] for entry in summary["per_shard"]} == {SMALL_SHARD_SIZE}
    assert summary["config_hash"] == config_hash(reference.config)
    assert (summary["seed"], summary["code_version"], summary["hardware_class"]) == (
        1,
        "0.1.0+test",
        "cpu",
    )
    marker = read_done_marker(data)
    assert marker is not None
    assert (marker.stage, marker.config_hash, marker.seed) == (
        STAGE_NAME,
        config_hash(reference.config),
        1,
    )
    assert marker.inputs == {}
    assert (marker.code_version, marker.hardware_class) == ("0.1.0+test", "cpu")


def test_the_done_marker_is_accepted_by_the_stage_check_of_the_same_configuration_only(
    reference, tiny_config_path, small_config
):
    data = data_directory(reference.out)
    same = check_stage(data, stage=STAGE_NAME, config_hash=config_hash(reference.config), inputs={})
    assert same.status is StageStatus.DONE
    other = build_config(tiny_config_path, small_config, {"seed": 2})
    different = check_stage(data, stage=STAGE_NAME, config_hash=config_hash(other), inputs={})
    assert different.status is StageStatus.STALE


def test_two_generations_with_the_small_configuration_write_identical_shard_bytes(
    reference, tiny_config_path, small_config, tmp_path
):
    config = build_config(tiny_config_path, small_config)
    second = generate(config, tmp_path / "second")
    first_data = data_directory(reference.out)
    second_data = data_directory(tmp_path / "second")
    assert list(shard_files(first_data)) == all_shard_names(4)
    assert_same_data(first_data, second_data)
    # With the same provenance, the summary is identical too. DONE.json holds the finishing time.
    assert (first_data / SUMMARY_NAME).read_bytes() == (second_data / SUMMARY_NAME).read_bytes()
    assert dict(second.summary) == dict(reference.result.summary)


def test_a_fresh_process_with_another_hash_seed_and_thread_count_writes_the_same_bytes(
    shrunk_config, small_config, tiny_config_path, tmp_path
):
    # Two runs in one process cannot show a dependence on the hash seed of Python or on the number
    # of threads, so the second run is a new interpreter with both changed (FR-024).
    generate(shrunk_config, tmp_path / "here")
    merged = {**small_config, **SHRUNK_OVERRIDES}
    overrides = [f"{key}={json.dumps(value)}" for key, value in merged.items()]
    script = (
        "import sys\n"
        "from strike_a_pose.config import load_config\n"
        "from strike_a_pose.data.generate import generate_dataset\n"
        "generate_dataset(load_config(sys.argv[1], sys.argv[3:]), sys.argv[2])\n"
    )
    environment = {**os.environ, "PYTHONHASHSEED": "12345", "OMP_NUM_THREADS": "1"}
    subprocess.run(
        [sys.executable, "-c", script, str(tiny_config_path), str(tmp_path / "there"), *overrides],
        check=True,
        env=environment,
        timeout=120,
    )
    assert_same_data(data_directory(tmp_path / "here"), data_directory(tmp_path / "there"))


def test_the_seed_changes_the_bodies(shrunk_config, tiny_config_path, small_config, tmp_path):
    generate(shrunk_config, tmp_path / "first")
    other = build_config(tiny_config_path, small_config, SHRUNK_OVERRIDES, {"seed": 2})
    generate(other, tmp_path / "second")
    first = read_manifest(data_directory(tmp_path / "first") / MANIFEST_NAME)
    second = read_manifest(data_directory(tmp_path / "second") / MANIFEST_NAME)
    assert [row.body_id for row in first] == [row.body_id for row in second]
    pairs = list(zip(first, second, strict=True))
    assert all(a.betas != b.betas and a.pose_body != b.pose_body for a, b in pairs)
    assert all(a.azimuth_deg != b.azimuth_deg for a, b in pairs)


def test_each_shard_holds_its_bodies_in_the_layout_of_the_artifact_contract(reference):
    data = data_directory(reference.out)
    for index in range(4):
        shard = read_shard(shard_path(data, index), 32)
        first = SMALL_SHARD_SIZE * index
        assert shard.body_id.tolist() == list(range(first, first + SMALL_SHARD_SIZE))
        assert shard.masks.shape == (SMALL_SHARD_SIZE, 4, 32, 32)
        np.testing.assert_array_equal(shard.K, intrinsics(32, 32.0))
        assert shard.R_true.shape == (SMALL_SHARD_SIZE, 4, 3, 3)
        assert shard.t_true.shape == (SMALL_SHARD_SIZE, 4, 3)
        assert shard.noise_axis.shape == (SMALL_SHARD_SIZE, 4, 3)
        assert shard.betas.shape == (SMALL_SHARD_SIZE, 10)
        assert shard.pose_body.shape == (SMALL_SHARD_SIZE, 63)
        assert shard.measurements.shape == (SMALL_SHARD_SIZE, 5)
        assert np.abs(shard.betas).max() <= 3.0
        np.testing.assert_array_equal(shard.pose_root, 0.0)


# ---- the content of the bodies -------------------------------------------------------------


def test_bodies_can_be_rebuilt_from_the_seed_and_the_stored_arrays(reference):
    config, data = reference.config, data_directory(reference.out)
    body = StandInBody()
    source = create_pose_source(config)
    pose_filter = PoseFilter.from_config(config, body)
    seed = config["seed"]
    for index, positions in ((0, (0, 9, 31)), (3, (0, 17))):
        shard = read_shard(shard_path(data, index), 32)
        rows = read_manifest(shard_path(data, index).with_suffix(".csv"))
        for position in positions:
            body_id = SMALL_SHARD_SIZE * index + position

            # The shape: standard normal draws of the first stream, clipped.
            betas = np.clip(
                rng_for(seed, GENERATE_STAGE_ID, index, body_id, BETAS_PART).standard_normal(10),
                -3.0,
                3.0,
            )
            np.testing.assert_array_equal(shard.betas[position], betas)

            # The pose: drawn from the second stream until the filter accepts it.
            rng = rng_for(seed, GENERATE_STAGE_ID, index, body_id, POSE_PART)
            rejections = 0
            pose = source.draw(rng)
            while not pose_filter.accept(pose).accepted:
                rejections += 1
                pose = source.draw(rng)
            np.testing.assert_array_equal(shard.pose_root[position], pose.pose_root)
            np.testing.assert_array_equal(shard.pose_body[position], pose.pose_body)
            assert rows[position].pose_rejections == rejections

            # The rig: sampled around the center of the bounding box of the mesh that stands on
            # the floor, from the third stream.
            vertices = body.vertices(betas, pose.pose_root, pose.pose_body)
            vertices = vertices - np.array([0.0, vertices[:, 1].min(), 0.0])
            center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
            rig = sample_rig(
                rng_for(seed, GENERATE_STAGE_ID, index, body_id, RIG_PART), config["camera"], center
            )
            np.testing.assert_array_equal(shard.R_true[position], rig.R_true)
            np.testing.assert_array_equal(shard.t_true[position], rig.t_true)
            np.testing.assert_array_equal(shard.noise_axis[position], rig.noise_axis)
            np.testing.assert_array_equal(rows[position].azimuth_deg, rig.azimuth_deg)

            # The four silhouettes: rendered from the true placement of each camera.
            for view in range(4):
                silhouette = render_silhouette(
                    vertices, body.faces, rig.K, rig.R_true[view], rig.t_true[view], 32
                )
                np.testing.assert_array_equal(shard.masks[position, view], silhouette.mask)

            # The measurements: those of the canonical-pose mesh of the shape, at measure.step_cm.
            canonical_vertices, canonical_joints = canonical_mesh(body, betas)
            measured = measure_mesh(
                canonical_vertices,
                body.faces,
                body.part_ids,
                canonical_joints,
                config["measure"]["step_cm"],
                device="cpu",
            )
            np.testing.assert_array_equal(shard.measurements[position], measured.numpy())


def test_every_camera_looks_at_the_center_of_the_bounding_box_of_the_posed_mesh(reference):
    # An independent check of the aim, built from the stored arrays alone: the center of the
    # axis-aligned bounding box of the posed mesh, stood on the floor, must project to within the
    # look-at jitter of the image center in all four views. The jitter is at most 0.05 m per axis,
    # which is under 2 pixels at the nearest distance of 2.5 m with a focal length of 32 pixels.
    data = data_directory(reference.out)
    body = StandInBody()
    principal = (32 - 1) / 2
    checked = 0
    for index in range(4):
        shard = read_shard(shard_path(data, index), 32)
        for position in range(0, SMALL_SHARD_SIZE, 3):
            vertices = body.vertices(
                shard.betas[position], shard.pose_root[position], shard.pose_body[position]
            )
            vertices = vertices - np.array([0.0, vertices[:, 1].min(), 0.0])
            assert vertices[:, 1].min() == 0.0
            center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
            for view in range(4):
                pixel, depth = project_points(
                    center, shard.K, shard.R_true[position, view], shard.t_true[position, view]
                )
                assert depth > 0.0
                assert np.abs(pixel - principal).max() < 2.0
                checked += 1
    assert checked == 4 * 11 * 4


def test_every_stored_pose_passes_the_filter_and_every_rig_follows_the_configuration(reference):
    config = reference.config
    pose_filter = PoseFilter.from_config(config, StandInBody())
    camera = config["camera"]
    for row in read_manifest(data_directory(reference.out) / MANIFEST_NAME):
        assert pose_filter.accept((row.pose_root, row.pose_body)).accepted
        assert row.pose_root == (0.0, 0.0, 0.0)
        assert all(
            camera["distance_m"][0] <= value <= camera["distance_m"][1] for value in row.distance_m
        )
        assert all(
            camera["height_m"][0] <= value <= camera["height_m"][1] for value in row.height_m
        )
        assert azimuths_are_separated(row.azimuth_deg, camera["min_separation_deg"])
        norms = np.linalg.norm(np.array(row.noise_axis), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_stored_flags_agree_with_the_masks_and_the_measurements(tiny_config_path, tmp_path):
    # At focal_px 48 on 32 pixels the field of view is 37 degrees, and about half of the bodies
    # reach the border of the image in some view. The flags are therefore checked in both
    # directions: a flag is set exactly when its condition holds in the stored masks.
    config = build_config(
        tiny_config_path,
        {"camera.image_size": 32, "camera.focal_px": 48},
        MIXED_OVERRIDES,
    )
    result = generate(config, tmp_path)
    data = data_directory(tmp_path)
    counts = dict.fromkeys(("empty_mask", "out_of_frame", "slice_nan", "any", "none"), 0)
    for index in range(MIXED_SHARDS):
        shard = read_shard(shard_path(data, index), 32)
        rows = read_manifest(shard_path(data, index).with_suffix(".csv"))
        for position in range(len(shard.body_id)):
            masks = shard.masks[position]
            expected = 0
            if not masks.reshape(4, -1).any(axis=1).all():
                expected |= FLAG_BITS["empty_mask"]
            on_border = [
                m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() for m in masks
            ]
            if any(on_border):
                expected |= FLAG_BITS["out_of_frame"]
            if not np.isfinite(shard.measurements[position]).all():
                expected |= FLAG_BITS["slice_nan"]
            assert shard.flags[position] == expected
            assert rows[position].flags == expected
            for name in ("empty_mask", "out_of_frame", "slice_nan"):
                counts[name] += bool(expected & FLAG_BITS[name])
            counts["any" if expected else "none"] += 1

    # The test has power only when both kinds of body occur, and the summary counts the same.
    assert counts["any"] > 0
    assert counts["none"] > 0
    assert counts["any"] + counts["none"] == MIXED_BODIES
    summary = result.summary
    assert summary["n_flagged"] == {name: counts[name] for name in summary["n_flagged"]}
    assert sum(summary["n_unflagged"].values()) == counts["none"]


def test_a_zoomed_camera_flags_the_bodies_that_leave_the_frame(tiny_config_path, tmp_path):
    # At focal_px 400 on 32 pixels the view is under 0.4 m wide at the far distance of 4 m, so no
    # body fits in it, in any pose. A view holds either part of a body that reaches the border
    # of the image, or nothing at all, and the body carries the matching flag.
    config = build_config(
        tiny_config_path,
        {"camera.image_size": 32, "camera.focal_px": 400},
        SHRUNK_OVERRIDES,
        {"data.min_unflagged": 0},
    )
    result = generate(config, tmp_path)
    rows = read_manifest(data_directory(tmp_path) / MANIFEST_NAME)
    assert len(rows) == SHRUNK_BODIES
    assert all(row.flags for row in rows)
    cut_off = sum(1 for row in rows if row.flags & FLAG_BITS["out_of_frame"])
    empty = sum(1 for row in rows if row.flags & FLAG_BITS["empty_mask"])
    assert cut_off > 0
    assert result.summary["n_flagged"]["out_of_frame"] == cut_off
    assert result.summary["n_flagged"]["empty_mask"] == empty
    assert result.summary["n_unflagged"] == {"train": 0, "cal": 0, "test": 0}


# ---- the rejection rate --------------------------------------------------------------------


def test_the_rejection_rate_is_reported_in_the_summary_and_the_result(reference):
    summary = reference.result.summary
    data = data_directory(reference.out)
    rows = read_manifest(data / MANIFEST_NAME)
    rejections = sum(row.pose_rejections for row in rows)
    assert summary["n_rejections"] == rejections
    assert summary["n_draws"] == rejections + SMALL_BODIES
    assert summary["rejection_rate"] == rejections / (rejections + SMALL_BODIES)
    assert reference.result.rejection_rate == summary["rejection_rate"]
    assert sum(entry["n_rejections"] for entry in summary["per_shard"]) == rejections
    # The limits sampler hits the filter in most draws, but accepts one in every few.
    assert 0.0 < summary["rejection_rate"] < 1.0
    assert read_summary(data)["rejection_rate"] == summary["rejection_rate"]


def test_the_rejection_rate_is_logged(shrunk_config, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=generate_module.logger.name):
        result = generate(shrunk_config, tmp_path)
    summary = result.summary
    expected = (
        f"pose rejections: {summary['n_rejections']} of {summary['n_draws']} draws were rejected "
        f"(rate {100.0 * summary['rejection_rate']:.1f}%) for {SHRUNK_BODIES} bodies"
    )
    messages = [record.getMessage() for record in caplog.records]
    assert expected in messages


def test_the_rejection_limit_counts_rejected_draws_not_draws(
    tiny_config_path, small_config, tmp_path
):
    # The most rejections of any body is the limit that just lets every body through.
    free_config = build_config(tiny_config_path, small_config, SHRUNK_OVERRIDES)
    free = generate(free_config, tmp_path / "free")
    rows = read_manifest(data_directory(tmp_path / "free") / MANIFEST_NAME)
    most = max(row.pose_rejections for row in rows)
    assert most > 0
    assert free.completed
    just_enough = build_config(
        tiny_config_path, small_config, SHRUNK_OVERRIDES, {"pose.max_rejections": most}
    )
    assert generate(just_enough, tmp_path / "enough").completed
    too_few = build_config(
        tiny_config_path, small_config, SHRUNK_OVERRIDES, {"pose.max_rejections": most - 1}
    )
    with pytest.raises(GenerationRefusedError, match=f"pose.max_rejections \\({most - 1}\\)"):
        generate(too_few, tmp_path / "short")


def test_a_body_with_too_many_rejected_poses_refuses_with_exit_4_and_names_the_body(
    tiny_config_path, small_config, tmp_path
):
    config = build_config(
        tiny_config_path, small_config, SHRUNK_OVERRIDES, {"pose.max_rejections": 0}
    )
    with pytest.raises(GenerationRefusedError) as refusal:
        generate(config, tmp_path)
    assert refusal.value.exit_code == 4
    assert "body " in str(refusal.value)
    assert "pose.max_rejections (0)" in str(refusal.value)
    assert not (data_directory(tmp_path) / DONE_MARKER_NAME).exists()


# ---- provenance ----------------------------------------------------------------------------


def test_without_a_run_record_the_provenance_comes_from_the_configuration(shrunk_config, tmp_path):
    result = generate_dataset(shrunk_config, tmp_path)
    assert result.summary["config_hash"] == config_hash(shrunk_config)
    assert result.summary["hardware_class"] == "cpu"
    assert result.summary["code_version"].startswith("0.1.0")


def test_a_run_record_of_another_configuration_is_refused(
    shrunk_config, tiny_config_path, tmp_path
):
    other = build_config(tiny_config_path, {"seed": 99})
    with pytest.raises(ValueError, match="belongs to another configuration"):
        generate_dataset(shrunk_config, tmp_path, run_record=fixed_record(other))
    assert not tmp_path.joinpath(DATA_DIRECTORY_NAME).exists()


# ---- resume --------------------------------------------------------------------------------


def test_resume_skips_the_shards_that_are_done_and_makes_the_missing_ones(
    shrunk_config, tmp_path, written_shards
):
    complete = generate(shrunk_config, tmp_path)
    assert written_shards == [shard_name(index, "npz") for index in range(SHRUNK_SHARDS)]
    assert complete.shards_generated == SHRUNK_SHARDS

    # Shard 2 lost its npz and shard 3 its table, so neither counts as done. The marker goes too.
    data = data_directory(tmp_path)
    untouched = {index: shard_path(data, index) for index in (0, 1)}
    stamps = {
        index: (path.stat().st_mtime_ns, path.with_suffix(".csv").stat().st_mtime_ns)
        for index, path in untouched.items()
    }
    expected = snapshot(data)
    shard_path(data, 2).unlink()
    shard_path(data, 3).with_suffix(".csv").unlink()
    (data / DONE_MARKER_NAME).unlink()
    written_shards.clear()

    resumed = generate(shrunk_config, tmp_path, resume=True)
    assert written_shards == [shard_name(2, "npz"), shard_name(3, "npz")]
    assert (resumed.shards_generated, resumed.shards_reused) == (2, 2)
    assert resumed.completed
    assert read_done_marker(data) is not None
    for index, path in untouched.items():
        assert (path.stat().st_mtime_ns, path.with_suffix(".csv").stat().st_mtime_ns) == stamps[
            index
        ]
    assert snapshot(data) == expected
    assert dict(resumed.summary) == dict(complete.summary)


def test_resume_with_every_shard_done_generates_nothing(shrunk_config, tmp_path, written_shards):
    first = generate(shrunk_config, tmp_path)
    expected = snapshot(data_directory(tmp_path))
    written_shards.clear()
    again = generate(shrunk_config, tmp_path, resume=True)
    assert written_shards == []
    assert (again.shards_generated, again.shards_reused) == (0, SHRUNK_SHARDS)
    assert again.completed
    assert read_done_marker(data_directory(tmp_path)) is not None
    assert snapshot(data_directory(tmp_path)) == expected
    assert dict(again.summary) == dict(first.summary)


def test_without_resume_every_shard_is_generated_again(shrunk_config, tmp_path, written_shards):
    generate(shrunk_config, tmp_path)
    data = data_directory(tmp_path)
    expected = snapshot(data)
    written_shards.clear()
    again = generate(shrunk_config, tmp_path)
    assert written_shards == [shard_name(index, "npz") for index in range(SHRUNK_SHARDS)]
    assert (again.shards_generated, again.shards_reused) == (SHRUNK_SHARDS, 0)
    assert snapshot(data) == expected


def test_resume_does_not_reuse_shards_of_another_configuration(
    shrunk_config, tiny_config_path, small_config, tmp_path, written_shards
):
    generate(shrunk_config, tmp_path / "mixed")
    other = build_config(tiny_config_path, small_config, SHRUNK_OVERRIDES, {"seed": 2})
    written_shards.clear()
    result = generate(other, tmp_path / "mixed", resume=True)
    assert written_shards == [shard_name(index, "npz") for index in range(SHRUNK_SHARDS)]
    assert result.shards_reused == 0
    fresh = generate(other, tmp_path / "fresh")
    assert fresh.summary["config_hash"] == result.summary["config_hash"]
    assert_same_data(data_directory(tmp_path / "mixed"), data_directory(tmp_path / "fresh"))


@pytest.mark.parametrize("summary_text", [None, "not json", "[]"])
def test_resume_without_a_readable_configuration_hash_generates_every_shard_again(
    shrunk_config, tmp_path, written_shards, summary_text
):
    generate(shrunk_config, tmp_path)
    data = data_directory(tmp_path)
    expected = snapshot(data)
    if summary_text is None:
        (data / SUMMARY_NAME).unlink()
    else:
        (data / SUMMARY_NAME).write_text(summary_text, encoding="utf-8")
    written_shards.clear()
    result = generate(shrunk_config, tmp_path, resume=True)
    assert written_shards == [shard_name(index, "npz") for index in range(SHRUNK_SHARDS)]
    assert (result.shards_generated, result.shards_reused) == (SHRUNK_SHARDS, 0)
    assert snapshot(data) == expected


def test_a_time_budget_stops_between_shards_and_resume_finishes_the_dataset(
    shrunk_config, tmp_path, monkeypatch
):
    clock = FakeClock()
    with monkeypatch.context() as patch:
        slow_shards(patch, clock, seconds=10.0)  # every shard takes ten seconds of the budget
        partial = generate(
            shrunk_config, tmp_path / "stopped", time_budget=TimeBudget(25.0, clock=clock)
        )
    data = data_directory(tmp_path / "stopped")

    # Two shards fit (0 + 10 + 10 s), and a third would end at 30 s, after the budget.
    assert not partial.completed
    assert (partial.shards_generated, partial.shards_total) == (2, SHRUNK_SHARDS)
    assert partial.summary["n_bodies"] == 6
    assert [entry["shard"] for entry in partial.summary["per_shard"]] == [0, 1]
    assert not (data / DONE_MARKER_NAME).exists()
    assert not (data / MANIFEST_NAME).exists()
    assert list(shard_files(data)) == all_shard_names(2)
    assert read_summary(data)["config_hash"] == config_hash(shrunk_config)

    # Resuming finishes the other two shards, and the result equals an uninterrupted run.
    finished = generate(shrunk_config, tmp_path / "stopped", resume=True)
    assert finished.completed
    assert (finished.shards_generated, finished.shards_reused) == (2, 2)
    generate(shrunk_config, tmp_path / "uninterrupted")
    assert_same_data(data, data_directory(tmp_path / "uninterrupted"))


def test_a_time_budget_that_is_already_spent_generates_no_shard(shrunk_config, tmp_path):
    clock = FakeClock()
    budget = TimeBudget(5.0, clock=clock)
    clock.now = 6.0
    result = generate(shrunk_config, tmp_path, time_budget=budget)
    assert not result.completed
    assert (result.shards_generated, result.summary["n_bodies"]) == (0, 0)
    assert not (data_directory(tmp_path) / DONE_MARKER_NAME).exists()


def test_a_time_budget_with_room_for_every_shard_completes_the_stage(shrunk_config, tmp_path):
    result = generate(shrunk_config, tmp_path, time_budget=TimeBudget(3600.0))
    assert result.completed
    assert read_done_marker(data_directory(tmp_path)) is not None


def test_stale_shards_never_mix_with_the_shards_of_a_new_run(
    shrunk_config, tiny_config_path, small_config, tmp_path, monkeypatch
):
    generate(shrunk_config, tmp_path / "out")
    other = build_config(tiny_config_path, small_config, SHRUNK_OVERRIDES, {"seed": 2})

    # A new run with another seed is stopped after one shard. It must not leave the old shards.
    clock = FakeClock()
    with monkeypatch.context() as patch:
        slow_shards(patch, clock, seconds=10.0)
        partial = generate(other, tmp_path / "out", time_budget=TimeBudget(15.0, clock=clock))
    data = data_directory(tmp_path / "out")
    assert not partial.completed
    assert list(shard_files(data)) == all_shard_names(1)
    assert not (data / MANIFEST_NAME).exists()  # the manifest of the first run is gone as well
    assert not (data / DONE_MARKER_NAME).exists()

    generate(other, tmp_path / "out", resume=True)
    generate(other, tmp_path / "fresh")
    assert_same_data(data, data_directory(tmp_path / "fresh"))


def test_a_shard_with_only_one_of_its_two_files_is_made_again(
    shrunk_config, tmp_path, monkeypatch, written_shards
):
    real_write_manifest = generate_module.write_manifest

    def failing_write_manifest(destination, rows):
        if Path(destination).name == shard_name(1, "csv"):
            raise OSError("simulated failure while writing the shard table")
        return real_write_manifest(destination, rows)

    monkeypatch.setattr(generate_module, "write_manifest", failing_write_manifest)
    with pytest.raises(OSError, match="simulated failure"):
        generate(shrunk_config, tmp_path)
    data = data_directory(tmp_path)
    assert list(shard_files(data)) == [
        shard_name(0, "csv"),
        shard_name(0, "npz"),
        shard_name(1, "npz"),
    ]
    assert not (data / DONE_MARKER_NAME).exists()

    monkeypatch.setattr(generate_module, "write_manifest", real_write_manifest)
    written_shards.clear()
    result = generate(shrunk_config, tmp_path, resume=True)
    assert written_shards == [shard_name(index, "npz") for index in (1, 2, 3)]
    assert result.completed
    generate(shrunk_config, tmp_path / "uninterrupted")
    assert_same_data(data, data_directory(tmp_path / "uninterrupted"))


def test_a_shard_whose_files_are_damaged_is_made_again(shrunk_config, tmp_path, written_shards):
    generate(shrunk_config, tmp_path)
    data = data_directory(tmp_path)
    expected = snapshot(data)

    # Shard 0 lost the second half of its npz, shard 1 lost the last row of its table, shard 2 has
    # the npz of another shard under its name, and shard 3 has a table that is not a table.
    first = shard_path(data, 0)
    first.write_bytes(first.read_bytes()[: first.stat().st_size // 2])
    table = shard_path(data, 1).with_suffix(".csv")
    lines = table.read_text(encoding="utf-8").splitlines(keepends=True)
    table.write_text("".join(lines[:-1]), encoding="utf-8")
    shard_path(data, 2).write_bytes(shard_path(data, 1).read_bytes())
    shard_path(data, 3).with_suffix(".csv").write_text("not a manifest\n", encoding="utf-8")
    written_shards.clear()

    result = generate(shrunk_config, tmp_path, resume=True)
    assert written_shards == [shard_name(index, "npz") for index in range(SHRUNK_SHARDS)]
    assert (result.shards_generated, result.shards_reused) == (SHRUNK_SHARDS, 0)
    assert snapshot(data) == expected


def test_reading_a_damaged_shard_does_not_leave_its_file_open(shrunk_config, tmp_path):
    generate(shrunk_config, tmp_path)
    data = data_directory(tmp_path)
    damaged = [shard_path(data, 0), shard_path(data, 1), shard_path(data, 2)]
    damaged[0].write_bytes(damaged[0].read_bytes()[: damaged[0].stat().st_size // 2])  # truncated
    damaged[1].write_bytes(b"")  # empty
    damaged[2].write_bytes(b"this is not an archive" * 20)  # garbage
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        generate(shrunk_config, tmp_path, resume=True)
        gc.collect()  # a file that is still open gives its warning when it is collected
    leaked = [str(warning.message) for warning in caught if warning.category is ResourceWarning]
    assert leaked == []


def test_a_file_that_is_not_an_npz_archive_does_not_count_as_a_done_shard(
    shrunk_config, tmp_path, written_shards
):
    generate(shrunk_config, tmp_path)
    data = data_directory(tmp_path)
    expected = snapshot(data)
    np.save(data / "shards" / "plain", np.arange(3))  # a plain .npy file written as a shard npz
    (data / "shards" / "plain.npy").replace(shard_path(data, 1))
    shard_path(data, 2).write_bytes(b"")
    written_shards.clear()

    generate(shrunk_config, tmp_path, resume=True)
    assert written_shards == [shard_name(1, "npz"), shard_name(2, "npz")]
    assert snapshot(data) == expected


def test_temporary_files_of_an_interrupted_write_are_removed(shrunk_config, tmp_path):
    generate(shrunk_config, tmp_path)
    data = data_directory(tmp_path)
    leftovers = [
        data / "shards" / ".shard_0001.0123456789abcdef.temporary.npz",
        data / "shards" / ".shard_0002.0123456789abcdef.temporary.csv",
        data / ".summary.0123456789abcdef.temporary.json",
    ]
    for path in leftovers:
        path.write_bytes(b"partial")
    generate(shrunk_config, tmp_path, resume=True)
    assert not any(path.exists() for path in leftovers)


# ---- exit code 4 ---------------------------------------------------------------------------


def test_a_min_unflagged_above_the_unflagged_count_exits_4_and_names_both_counts(
    tiny_config_path, small_config, tmp_path
):
    # Three calibration bodies and three test bodies exist, so a floor of four can never be met.
    config = build_config(
        tiny_config_path, small_config, SHRUNK_OVERRIDES, {"data.min_unflagged": 4}
    )
    with pytest.raises(GenerationRefusedError) as refusal:
        generate(config, tmp_path)
    assert refusal.value.exit_code == STAGE_REFUSED_EXIT_CODE == 4

    # The counts are written, the stage is not done, and the message names both of them.
    data = data_directory(tmp_path)
    summary = read_summary(data)
    cal, test = summary["n_unflagged"]["cal"], summary["n_unflagged"]["test"]
    assert cal <= 3 and test <= 3
    assert summary["min_unflagged"] == 4
    message = str(refusal.value)
    assert f"only {cal} unflagged calibration bodies and {test} unflagged test bodies" in message
    assert "data.min_unflagged requires at least 4" in message
    assert not (data / DONE_MARKER_NAME).exists()
    assert (data / MANIFEST_NAME).is_file()

    # A resumed call finds every shard done and still refuses, so DONE.json cannot appear.
    with pytest.raises(GenerationRefusedError, match="data.min_unflagged requires at least 4"):
        generate(config, tmp_path, resume=True)
    assert not (data / DONE_MARKER_NAME).exists()


def test_the_unflagged_floor_is_inclusive_and_a_refusal_removes_an_earlier_marker(
    tiny_config_path, small_config, tmp_path
):
    easy = build_config(tiny_config_path, small_config, SHRUNK_OVERRIDES, {"data.min_unflagged": 1})
    unflagged = generate(easy, tmp_path / "easy").summary["n_unflagged"]
    lowest = min(unflagged["cal"], unflagged["test"])
    assert lowest >= 1

    at_the_floor = build_config(
        tiny_config_path, small_config, SHRUNK_OVERRIDES, {"data.min_unflagged": lowest}
    )
    result = generate(at_the_floor, tmp_path / "floor")
    assert result.completed
    assert read_done_marker(data_directory(tmp_path / "floor")) is not None

    # One more than the lowest count is refused, and the marker of the run before it is gone.
    above = build_config(
        tiny_config_path, small_config, SHRUNK_OVERRIDES, {"data.min_unflagged": lowest + 1}
    )
    with pytest.raises(GenerationRefusedError):
        generate(above, tmp_path / "floor", resume=True)
    assert not (data_directory(tmp_path / "floor") / DONE_MARKER_NAME).exists()


def test_the_refusal_names_both_counts_when_only_the_test_split_is_short(
    shrunk_config, tmp_path, monkeypatch
):
    real_render = generate_module.render_silhouette
    calls: list[int] = []

    def render_with_test_bodies_out_of_frame(*arguments):
        silhouette = real_render(*arguments)
        calls.append(0)
        body_id = (len(calls) - 1) // 4  # four views per body, rendered in body order
        return dataclasses.replace(
            silhouette, touches_border=silhouette.touches_border or body_id >= 7
        )

    monkeypatch.setattr(generate_module, "render_silhouette", render_with_test_bodies_out_of_frame)
    with pytest.raises(GenerationRefusedError) as refusal:
        generate(shrunk_config, tmp_path)
    cal = read_summary(data_directory(tmp_path))["n_unflagged"]["cal"]
    message = str(refusal.value)
    assert f"only {cal} unflagged calibration bodies and 0 unflagged test bodies remain" in message
    assert "data.min_unflagged requires at least 2" in message
    assert not (data_directory(tmp_path) / DONE_MARKER_NAME).exists()


def test_the_refusal_names_both_counts_when_only_the_calibration_split_is_short(
    shrunk_config, tmp_path, monkeypatch
):
    real_render = generate_module.render_silhouette
    calls: list[int] = []

    def render_with_calibration_bodies_out_of_frame(*arguments):
        silhouette = real_render(*arguments)
        calls.append(0)
        body_id = (len(calls) - 1) // 4
        return dataclasses.replace(
            silhouette, touches_border=silhouette.touches_border or 4 <= body_id <= 6
        )

    monkeypatch.setattr(
        generate_module, "render_silhouette", render_with_calibration_bodies_out_of_frame
    )
    with pytest.raises(GenerationRefusedError) as refusal:
        generate(shrunk_config, tmp_path)
    test = read_summary(data_directory(tmp_path))["n_unflagged"]["test"]
    message = str(refusal.value)
    assert f"only 0 unflagged calibration bodies and {test} unflagged test bodies remain" in message


# ---- flags from the renderer and the measurements ------------------------------------------


def test_flags_follow_the_views_and_the_measurements(
    tiny_config_path, small_config, tmp_path, monkeypatch
):
    config = build_config(
        tiny_config_path, small_config, SHRUNK_OVERRIDES, {"data.min_unflagged": 1}
    )
    empty_views = {(1, 2), (5, 1)}  # (body, view) pairs whose silhouette has no pixel
    border_views = {(4, 0), (5, 3)}  # (body, view) pairs whose silhouette touches the border
    real_measure = generate_module.measure_batch
    renders: list[int] = []
    measured_batches: list[int] = []

    def render_with_chosen_faults(vertices, faces, K, R, t, image_size):
        # Every view that is not faulted gets a clean block in the middle, so that no flag comes
        # from the real silhouettes, and the flags below come from the faults alone.
        body_id, view = divmod(len(renders), 4)
        renders.append(0)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        if (body_id, view) not in empty_views:
            mask[image_size // 4 : image_size // 2, image_size // 4 : image_size // 2] = 1
        if (body_id, view) in border_views:
            mask[0, image_size // 4] = 1
        return silhouette_from_mask(mask)

    def measure_with_a_degenerate_slice(*arguments, **options):
        measurements = real_measure(*arguments, **options)
        measured_batches.append(0)
        if len(measured_batches) == 3:  # the third shard holds bodies 6 to 8
            measurements[1, 1] = float("nan")  # body 7 has no chest slice
        return measurements

    monkeypatch.setattr(generate_module, "render_silhouette", render_with_chosen_faults)
    monkeypatch.setattr(generate_module, "measure_batch", measure_with_a_degenerate_slice)
    result = generate(config, tmp_path)

    data = data_directory(tmp_path)
    rows = read_manifest(data / MANIFEST_NAME)
    flags = {row.body_id: row.flags for row in rows if row.flags}
    assert flags == {
        1: FLAG_BITS["empty_mask"],
        4: FLAG_BITS["out_of_frame"],
        5: FLAG_BITS["empty_mask"] | FLAG_BITS["out_of_frame"],
        7: FLAG_BITS["slice_nan"],
    }
    stored = {}
    for index in range(SHRUNK_SHARDS):
        shard = read_shard(shard_path(data, index), 32)
        pairs = zip(shard.body_id, shard.flags, strict=True)
        stored.update({int(body_id): int(flag) for body_id, flag in pairs if flag})
    assert stored == flags
    assert np.isnan(rows[7].measurements_cm[1])
    summary = result.summary
    assert summary["n_flagged"] == {"empty_mask": 2, "out_of_frame": 2, "slice_nan": 1}
    assert summary["n_unflagged"] == {"train": 3, "cal": 1, "test": 2}
    assert [entry["n_flagged"] for entry in summary["per_shard"]] == [1, 2, 1, 0]


# ---- configurations and assets the stage refuses --------------------------------------------


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ({"body.n_betas": 5}, "body.n_betas"),
        ({"camera.n_cameras": 5}, "camera.n_cameras"),
        (
            {"camera.n_cameras": 3, "train.views_train": [1, 2, 3], "evaluate.views": [1, 3]},
            "camera.n_cameras",
        ),
    ],
)
def test_a_configuration_that_the_shard_format_cannot_hold_exits_2_and_writes_nothing(
    tiny_config_path, small_config, tmp_path, override, key
):
    config = build_config(tiny_config_path, small_config, SHRUNK_OVERRIDES, override)
    with pytest.raises(ConfigError) as refusal:
        generate(config, tmp_path / "out")
    assert refusal.value.key == key
    assert key in str(refusal.value)
    assert not (tmp_path / "out").exists()


def test_a_single_training_body_is_a_configuration_error(tiny_config_path, small_config, tmp_path):
    config = build_config(tiny_config_path, small_config, SHRUNK_OVERRIDES, {"data.n_train": 1})
    with pytest.raises(ConfigError, match="n_train must be at least 2") as refusal:
        generate(config, tmp_path / "out")
    assert refusal.value.key == "data.n_train"
    assert not (tmp_path / "out").exists()


def test_a_configuration_that_does_not_validate_is_a_configuration_error(shrunk_config, tmp_path):
    broken = {**shrunk_config, "seed": -1}
    with pytest.raises(ConfigError):
        generate_dataset(broken, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("override", "key"),
    [({"body.model": "smplx"}, "body.model"), ({"pose.source": "amass"}, "pose.source")],
)
def test_a_missing_licensed_asset_stops_the_stage_before_anything_is_written(
    tiny_config_path, small_config, tmp_path, override, key
):
    config = build_config(tiny_config_path, small_config, SHRUNK_OVERRIDES, override)
    with pytest.raises(MissingAssetError, match=key):
        generate(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_body_model_file_is_looked_up_under_the_configured_asset_root(
    tiny_config_path, small_config, tmp_path
):
    root = tmp_path / "assets"
    config = build_config(
        tiny_config_path,
        small_config,
        SHRUNK_OVERRIDES,
        {"body.model": "smplx", "assets.root": str(root)},
    )
    with pytest.raises(MissingAssetError) as refusal:
        generate(config, tmp_path / "out")
    assert str(root / "smplx" / "SMPLX_NEUTRAL.npz") in str(refusal.value)
    assert "body.model" in str(refusal.value)
    assert not (tmp_path / "out").exists()


class _FakeSmplxModel:
    """The model that a fake smplx.create returns: the geometry of the stand-in mannequin.

    It has the members that SmplxBody reads from a model of the smplx package (the parent table,
    the skinning weights, the faces, and a forward pass), so the stage can drive the SmplxBody class
    without a licensed file.
    """

    def __init__(self) -> None:
        self._body = StandInBody()
        part_ids = self._body.part_ids
        weights = np.zeros((len(part_ids), NUM_JOINTS))
        weights[np.arange(len(part_ids)), part_ids] = 1.0
        self.parents = torch.as_tensor(PARENTS, dtype=torch.long)
        self.lbs_weights = torch.as_tensor(weights, dtype=torch.float64)
        self.faces = self._body.faces

    def __call__(self, *, betas, global_orient, body_pose, **unused):
        arguments = (betas.numpy(), global_orient.numpy(), body_pose.numpy())
        return SimpleNamespace(
            vertices=torch.as_tensor(self._body.vertices(*arguments)),
            joints=torch.as_tensor(self._body.joints(*arguments)),
        )


def test_the_stage_writes_the_same_bytes_through_the_smplx_body_class_as_through_the_stand_in(
    shrunk_config, tiny_config_path, small_config, tmp_path, monkeypatch
):
    fake_package = ModuleType("smplx")
    fake_package.create = lambda *arguments, **options: _FakeSmplxModel()
    monkeypatch.setitem(sys.modules, "smplx", fake_package)
    root = tmp_path / "assets"
    (root / "smplx").mkdir(parents=True)
    (root / "smplx" / "SMPLX_NEUTRAL.npz").write_bytes(b"a placeholder, not a model")
    config = build_config(
        tiny_config_path,
        small_config,
        SHRUNK_OVERRIDES,
        {"body.model": "smplx", "assets.root": str(root)},
    )
    assert isinstance(create_body_model(config), SmplxBody)

    generate(shrunk_config, tmp_path / "standin")
    through_smplx = generate(config, tmp_path / "smplx")
    assert through_smplx.completed
    assert_same_data(data_directory(tmp_path / "standin"), data_directory(tmp_path / "smplx"))


def test_the_body_model_follows_the_configuration(shrunk_config):
    assert isinstance(create_body_model(shrunk_config), StandInBody)
