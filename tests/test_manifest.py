"""Smoke tests for data/manifest.py: manifest rows, shard npz files, packed masks, atomic writes."""

import csv
import dataclasses
import math
import time
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from strike_a_pose.body.base import BODY_POSE_SIZE, ROOT_POSE_SIZE
from strike_a_pose.data import manifest as manifest_module
from strike_a_pose.data.manifest import (
    BETA_COUNT,
    CAMERA_COUNT,
    FLAG_BITS,
    MANIFEST_COLUMNS,
    MANIFEST_NAME,
    MEASUREMENT_NAMES,
    SHARD_KEYS,
    ManifestRow,
    flags_from_text,
    flags_to_text,
    read_manifest,
    read_shard,
    render_manifest,
    shard_path,
    write_manifest,
    write_shard,
)


def make_row(body_id: int, *, split: str = "train", flags: int = 0) -> ManifestRow:
    """Return a row of reproducible random values, given as NumPy arrays on purpose."""
    rng = np.random.default_rng(1000 + body_id)
    return ManifestRow(
        body_id=body_id,
        shard=body_id // 32,
        split=split,
        pose_source="limits",
        pose_rejections=body_id % 4,
        flags=flags,
        betas=rng.standard_normal(BETA_COUNT),
        pose_root=np.zeros(ROOT_POSE_SIZE),
        pose_body=rng.uniform(-1.0, 1.0, BODY_POSE_SIZE),
        azimuth_deg=rng.uniform(0.0, 360.0, CAMERA_COUNT),
        distance_m=rng.uniform(2.5, 4.0, CAMERA_COUNT),
        height_m=rng.uniform(0.8, 1.8, CAMERA_COUNT),
        noise_axis=rng.standard_normal((CAMERA_COUNT, 3)),
        measurements_cm=rng.uniform(20.0, 120.0, len(MEASUREMENT_NAMES)),
    )


def make_shard_arrays(count: int, side: int, seed: int = 7) -> dict[str, np.ndarray]:
    """Return the keyword arguments of write_shard for count bodies with side by side masks."""
    rng = np.random.default_rng(seed)
    return {
        "body_id": np.arange(count, dtype=np.int64) + 100,
        "masks": rng.integers(0, 2, size=(count, CAMERA_COUNT, side, side), dtype=np.uint8),
        "K": np.array([[side, 0.0, side / 2], [0.0, side, side / 2], [0.0, 0.0, 1.0]]),
        "R_true": rng.standard_normal((count, CAMERA_COUNT, 3, 3)),
        "t_true": rng.standard_normal((count, CAMERA_COUNT, 3)),
        "noise_axis": rng.standard_normal((count, CAMERA_COUNT, 3)),
        "betas": rng.standard_normal((count, BETA_COUNT)),
        "pose_root": np.zeros((count, ROOT_POSE_SIZE)),
        "pose_body": rng.uniform(-1.0, 1.0, (count, BODY_POSE_SIZE)),
        "measurements": rng.uniform(20.0, 120.0, (count, len(MEASUREMENT_NAMES))),
        "flags": rng.integers(0, 8, size=count, dtype=np.int64),
    }


def test_header_has_the_contract_columns_in_order():
    assert len(MANIFEST_COLUMNS) == 111
    assert MANIFEST_COLUMNS[:6] == (
        "body_id",
        "shard",
        "split",
        "pose_source",
        "pose_rejections",
        "flags",
    )
    assert MANIFEST_COLUMNS[6] == "betas_0"
    assert MANIFEST_COLUMNS[15] == "betas_9"
    assert MANIFEST_COLUMNS[16:19] == ("pose_root_0", "pose_root_1", "pose_root_2")
    assert MANIFEST_COLUMNS[19] == "pose_body_0"
    assert MANIFEST_COLUMNS[81] == "pose_body_62"
    assert MANIFEST_COLUMNS[82:88] == (
        "cam0_azimuth_deg",
        "cam0_distance_m",
        "cam0_height_m",
        "cam0_noise_axis_x",
        "cam0_noise_axis_y",
        "cam0_noise_axis_z",
    )
    assert MANIFEST_COLUMNS[88] == "cam1_azimuth_deg"
    assert MANIFEST_COLUMNS[105] == "cam3_noise_axis_z"
    assert MANIFEST_COLUMNS[106:] == ("height_cm", "chest_cm", "waist_cm", "hip_cm", "thigh_cm")


def test_round_trip_restores_every_row_and_leaves_one_file(tmp_path):
    rows = [
        make_row(0),
        make_row(1, split="cal", flags=FLAG_BITS["out_of_frame"]),
        make_row(2, split="test", flags=7),
    ]
    destination = tmp_path / "data" / MANIFEST_NAME
    written = write_manifest(destination, rows)
    assert written == destination
    assert read_manifest(destination) == rows
    assert [entry.name for entry in destination.parent.iterdir()] == [MANIFEST_NAME]


def test_a_nan_measurement_round_trips(tmp_path):
    row = dataclasses.replace(
        make_row(3, flags=FLAG_BITS["slice_nan"]),
        measurements_cm=[math.nan, 35.5, 80.25, 95.0, 55.0],
    )
    (back,) = read_manifest(write_manifest(tmp_path / MANIFEST_NAME, [row]))
    assert math.isnan(back.measurements_cm[0])
    assert back.measurements_cm[1:] == (35.5, 80.25, 95.0, 55.0)
    assert back.flags == FLAG_BITS["slice_nan"]


class _SplitName(str, Enum):
    """A split given as a member of a str-based Enum, the pattern of data/splits.py."""

    TRAIN = "train"
    CALIBRATION = "cal"
    TEST = "test"


def test_enum_members_are_written_as_their_values(tmp_path):
    row = dataclasses.replace(make_row(0), split=_SplitName.CALIBRATION)
    text = render_manifest([row])
    cells = dict(zip(MANIFEST_COLUMNS, next(csv.reader([text.splitlines()[1]])), strict=True))
    assert cells["split"] == "cal"
    (back,) = read_manifest(write_manifest(tmp_path / MANIFEST_NAME, [row]))
    assert back.split == "cal"


def test_repeated_writes_give_identical_bytes(tmp_path):
    rows = [make_row(index) for index in range(5)]
    first = write_manifest(tmp_path / "first" / MANIFEST_NAME, rows).read_bytes()
    second = write_manifest(tmp_path / "second" / MANIFEST_NAME, rows).read_bytes()
    assert first == second
    assert render_manifest(rows) == render_manifest(rows)
    assert first.endswith(b"\n")
    assert b"\r" not in first


def test_floats_are_written_as_repr_text_and_stored_as_built_in_floats():
    row = dataclasses.replace(
        make_row(0),
        betas=[np.float64(0.1)] + [0.0] * (BETA_COUNT - 1),
        measurements_cm=[0.1, 1 / 3, 1e-300, -0.0, 123456789.123456789],
    )
    assert type(row.betas[0]) is float
    text = render_manifest([row])
    cells = dict(zip(MANIFEST_COLUMNS, next(csv.reader([text.splitlines()[1]])), strict=True))
    assert cells["betas_0"] == "0.1"
    assert cells["height_cm"] == "0.1"
    assert cells["chest_cm"] == "0.3333333333333333"
    assert cells["waist_cm"] == "1e-300"
    assert cells["hip_cm"] == "-0.0"
    assert cells["thigh_cm"] == repr(123456789.123456789)
    assert "np." not in text


def test_flag_bits_and_their_text():
    assert FLAG_BITS == {"empty_mask": 1, "out_of_frame": 2, "slice_nan": 4}
    assert flags_to_text(0) == ""
    assert flags_to_text(7) == "empty_mask;out_of_frame;slice_nan"
    assert flags_to_text(FLAG_BITS["slice_nan"]) == "slice_nan"
    for bits in range(8):
        assert flags_from_text(flags_to_text(bits)) == bits
    assert flags_from_text("") == 0
    with pytest.raises(ValueError, match="unknown flag 'blurred'"):
        flags_from_text("blurred")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"split": "val"}, "split must be one of"),
        ({"pose_source": "smplx"}, "pose_source must be one of"),
        ({"betas": [0.0] * (BETA_COUNT - 1)}, "betas must have shape"),
        ({"noise_axis": [[0.0, 0.0]] * CAMERA_COUNT}, "noise_axis must have shape"),
        ({"body_id": -1}, "body_id must be at least 0"),
        ({"body_id": 1.5}, "body_id must be an integer"),
        ({"flags": 8}, "flags must be a bit field"),
    ],
)
def test_invalid_rows_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(make_row(0), **changes)


def test_read_refuses_a_file_with_another_header(tmp_path):
    path = tmp_path / MANIFEST_NAME
    path.write_text("body_id,shard\n0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest header"):
        read_manifest(path)


@pytest.mark.parametrize("side", [8, 10])
def test_shard_round_trip_restores_every_array(tmp_path, side):
    arrays = make_shard_arrays(count=3, side=side)
    path = write_shard(shard_path(tmp_path / "data", 0), **arrays)
    back = read_shard(path, image_size=side)
    np.testing.assert_array_equal(back.masks, arrays["masks"])
    assert back.masks.dtype == np.uint8
    assert back.masks.shape == (3, CAMERA_COUNT, side, side)
    for key in SHARD_KEYS:
        if key != "masks":
            np.testing.assert_array_equal(getattr(back, key), arrays[key])
    assert back.body_id.dtype == np.int64
    assert back.flags.dtype == np.int64


def test_shard_file_holds_the_contract_keys_and_packed_masks(tmp_path):
    side = 10
    path = write_shard(tmp_path / "shard_0000.npz", **make_shard_arrays(count=2, side=side))
    with np.load(path) as archive:
        assert sorted(archive.files) == sorted(SHARD_KEYS)
        packed = archive["masks"]
        # One bit per pixel, padded to whole bytes: 100 pixels need 13 bytes.
        assert packed.dtype == np.uint8
        assert packed.shape == (2, CAMERA_COUNT, 13)
        assert archive["K"].shape == (3, 3)
        assert archive["flags"].dtype == np.int64


def test_shard_bytes_do_not_depend_on_the_clock(tmp_path, monkeypatch):
    arrays = make_shard_arrays(count=4, side=8)
    first = write_shard(tmp_path / "first.npz", **arrays).read_bytes()
    monkeypatch.setattr(time, "time", lambda: 2_000_000_000.0)
    second = write_shard(tmp_path / "second.npz", **arrays).read_bytes()
    assert first == second


def test_shard_refuses_bad_masks_and_shapes_and_writes_nothing(tmp_path):
    arrays = make_shard_arrays(count=2, side=8)
    bad_value = arrays["masks"].copy()
    bad_value[0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="only the values 0 and 1"):
        write_shard(tmp_path / "a.npz", **{**arrays, "masks": bad_value})
    with pytest.raises(ValueError, match="masks must have shape"):
        write_shard(tmp_path / "b.npz", **{**arrays, "masks": arrays["masks"][:, :, :6, :]})
    with pytest.raises(ValueError, match="betas must have shape"):
        write_shard(tmp_path / "c.npz", **{**arrays, "betas": arrays["betas"][:, :9]})
    with pytest.raises(ValueError, match="body_id cannot be stored as int64"):
        write_shard(tmp_path / "d.npz", **{**arrays, "body_id": np.array([0.5, 1.5])})
    assert list(tmp_path.iterdir()) == []


def test_read_refuses_a_shard_unpacked_at_the_wrong_size(tmp_path):
    path = write_shard(tmp_path / "shard_0000.npz", **make_shard_arrays(count=2, side=10))
    with pytest.raises(ValueError, match="masks must have shape"):
        read_shard(path, image_size=8)


def test_a_failed_shard_write_keeps_the_previous_file(tmp_path, monkeypatch):
    arrays = make_shard_arrays(count=2, side=8)
    destination = write_shard(tmp_path / "shard_0000.npz", **arrays)
    previous = destination.read_bytes()

    def broken_savez(file, *args, **kwargs):
        # Leave a partial temporary file behind, as a crash midway through the write would.
        Path(file).write_bytes(b"partial")
        raise OSError("simulated failure while writing")

    monkeypatch.setattr(manifest_module.np, "savez", broken_savez)
    with pytest.raises(OSError, match="simulated failure"):
        write_shard(destination, **{**arrays, "flags": arrays["flags"] + 1})
    assert destination.read_bytes() == previous
    assert [entry.name for entry in tmp_path.iterdir()] == ["shard_0000.npz"]


def test_shard_paths_are_zero_padded_by_index(tmp_path):
    assert shard_path(tmp_path, 0) == tmp_path / "shards" / "shard_0000.npz"
    assert shard_path(tmp_path, 12) == tmp_path / "shards" / "shard_0012.npz"
    with pytest.raises(ValueError, match="shard index must be at least 0"):
        shard_path(tmp_path, -1)
