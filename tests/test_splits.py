"""Smoke tests for data/splits.py: contiguous ranges, split_of, disjointness, and monitoring."""

import math
from fractions import Fraction

import numpy as np
import pytest

from strike_a_pose.data.splits import MONITOR_PERCENT, DataSplit, Split, check_disjoint

TRAIN = Split.TRAIN
CAL = Split.CALIBRATION
TEST = Split.TEST


def test_ranges_are_contiguous_in_train_calibration_test_order():
    split = DataSplit(n_train=64, n_cal=32, n_test=16)
    assert split.range_of(TRAIN) == range(0, 64)
    assert split.range_of(CAL) == range(64, 96)
    assert split.range_of(TEST) == range(96, 112)
    assert split.n_bodies == 112


@pytest.mark.parametrize(
    ("body_id", "expected"),
    [(0, TRAIN), (63, TRAIN), (64, CAL), (95, CAL), (96, TEST), (111, TEST)],
)
def test_split_of_places_each_boundary_body_in_the_right_split(body_id, expected):
    assert DataSplit(n_train=64, n_cal=32, n_test=16).split_of(body_id) is expected


def test_split_of_agrees_with_the_ranges_for_every_body_of_the_small_configuration(small_config):
    split = DataSplit(
        n_train=small_config["data.n_train"],
        n_cal=small_config["data.n_cal"],
        n_test=small_config["data.n_test"],
    )
    for name in Split:
        for body_id in split.range_of(name):
            assert split.split_of(body_id) is name
    assert sum(len(split.range_of(name)) for name in Split) == split.n_bodies


@pytest.mark.parametrize("body_id", [-1, 128])
def test_split_of_refuses_a_body_id_outside_the_population(body_id):
    with pytest.raises(ValueError, match="outside 0 to 127"):
        DataSplit(n_train=64, n_cal=32, n_test=32).split_of(body_id)


@pytest.mark.parametrize("body_id", [1.0, "3", True])
def test_split_of_refuses_a_body_id_that_is_not_an_integer(body_id):
    with pytest.raises(TypeError):
        DataSplit(n_train=64, n_cal=32, n_test=32).split_of(body_id)


def test_full_size_splits_are_disjoint_and_cover_every_body_once():
    # The full configuration of contracts/config.md: 20,000 training, 2,500 calibration, 2,500 test.
    split = DataSplit(n_train=20000, n_cal=2500, n_test=2500)
    train, cal, test = (set(split.range_of(name)) for name in (TRAIN, CAL, TEST))
    assert train.isdisjoint(cal) and train.isdisjoint(test) and cal.isdisjoint(test)
    assert train | cal | test == set(range(25000))


@pytest.mark.parametrize(
    "ranges",
    [
        {TRAIN: range(0, 10), CAL: range(10, 12), TEST: range(12, 20)},
        {TRAIN: range(0, 5), CAL: range(5, 5), TEST: range(5, 9)},  # an empty split
        {TEST: range(10, 12), TRAIN: range(0, 10), CAL: range(20, 20)},  # any key order
    ],
)
def test_check_disjoint_accepts_ranges_that_tile_the_body_ids(ranges):
    check_disjoint(ranges)


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ({TRAIN: range(0, 10), CAL: range(9, 12)}, "the cal split starts at body id 9"),
        ({TRAIN: range(0, 10), CAL: range(11, 12)}, "body ids 10 to 10 belong to no split"),
        ({TRAIN: range(1, 10)}, "body ids 0 to 0 belong to no split"),
        ({TRAIN: range(-2, 3)}, "body ids start at 0, but the train split starts at -2"),
        ({TRAIN: range(0, 10, 2)}, "the train split is not a contiguous range"),
        (
            {TRAIN: range(0, 5), CAL: range(5, 10), TEST: range(3, 4)},
            "the test split starts at body id 3, but the train split already holds body id 4",
        ),
    ],
)
def test_check_disjoint_refuses_overlaps_gaps_and_stepped_ranges(ranges, message):
    with pytest.raises(ValueError, match=message):
        check_disjoint(ranges)


def test_empty_calibration_and_test_splits_are_allowed():
    # calibrate refuses a calibration split below calibrate.min_cal (FR-011); the ranges still hold.
    split = DataSplit(n_train=8, n_cal=0, n_test=0)
    assert split.range_of(CAL) == range(8, 8)
    assert split.range_of(TEST) == range(8, 8)
    assert split.split_of(7) is TRAIN


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"n_train": 1}, ValueError, "n_train must be at least 2"),
        ({"n_train": 0}, ValueError, "n_train must be at least 2"),
        ({"n_cal": -1}, ValueError, "n_cal must be non-negative"),
        ({"n_test": 2.0}, TypeError, "n_test must be an integer"),
        ({"n_train": True}, TypeError, "n_train must be an integer"),
    ],
)
def test_refuses_sizes_that_cannot_form_the_splits(overrides, error, message):
    sizes = {"n_train": 64, "n_cal": 32, "n_test": 32, **overrides}
    with pytest.raises(error, match=message):
        DataSplit(**sizes)


def test_sizes_given_as_numpy_integers_are_stored_as_plain_ints():
    split = DataSplit(n_train=np.int64(64), n_cal=np.int32(32), n_test=32)
    assert type(split.n_train) is int
    assert type(split.n_cal) is int
    assert type(split.n_monitor) is int


@pytest.mark.parametrize(
    ("n_train", "expected"),
    [(2, 1), (20, 1), (21, 2), (64, 4), (256, 13), (20000, 1000)],
)
def test_monitor_slice_is_five_percent_of_the_training_range_rounded_up(n_train, expected):
    assert DataSplit(n_train=n_train, n_cal=0, n_test=0).n_monitor == expected


def test_monitor_size_is_the_exact_ceiling_for_every_training_size_up_to_2000():
    # Fraction arithmetic is exact, so its ceiling is the true value to compare against.
    share = Fraction(MONITOR_PERCENT, 100)
    for n_train in range(2, 2001):
        assert DataSplit(n_train=n_train, n_cal=0, n_test=0).n_monitor == math.ceil(share * n_train)


def test_the_training_sampler_never_reaches_the_monitoring_slice():
    split = DataSplit(n_train=256, n_cal=64, n_test=64)
    assert split.n_monitor == 13
    assert split.sampler_range == range(0, 243)
    assert split.monitor_range == range(243, 256)
    assert set(split.sampler_range) | set(split.monitor_range) == set(split.range_of(TRAIN))
    assert set(split.sampler_range).isdisjoint(split.monitor_range)
    # The monitoring bodies keep the split name train in the manifest.
    assert all(split.split_of(body_id) is TRAIN for body_id in split.monitor_range)


def test_split_names_are_the_values_written_to_the_manifest_and_predict_files():
    assert [name.value for name in Split] == ["train", "cal", "test"]
    assert Split("cal") is CAL
    assert DataSplit(n_train=64, n_cal=32, n_test=32).range_of("cal") == range(64, 96)
