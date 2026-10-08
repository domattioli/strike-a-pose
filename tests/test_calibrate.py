"""Smoke tests for calibrate.py: the quantile rank, the scores, the refusals, and the coverage."""

import csv
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strike_a_pose.calibrate import (
    QUANTILE_COLUMNS,
    STAGE_REFUSED_EXIT_CODE,
    CalibrationQuantile,
    CalibrationRefusedError,
    calibrate_directory,
    calibrate_quantiles,
    cell_id,
    compute_quantiles,
    conformal_quantile,
    normalized_scores,
    quantile_rank,
    read_quantiles,
    smallest_calibration_size,
)
from strike_a_pose.checkpoint import (
    StageStatus,
    check_stage,
    input_hashes,
    read_done_marker,
    write_done_marker,
)
from strike_a_pose.config import ConfigError, config_hash, load_config
from strike_a_pose.runrecord import RunRecord, start_run_record
from strike_a_pose.seeding import rng_for

MEASUREMENTS = ("height", "chest", "waist", "hip", "thigh")
CODE_VERSION = "0.1.0+test"
# The calibration bodies of the small_config: data.n_train is 64 and data.n_cal is 32.
FIRST_BODY = 64
LAST_BODY = 95
QUANTILES_HEADER = "cell_id,views,noise_deg,measurement,n_cal,alpha,q_hat,spread_floor_cm"

# A list of 19 distinct ranks in no particular order. Every measurement of the known-list tests
# sorts the same ranks 1 to 19, so the k-th smallest score of a column is k times its scale.
SHUFFLED_RANKS = [10, 3, 17, 1, 8, 15, 5, 12, 19, 2, 14, 7, 18, 4, 11, 16, 6, 13, 9]
# Powers of two, so that every score of the known-list tests is exact in binary floating point.
SCALES = (1.0, 2.0, 0.5, 4.0, 0.25)


def build_config(tiny_config_path: Path, *override_sets: Mapping[str, object]) -> dict[str, Any]:
    """Return tiny.yaml with the given dotted-key overrides applied; a later set wins a key."""
    merged: dict[str, object] = {}
    for overrides in override_sets:
        merged.update(overrides)
    return load_config(
        tiny_config_path, [f"{key}={json.dumps(value)}" for key, value in merged.items()]
    )


def fixed_record(config: dict[str, Any]) -> RunRecord:
    """Return a run record with fixed provenance, so DONE.json ignores the checkout."""
    return start_run_record(
        config,
        hardware_class="cpu",
        device_name="cpu",
        started_at=datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc),
        code_version=CODE_VERSION,
    )


def predict_fields(
    config: dict[str, Any], views: int, noise: float, count: int = 20
) -> dict[str, np.ndarray]:
    """Return the keys of one calibration predict file, in the layout of contracts/artifacts.md.

    The bodies are the first `count` calibration bodies of the configuration. The spreads are well
    above the spread floor, and the medians are close to the truths.
    """
    rng = rng_for(7, views, int(noise * 10))
    first = config["data"]["n_train"]
    m_median = rng.normal(100.0, 15.0, size=(count, 5))
    return {
        "body_id": np.arange(first, first + count, dtype=np.int64),
        "m_true": m_median + rng.normal(0.0, 2.0, size=(count, 5)),
        "m_median": m_median,
        "spread": rng.uniform(0.5, 2.0, size=(count, 5)),
        "latent_var_mean": rng.uniform(0.1, 0.5, size=count),
        "samples": np.zeros((count, 4, 5), dtype=np.float32),
        "views": np.int64(views),
        "noise_deg": np.float64(noise),
        "split": np.array("cal"),
    }


def known_fields(
    config: dict[str, Any], views: int, noise: float, *, spread: float, unit: float = 1.0
) -> dict[str, np.ndarray]:
    """Return a predict file whose scores are known by hand.

    Measurement j of body i has an error of size SHUFFLED_RANKS[(i - 3 j) mod 19] * SCALES[j] * unit
    with the sign alternating by body, and the constant spread given. Its score is that size over
    the larger of the spread and the floor.
    """
    count = len(SHUFFLED_RANKS)
    ranks = np.stack([np.roll(SHUFFLED_RANKS, 3 * column) for column in range(5)], axis=1)
    signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)[:, np.newaxis]
    fields = predict_fields(config, views, noise, count)
    fields["m_median"] = np.full((count, 5), 100.0)
    fields["m_true"] = 100.0 + signs * ranks * np.array(SCALES) * unit
    fields["spread"] = np.full((count, 5), spread)
    return fields


def save_predict_file(
    out: Path, views: int, noise: float, fields: Mapping[str, Any], *, split_in_name: str = "cal"
) -> Path:
    """Write predict/<split>_v<views>_n<noise>.npz; the file name is spelled out here on purpose."""
    path = out / "predict" / f"{split_in_name}_v{views}_n{noise:g}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fields)
    return path


def write_grid(out: Path, config: dict[str, Any], count: int = 20) -> None:
    """Write a valid calibration predict file for every cell of the configuration."""
    for views in config["evaluate"]["views"]:
        for noise in config["evaluate"]["noise_deg"]:
            save_predict_file(out, views, noise, predict_fields(config, views, noise, count))


@pytest.fixture
def config(tiny_config_path: Path, small_config: dict[str, object]) -> dict[str, Any]:
    """The small_config of contracts/config.md: two cells (1 and 4 views at 0 degrees)."""
    built = build_config(tiny_config_path, small_config)
    assert built["data"]["n_train"] == FIRST_BODY
    assert built["data"]["n_train"] + built["data"]["n_cal"] - 1 == LAST_BODY
    return built


def refusal_of(config: dict[str, Any], out: Path) -> CalibrationRefusedError:
    """Run the stage, expect it to refuse, and return the error."""
    with pytest.raises(CalibrationRefusedError) as caught:
        calibrate_quantiles(config, out, run_record=fixed_record(config))
    return caught.value


def assert_nothing_written(out: Path) -> None:
    """A refused run leaves neither quantiles.csv nor DONE.json."""
    assert not (calibrate_directory(out) / "quantiles.csv").exists()
    assert not (calibrate_directory(out) / "DONE.json").exists()


# --- The quantile index (research R8) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "alpha", "rank"),
    [
        (9, 0.1, 9),  # 10 * 0.9 = 9
        (19, 0.1, 18),  # 20 * 0.9 = 18
        (99, 0.1, 90),  # 100 * 0.9 = 90
        (100, 0.1, 91),  # 101 * 0.9 = 90.9
        (200, 0.1, 181),  # 201 * 0.9 = 180.9
        (2000, 0.1, 1801),  # 2001 * 0.9 = 1800.9
        (7, 0.05, 8),  # 8 * 0.95 = 7.6, above n: no finite quantile exists
        (0, 0.1, 1),  # 1 * 0.9 = 0.9
        # In floating point, 10 * (1 - 0.7) is 3.0000000000000004 and 20 * (1 - 0.7) is
        # 6.000000000000001, which round up to 4 and 7. The exact ranks are 3 and 6.
        (9, 0.7, 3),
        (19, 0.7, 6),
    ],
)
def test_quantile_rank_is_the_ceiling_of_n_plus_one_times_one_minus_alpha(
    n: int, alpha: float, rank: int
) -> None:
    assert quantile_rank(n, alpha) == rank


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.5, 0.7, 0.01])
def test_smallest_calibration_size_is_where_the_rank_first_fits(alpha: float) -> None:
    smallest = smallest_calibration_size(alpha)
    for n in range(0, 300):
        assert (quantile_rank(n, alpha) <= n) == (n >= smallest)


def test_smallest_calibration_size_on_known_values() -> None:
    assert smallest_calibration_size(0.1) == 9
    assert smallest_calibration_size(0.05) == 19
    assert smallest_calibration_size(0.5) == 1


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_quantile_rank_refuses_an_alpha_outside_zero_and_one(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be above 0 and below 1"):
        quantile_rank(10, alpha)


def test_quantile_rank_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="zero or more"):
        quantile_rank(-1, 0.1)


def test_conformal_quantile_is_the_ranked_score_of_a_known_list() -> None:
    scores = np.array(SHUFFLED_RANKS, dtype=np.float64)
    before = scores.copy()
    # n = 19 scores, 1 to 19 in no order: the rank is 18 at alpha 0.1, and 10 at alpha 0.5.
    assert conformal_quantile(scores, 0.1) == 18.0
    assert conformal_quantile(scores, 0.5) == 10.0
    # The k-th smallest of a list with ties is a value of the list, and the input is not sorted.
    assert conformal_quantile([2.0] * 9, 0.1) == 2.0
    assert conformal_quantile([0.5, 0.25, 0.75, 0.25, 0.5, 0.125, 1.0, 0.0, 0.375], 0.1) == 1.0
    np.testing.assert_array_equal(scores, before)


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([], "non-empty"),
        ([[1.0, 2.0], [3.0, 4.0]], "one-dimensional"),
        ([1.0] * 18 + [float("nan")], "finite"),
        ([1.0] * 18 + [float("inf")], "finite"),
        ([1.0] * 8, "at least 9 scores are needed"),
    ],
)
def test_conformal_quantile_refuses_scores_it_cannot_rank(scores: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        conformal_quantile(scores, 0.1)


def test_normalized_scores_divide_by_the_spread_or_by_the_floor() -> None:
    m_true = [10.0, 10.0, 10.0, 10.0]
    m_median = [8.0, 12.0, 10.0, 9.0]
    spread = [0.5, 0.01, 0.0, 2.0]
    # The floor is 0.125 cm: a spread below it is replaced, and a spread above it is kept.
    scores = normalized_scores(m_true, m_median, spread, 0.125)
    np.testing.assert_array_equal(scores, [4.0, 16.0, 0.0, 0.5])


def test_cell_id_writes_the_noise_without_a_trailing_zero() -> None:
    assert cell_id(1, 0.0) == "v1_n0"
    assert cell_id(2, 2.0) == "v2_n2"
    assert cell_id(4, 5.0) == "v4_n5"
    assert cell_id(4, 2.5) == "v4_n2.5"


# --- The stage on known scores ----------------------------------------------------------------


@pytest.mark.parametrize(("alpha", "rank"), [(0.1, 18), (0.2, 16), (0.5, 10), (0.7, 6)])
def test_quantile_index_on_a_known_score_list(
    tiny_config_path: Path,
    small_config: dict[str, object],
    tmp_path: Path,
    alpha: float,
    rank: int,
) -> None:
    config = build_config(tiny_config_path, small_config, {"calibrate.alpha": alpha})
    out = tmp_path / "out"
    # The second cell has twice the spread, so each of its scores is half the first cell's.
    cells = ((1, 1.0), (4, 2.0))
    for views, spread in cells:
        save_predict_file(out, views, 0.0, known_fields(config, views, 0.0, spread=spread))

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    expected = [QUANTILES_HEADER]
    for views, spread in cells:
        for measurement, scale in zip(MEASUREMENTS, SCALES, strict=True):
            q_hat = rank * scale / spread
            expected.append(f"v{views}_n0,{views},0.0,{measurement},19,{alpha!r},{q_hat!r},0.1")
    assert result.quantiles_path.read_bytes() == ("\n".join(expected) + "\n").encode("utf-8")
    # compute_quantiles gives the same rows without writing anything.
    assert compute_quantiles(config, out / "predict") == list(result.quantiles)


def test_the_spread_floor_replaces_a_spread_below_it(
    tiny_config_path: Path, small_config: dict[str, object], tmp_path: Path
) -> None:
    config = build_config(tiny_config_path, small_config, {"calibrate.spread_floor_cm": 0.125})
    out = tmp_path / "out"
    # Errors are multiples of 0.5 cm and the spread is 0.01 cm, so the floor of 0.125 cm divides
    # every score: the 18th smallest is 18 * scale * 0.5 / 0.125.
    for views in (1, 4):
        save_predict_file(out, views, 0.0, known_fields(config, views, 0.0, spread=0.01, unit=0.5))

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    for row in result.quantiles:
        scale = SCALES[MEASUREMENTS.index(row.measurement)]
        assert row.q_hat == 18 * scale * 0.5 / 0.125
        assert row.spread_floor_cm == 0.125


def test_a_cell_with_exactly_the_minimum_of_bodies_uses_the_largest_score(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert config["calibrate"]["min_cal"] == 16
    for views in (1, 4):
        fields = predict_fields(config, views, 0.0, count=16)
        save_predict_file(out, views, 0.0, fields)

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    # n = 16 and alpha = 0.1 give the rank ceil(17 * 0.9) = 16, the largest score.
    fields = predict_fields(config, 1, 0.0, count=16)
    largest = np.max(np.abs(fields["m_true"] - fields["m_median"]) / fields["spread"], axis=0)
    first_cell = [row for row in result.quantiles if row.cell_id == "v1_n0"]
    assert [row.q_hat for row in first_cell] == list(largest)
    assert {row.n_cal for row in result.quantiles} == {16}


def test_the_full_grid_has_45_rows_in_cell_order_then_measurement_order(
    tiny_config_path: Path, tmp_path: Path
) -> None:
    config = build_config(tiny_config_path)
    out = tmp_path / "out"
    write_grid(out, config, count=40)

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    cells = [f"v{views}_n{noise}" for views in (1, 2, 4) for noise in (0, 2, 5)]
    assert [(row.cell_id, row.measurement) for row in result.quantiles] == [
        (cell, measurement) for cell in cells for measurement in MEASUREMENTS
    ]
    assert {row.n_cal for row in result.quantiles} == {40}
    assert len(read_quantiles(result.quantiles_path)) == 45


# --- Names and columns that the evaluate stage reads (contracts/artifacts.md) -----------------


def test_outputs_and_inputs_use_the_names_the_evaluate_stage_reads(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)

    calibrate_quantiles(config, out, run_record=fixed_record(config))

    assert sorted(path.name for path in (out / "predict").iterdir()) == [
        "cal_v1_n0.npz",
        "cal_v4_n0.npz",
    ]
    assert sorted(path.name for path in (out / "calibrate").iterdir()) == [
        "DONE.json",
        "quantiles.csv",
    ]
    with (out / "calibrate" / "quantiles.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == QUANTILES_HEADER.split(",")
    assert tuple(rows[0]) == QUANTILE_COLUMNS
    assert [row["cell_id"] for row in rows] == ["v1_n0"] * 5 + ["v4_n0"] * 5
    assert [row["measurement"] for row in rows] == list(MEASUREMENTS) * 2
    # The nominal level is 1 - alpha: alpha 0.1 is 90% coverage. The floor goes with the scores.
    assert {row["alpha"] for row in rows} == {"0.1"}
    assert {row["spread_floor_cm"] for row in rows} == {"0.1"}
    assert {row["noise_deg"] for row in rows} == {"0.0"}
    assert b"\r" not in (out / "calibrate" / "quantiles.csv").read_bytes()


def test_a_file_of_another_split_or_another_cell_is_never_read(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    other = predict_fields(config, 1, 0.0)
    other["split"] = np.array("test")
    save_predict_file(out, 1, 0.0, other, split_in_name="test")
    save_predict_file(out, 2, 5.0, predict_fields(config, 2, 5.0))  # a cell outside the grid

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    assert {row.cell_id for row in result.quantiles} == {"v1_n0", "v4_n0"}


# --- Refusals (FR-010, FR-011) ----------------------------------------------------------------


def test_a_cell_below_the_minimum_is_refused_and_the_message_names_the_minimum(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    save_predict_file(out, 1, 0.0, predict_fields(config, 1, 0.0, count=20))
    save_predict_file(out, 4, 0.0, predict_fields(config, 4, 0.0, count=15))

    error = refusal_of(config, out)

    message = str(error)
    assert error.exit_code == STAGE_REFUSED_EXIT_CODE == 4
    assert "calibrate.min_cal" in message
    assert "minimum of 16" in message
    assert "15 calibration bodies" in message  # the set size that calibration found
    assert "v4_n0" in message
    assert_nothing_written(out)


def test_the_quantile_needs_more_bodies_than_a_low_minimum_allows(
    tiny_config_path: Path, small_config: dict[str, object], tmp_path: Path
) -> None:
    config = build_config(tiny_config_path, small_config, {"calibrate.min_cal": 1})
    out = tmp_path / "out"
    for views in (1, 4):
        save_predict_file(out, views, 0.0, predict_fields(config, views, 0.0, count=8))

    message = str(refusal_of(config, out))

    # n + 1 >= 1 / alpha: for alpha 0.1 the least is 9 bodies, so 8 are refused.
    assert "n + 1 >= 1 / alpha" in message
    assert "at least 9 bodies" in message
    assert_nothing_written(out)

    for views in (1, 4):
        save_predict_file(out, views, 0.0, predict_fields(config, views, 0.0, count=9))
    result = calibrate_quantiles(config, out, run_record=fixed_record(config))
    assert {row.n_cal for row in result.quantiles} == {9}


@pytest.mark.parametrize("label", ["test", "train", "validation", "CAL", ""])
def test_a_predict_file_whose_split_is_not_cal_is_refused(
    config: dict[str, Any], tmp_path: Path, label: str
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    fields = predict_fields(config, 4, 0.0)
    fields["split"] = np.array(label)
    save_predict_file(out, 4, 0.0, fields)

    error = refusal_of(config, out)

    message = str(error)
    assert error.exit_code == 4
    assert "cal_v4_n0.npz" in message
    assert f"has split {label!r}" in message
    assert "reads only the 'cal' split" in message
    assert_nothing_written(out)


def test_the_split_is_checked_before_any_other_key(config: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    save_predict_file(out, 1, 0.0, {"split": np.array("test")})

    message = str(refusal_of(config, out))

    assert "has split 'test'" in message
    assert "lacks" not in message


def test_a_split_stored_as_bytes_is_read_as_text(config: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    fields = predict_fields(config, 1, 0.0)
    fields["split"] = np.array(b"cal")
    save_predict_file(out, 1, 0.0, fields)
    assert len(calibrate_quantiles(config, out, run_record=fixed_record(config)).quantiles) == 10

    fields["split"] = np.array(b"test")
    save_predict_file(out, 1, 0.0, fields)
    assert "has split 'test'" in str(refusal_of(config, out))


@pytest.mark.parametrize(
    "key", ["split", "views", "noise_deg", "body_id", "m_true", "m_median", "spread"]
)
def test_a_predict_file_without_a_needed_key_is_refused(
    config: dict[str, Any], tmp_path: Path, key: str
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    fields = predict_fields(config, 1, 0.0)
    del fields[key]
    save_predict_file(out, 1, 0.0, fields)

    message = str(refusal_of(config, out))

    assert f"lacks the key {key}" in message
    assert "cal_v1_n0.npz" in message
    assert_nothing_written(out)


def test_a_missing_predict_file_is_refused_and_named(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    (out / "predict" / "cal_v4_n0.npz").unlink()

    message = str(refusal_of(config, out))

    assert "cal_v4_n0.npz" in message
    assert "missing" in message
    assert_nothing_written(out)


def test_a_file_that_is_not_a_readable_npz_archive_is_refused(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    target = out / "predict" / "cal_v1_n0.npz"

    target.write_bytes(b"this is not an archive")
    assert "cannot read the predict file" in str(refusal_of(config, out))

    with target.open("wb") as handle:  # a plain npy array under an npz name
        np.save(handle, np.arange(3))
    assert "not an npz archive" in str(refusal_of(config, out))

    save_predict_file(out, 1, 0.0, predict_fields(config, 1, 0.0))
    data = target.read_bytes()
    target.write_bytes(data[: len(data) // 2])  # a damaged archive
    assert "cannot read the predict file" in str(refusal_of(config, out))
    assert_nothing_written(out)


@pytest.mark.parametrize(("key", "value"), [("views", np.int64(2)), ("noise_deg", np.float64(2.0))])
def test_a_file_that_belongs_to_another_cell_is_refused(
    config: dict[str, Any], tmp_path: Path, key: str, value: np.generic
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    fields = predict_fields(config, 1, 0.0)
    fields[key] = value
    save_predict_file(out, 1, 0.0, fields)

    message = str(refusal_of(config, out))

    assert "cell v1_n0" in message
    assert "cal_v1_n0.npz" in message
    assert_nothing_written(out)


def test_a_noise_level_stored_in_single_precision_still_matches_its_cell(
    tiny_config_path: Path, small_config: dict[str, object], tmp_path: Path
) -> None:
    config = build_config(tiny_config_path, small_config, {"evaluate.noise_deg": [0.1]})
    out = tmp_path / "out"
    for views in (1, 4):
        fields = predict_fields(config, views, 0.1)
        fields["noise_deg"] = np.float32(0.1)  # not equal to the double 0.1
        save_predict_file(out, views, 0.1, fields)

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    assert {row.cell_id for row in result.quantiles} == {"v1_n0.1", "v4_n0.1"}


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("split", np.array(["cal", "cal"]), "must hold one value"),
        ("body_id", np.arange(FIRST_BODY, FIRST_BODY + 20, dtype=np.float64), "list of integers"),
        (
            "body_id",
            np.arange(FIRST_BODY, FIRST_BODY + 20, dtype=np.int64).reshape(4, 5),
            "list of integers",
        ),
        ("m_true", np.zeros((20, 4)), "shape (20, 5)"),
        ("m_median", np.zeros((19, 5)), "shape (20, 5)"),
        ("spread", np.zeros((20,)), "shape (20, 5)"),
        ("spread", np.full((20, 5), "1.0"), "dtype"),
    ],
)
def test_a_value_of_the_wrong_shape_or_type_is_refused(
    config: dict[str, Any], tmp_path: Path, key: str, value: np.ndarray, expected: str
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    fields = predict_fields(config, 1, 0.0)
    fields[key] = value
    save_predict_file(out, 1, 0.0, fields)

    message = str(refusal_of(config, out))

    assert expected in message
    assert_nothing_written(out)


def test_body_ids_outside_the_calibration_bodies_are_refused(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    for body_id in (FIRST_BODY - 1, LAST_BODY + 1):  # the last training body, the first test body
        fields = predict_fields(config, 1, 0.0)
        fields["body_id"][5] = body_id
        save_predict_file(out, 1, 0.0, fields)

        message = str(refusal_of(config, out))

        assert f"body id {body_id} is not a calibration body" in message
        assert f"{FIRST_BODY} to {LAST_BODY}" in message
        assert_nothing_written(out)


def test_a_body_id_that_appears_twice_is_refused(config: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    fields = predict_fields(config, 1, 0.0)
    fields["body_id"][7] = fields["body_id"][6]
    save_predict_file(out, 1, 0.0, fields)

    message = str(refusal_of(config, out))

    assert f"body id {fields['body_id'][6]} appears more than once" in message
    assert_nothing_written(out)


@pytest.mark.parametrize("key", ["m_true", "m_median", "spread"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_value_that_is_not_finite_is_refused_and_names_its_body(
    config: dict[str, Any], tmp_path: Path, key: str, bad: float
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    fields = predict_fields(config, 1, 0.0)
    fields[key][3, 2] = bad
    save_predict_file(out, 1, 0.0, fields)

    message = str(refusal_of(config, out))

    assert f"body {FIRST_BODY + 3} and measurement waist" in message
    assert f"{key} is {bad}" in message
    assert "cal_v1_n0.npz" in message
    assert_nothing_written(out)


def test_a_negative_spread_is_refused(config: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    fields = predict_fields(config, 1, 0.0)
    fields["spread"][2, 1] = -0.5
    save_predict_file(out, 1, 0.0, fields)

    message = str(refusal_of(config, out))

    assert f"spread is -0.5, below 0, for body {FIRST_BODY + 2} and measurement chest" in message


def test_a_zero_spread_needs_a_positive_floor(
    tiny_config_path: Path, small_config: dict[str, object], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    no_floor = build_config(tiny_config_path, small_config, {"calibrate.spread_floor_cm": 0.0})
    write_grid(out, no_floor)
    fields = predict_fields(no_floor, 1, 0.0)
    fields["spread"][4, 4] = 0.0
    save_predict_file(out, 1, 0.0, fields)

    message = str(refusal_of(no_floor, out))

    assert f"spread of body {FIRST_BODY + 4} for measurement thigh is 0" in message
    assert "calibrate.spread_floor_cm" in message

    # With the default floor of 0.1 cm the same file is calibrated, and the floor divides the score.
    floored = build_config(tiny_config_path, small_config)
    assert len(calibrate_quantiles(floored, out, run_record=fixed_record(floored)).quantiles) == 10


def test_a_refusal_removes_the_outputs_of_an_earlier_run(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    calibrate_quantiles(config, out, run_record=fixed_record(config))
    assert (out / "calibrate" / "quantiles.csv").is_file()
    assert (out / "calibrate" / "DONE.json").is_file()

    fields = predict_fields(config, 4, 0.0)
    fields["split"] = np.array("test")
    save_predict_file(out, 4, 0.0, fields)
    refusal_of(config, out)

    assert_nothing_written(out)


def test_a_run_record_of_another_configuration_is_refused_before_anything_changes(
    config: dict[str, Any], tiny_config_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    calibrate_quantiles(config, out, run_record=fixed_record(config))
    other = build_config(tiny_config_path)  # another configuration, so another hash

    with pytest.raises(ValueError, match="belongs to another configuration"):
        calibrate_quantiles(config, out, run_record=fixed_record(other))

    assert (out / "calibrate" / "quantiles.csv").is_file()


def test_a_repeated_view_count_or_noise_level_is_a_configuration_error(
    config: dict[str, Any], tiny_config_path: Path, small_config: dict[str, object], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    calibrate_quantiles(config, out, run_record=fixed_record(config))
    for override in ({"evaluate.views": [1, 1]}, {"evaluate.noise_deg": [0.0, 0.0]}):
        repeated = build_config(tiny_config_path, small_config, override)
        with pytest.raises(ConfigError) as caught:
            calibrate_quantiles(repeated, out, run_record=fixed_record(repeated))
        assert caught.value.key == "evaluate.views"
        assert "v1_n0" in str(caught.value)
        assert "more than once" in str(caught.value)
        # A configuration error is found before any file changes, so the earlier outputs stay.
        assert (out / "calibrate" / "quantiles.csv").is_file()
        assert (out / "calibrate" / "DONE.json").is_file()


def test_a_training_split_too_small_for_the_split_is_a_configuration_error(
    tiny_config_path: Path, small_config: dict[str, object], tmp_path: Path
) -> None:
    config = build_config(tiny_config_path, small_config, {"data.n_train": 1})
    with pytest.raises(ConfigError) as caught:
        calibrate_quantiles(config, tmp_path / "out", run_record=fixed_record(config))
    assert caught.value.key == "data.n_train"


# --- DONE.json, determinism, and the quantiles file -------------------------------------------


def test_done_marker_records_the_run_and_the_predict_input(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    digest = config_hash(config)
    predict_marker = {
        "stage": "predict",
        "config_hash": digest,
        "seed": config["seed"],
        "code_version": CODE_VERSION,
        "hardware_class": "cpu",
        "inputs": {"generate": "a" * 64, "train": "b" * 64},
    }
    write_done_marker(out / "predict", **predict_marker)

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    marker = read_done_marker(result.calibrate_dir)
    assert marker is not None
    assert marker.stage == "calibrate"
    assert marker.config_hash == digest
    assert marker.seed == config["seed"]
    assert marker.code_version == CODE_VERSION
    assert marker.hardware_class == "cpu"
    assert marker.inputs == {"predict": digest}
    predict_inputs = input_hashes({"predict": out / "predict"})
    current = check_stage(
        result.calibrate_dir, stage="calibrate", config_hash=digest, inputs=predict_inputs
    )
    assert current.status is StageStatus.DONE

    # A predict stage that ran again under another configuration makes this stage stale.
    write_done_marker(out / "predict", **{**predict_marker, "config_hash": "c" * 64})
    stale = check_stage(
        result.calibrate_dir,
        stage="calibrate",
        config_hash=digest,
        inputs=input_hashes({"predict": out / "predict"}),
    )
    assert stale.status is StageStatus.STALE
    assert sorted(path.name for path in result.calibrate_dir.iterdir()) == [
        "DONE.json",
        "quantiles.csv",
    ]


def test_without_a_predict_marker_the_input_list_is_empty(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    marker = read_done_marker(result.calibrate_dir)
    assert marker is not None
    assert marker.inputs == {}


def test_without_a_run_record_the_marker_takes_the_device_of_the_configuration(
    config: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    write_grid(out, config)

    result = calibrate_quantiles(config, out)

    marker = read_done_marker(result.calibrate_dir)
    assert marker is not None
    assert marker.hardware_class == "cpu"  # tiny.yaml sets device: cpu
    assert marker.config_hash == config_hash(config)
    assert marker.code_version != ""


def test_equal_inputs_give_equal_bytes(config: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_grid(out, config)
    quantiles = out / "calibrate" / "quantiles.csv"

    first = calibrate_quantiles(config, out, run_record=fixed_record(config))
    first_bytes = quantiles.read_bytes()
    second = calibrate_quantiles(config, out, run_record=fixed_record(config))

    assert quantiles.read_bytes() == first_bytes
    assert first.quantiles == second.quantiles


def test_compute_quantiles_writes_nothing(config: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_grid(out, config)

    rows = compute_quantiles(config, out / "predict")

    assert len(rows) == 10
    assert not (out / "calibrate").exists()


def test_quantiles_read_back_as_they_were_written(config: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_grid(out, config)

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    assert read_quantiles(result.quantiles_path) == list(result.quantiles)


def test_read_quantiles_refuses_a_file_that_breaks_the_contract(tmp_path: Path) -> None:
    good_row = "v1_n0,1,0.0,height,20,0.1,2.5,0.1"
    cases = {
        "wrong header": ("cell_id,views\n" + good_row + "\n", "quantiles header"),
        "short row": (QUANTILES_HEADER + "\nv1_n0,1,0.0,height\n", "line 2 has 4 columns"),
        "text number": (
            QUANTILES_HEADER + "\nv1_n0,one,0.0,height,20,0.1,2.5,0.1\n",
            "line 2",
        ),
        "negative quantile": (
            QUANTILES_HEADER + "\nv1_n0,1,0.0,height,20,0.1,-2.5,0.1\n",
            "q_hat must be a finite number",
        ),
        "not finite": (
            QUANTILES_HEADER + "\nv1_n0,1,0.0,height,20,0.1,nan,0.1\n",
            "q_hat must be a finite number",
        ),
    }
    for name, (text, message) in cases.items():
        path = tmp_path / f"{name.replace(' ', '_')}.csv"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            read_quantiles(path)
    ok = tmp_path / "ok.csv"
    ok.write_text(QUANTILES_HEADER + "\n" + good_row + "\n", encoding="utf-8")
    assert read_quantiles(ok) == [CalibrationQuantile("v1_n0", 1, 0.0, "height", 20, 0.1, 2.5, 0.1)]


@pytest.mark.parametrize(
    "changes",
    [
        {"cell_id": ""},
        {"measurement": ""},
        {"views": 0},
        {"n_cal": 0},
        {"noise_deg": -1.0},
        {"noise_deg": float("nan")},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"q_hat": -0.1},
        {"q_hat": float("inf")},
        {"spread_floor_cm": -0.1},
        {"spread_floor_cm": float("nan")},
    ],
)
def test_a_quantile_row_with_an_invalid_field_is_refused(changes: dict[str, Any]) -> None:
    fields: dict[str, Any] = {
        "cell_id": "v1_n0",
        "views": 1,
        "noise_deg": 0.0,
        "measurement": "height",
        "n_cal": 20,
        "alpha": 0.1,
        "q_hat": 2.5,
        "spread_floor_cm": 0.1,
    }
    CalibrationQuantile(**fields)  # the unchanged row is valid
    with pytest.raises(ValueError):
        CalibrationQuantile(**{**fields, **changes})


# --- Coverage ---------------------------------------------------------------------------------


def test_coverage_on_synthetic_gaussian_data_is_within_two_points_of_ninety_percent(
    tiny_config_path: Path, tmp_path: Path
) -> None:
    n_cal, n_test = 5000, 50_000
    config = build_config(
        tiny_config_path,
        {
            "data.n_cal": n_cal,
            "calibrate.min_cal": 200,
            "evaluate.views": [1],
            "evaluate.noise_deg": [0.0],
        },
    )
    # The model reports its own scale times a factor that is wrong by up to a factor of 3 in
    # either direction. The true error of a body is Gaussian with the body's own scale, so a
    # score is |z| over the factor, and the calibrated quantile must undo the factor.
    factors = np.array([0.5, 0.8, 1.0, 1.5, 3.0])

    def draw(rng: np.random.Generator, count: int) -> tuple[np.ndarray, ...]:
        median = rng.normal(100.0, 15.0, size=(count, 5))
        scale = rng.uniform(0.5, 3.0, size=(count, 5))  # the body's own error scale, in cm
        return median, median + scale * rng.standard_normal((count, 5)), scale * factors

    median, m_true, spread = draw(rng_for(20261007, 0), n_cal)
    assert spread.min() > config["calibrate"]["spread_floor_cm"]  # the floor plays no part here
    fields = predict_fields(config, 1, 0.0, count=n_cal)
    fields.update(m_median=median, m_true=m_true, spread=spread)
    out = tmp_path / "out"
    save_predict_file(out, 1, 0.0, fields)

    result = calibrate_quantiles(config, out, run_record=fixed_record(config))

    q_hat = np.array([row.q_hat for row in result.quantiles])
    assert [row.n_cal for row in result.quantiles] == [n_cal] * 5
    # P(|z| <= 1.6449) = 0.90 for a standard normal z, so q_hat is about 1.6449 over the factor.
    np.testing.assert_allclose(q_hat, 1.6448536 / factors, rtol=0.06)

    median, m_true, spread = draw(rng_for(20261007, 1), n_test)  # new bodies, same process
    covered = np.abs(m_true - median) <= q_hat * spread
    coverage = covered.mean(axis=0)
    assert np.all(np.abs(coverage - 0.90) <= 0.02), coverage
    # With q_hat set to 1 instead, coverage misses 90% by a wide margin for the factors 0.5 and 3.
    uncalibrated = (np.abs(m_true - median) <= spread).mean(axis=0)
    assert abs(uncalibrated[0] - 0.90) > 0.3
    assert abs(uncalibrated[4] - 0.90) > 0.05
