"""Smoke tests for evaluate.py: result rows, band flags, clipping, SC-004, and refusals."""

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strike_a_pose.checkpoint import read_done_marker, write_done_marker
from strike_a_pose.config import resolve_config
from strike_a_pose.evaluate import (
    PER_SAMPLE_COLUMNS,
    RESULT_COLUMNS,
    SC004_TOLERANCE,
    EvaluateError,
    cell_identifier,
    is_in_band,
    run_evaluate,
)
from strike_a_pose.runrecord import RunRecord, start_run_record, write_run_record

MEASUREMENTS = ("height", "chest", "waist", "hip", "thigh")
VIEWS = (1, 2, 4)
NOISE = (0.0, 2.0, 5.0)
CELLS = [cell_identifier(views, noise) for views in VIEWS for noise in NOISE]
BODY_COUNT = 100
CODE_VERSION = "0.1.0+abcdef012345"
VERSIONS: dict[str, str | None] = {
    "python": "3.13.16",
    "numpy": "2.5.3",
    "torch": "2.14.1",
    "opencv": "4.14.0.94",
    "smplx": None,
    "sam2": None,
}
# Fused latent variance shrinks as views are added: v4 <= v2 <= v1 for every body.
LATENT_FACTOR = {1: 1.0, 2: 0.6, 4: 0.35}
# With spread 1 cm and quantile 2, every interval is 4 cm wide before clipping.
Q_HAT = 2.0
SPREAD_FLOOR_CM = 0.1
ALPHA = 0.1
N_CAL = 100
CALIBRATE_INPUT_HASH = "b" * 64


def _prepare_run(out: Path, config: dict[str, Any]) -> RunRecord:
    """Write the run record and the calibration outputs of one run, and return the record."""
    record = start_run_record(
        config,
        hardware_class="cpu",
        device_name="cpu (test machine)",
        code_version=CODE_VERSION,
        versions=VERSIONS,
    )
    record.finish()
    write_run_record(record, out / "run_record.json")
    quantiles = out / "calibrate" / "quantiles.csv"
    quantiles.parent.mkdir(parents=True, exist_ok=True)
    with quantiles.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "cell_id",
                "views",
                "noise_deg",
                "measurement",
                "n_cal",
                "alpha",
                "q_hat",
                "spread_floor_cm",
            ]
        )
        for views in VIEWS:
            for noise in NOISE:
                for measurement in MEASUREMENTS:
                    writer.writerow(
                        [
                            cell_identifier(views, noise),
                            views,
                            noise,
                            measurement,
                            N_CAL,
                            ALPHA,
                            Q_HAT,
                            SPREAD_FLOOR_CM,
                        ]
                    )
    write_done_marker(
        out / "calibrate",
        stage="calibrate",
        config_hash=record.config_hash,
        seed=record.seed,
        code_version=record.code_version,
        hardware_class=record.hardware_class,
        inputs={"predict": CALIBRATE_INPUT_HASH},
    )
    return record


def _default_arrays(body_count: int = BODY_COUNT) -> dict[str, np.ndarray]:
    """Return seeded predictions: truths, medians, spreads, and the 1-view latent variance."""
    generator = np.random.default_rng(3)
    m_true = generator.uniform(60.0, 120.0, size=(body_count, len(MEASUREMENTS)))
    return {
        "m_true": m_true,
        "m_median": m_true + generator.normal(0.0, 1.0, size=m_true.shape),
        "spread": generator.uniform(1.0, 3.0, size=m_true.shape),
        "base_latent": generator.uniform(0.02, 0.05, size=body_count),
    }


def _save_predict_file(
    path: Path,
    views: int,
    noise: float,
    arrays: dict[str, np.ndarray],
    *,
    latent: np.ndarray,
    split: str = "test",
) -> None:
    """Write one predict/<split>_<cell>.npz file in the layout of contracts/artifacts.md."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = arrays["m_true"].shape[0]
    np.savez(
        path,
        body_id=np.arange(count, dtype=np.int64),
        m_true=arrays["m_true"],
        m_median=arrays["m_median"],
        spread=arrays["spread"],
        latent_var_mean=latent,
        samples=np.zeros((count, 2, len(MEASUREMENTS)), dtype=np.float32),
        views=np.int64(views),
        noise_deg=np.float64(noise),
        split=np.array(split),
    )


def _write_grid(
    out: Path,
    arrays: dict[str, np.ndarray],
    overrides: dict[tuple[int, float], np.ndarray] | None = None,
) -> None:
    """Write the test-split predict file of every cell, with optional latent variance overrides."""
    overrides = overrides or {}
    for views in VIEWS:
        for noise in NOISE:
            latent = overrides.get((views, noise), arrays["base_latent"] * LATENT_FACTOR[views])
            _save_predict_file(
                out / "predict" / f"test_{cell_identifier(views, noise)}.npz",
                views,
                noise,
                arrays,
                latent=latent,
            )


def _read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return the header and the data rows of a CSV file."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _expected_row(m_true: np.ndarray, m_median: np.ndarray, spread: np.ndarray) -> dict[str, Any]:
    """Recompute one cell and measurement with plain Python loops, for comparison with the table."""
    widths: list[float] = []
    errors: list[float] = []
    covered_count = 0
    clipped_count = 0
    for true, median, width_spread in zip(m_true, m_median, spread, strict=True):
        half_width = Q_HAT * max(float(width_spread), SPREAD_FLOOR_CM)
        lower = float(median) - half_width
        upper = float(median) + half_width
        if lower < 0.0:
            clipped_count += 1
            lower = 0.0
        widths.append(upper - lower)
        if lower <= float(true) <= upper:
            covered_count += 1
        errors.append(float(median) - float(true))
    count = len(widths)
    return {
        "coverage": covered_count / count,
        "median_width_cm": statistics.median(widths),
        "mae_cm": sum(abs(error) for error in errors) / count,
        "mean_signed_error_cm": sum(errors) / count,
        "clipped_count": clipped_count,
        "in_band": 100 * covered_count >= 87 * count and 100 * covered_count <= 93 * count,
    }


def test_results_table_holds_one_row_per_cell_and_measurement(tmp_path):
    config = resolve_config({"seed": 7})
    out = tmp_path / "out"
    record = _prepare_run(out, config)
    _write_grid(out, _default_arrays())

    result = run_evaluate(config, out)

    header, rows = _read_table(out / "evaluate" / "results.csv")
    assert header == list(RESULT_COLUMNS)
    assert len(rows) == 45
    assert len(result.rows) == 45
    assert [(row["cell_id"], row["measurement"]) for row in rows] == [
        (cell, measurement) for cell in CELLS for measurement in MEASUREMENTS
    ]
    for row in rows:
        assert row["seed"] == str(record.seed)
        assert row["config_hash"] == record.config_hash
        assert row["code_version"] == CODE_VERSION
        assert row["hardware_class"] == "cpu"
        assert row["n_test"] == str(BODY_COUNT)
        assert row["n_cal"] == str(N_CAL)
        assert float(row["nominal_level"]) == pytest.approx(0.9)
        assert row["in_band"] in ("true", "false")


def test_metrics_match_a_plain_recomputation(tmp_path):
    config = resolve_config({"seed": 7})
    out = tmp_path / "out"
    _prepare_run(out, config)
    arrays = _default_arrays()
    _write_grid(out, arrays)

    run_evaluate(config, out)

    _, rows = _read_table(out / "evaluate" / "results.csv")
    by_key = {(row["cell_id"], row["measurement"]): row for row in rows}
    for cell in CELLS:
        for column, measurement in enumerate(MEASUREMENTS):
            expected = _expected_row(
                arrays["m_true"][:, column],
                arrays["m_median"][:, column],
                arrays["spread"][:, column],
            )
            row = by_key[(cell, measurement)]
            assert float(row["coverage"]) == pytest.approx(expected["coverage"], rel=1e-12)
            assert float(row["median_width_cm"]) == pytest.approx(
                expected["median_width_cm"], rel=1e-12
            )
            assert float(row["mae_cm"]) == pytest.approx(expected["mae_cm"], rel=1e-12)
            assert float(row["mean_signed_error_cm"]) == pytest.approx(
                expected["mean_signed_error_cm"], rel=1e-9, abs=1e-12
            )
            assert int(row["clipped_count"]) == expected["clipped_count"]
            assert row["in_band"] == ("true" if expected["in_band"] else "false")


def test_per_sample_files_and_done_marker_are_written(tmp_path):
    config = resolve_config({"seed": 7})
    out = tmp_path / "out"
    record = _prepare_run(out, config)
    _write_grid(out, _default_arrays())

    result = run_evaluate(config, out)

    for cell in CELLS:
        header, rows = _read_table(out / "evaluate" / "per_sample" / f"test_{cell}.csv")
        assert header == list(PER_SAMPLE_COLUMNS)
        assert len(rows) == BODY_COUNT * len(MEASUREMENTS)
        assert {row["covered"] for row in rows} <= {"true", "false"}
    assert len(result.per_sample_paths) == len(CELLS)
    marker = read_done_marker(out / "evaluate")
    assert marker is not None
    assert marker.stage == "evaluate"
    assert marker.config_hash == record.config_hash
    assert marker.inputs == {"calibrate": record.config_hash}


@pytest.mark.parametrize(
    ("covered_count", "expected_in_band"),
    [(86, False), (87, True), (93, True), (94, False)],
)
def test_band_boundaries_are_inclusive(tmp_path, covered_count, expected_in_band):
    config = resolve_config({"seed": 7})
    out = tmp_path / "out"
    _prepare_run(out, config)
    m_true = np.full((BODY_COUNT, len(MEASUREMENTS)), 50.0)
    m_true[covered_count:] = 60.0
    arrays = {
        "m_true": m_true,
        "m_median": np.full((BODY_COUNT, len(MEASUREMENTS)), 50.0),
        "spread": np.ones((BODY_COUNT, len(MEASUREMENTS))),
        "base_latent": np.full(BODY_COUNT, 0.03),
    }
    _write_grid(out, arrays)

    run_evaluate(config, out)

    _, rows = _read_table(out / "evaluate" / "results.csv")
    assert {row["in_band"] for row in rows} == {"true" if expected_in_band else "false"}
    for row in rows:
        assert float(row["coverage"]) == pytest.approx(covered_count / BODY_COUNT)


@pytest.mark.parametrize(
    ("covered_count", "count", "expected"),
    [
        (2174, 2500, False),
        (2175, 2500, True),
        (2325, 2500, True),
        (2326, 2500, False),
        (87, 100, True),
        (86, 100, False),
        (93, 100, True),
        (94, 100, False),
    ],
)
def test_is_in_band_compares_integers_at_the_limits(covered_count, count, expected):
    assert is_in_band(covered_count, count) is expected


def test_lower_bound_below_zero_is_clipped_and_counted(tmp_path):
    config = resolve_config({"seed": 7})
    out = tmp_path / "out"
    _prepare_run(out, config)
    m_true = np.full((BODY_COUNT, len(MEASUREMENTS)), 50.0)
    m_median = np.full((BODY_COUNT, len(MEASUREMENTS)), 50.0)
    # Waist of body 0: the median is 0.5 cm, so the lower bound before clipping is -1.5 cm.
    m_true[0, 2] = 0.5
    m_median[0, 2] = 0.5
    arrays = {
        "m_true": m_true,
        "m_median": m_median,
        "spread": np.ones((BODY_COUNT, len(MEASUREMENTS))),
        "base_latent": np.full(BODY_COUNT, 0.03),
    }
    _write_grid(out, arrays)

    run_evaluate(config, out)

    _, rows = _read_table(out / "evaluate" / "results.csv")
    waist_rows = [row for row in rows if row["measurement"] == "waist"]
    assert len(waist_rows) == len(CELLS)
    assert {int(row["clipped_count"]) for row in waist_rows} == {1}
    assert {int(row["clipped_count"]) for row in rows if row["measurement"] != "waist"} == {0}
    # The clipped interval of body 0 is [0, 2.5] (width 2.5); the other 99 bodies are 4.0 wide.
    assert all(float(row["median_width_cm"]) == pytest.approx(4.0) for row in waist_rows)
    _, per_sample = _read_table(out / "evaluate" / "per_sample" / "test_v1_n0.csv")
    body_zero_waist = next(
        row for row in per_sample if row["body_id"] == "0" and row["measurement"] == "waist"
    )
    assert float(body_zero_waist["lower_cm"]) == 0.0
    assert float(body_zero_waist["width_cm"]) == pytest.approx(2.5)
    assert body_zero_waist["clipped"] == "true"
    assert body_zero_waist["covered"] == "true"


def test_one_sc004_violation_is_counted_and_the_tolerance_is_respected(tmp_path):
    config = resolve_config({"seed": 7})
    out = tmp_path / "out"
    _prepare_run(out, config)
    arrays = _default_arrays()
    base = arrays["base_latent"]
    two_view = base * LATENT_FACTOR[2]
    # Body 3 at 0 degrees: two views exceed one view by 2e-5, more than the 1e-5 tolerance.
    two_view[3] = base[3] * (1.0 + 2.0 * SC004_TOLERANCE)
    # Body 5: the excess of 0.5e-5 is inside the tolerance, so it is not a violation.
    two_view[5] = base[5] * (1.0 + 0.5 * SC004_TOLERANCE)
    _write_grid(out, arrays, overrides={(2, 0.0): two_view})

    result = run_evaluate(config, out)

    sc004 = json.loads((out / "evaluate" / "sc004.json").read_text(encoding="utf-8"))
    assert result.sc004_violations == 1
    assert sc004["violations"] == 1
    assert sc004["n_bodies"] == BODY_COUNT
    assert sc004["pairs"] == ["v2_vs_v1", "v4_vs_v2"]
    assert sc004["noise_levels"] == [0.0, 2.0, 5.0]
    assert sc004["tolerance"] == SC004_TOLERANCE
    assert sc004["max_relative_excess"] == pytest.approx(2.0 * SC004_TOLERANCE, rel=1e-6)


def test_a_predict_file_of_the_calibration_split_is_refused(tmp_path):
    config = resolve_config({"seed": 7})
    out = tmp_path / "out"
    _prepare_run(out, config)
    arrays = _default_arrays()
    _write_grid(out, arrays)
    _save_predict_file(
        out / "predict" / "test_v1_n0.npz",
        1,
        0.0,
        arrays,
        latent=arrays["base_latent"],
        split="cal",
    )

    with pytest.raises(EvaluateError, match="split 'cal'"):
        run_evaluate(config, out)
    assert not (out / "evaluate" / "results.csv").exists()


def test_a_non_finite_prediction_is_refused_with_its_body(tmp_path):
    config = resolve_config({"seed": 7})
    out = tmp_path / "out"
    _prepare_run(out, config)
    arrays = _default_arrays()
    arrays["m_median"][4, 1] = np.nan
    _write_grid(out, arrays)

    with pytest.raises(EvaluateError, match="non-finite m_median for body 4"):
        run_evaluate(config, out)


def test_a_run_record_of_another_configuration_is_refused(tmp_path):
    out = tmp_path / "out"
    _prepare_run(out, resolve_config({"seed": 7}))
    _write_grid(out, _default_arrays())

    with pytest.raises(EvaluateError, match="one run only"):
        run_evaluate(resolve_config({"seed": 8}), out)
