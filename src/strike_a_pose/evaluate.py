"""Evaluate the test split of every cell: intervals, coverage, widths, errors, and the SC-004 check.

Each test body's interval is its median prediction plus or minus the calibrated quantile times the
spread, with the lower bound clipped at 0 cm. This is the split conformal construction of research
R8 (Lei, G'Sell, Rinaldo, Tibshirani and Wasserman 2018, https://arxiv.org/abs/1604.04173;
Angelopoulos and Bates 2021, https://arxiv.org/abs/2107.07511). The table columns follow the
ResultCell entity of data-model.md, the file layout follows contracts/artifacts.md, and the band
flag is the inclusive integer comparison of FR-013. The SC-004 check follows spec.md: adding a view
may not raise the mean fused latent variance.
"""

import csv
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from strike_a_pose.checkpoint import (
    atomic_write_text,
    clear_done_marker,
    input_hashes,
    write_done_marker,
)
from strike_a_pose.config import config_hash, resolve_config
from strike_a_pose.runrecord import RunRecord, RunRecordError, read_run_record

__all__ = [
    "BAND_HIGH_PERCENT",
    "BAND_LOW_PERCENT",
    "PER_SAMPLE_COLUMNS",
    "RESULT_COLUMNS",
    "SC004_TOLERANCE",
    "EvaluateError",
    "EvaluationResult",
    "ResultRow",
    "cell_identifier",
    "is_in_band",
    "run_evaluate",
]

# The band of FR-013 around the 90% nominal level, in whole percent (contracts/config.md fixes it).
BAND_LOW_PERCENT = 87
BAND_HIGH_PERCENT = 93
# The relative tolerance of the SC-004 check (contracts/artifacts.md, evaluate/sc004.json).
SC004_TOLERANCE = 1e-5
# The stage name written to DONE.json, and the stage whose marker this stage consumes.
STAGE_NAME = "evaluate"
CALIBRATE_STAGE = "calibrate"
# Two noise levels are the same cell when they differ by less than this many degrees.
_NOISE_MATCH_TOLERANCE = 1e-9

# The columns of evaluate/results.csv, in the order of contracts/artifacts.md.
RESULT_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "views",
    "noise_deg",
    "measurement",
    "nominal_level",
    "n_cal",
    "n_test",
    "coverage",
    "median_width_cm",
    "mae_cm",
    "mean_signed_error_cm",
    "clipped_count",
    "in_band",
    "q_hat",
    "seed",
    "config_hash",
    "code_version",
    "hardware_class",
)
# The columns of evaluate/per_sample/test_<cell>.csv: one row per test body and measurement.
PER_SAMPLE_COLUMNS: tuple[str, ...] = (
    "body_id",
    "measurement",
    "m_true_cm",
    "m_median_cm",
    "spread_cm",
    "q_hat",
    "lower_cm",
    "upper_cm",
    "width_cm",
    "covered",
    "clipped",
)
# The columns of calibrate/quantiles.csv that this stage reads (contracts/artifacts.md).
_QUANTILE_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "views",
    "noise_deg",
    "measurement",
    "n_cal",
    "alpha",
    "q_hat",
    "spread_floor_cm",
)
# The keys of a predict/test_<cell>.npz file (contracts/artifacts.md).
_PREDICT_KEYS: tuple[str, ...] = (
    "body_id",
    "m_true",
    "m_median",
    "spread",
    "latent_var_mean",
    "views",
    "noise_deg",
    "split",
)


class EvaluateError(ValueError):
    """An evaluation input is missing, belongs to another run, or fails a check. The CLI exits 4."""


@dataclass(frozen=True)
class ResultRow:
    """One row of evaluate/results.csv before the run-record columns are added (ResultCell)."""

    cell_id: str
    views: int
    noise_deg: float
    measurement: str
    nominal_level: float
    n_cal: int
    n_test: int
    coverage: float
    median_width_cm: float
    mae_cm: float
    mean_signed_error_cm: float
    clipped_count: int
    in_band: bool
    q_hat: float


@dataclass(frozen=True)
class EvaluationResult:
    """The files one evaluation wrote, its result rows, and its SC-004 violation count."""

    results_path: Path
    sc004_path: Path
    per_sample_paths: tuple[Path, ...]
    rows: tuple[ResultRow, ...]
    sc004_violations: int


@dataclass(frozen=True)
class _Quantile:
    """The calibration values of one cell and measurement, from calibrate/quantiles.csv."""

    n_cal: int
    nominal_level: float
    q_hat: float
    spread_floor_cm: float


@dataclass(frozen=True)
class _Predictions:
    """The test-split predictions of one cell, with the bodies in ascending body id order."""

    body_id: np.ndarray  # shape (n,), integers
    m_true: np.ndarray  # shape (n, measurements), centimetres
    m_median: np.ndarray  # shape (n, measurements), centimetres
    spread: np.ndarray  # shape (n, measurements), centimetres
    latent_variance: np.ndarray  # shape (n,), mean fused latent variance


def cell_identifier(views: int, noise_deg: float) -> str:
    """Return the cell name v<views>_n<noise>, for example v4_n0, used in file names and tables."""
    return f"v{views}_n{noise_deg:g}"


def is_in_band(covered_count: int, n_test: int) -> bool:
    """Return True when covered_count of n_test bodies lies inside the band, both ends inclusive.

    The test runs on integers, 100 * covered_count >= 87 * n_test and 100 * covered_count <=
    93 * n_test, so no rounding of the coverage fraction can move a boundary value out of the band.
    """
    return (
        100 * covered_count >= BAND_LOW_PERCENT * n_test
        and 100 * covered_count <= BAND_HIGH_PERCENT * n_test
    )


def run_evaluate(
    config: Mapping[str, Any], out_directory: str | os.PathLike[str]
) -> EvaluationResult:
    """Evaluate every cell on the test split and write evaluate/ under the output directory.

    config is a resolved configuration, and the run record in the output directory must name its
    configuration hash. The inputs are predict/test_<cell>.npz for each cell, calibrate/
    quantiles.csv, and the DONE.json of calibrate. The outputs are per_sample/, results.csv,
    sc004.json, and DONE.json. Every refused input raises EvaluateError before anything is written.
    """
    resolved = resolve_config(config)
    out = Path(out_directory)
    record = _read_record(out / "run_record.json")
    expected_hash = config_hash(resolved)
    if record.config_hash != expected_hash:
        raise EvaluateError(
            f"the run record in '{out}' names configuration {record.config_hash[:12]}, but the "
            f"configuration given holds {expected_hash[:12]}; evaluate reads the outputs of "
            "one run only"
        )
    calibrate_directory = out / CALIBRATE_STAGE
    inputs = input_hashes({CALIBRATE_STAGE: calibrate_directory})
    if CALIBRATE_STAGE not in inputs:
        raise EvaluateError(
            f"the calibrate stage has no readable DONE.json in '{calibrate_directory}'; "
            "run sap calibrate first"
        )

    settings = resolved["evaluate"]
    measurements = list(settings["measurements"])
    cells = [
        (int(views), float(noise)) for views in settings["views"] for noise in settings["noise_deg"]
    ]
    names = [cell_identifier(views, noise) for views, noise in cells]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise EvaluateError(
            "evaluate.views and evaluate.noise_deg repeat the cell(s) " + ", ".join(repeated)
        )

    quantiles = _read_quantiles(calibrate_directory / "quantiles.csv")
    predictions = {
        key: _read_predictions(
            out / "predict" / f"test_{name}.npz", key[0], key[1], len(measurements)
        )
        for key, name in zip(cells, names, strict=True)
    }
    reference_ids = predictions[cells[0]].body_id
    for key, name in zip(cells, names, strict=True):
        if not np.array_equal(predictions[key].body_id, reference_ids):
            raise EvaluateError(
                f"the predict file of cell {name} holds other test bodies than the first cell; "
                "every cell must evaluate the same test split"
            )

    rows: list[ResultRow] = []
    per_sample_texts: dict[str, str] = {}
    for (views, noise), name in zip(cells, names, strict=True):
        block = predictions[(views, noise)]
        cell_quantiles = [
            _quantile_for(quantiles, name, measurement) for measurement in measurements
        ]
        floors = np.array([quantile.spread_floor_cm for quantile in cell_quantiles])
        q_hats = np.array([quantile.q_hat for quantile in cell_quantiles])
        lower, upper, width, covered, clipped, spread_used = _intervals(block, floors, q_hats)
        n_test = int(block.body_id.shape[0])
        for column, measurement in enumerate(measurements):
            errors = block.m_median[:, column] - block.m_true[:, column]
            covered_count = int(np.count_nonzero(covered[:, column]))
            quantile = cell_quantiles[column]
            rows.append(
                ResultRow(
                    cell_id=name,
                    views=views,
                    noise_deg=noise,
                    measurement=measurement,
                    nominal_level=quantile.nominal_level,
                    n_cal=quantile.n_cal,
                    n_test=n_test,
                    coverage=covered_count / n_test,
                    median_width_cm=float(np.median(width[:, column])),
                    mae_cm=float(np.mean(np.abs(errors))),
                    mean_signed_error_cm=float(np.mean(errors)),
                    clipped_count=int(np.count_nonzero(clipped[:, column])),
                    in_band=is_in_band(covered_count, n_test),
                    q_hat=quantile.q_hat,
                )
            )
        per_sample_texts[name] = _per_sample_csv(
            block, measurements, q_hats, spread_used, lower, upper, width, covered, clipped
        )

    latent = {key: predictions[key].latent_variance for key in cells}
    violations, largest_excess, pair_labels = _sc004(
        latent, settings["views"], settings["noise_deg"]
    )
    sc004_text = _sc004_json(
        record,
        n_bodies=int(reference_ids.shape[0]),
        noise_levels=[float(noise) for noise in settings["noise_deg"]],
        pair_labels=pair_labels,
        violations=violations,
        largest_excess=largest_excess,
    )

    evaluate_directory = out / STAGE_NAME
    clear_done_marker(evaluate_directory)
    per_sample_paths = tuple(
        atomic_write_text(
            evaluate_directory / "per_sample" / f"test_{name}.csv", per_sample_texts[name]
        )
        for name in names
    )
    results_path = atomic_write_text(evaluate_directory / "results.csv", _results_csv(rows, record))
    sc004_path = atomic_write_text(evaluate_directory / "sc004.json", sc004_text)
    write_done_marker(
        evaluate_directory,
        stage=STAGE_NAME,
        config_hash=record.config_hash,
        seed=record.seed,
        code_version=record.code_version,
        hardware_class=record.hardware_class,
        inputs=inputs,
    )
    return EvaluationResult(
        results_path=results_path,
        sc004_path=sc004_path,
        per_sample_paths=per_sample_paths,
        rows=tuple(rows),
        sc004_violations=violations,
    )


def _intervals(
    block: _Predictions, floors: np.ndarray, q_hats: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Return the lower and upper bounds, width, coverage and clip flags, and the spread used.

    The spread is floored at the calibration floor, as in the score of calibrate (research R8). The
    lower bound is clipped at 0 cm after the interval is formed, and the clip flag records it.
    """
    spread_used = np.maximum(block.spread, floors[np.newaxis, :])
    half_width = q_hats[np.newaxis, :] * spread_used
    lower_before_clip = block.m_median - half_width
    upper = block.m_median + half_width
    clipped = lower_before_clip < 0.0
    lower = np.where(clipped, 0.0, lower_before_clip)
    width = upper - lower
    covered = (lower <= block.m_true) & (block.m_true <= upper)
    return lower, upper, width, covered, clipped, spread_used


def _sc004(
    latent: Mapping[tuple[int, float], np.ndarray],
    views: Sequence[int],
    noise_levels: Sequence[float],
) -> tuple[int, float | None, list[str]]:
    """Count the SC-004 violations and return the largest relative excess and the pair labels.

    Consecutive view counts in ascending order form the pairs (fewer, more). A violation is one
    body, one noise level, and one pair whose latent variance with more views exceeds the one with
    fewer views by more than SC004_TOLERANCE times the fewer-view value. The largest relative excess
    is None when there are no pairs.
    """
    pairs = list(pairwise(sorted(views)))
    labels = [f"v{more}_vs_v{fewer}" for fewer, more in pairs]
    violations = 0
    relative_excesses: list[np.ndarray] = []
    for noise in noise_levels:
        for fewer, more in pairs:
            fewer_variance = latent[(fewer, noise)]
            excess = latent[(more, noise)] - fewer_variance
            violations += int(np.count_nonzero(excess > SC004_TOLERANCE * fewer_variance))
            relative_excesses.append(excess / fewer_variance)
    largest = float(np.max(np.concatenate(relative_excesses))) if relative_excesses else None
    return violations, largest, labels


def _sc004_json(
    record: RunRecord,
    *,
    n_bodies: int,
    noise_levels: list[float],
    pair_labels: list[str],
    violations: int,
    largest_excess: float | None,
) -> str:
    """Return the text of evaluate/sc004.json, with the run record fields of the contract."""
    document: dict[str, Any] = {
        "n_bodies": n_bodies,
        "noise_levels": noise_levels,
        "pairs": pair_labels,
        "violations": violations,
        "max_relative_excess": largest_excess,
        "tolerance": SC004_TOLERANCE,
        "config_hash": record.config_hash,
        "seed": record.seed,
        "code_version": record.code_version,
        "hardware_class": record.hardware_class,
    }
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def _results_csv(rows: Sequence[ResultRow], record: RunRecord) -> str:
    """Return evaluate/results.csv, with one row per cell and measurement and the run record."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(RESULT_COLUMNS)
    for row in rows:
        writer.writerow(
            [
                row.cell_id,
                row.views,
                _number(row.noise_deg),
                row.measurement,
                _number(row.nominal_level),
                row.n_cal,
                row.n_test,
                _number(row.coverage),
                _number(row.median_width_cm),
                _number(row.mae_cm),
                _number(row.mean_signed_error_cm),
                row.clipped_count,
                _flag(row.in_band),
                _number(row.q_hat),
                record.seed,
                record.config_hash,
                record.code_version,
                record.hardware_class,
            ]
        )
    return buffer.getvalue()


def _per_sample_csv(
    block: _Predictions,
    measurements: Sequence[str],
    q_hats: np.ndarray,
    spread_used: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    width: np.ndarray,
    covered: np.ndarray,
    clipped: np.ndarray,
) -> str:
    """Return the per-sample CSV of one cell: one row per test body and measurement."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(PER_SAMPLE_COLUMNS)
    for body_index, body in enumerate(block.body_id):
        for column, measurement in enumerate(measurements):
            writer.writerow(
                [
                    int(body),
                    measurement,
                    _number(block.m_true[body_index, column]),
                    _number(block.m_median[body_index, column]),
                    _number(spread_used[body_index, column]),
                    _number(q_hats[column]),
                    _number(lower[body_index, column]),
                    _number(upper[body_index, column]),
                    _number(width[body_index, column]),
                    _flag(bool(covered[body_index, column])),
                    _flag(bool(clipped[body_index, column])),
                ]
            )
    return buffer.getvalue()


def _number(value: float) -> str:
    """Return a finite number as text with the shortest form that reads back to the same value."""
    return repr(float(value))


def _flag(value: bool) -> str:
    """Return a flag as the lowercase words that the report and the verdict read."""
    return "true" if value else "false"


def _read_record(path: Path) -> RunRecord:
    """Read run_record.json, turning a defect in it into an EvaluateError."""
    try:
        return read_run_record(path)
    except RunRecordError as error:
        raise EvaluateError(str(error)) from error


def _read_text(path: Path, label: str) -> str:
    """Return the text of a file this stage reads, refusing one that is missing or unreadable."""
    if not path.is_file():
        raise EvaluateError(f"the {label} '{path}' is missing")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise EvaluateError(f"cannot read the {label} '{path}': {error}") from error


def _text_field(row: Mapping[str, Any], column: str, where: str) -> str:
    """Return one non-empty text field of a CSV row."""
    value = row.get(column)
    if value is None or value.strip() == "":
        raise EvaluateError(f"{where} has no value in the column '{column}'")
    return value.strip()


def _number_field(row: Mapping[str, Any], column: str, where: str) -> float:
    """Return one finite number field of a CSV row."""
    text = _text_field(row, column, where)
    try:
        number = float(text)
    except ValueError as error:
        raise EvaluateError(
            f"{where}: the column '{column}' must be a number; got {text!r}"
        ) from error
    if not math.isfinite(number):
        raise EvaluateError(f"{where}: the column '{column}' must be finite; got {text!r}")
    return number


def _integer_field(row: Mapping[str, Any], column: str, where: str) -> int:
    """Return one whole-number field of a CSV row."""
    text = _text_field(row, column, where)
    try:
        return int(text)
    except ValueError as error:
        raise EvaluateError(
            f"{where}: the column '{column}' must be a whole number; got {text!r}"
        ) from error


def _read_quantiles(path: Path) -> dict[tuple[str, str], _Quantile]:
    """Read calibrate/quantiles.csv into one entry per cell name and measurement."""
    reader = csv.DictReader(io.StringIO(_read_text(path, "calibration quantiles file")))
    header = reader.fieldnames or []
    missing = [column for column in _QUANTILE_COLUMNS if column not in header]
    if missing:
        raise EvaluateError(
            f"the calibration quantiles file '{path}' lacks the column(s) {', '.join(missing)}"
        )
    quantiles: dict[tuple[str, str], _Quantile] = {}
    for number, row in enumerate(reader, start=2):
        where = f"row {number} of '{path}'"
        key = (_text_field(row, "cell_id", where), _text_field(row, "measurement", where))
        if key in quantiles:
            raise EvaluateError(f"{where} repeats the cell {key[0]} and measurement {key[1]}")
        quantiles[key] = _Quantile(
            n_cal=_integer_field(row, "n_cal", where),
            nominal_level=1.0 - _number_field(row, "alpha", where),
            q_hat=_number_field(row, "q_hat", where),
            spread_floor_cm=_number_field(row, "spread_floor_cm", where),
        )
    if not quantiles:
        raise EvaluateError(f"the calibration quantiles file '{path}' has no data rows")
    return quantiles


def _quantile_for(
    quantiles: Mapping[tuple[str, str], _Quantile], cell: str, measurement: str
) -> _Quantile:
    """Return the calibration values of one cell and measurement, or refuse when they are absent."""
    try:
        return quantiles[(cell, measurement)]
    except KeyError:
        raise EvaluateError(
            f"calibrate/quantiles.csv lacks the cell {cell} and measurement {measurement}; "
            "run sap calibrate for this configuration"
        ) from None


def _scalar(array: np.ndarray, key: str, path: Path) -> Any:
    """Return the one value a predict file holds under key, as a Python value."""
    if array.size != 1:
        raise EvaluateError(f"the key '{key}' of the predict file '{path}' must hold one value")
    value = array.reshape(()).item()
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _finite_floats(name: str, array: np.ndarray, body_id: np.ndarray, path: Path) -> np.ndarray:
    """Return a predict array as float64, refusing any non-finite value and naming its body."""
    try:
        values = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise EvaluateError(
            f"the key '{name}' of the predict file '{path}' must hold numbers"
        ) from error
    bad = np.argwhere(~np.isfinite(values))
    if bad.size:
        row = int(bad[0][0])
        raise EvaluateError(
            f"the predict file '{path}' holds a non-finite {name} for body {int(body_id[row])}; "
            "a non-finite prediction is refused"
        )
    return values


def _read_predictions(
    path: Path, views: int, noise_deg: float, measurement_count: int
) -> _Predictions:
    """Read the test-split predictions of one cell, refusing a file of any other split or cell."""
    if not path.is_file():
        raise EvaluateError(f"the predict file '{path}' is missing; run sap predict first")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name]) for name in _PREDICT_KEYS if name in archive.files
            }
    except (OSError, ValueError) as error:
        raise EvaluateError(f"cannot read the predict file '{path}': {error}") from error
    missing = [name for name in _PREDICT_KEYS if name not in arrays]
    if missing:
        raise EvaluateError(f"the predict file '{path}' lacks the key(s) {', '.join(missing)}")

    split = _scalar(arrays["split"], "split", path)
    if split != "test":
        raise EvaluateError(
            f"the predict file '{path}' holds split '{split}'; evaluate reads only the test split"
        )
    stored_views = _scalar(arrays["views"], "views", path)
    stored_noise = _scalar(arrays["noise_deg"], "noise_deg", path)
    if int(stored_views) != views or not math.isclose(
        float(stored_noise), noise_deg, rel_tol=0.0, abs_tol=_NOISE_MATCH_TOLERANCE
    ):
        raise EvaluateError(
            f"the predict file '{path}' holds views {stored_views} and noise {stored_noise}, "
            f"but its cell is {cell_identifier(views, noise_deg)}"
        )

    body_id = arrays["body_id"]
    if body_id.ndim != 1 or body_id.shape[0] == 0 or not np.issubdtype(body_id.dtype, np.integer):
        raise EvaluateError(
            f"the key 'body_id' of the predict file '{path}' must be a non-empty list of integers"
        )
    count = int(body_id.shape[0])
    expected_shapes = {
        "m_true": (count, measurement_count),
        "m_median": (count, measurement_count),
        "spread": (count, measurement_count),
        "latent_var_mean": (count,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise EvaluateError(
                f"the key '{name}' of the predict file '{path}' must have shape {shape}; "
                f"it has shape {arrays[name].shape}"
            )
    m_true = _finite_floats("m_true", arrays["m_true"], body_id, path)
    m_median = _finite_floats("m_median", arrays["m_median"], body_id, path)
    spread = _finite_floats("spread", arrays["spread"], body_id, path)
    latent = _finite_floats("latent_var_mean", arrays["latent_var_mean"], body_id, path)
    not_positive = np.flatnonzero(latent <= 0.0)
    if not_positive.size:
        row = int(not_positive[0])
        raise EvaluateError(
            f"the predict file '{path}' holds a latent_var_mean of {latent[row]!r} for body "
            f"{int(body_id[row])}; a posterior variance must be positive"
        )

    order = np.argsort(body_id, kind="stable")
    sorted_ids = body_id[order].astype(np.int64)
    if sorted_ids.shape[0] > 1 and np.any(np.diff(sorted_ids) == 0):
        raise EvaluateError(f"the predict file '{path}' lists a body id more than once")
    return _Predictions(
        body_id=sorted_ids,
        m_true=m_true[order],
        m_median=m_median[order],
        spread=spread[order],
        latent_variance=latent[order],
    )
