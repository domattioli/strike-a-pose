"""Split conformal calibration per cell and measurement: quantiles.csv and DONE.json.

This is the calibrate stage of specs/001-kill-test-mvp (FR-010, FR-011, FR-029; research R8). It
reads ``<out>/predict/cal_<cell_id>.npz`` for every cell of ``evaluate.views`` by
``evaluate.noise_deg``, and writes ``<out>/calibrate/quantiles.csv`` and then
``<out>/calibrate/DONE.json`` (contracts/artifacts.md). A cell is named ``v{views}_n{noise_deg}``
with the noise written in the ``g`` format, so 4 views at 2.0 degrees is ``v4_n2`` and the predict
file of that cell is ``cal_v4_n2.npz`` (data-model.md, ExperimentCondition).

The method is split conformal prediction with normalized residual scores (research R8). For one
cell and one measurement, a calibration body has the score

    score = |m_true - m_median| / max(spread, calibrate.spread_floor_cm)

so that the interval width follows the model's own spread. With n scores and the nominal
miscoverage alpha (``calibrate.alpha``), ``q_hat`` is the k-th smallest score, where
k = ceil((n + 1)(1 - alpha)). The interval ``m_median -/+ q_hat * max(spread, floor)`` then covers
the true value of a new body with probability at least 1 - alpha, for any score distribution, when
the calibration bodies and the new body are exchangeable. The rank is computed in exact arithmetic
from the decimal text of alpha, so that 0.7 does not carry the rounding error of its binary form:
for n = 9 and alpha = 0.7, ``10 * (1 - 0.7)`` is 3.0000000000000004 in floating point and would
round up to rank 4, where the exact rank is 3.

Calibration set. A row of a predict file is one unflagged calibration body, because the predict
stage skips flagged bodies (data-model.md, Body flags). The number of rows is therefore ``n_cal``
of that cell, and it is written to ``quantiles.csv`` next to ``q_hat``, which reports the set sizes
that calibration used (FR-011). Calibration and test bodies are reused across the cells, so each
cell is calibrated on its own scores (FR-010).

The stage refuses with ``CalibrationRefusedError`` (exit code 4 of contracts/cli.md), before it
writes anything, when:

* a predict file is missing, unreadable, or lacks a key it needs;
* the file's ``split`` is not ``cal``, because calibrating on any other split would leak held-out
  bodies into the calibration (FR-010; calibrate refuses any value but ``cal``);
* the file belongs to another cell than its name says, or holds a body id outside the calibration
  range of the configuration, or the same body id twice;
* a cell has fewer than ``calibrate.min_cal`` bodies (FR-011), or too few for the quantile to
  exist: the rank k is above n unless n + 1 is at least 1 / alpha;
* a measurement, a median, or a spread is not finite, or a spread is negative, or the divisor of
  a score would be zero. The message names the body, the measurement, and the file.

The stage clears an earlier ``DONE.json`` and ``quantiles.csv`` before it reads a predict file, so
a refusal or a crash never leaves outputs that look current. Calibration takes milliseconds, so the
stage has no checkpoints inside it. A resumed run skips the whole stage when
``checkpoint.check_stage`` finds its ``DONE.json`` current.

Public sources (research R8): split conformal prediction with locally weighted residual scores,
Lei, G'Sell, Rinaldo, Tibshirani, and Wasserman, "Distribution-Free Predictive Inference for
Regression" (2018, https://arxiv.org/abs/1604.04173); and the finite-sample quantile
ceil((n + 1)(1 - alpha)) / n in Angelopoulos and Bates, "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification" (2021, https://arxiv.org/abs/2107.07511).
"""

import csv
import io
import logging
import math
import operator
import os
import zipfile
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import ArrayLike, NDArray

from strike_a_pose.checkpoint import (
    atomic_path,
    clear_done_marker,
    input_hashes,
    write_done_marker,
)
from strike_a_pose.config import ConfigError, config_hash
from strike_a_pose.data.manifest import format_float
from strike_a_pose.data.splits import DataSplit, Split
from strike_a_pose.device import select_device
from strike_a_pose.runrecord import RunRecord, start_run_record

__all__ = [
    "CALIBRATE_DIRECTORY_NAME",
    "PREDICT_DIRECTORY_NAME",
    "QUANTILES_NAME",
    "QUANTILE_COLUMNS",
    "STAGE_NAME",
    "STAGE_REFUSED_EXIT_CODE",
    "CalibrationQuantile",
    "CalibrationRefusedError",
    "CalibrationResult",
    "calibrate_directory",
    "calibrate_quantiles",
    "cell_id",
    "compute_quantiles",
    "conformal_quantile",
    "normalized_scores",
    "quantile_rank",
    "read_quantiles",
    "smallest_calibration_size",
]

logger = logging.getLogger(__name__)

# The stage name in run_record.json and DONE.json, and the folders of this stage and of its input
# under --out (contracts/artifacts.md).
STAGE_NAME = "calibrate"
CALIBRATE_DIRECTORY_NAME = "calibrate"
PREDICT_DIRECTORY_NAME = "predict"
QUANTILES_NAME = "quantiles.csv"
# The exit code of contracts/cli.md for a stage that refuses to go on.
STAGE_REFUSED_EXIT_CODE = 4

# The columns of quantiles.csv, in the order of contracts/artifacts.md.
QUANTILE_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "views",
    "noise_deg",
    "measurement",
    "n_cal",
    "alpha",
    "q_hat",
    "spread_floor_cm",
)

# The split that calibration reads, as the predict files and the manifest write it.
_CALIBRATION_SPLIT = Split.CALIBRATION.value
# The keys of a predict file that calibration reads. The others (latent_var_mean and samples) are
# for the evaluate stage and for recomputation.
_FIELD_NAMES = ("split", "views", "noise_deg", "body_id", "m_true", "m_median", "spread")
_VALUE_FIELDS = ("m_true", "m_median", "spread")
# Noise levels in a predict file and in the configuration come from one number, so they agree
# closely. The tolerance only absorbs a file that stored the noise level in single precision.
_NOISE_RELATIVE_TOLERANCE = 1e-6
_NOISE_ABSOLUTE_TOLERANCE = 1e-9


class CalibrationRefusedError(RuntimeError):
    """The stage refuses to go on, which the sap command reports with exit code 4.

    The causes are listed in the module docstring. The message names the file, the cell, the body,
    or the limit, and no quantiles.csv or DONE.json is left behind.
    """

    exit_code = STAGE_REFUSED_EXIT_CODE


@dataclass(frozen=True)
class CalibrationQuantile:
    """One row of quantiles.csv: the conformal quantile of one cell and one measurement.

    The fields are the columns of contracts/artifacts.md (CalibrationQuantile in data-model.md).
    ``n_cal`` counts the unflagged calibration bodies of the cell. The evaluate stage builds the
    interval ``m_median -/+ q_hat * max(spread, spread_floor_cm)`` from this row.
    """

    cell_id: str
    views: int
    noise_deg: float
    measurement: str
    n_cal: int
    alpha: float
    q_hat: float
    spread_floor_cm: float

    def __post_init__(self) -> None:
        if not (self.cell_id and self.measurement):
            raise ValueError("a calibration quantile needs a cell id and a measurement name")
        if self.views < 1 or self.n_cal < 1:
            raise ValueError(
                f"views and n_cal must be at least 1; got views {self.views}, n_cal {self.n_cal}"
            )
        if not (math.isfinite(self.noise_deg) and self.noise_deg >= 0):
            raise ValueError(
                f"noise_deg must be a finite number of 0 or more; got {self.noise_deg}"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be above 0 and below 1; got {self.alpha}")
        if not (math.isfinite(self.q_hat) and self.q_hat >= 0):
            raise ValueError(f"q_hat must be a finite number of 0 or more; got {self.q_hat}")
        if not (math.isfinite(self.spread_floor_cm) and self.spread_floor_cm >= 0):
            raise ValueError(
                f"spread_floor_cm must be a finite number of 0 or more; got {self.spread_floor_cm}"
            )


@dataclass(frozen=True, eq=False)
class CalibrationResult:
    """What a call of ``calibrate_quantiles`` wrote: the folder, the file, and the rows."""

    calibrate_dir: Path
    quantiles_path: Path
    quantiles: tuple[CalibrationQuantile, ...]


@dataclass(frozen=True)
class _Settings:
    """The configuration values that calibration reads, taken from the resolved configuration."""

    cells: tuple[tuple[int, float], ...]  # (views, noise_deg), views first, then noise
    measurements: tuple[str, ...]
    alpha: float
    min_cal: int
    spread_floor_cm: float
    calibration_ids: range
    n_train: int
    n_cal: int


def cell_id(views: int, noise_deg: float) -> str:
    """Return the cell name ``v{views}_n{noise_deg}``, with the noise in ``g`` format.

    The format drops a trailing ``.0``, so the cells of the default grid are ``v1_n0``, ``v2_n2``,
    and ``v4_n5``, and 2.5 degrees is ``n2.5``. The predict files are named
    ``<split>_<cell_id>.npz``, and evaluate/results.csv uses the same cell names.
    """
    return f"v{views}_n{noise_deg:g}"


def calibrate_directory(out_dir: str | os.PathLike[str]) -> Path:
    """Return the folder of this stage, ``<out>/calibrate``."""
    return Path(out_dir) / CALIBRATE_DIRECTORY_NAME


def _decimal_alpha(alpha: float) -> Fraction:
    """Return alpha as the exact decimal that repr prints, for example 0.1 as 1/10."""
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError(f"alpha must be above 0 and below 1; got {alpha!r}")
    return Fraction(repr(value))


def quantile_rank(n: int, alpha: float) -> int:
    """Return k = ceil((n + 1)(1 - alpha)), the 1-based rank of the quantile among n scores.

    The product is exact (see the module docstring). A rank above n means that no finite quantile
    exists for this n, which happens when n + 1 is below 1 / alpha.
    """
    count = operator.index(n)
    if count < 0:
        raise ValueError(f"the number of scores must be zero or more; got {n}")
    return math.ceil((count + 1) * (1 - _decimal_alpha(alpha)))


def smallest_calibration_size(alpha: float) -> int:
    """Return the fewest scores for which the conformal quantile exists: ceil(1 / alpha) - 1.

    The rank k = ceil((n + 1)(1 - alpha)) is at most n exactly when n + 1 >= 1 / alpha. For alpha
    of 0.10 that is 9 scores.
    """
    return math.ceil(1 / _decimal_alpha(alpha)) - 1


def normalized_scores(
    m_true: ArrayLike, m_median: ArrayLike, spread: ArrayLike, spread_floor_cm: float
) -> NDArray[np.float64]:
    """Return ``|m_true - m_median| / max(spread, spread_floor_cm)`` elementwise (research R8).

    The inputs broadcast against each other. They must be finite, and the divisor must be above
    zero; the stage checks both before it calls this function.
    """
    error = np.abs(np.asarray(m_true, dtype=np.float64) - np.asarray(m_median, dtype=np.float64))
    return error / np.maximum(np.asarray(spread, dtype=np.float64), spread_floor_cm)


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Return the k-th smallest score, k = ``quantile_rank(n, alpha)`` (research R8).

    Raises ValueError for scores that are not a finite, non-empty, one-dimensional list, and for an
    n that is below ``smallest_calibration_size(alpha)``, where the quantile does not exist.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("the scores must be a non-empty, one-dimensional list of finite numbers")
    rank = quantile_rank(values.size, alpha)
    if rank > values.size:
        raise ValueError(
            f"{values.size} scores are too few for alpha {alpha}: the rank {rank} is above "
            f"{values.size}; at least {smallest_calibration_size(alpha)} scores are needed"
        )
    return float(np.sort(values)[rank - 1])


def compute_quantiles(
    config: Mapping[str, Any], predict_dir: str | os.PathLike[str]
) -> list[CalibrationQuantile]:
    """Return the quantile of every cell and measurement, in cell order then measurement order.

    This reads the calibration predict files of ``predict_dir`` and writes nothing, so a
    recomputation (sap verify) can call it too. The cell order is each view count of
    ``evaluate.views`` with every noise level of ``evaluate.noise_deg``. Raises
    CalibrationRefusedError for the causes listed in the module docstring. The configuration must
    be a resolved one, as ``config.load_config`` returns.
    """
    settings = _settings(config)
    rows: list[CalibrationQuantile] = []
    for views, noise_deg in settings.cells:
        name = cell_id(views, noise_deg)
        path = Path(predict_dir) / f"{_CALIBRATION_SPLIT}_{name}.npz"
        scores = _cell_scores(path, name, views, noise_deg, settings)
        n_cal = scores.shape[0]
        logger.info(
            "calibrating cell %s on %d calibration bodies (alpha %s, spread floor %s cm)",
            name,
            n_cal,
            settings.alpha,
            settings.spread_floor_cm,
        )
        for column, measurement in enumerate(settings.measurements):
            rows.append(
                CalibrationQuantile(
                    cell_id=name,
                    views=views,
                    noise_deg=noise_deg,
                    measurement=measurement,
                    n_cal=n_cal,
                    alpha=settings.alpha,
                    q_hat=conformal_quantile(scores[:, column], settings.alpha),
                    spread_floor_cm=settings.spread_floor_cm,
                )
            )
    return rows


def calibrate_quantiles(
    config: Mapping[str, Any],
    out_dir: str | os.PathLike[str],
    *,
    run_record: RunRecord | None = None,
) -> CalibrationResult:
    """Calibrate every cell from ``<out_dir>/predict`` and write ``<out_dir>/calibrate``.

    ``run_record`` supplies the code version and the hardware class for DONE.json. Without it, a
    record is made from the configuration and the device that its ``device`` key requests. The
    marker records the configuration hash of the predict stage's own marker as its input, when
    that marker exists, so a later change of the predict stage makes this stage stale.

    Raises ConfigError (exit code 2) for a configuration that cannot be calibrated, and ValueError
    for a run record of another configuration. Both are raised before any file changes.
    Raises CalibrationRefusedError (exit code 4) for the causes in the module docstring. That
    error is raised after an earlier DONE.json and quantiles.csv were removed, and before
    anything is written.
    """
    digest = config_hash(config)  # validates the configuration first
    _settings(config)  # refuses a configuration that calibration cannot use, before files change
    record = _provenance(config, digest, run_record)
    predict_dir = Path(out_dir) / PREDICT_DIRECTORY_NAME
    stage_dir = calibrate_directory(out_dir)
    quantiles_path = stage_dir / QUANTILES_NAME
    clear_done_marker(stage_dir)
    quantiles_path.unlink(missing_ok=True)

    rows = compute_quantiles(config, predict_dir)
    _write_quantiles(quantiles_path, rows)
    write_done_marker(
        stage_dir,
        stage=STAGE_NAME,
        config_hash=record.config_hash,
        seed=record.seed,
        code_version=record.code_version,
        hardware_class=record.hardware_class,
        inputs=input_hashes({PREDICT_DIRECTORY_NAME: predict_dir}),
    )
    logger.info("wrote %d quantiles to %s", len(rows), quantiles_path)
    return CalibrationResult(
        calibrate_dir=stage_dir, quantiles_path=quantiles_path, quantiles=tuple(rows)
    )


def read_quantiles(source: str | os.PathLike[str]) -> list[CalibrationQuantile]:
    """Read quantiles.csv back into rows, after checking its header against QUANTILE_COLUMNS.

    Raises ValueError, naming the file and the line, for a wrong header, a row with the wrong
    number of cells, or a cell that is not a valid value of its column.
    """
    path = Path(source)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if tuple(header) != QUANTILE_COLUMNS:
            raise ValueError(
                f"{path} does not carry the quantiles header of contracts/artifacts.md"
            )
        rows: list[CalibrationQuantile] = []
        for line_number, cells in enumerate(reader, start=2):
            if len(cells) != len(QUANTILE_COLUMNS):
                raise ValueError(
                    f"{path} line {line_number} has {len(cells)} columns; "
                    f"the quantiles file has {len(QUANTILE_COLUMNS)}"
                )
            try:
                rows.append(
                    CalibrationQuantile(
                        cell_id=cells[0],
                        views=int(cells[1]),
                        noise_deg=float(cells[2]),
                        measurement=cells[3],
                        n_cal=int(cells[4]),
                        alpha=float(cells[5]),
                        q_hat=float(cells[6]),
                        spread_floor_cm=float(cells[7]),
                    )
                )
            except ValueError as error:
                raise ValueError(f"{path} line {line_number}: {error}") from error
    return rows


def _settings(config: Mapping[str, Any]) -> _Settings:
    """Collect the configuration values that calibration reads, and the calibration body ids."""
    data = config["data"]
    try:
        split = DataSplit(data["n_train"], data["n_cal"], data["n_test"])
    except ValueError as error:
        raise ConfigError(f"configuration key 'data.n_train': {error}", "data.n_train") from error
    evaluate = config["evaluate"]
    calibrate = config["calibrate"]
    cells = tuple(
        (int(views), float(noise_deg))
        for views in evaluate["views"]
        for noise_deg in evaluate["noise_deg"]
    )
    names = [cell_id(views, noise_deg) for views, noise_deg in cells]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        # evaluate refuses the same repeat, and it reads quantiles.csv with one row per cell and
        # measurement, so a repeated cell would give a file that the next stage cannot read.
        raise ConfigError(
            "configuration keys 'evaluate.views' and 'evaluate.noise_deg' give the cell(s) "
            f"{', '.join(repeated)} more than once; give each view count and noise level once",
            "evaluate.views",
        )
    return _Settings(
        cells=cells,
        measurements=tuple(evaluate["measurements"]),
        alpha=calibrate["alpha"],
        min_cal=calibrate["min_cal"],
        spread_floor_cm=calibrate["spread_floor_cm"],
        calibration_ids=split.range_of(Split.CALIBRATION),
        n_train=split.n_train,
        n_cal=split.n_cal,
    )


def _provenance(config: Mapping[str, Any], digest: str, run_record: RunRecord | None) -> RunRecord:
    """Return the run record whose code version and hardware class DONE.json carries.

    A record that the caller passes must belong to this configuration. Without one, the record is
    made from the configuration and the device that its ``device`` key requests (FR-028).
    """
    if run_record is not None:
        if run_record.config_hash != digest:
            raise ValueError(
                "the run record belongs to another configuration: its config_hash is "
                f"{run_record.config_hash[:12]}, the configuration's is {digest[:12]}"
            )
        return run_record
    choice = select_device(config["device"])
    return start_run_record(
        config, hardware_class=choice.hardware_class, device_name=choice.device_name
    )


def _cell_scores(
    path: Path, name: str, views: int, noise_deg: float, settings: _Settings
) -> NDArray[np.float64]:
    """Read and check the predict file of one cell; return its scores, shape (bodies, measurements).

    The checks run in this order: the file, the split, the keys, the cell, the shapes, the number
    of bodies, the body ids, and the values. The first that fails raises CalibrationRefusedError.
    """
    if not path.is_file():
        raise CalibrationRefusedError(
            f"the predict file {path} for cell {name} is missing; run the predict stage first"
        )
    fields = _read_fields(path)
    _require_calibration_split(path, fields)
    missing = [key for key in _FIELD_NAMES if key not in fields]
    if missing:
        raise CalibrationRefusedError(f"{path} lacks the key {', '.join(missing)}")
    _require_same_cell(path, fields, name, views, noise_deg)
    body_ids = _body_ids(path, fields["body_id"])
    values = {
        key: _value_matrix(path, key, fields[key], len(body_ids), len(settings.measurements))
        for key in _VALUE_FIELDS
    }
    _require_enough_bodies(path, name, len(body_ids), settings)
    _require_calibration_bodies(path, body_ids, settings)
    _require_usable_values(path, body_ids, values, settings)
    return normalized_scores(
        values["m_true"], values["m_median"], values["spread"], settings.spread_floor_cm
    )


def _read_fields(path: Path) -> dict[str, NDArray[Any]]:
    """Read the arrays that calibration needs from a predict file; a key the file lacks is left out.

    Pickled data is never loaded. A file that is not an npz archive, or is damaged, raises
    CalibrationRefusedError.
    """
    try:
        with path.open("rb") as handle:
            loaded = np.load(handle, allow_pickle=False)
            if not isinstance(loaded, NpzFile):  # a plain .npy file under an .npz name
                raise ValueError("it is not an npz archive")
            with loaded:
                return {key: loaded[key] for key in _FIELD_NAMES if key in loaded.files}
    except (OSError, EOFError, ValueError, zipfile.BadZipFile, zlib.error) as error:
        raise CalibrationRefusedError(f"cannot read the predict file {path}: {error}") from error


def _single_value(path: Path, key: str, value: NDArray[Any]) -> Any:
    """Return the one value of a key that must hold a single number or text."""
    if value.size != 1:
        raise CalibrationRefusedError(
            f"{path}: the key {key} must hold one value; its shape is {value.shape}"
        )
    item = value.item()
    return item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item


def _require_calibration_split(path: Path, fields: Mapping[str, NDArray[Any]]) -> None:
    """Refuse a predict file whose split is not ``cal``, the only split that calibration reads."""
    if "split" not in fields:
        raise CalibrationRefusedError(f"{path} lacks the key split")
    split = _single_value(path, "split", fields["split"])
    if split != _CALIBRATION_SPLIT:
        raise CalibrationRefusedError(
            f"{path} has split {split!r}, but calibration reads only the {_CALIBRATION_SPLIT!r} "
            "split; calibrating on any other split would leak held-out bodies into the calibration"
        )


def _require_same_cell(
    path: Path, fields: Mapping[str, NDArray[Any]], name: str, views: int, noise_deg: float
) -> None:
    """Refuse a file whose own view count or noise level differs from the cell it was read for."""
    file_views = _single_value(path, "views", fields["views"])
    file_noise = _single_value(path, "noise_deg", fields["noise_deg"])
    try:
        same = file_views == views and math.isclose(
            float(file_noise),
            noise_deg,
            rel_tol=_NOISE_RELATIVE_TOLERANCE,
            abs_tol=_NOISE_ABSOLUTE_TOLERANCE,
        )
    except (TypeError, ValueError):  # the key holds text that is not a number
        same = False
    if not same:
        raise CalibrationRefusedError(
            f"{path} holds predictions for views {file_views!r} and noise_deg {file_noise!r}, "
            f"but it was read for cell {name} (views {views}, noise_deg {noise_deg:g})"
        )


def _body_ids(path: Path, value: NDArray[Any]) -> NDArray[np.int64]:
    """Return the body ids of a predict file after checking that they are a list of integers."""
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise CalibrationRefusedError(
            f"{path}: body_id must be a one-dimensional list of integers; "
            f"got dtype {value.dtype} and shape {value.shape}"
        )
    return value.astype(np.int64)


def _value_matrix(
    path: Path, key: str, value: NDArray[Any], bodies: int, width: int
) -> NDArray[np.float64]:
    """Return a (bodies, measurements) float matrix of a predict file after checking its shape."""
    if value.dtype.kind not in "fiu" or value.shape != (bodies, width):
        raise CalibrationRefusedError(
            f"{path}: {key} must be numbers of shape ({bodies}, {width}), one row per body and "
            f"one column per measurement; got dtype {value.dtype} and shape {value.shape}"
        )
    return value.astype(np.float64)


def _require_enough_bodies(path: Path, name: str, bodies: int, settings: _Settings) -> None:
    """Refuse a cell with fewer bodies than calibrate.min_cal, or than the quantile needs.

    The first refusal is FR-011. The second only applies when calibrate.min_cal is set below the
    smallest size for which the quantile of calibrate.alpha exists.
    """
    if bodies < settings.min_cal:
        raise CalibrationRefusedError(
            f"cell {name} has {bodies} calibration bodies in {path.name}, below the minimum of "
            f"{settings.min_cal} (calibrate.min_cal); raise data.n_cal and generate again, or "
            "lower calibrate.min_cal if the smaller calibration set is intended"
        )
    needed = smallest_calibration_size(settings.alpha)
    if bodies < needed:
        raise CalibrationRefusedError(
            f"cell {name} has {bodies} calibration bodies in {path.name}, too few for "
            f"calibrate.alpha {settings.alpha}: the conformal quantile needs n + 1 >= 1 / alpha, "
            f"that is at least {needed} bodies"
        )


def _require_calibration_bodies(
    path: Path, body_ids: NDArray[np.int64], settings: _Settings
) -> None:
    """Refuse body ids that are not distinct calibration bodies of this configuration (FR-010)."""
    ids = settings.calibration_ids
    outside = (body_ids < ids.start) | (body_ids >= ids.stop)
    if outside.any():
        raise CalibrationRefusedError(
            f"{path.name}: body id {int(body_ids[outside][0])} is not a calibration body; the "
            f"calibration bodies are {ids.start} to {ids.stop - 1} (data.n_train "
            f"{settings.n_train}, data.n_cal {settings.n_cal}), and calibration reads no others"
        )
    distinct, counts = np.unique(body_ids, return_counts=True)
    if (counts > 1).any():
        raise CalibrationRefusedError(
            f"{path.name}: body id {int(distinct[counts > 1][0])} appears more than once; "
            "the calibration scores must come from distinct bodies"
        )


def _require_usable_values(
    path: Path,
    body_ids: NDArray[np.int64],
    values: Mapping[str, NDArray[np.float64]],
    settings: _Settings,
) -> None:
    """Refuse a value that is not finite, a negative spread, or a score with no positive divisor.

    The message names the body and the measurement, so the sample can be found (spec edge case:
    a non-finite prediction).
    """
    for key in _VALUE_FIELDS:
        unusable = ~np.isfinite(values[key])
        if unusable.any():
            row, column = np.argwhere(unusable)[0]
            raise CalibrationRefusedError(
                f"{path.name}: {key} is {values[key][row, column]} (not finite) for body "
                f"{int(body_ids[row])} and measurement {settings.measurements[column]}"
            )
    spread = values["spread"]
    negative = spread < 0
    if negative.any():
        row, column = np.argwhere(negative)[0]
        raise CalibrationRefusedError(
            f"{path.name}: spread is {spread[row, column]}, below 0, for body {int(body_ids[row])} "
            f"and measurement {settings.measurements[column]}"
        )
    without_divisor = np.maximum(spread, settings.spread_floor_cm) <= 0
    if without_divisor.any():
        row, column = np.argwhere(without_divisor)[0]
        raise CalibrationRefusedError(
            f"{path.name}: the spread of body {int(body_ids[row])} for measurement "
            f"{settings.measurements[column]} is 0 and calibrate.spread_floor_cm is "
            f"{settings.spread_floor_cm}, so its score has no divisor; set "
            "calibrate.spread_floor_cm above 0"
        )


def _write_quantiles(destination: Path, rows: Iterable[CalibrationQuantile]) -> None:
    """Write quantiles.csv atomically. Floats use repr, so equal inputs give equal bytes."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(QUANTILE_COLUMNS)
    for row in rows:
        writer.writerow(
            [
                row.cell_id,
                row.views,
                format_float(row.noise_deg),
                row.measurement,
                row.n_cal,
                format_float(row.alpha),
                format_float(row.q_hat),
                format_float(row.spread_floor_cm),
            ]
        )
    # The text is encoded here and written as bytes, so every platform writes bare line feeds.
    with atomic_path(destination) as temporary:
        temporary.write_bytes(buffer.getvalue().encode("utf-8"))
