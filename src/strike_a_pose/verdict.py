"""The FR-014 kill verdict from the saved results table, written as verdict.json and verdict.md.

This is the verdict stage of specs/001-kill-test-mvp (FR-014, FR-015, SC-003). The rule and its
constants are held in this module, in code, so that no configuration file can move them after a
result exists (constitution Principle IV): the threshold 0.70, the circumferences chest, waist, hip,
and thigh, the placement noise of 0 degrees, the compared view counts 1 and 4, and the coverage band
0.87 to 0.93. FIXED_CONFIGURATION restates them under the configuration keys that config.py fixes,
and run_verdict refuses a configuration whose values differ (ConfigError, exit code 2).

The rule. For each circumference, the width ratio is the 4-view median interval width divided by the
1-view median interval width, both at 0 degrees placement noise, read from evaluate/results.csv. The
median ratio is numpy.median of the four ratios (the mean of the 2nd and 3rd smallest). The verdict
is PASS when the median ratio is at most 0.70, all eight compared cells (two view counts by four
circumferences) have in_band true, the comparison is valid, and evaluate/sc004.json counts no SC-004
violation. Otherwise it is KILL. Height and the 2 and 5 degree rows are copied next to the verdict
(reported_only) and never enter it.

Three situations give KILL, and they end differently (contracts/cli.md):

* A compared cell outside the band is a legitimate result. cells_in_band is false, out_of_band_cells
  names the cells, the verdict line ends with "; comparison invalid for cells <ids>", and
  run_verdict returns, which the sap command reports with exit code 0.
* A compared cell missing from the table, or a width ratio that is not finite, makes the comparison
  itself invalid (invalid_comparison is true). A width that is not finite or is negative counts as
  such a ratio, so a finite width over an infinite one can never read as the ratio 0.
* An SC-004 violation is a defect of the run, not a result.

For the last two, run_verdict writes verdict.json and verdict.md first, and then raises
VerdictRefusedError, which the sap command reports with exit code 4 after it prints the verdict line
(error.verdict.line). An input that is missing, unreadable, or from another run raises
VerdictInputError (also exit code 4) before anything is written. A fixed key that differs raises
ConfigError (exit code 2) before anything is read.

Use from the command line. ``sap verdict`` calls run_verdict(config, out), prints the returned
verdict's ``line`` on stdout, and exits with code 0 (a PASS, or a KILL by the threshold or by a band
miss). ``sap run`` goes on to the report stage after such a return, and stops after a
VerdictRefusedError. ``sap verify`` calls recompute(out) and compares its ``to_json()`` and
``to_markdown()`` with the stored files.

Files. The inputs are run_record.json, evaluate/results.csv, evaluate/sc004.json, and the DONE.json
of the evaluate stage (contracts/artifacts.md). The outputs are verdict/verdict.json, verdict/
verdict.md (the verdict line first, then the tables), and verdict/DONE.json, which only a verdict
that exits 0 receives. Both verdict files are plain functions of the inputs, with no clock and no
machine name, so recompute(out) rebuilds them byte for byte; sap verify compares them (SC-003).

Public source. This module implements no published method. The median is numpy.median
(https://numpy.org/doc/stable/reference/generated/numpy.median.html).
"""

import csv
import filecmp
import io
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from strike_a_pose.checkpoint import (
    atomic_write_text,
    clear_done_marker,
    input_hashes,
    write_done_marker,
)
from strike_a_pose.config import ConfigError, config_hash
from strike_a_pose.data.generate import data_directory
from strike_a_pose.data.manifest import MANIFEST_NAME, SHARD_DIRECTORY_NAME
from strike_a_pose.runrecord import RunRecord, RunRecordError, read_run_record

__all__ = [
    "BAND",
    "COMPARE_VIEWS",
    "EVALUATED_MEASUREMENTS",
    "FIXED_CONFIGURATION",
    "MEASUREMENTS",
    "NOISE_DEG",
    "REPORTED_NOISE_DEG",
    "STAGE_NAME",
    "STAGE_REFUSED_EXIT_CODE",
    "THRESHOLD",
    "VERDICT_JSON_NAME",
    "VERDICT_MARKDOWN_NAME",
    "SEED_COVERAGE_TOLERANCE",
    "SEED_WIDTH_TOLERANCE_CM",
    "KillVerdict",
    "ReportedRow",
    "ReproductionDifferenceError",
    "ResultCell",
    "VerdictError",
    "VerdictInputError",
    "VerdictRefusedError",
    "check_fixed_configuration",
    "check_second_run",
    "compute_verdict",
    "recompute",
    "run_verdict",
    "verdict_directory",
    "verify_output",
    "width_ratio",
]

logger = logging.getLogger(__name__)

# The stage name in DONE.json, the stage whose marker this stage consumes, and the output folder.
STAGE_NAME = "verdict"
EVALUATE_STAGE = "evaluate"
VERDICT_DIRECTORY_NAME = "verdict"
VERDICT_JSON_NAME = "verdict.json"
VERDICT_MARKDOWN_NAME = "verdict.md"
# The exit code of contracts/cli.md for a stage that refuses to go on.
STAGE_REFUSED_EXIT_CODE = 4

# The constants of FR-014. They are held here, in code, and the configuration only restates them.
THRESHOLD = 0.70
MEASUREMENTS = ("chest", "waist", "hip", "thigh")
NOISE_DEG = 0.0
COMPARE_VIEWS = (1, 4)
BAND = (0.87, 0.93)
# The measurements of evaluate/results.csv in their fixed order (config.py fixes this order too).
EVALUATED_MEASUREMENTS = ("height", "chest", "waist", "hip", "thigh")
# The configuration keys that config.FIXED_KEYS fixes, with the value that this module holds for
# each. tests/test_verdict.py asserts that the two never drift apart.
FIXED_CONFIGURATION: dict[str, Any] = {
    "evaluate.band": BAND,
    "evaluate.measurements": EVALUATED_MEASUREMENTS,
    "verdict.threshold": THRESHOLD,
    "verdict.measurements": MEASUREMENTS,
    "verdict.noise_deg": NOISE_DEG,
    "verdict.compare_views": COMPARE_VIEWS,
}
# The placement noise levels whose rows are reported next to the verdict and never enter it.
REPORTED_NOISE_DEG = (2.0, 5.0)

_RUN_RECORD_NAME = "run_record.json"
# The columns of evaluate/results.csv that the verdict reads (contracts/artifacts.md).
_READ_COLUMNS = (
    "cell_id",
    "views",
    "noise_deg",
    "measurement",
    "nominal_level",
    "n_cal",
    "n_test",
    "median_width_cm",
    "in_band",
    "seed",
    "config_hash",
    "code_version",
    "hardware_class",
)


class VerdictError(ValueError):
    """A verdict that cannot be made or used. The sap command reports it with exit code 4."""

    exit_code = STAGE_REFUSED_EXIT_CODE


class VerdictInputError(VerdictError):
    """An input of the verdict stage is missing, unreadable, or from another run.

    Nothing is written when this is raised, so the files of an earlier verdict stay as they were.
    The message names the file, the row, or the column.
    """


class VerdictRefusedError(VerdictError):
    """The verdict is KILL and its comparison cannot be used: it was written, and the stage refuses.

    verdict.json and verdict.md exist when this is raised. The cause is an invalid comparison (a
    compared cell is missing or a width ratio is not finite) or an SC-004 violation. ``verdict``
    holds the verdict, so the sap command can print its line (``verdict.line``) before the error.
    """

    def __init__(self, message: str, verdict: "KillVerdict") -> None:
        super().__init__(message)
        self.verdict = verdict


@dataclass(frozen=True)
class ResultCell:
    """One row of evaluate/results.csv as the verdict reads it: a cell and one measurement."""

    cell_id: str
    views: int
    noise_deg: float
    measurement: str
    nominal_level: float
    n_cal: int
    n_test: int
    median_width_cm: float
    in_band: bool


@dataclass(frozen=True)
class ReportedRow:
    """One reported-only row: a measurement in its 1-view and 4-view cells at one noise level.

    A width is nan and an in-band flag is None when that cell is missing from the table.
    """

    measurement: str
    one_view_width_cm: float
    four_view_width_cm: float
    ratio: float
    one_view_in_band: bool | None
    four_view_in_band: bool | None

    def to_dict(self) -> dict[str, Any]:
        """Return the row as a JSON-ready dict; a width or ratio that is not finite becomes null."""
        one_view, four_view = COMPARE_VIEWS
        return {
            "measurement": self.measurement,
            f"width_v{one_view}_cm": _json_number(self.one_view_width_cm),
            f"width_v{four_view}_cm": _json_number(self.four_view_width_cm),
            "ratio": _json_number(self.ratio),
            f"in_band_v{one_view}": self.one_view_in_band,
            f"in_band_v{four_view}": self.four_view_in_band,
        }


@dataclass(frozen=True, eq=False)
class KillVerdict:
    """The kill verdict of one run (KillVerdict in data-model.md), with the details for verdict.md.

    ``widths`` and ``in_band_flags`` are keyed by view label (``v1``, ``v4``) and then by
    circumference. A width is nan and a flag is None when that compared cell is missing from the
    results table. ``to_dict`` gives the fields of verdict.json in the order of
    contracts/artifacts.md; the other fields only feed verdict.md and the error message.
    """

    verdict: str
    median_ratio: float
    cells_in_band: bool
    out_of_band_cells: tuple[str, ...]
    invalid_comparison: bool
    sc004_violations: int
    ratios: Mapping[str, float]
    widths: Mapping[str, Mapping[str, float]]
    in_band_flags: Mapping[str, Mapping[str, bool | None]]
    missing_cells: tuple[str, ...]
    height: ReportedRow
    noise_rows: Mapping[float, tuple[ReportedRow, ...]]
    nominal_levels: tuple[float, ...]
    calibration_counts: tuple[int, ...]
    test_counts: tuple[int, ...]
    config_hash: str
    seed: int
    code_version: str
    hardware_class: str

    @property
    def threshold(self) -> float:
        """The threshold of FR-014."""
        return THRESHOLD

    @property
    def noise_deg(self) -> float:
        """The placement noise of the compared cells, in degrees."""
        return NOISE_DEG

    @property
    def compare_views(self) -> tuple[int, ...]:
        """The compared view counts."""
        return COMPARE_VIEWS

    @property
    def line(self) -> str:
        """The verdict line in the fixed format of contracts/cli.md, without a line ending."""
        one_view, four_view = COMPARE_VIEWS
        ratios = " ".join(f"{name}={_decimals(self.ratios[name])}" for name in MEASUREMENTS)
        widths = " ".join(
            f"{name}={_decimals(self.widths[f'v{one_view}'][name])}"
            f"/{_decimals(self.widths[f'v{four_view}'][name])}"
            for name in MEASUREMENTS
        )
        line = (
            f"VERDICT: {self.verdict} median_ratio={_decimals(self.median_ratio)} "
            f"threshold={THRESHOLD:.3f} cells_in_band={_word(self.cells_in_band)} "
            f"invalid={_word(self.invalid_comparison)} sc004_violations={self.sc004_violations} "
            f"({ratios}) widths_v{one_view}_v{four_view}_cm=({widths}) "
            f"config={self.config_hash[:12]} seed={self.seed}"
        )
        if not self.cells_in_band:
            line += "; comparison invalid for cells " + ",".join(self.out_of_band_cells)
        return line

    @property
    def exit_code(self) -> int:
        """The exit code of the sap command for this verdict: 0, or 4 when its files are refused."""
        return STAGE_REFUSED_EXIT_CODE if self.refusal_reasons() else 0

    def refusal_reasons(self) -> list[str]:
        """Return why the comparison cannot be used (exit code 4), or an empty list.

        An out-of-band cell is not a reason: it is a legitimate KILL that exits 0.
        """
        reasons: list[str] = []
        if self.missing_cells:
            reasons.append(
                "compared cell(s) missing from evaluate/results.csv: "
                + ", ".join(self.missing_cells)
            )
        one_view, four_view = (f"v{views}" for views in COMPARE_VIEWS)
        for name in MEASUREMENTS:
            present = (
                self.in_band_flags[one_view][name] is not None
                and self.in_band_flags[four_view][name] is not None
            )
            if present and not math.isfinite(self.ratios[name]):
                reasons.append(
                    f"the width ratio for {name} is not finite (1-view width "
                    f"{_decimals(self.widths[one_view][name])} cm, 4-view width "
                    f"{_decimals(self.widths[four_view][name])} cm)"
                )
        if self.sc004_violations > 0:
            reasons.append(
                f"the SC-004 check in evaluate/sc004.json counts {self.sc004_violations} "
                "violation(s), and the limit is 0"
            )
        return reasons

    def to_dict(self) -> dict[str, Any]:
        """Return the fields of verdict.json in the order of contracts/artifacts.md.

        A ratio or width that is not finite becomes null, because JSON has no nan or infinity.
        """
        return {
            "verdict": self.verdict,
            "median_ratio": _json_number(self.median_ratio),
            "threshold": THRESHOLD,
            "cells_in_band": self.cells_in_band,
            "out_of_band_cells": list(self.out_of_band_cells),
            "invalid_comparison": self.invalid_comparison,
            "sc004_violations": self.sc004_violations,
            "ratios": {name: _json_number(self.ratios[name]) for name in MEASUREMENTS},
            "widths": {
                label: {name: _json_number(self.widths[label][name]) for name in MEASUREMENTS}
                for label in (f"v{views}" for views in COMPARE_VIEWS)
            },
            "noise_deg": NOISE_DEG,
            "compare_views": list(COMPARE_VIEWS),
            "reported_only": {
                "height_ratio": _json_number(self.height.ratio),
                **{
                    f"rows_{noise:g}deg": [row.to_dict() for row in self.noise_rows[noise]]
                    for noise in REPORTED_NOISE_DEG
                },
            },
            "config_hash": self.config_hash,
            "seed": self.seed,
            "code_version": self.code_version,
            "hardware_class": self.hardware_class,
        }

    def to_json(self) -> str:
        """Return the text of verdict.json."""
        return json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"

    def to_markdown(self) -> str:
        """Return the text of verdict.md: the verdict line first, then the tables."""
        return _render_markdown(self)


def verdict_directory(out_directory: str | os.PathLike[str]) -> Path:
    """Return the folder of this stage, ``<out>/verdict``."""
    return Path(out_directory) / VERDICT_DIRECTORY_NAME


def width_ratio(*, one_view_cm: float, four_view_cm: float) -> float:
    """Return the 4-view median width divided by the 1-view median width (FR-014).

    The ratio is nan when either width is not finite or is negative, because such a width is not a
    width, and a finite width over an infinite one would otherwise read as the ratio 0. Otherwise
    the division follows IEEE 754: a zero 1-view width gives inf, or nan when both widths are zero.
    """
    for width in (one_view_cm, four_view_cm):
        if not math.isfinite(width) or width < 0.0:
            return math.nan
    if one_view_cm == 0.0:
        return math.inf if four_view_cm > 0.0 else math.nan
    return four_view_cm / one_view_cm


def check_fixed_configuration(config: Mapping[str, Any]) -> None:
    """Raise ConfigError (exit code 2) unless the configuration holds the FR-014 values held here.

    The keys are those of FIXED_CONFIGURATION. config.py refuses the same values when it loads a
    file or applies a --set override; this check also covers a configuration built in code, and it
    reads the values from this module, not from config.py. The error names the key.
    """
    for key, fixed in FIXED_CONFIGURATION.items():
        section, name = key.split(".")
        node = config.get(section) if isinstance(config, Mapping) else None
        if not isinstance(node, Mapping) or name not in node:
            raise ConfigError(
                f"missing configuration key '{key}', which FR-014 fixes at {_listed(fixed)!r}", key
            )
        if not _same_value(node[name], fixed):
            raise ConfigError(
                f"configuration key '{key}' is fixed at {_listed(fixed)!r} by FR-014 and is held "
                f"in verdict.py; got {node[name]!r}",
                key,
            )


def compute_verdict(
    cells: Sequence[ResultCell], *, sc004_violations: int, record: RunRecord
) -> KillVerdict:
    """Apply the FR-014 rule to the rows of a results table, and return the verdict.

    Nothing is read or written here. ``record`` supplies the four provenance fields. A compared
    cell that is missing, or a ratio that is not finite, sets ``invalid_comparison``; neither
    raises. Raises ValueError for a violation count that is not a non-negative integer, and
    VerdictInputError when two rows name the same cell and measurement.
    """
    if isinstance(sc004_violations, bool) or not isinstance(sc004_violations, int):
        raise ValueError(f"the SC-004 violation count must be an integer; got {sc004_violations!r}")
    if sc004_violations < 0:
        raise ValueError(f"the SC-004 violation count cannot be negative; got {sc004_violations}")
    table = _index_cells(cells)
    one_view, four_view = COMPARE_VIEWS
    one_label, four_label = f"v{one_view}", f"v{four_view}"

    widths: dict[str, dict[str, float]] = {one_label: {}, four_label: {}}
    flags: dict[str, dict[str, bool | None]] = {one_label: {}, four_label: {}}
    missing: list[str] = []
    out_of_band: list[str] = []
    compared: list[ResultCell] = []
    for name in MEASUREMENTS:  # circumference first, then 1 view before 4 views
        for views, label in ((one_view, one_label), (four_view, four_label)):
            cell = table.get((views, NOISE_DEG, name))
            identifier = f"{_cell_identifier(views, NOISE_DEG)}:{name}"
            if cell is None:
                widths[label][name] = math.nan
                flags[label][name] = None
                missing.append(identifier)
                continue
            widths[label][name] = cell.median_width_cm
            flags[label][name] = cell.in_band
            compared.append(cell)
            if not cell.in_band:
                out_of_band.append(identifier)

    ratios = {
        name: width_ratio(
            one_view_cm=widths[one_label][name], four_view_cm=widths[four_label][name]
        )
        for name in MEASUREMENTS
    }
    median_ratio = float(np.median(np.array(list(ratios.values()), dtype=np.float64)))
    invalid = bool(missing) or any(not math.isfinite(ratio) for ratio in ratios.values())
    cells_in_band = not out_of_band
    passes = median_ratio <= THRESHOLD and cells_in_band and not invalid and sc004_violations == 0
    return KillVerdict(
        verdict="PASS" if passes else "KILL",
        median_ratio=median_ratio,
        cells_in_band=cells_in_band,
        out_of_band_cells=tuple(out_of_band),
        invalid_comparison=invalid,
        sc004_violations=sc004_violations,
        ratios=ratios,
        widths=widths,
        in_band_flags=flags,
        missing_cells=tuple(missing),
        height=_reported_row(table, "height", NOISE_DEG),
        noise_rows={noise: _noise_rows(table, noise) for noise in REPORTED_NOISE_DEG},
        nominal_levels=tuple(sorted({cell.nominal_level for cell in compared})),
        calibration_counts=tuple(sorted({cell.n_cal for cell in compared})),
        test_counts=tuple(sorted({cell.n_test for cell in compared})),
        config_hash=record.config_hash,
        seed=record.seed,
        code_version=record.code_version,
        hardware_class=record.hardware_class,
    )


def recompute(out_directory: str | os.PathLike[str]) -> KillVerdict:
    """Recompute the verdict from the saved results of an output directory, and write nothing.

    This is the recomputation of ``sap verify`` (SC-003). It needs no configuration: the rule is
    held in this module, and the provenance comes from ``run_record.json``. The inputs are
    ``evaluate/results.csv`` and ``evaluate/sc004.json``, and each row and field must carry the
    run record's seed, configuration hash, code version, and hardware class. The result's
    ``to_dict()``, ``to_json()``, and ``to_markdown()`` equal what ``run_verdict`` wrote from the
    same files. A verdict whose comparison is invalid is returned, not raised; its ``exit_code``
    says 4.

    Raises VerdictInputError when a file is missing, unreadable, from another run, or malformed.
    """
    out = Path(out_directory)
    return _verdict_from_files(out, _read_record(out / _RUN_RECORD_NAME))


def run_verdict(config: Mapping[str, Any], out_directory: str | os.PathLike[str]) -> KillVerdict:
    """Compute the verdict, write verdict/verdict.json and verdict/verdict.md, and return it.

    config is a resolved configuration, and the run record in the output directory must name its
    configuration hash. The evaluate stage must be done (its DONE.json is read). The files are
    replaced atomically and are written before this function decides how the command ends:

    * A verdict whose comparison is valid and whose SC-004 count is 0 is returned, PASS or KILL, and
      verdict/DONE.json is written last. A band miss returns here, as a KILL for exit code 0.
    * Otherwise the files are written, DONE.json is left out, and VerdictRefusedError (exit code 4)
      is raised with the verdict attached. An earlier DONE.json is removed, so a stage that failed
      this run cannot look done.

    Raises ConfigError (exit code 2) when a fixed configuration key differs from FR-014, before any
    file is read. Raises VerdictInputError (exit code 4) when an input is missing, unreadable, or
    from another run, before any file is written.
    """
    check_fixed_configuration(config)
    digest = config_hash(config)
    out = Path(out_directory)
    record = _read_record(out / _RUN_RECORD_NAME)
    if record.config_hash != digest:
        raise VerdictInputError(
            f"the run record in '{out}' names configuration {record.config_hash[:12]}, but the "
            f"configuration given holds {digest[:12]}; the verdict reads the outputs of one run "
            "only"
        )
    evaluate_directory = out / EVALUATE_STAGE
    inputs = input_hashes({EVALUATE_STAGE: evaluate_directory})
    if EVALUATE_STAGE not in inputs:
        raise VerdictInputError(
            f"the evaluate stage has no readable DONE.json in '{evaluate_directory}'; "
            "run sap evaluate first"
        )
    if inputs[EVALUATE_STAGE] != digest:
        raise VerdictInputError(
            f"the evaluate stage in '{evaluate_directory}' was done under configuration "
            f"{inputs[EVALUATE_STAGE][:12]}, but the configuration given holds {digest[:12]}; "
            "run sap evaluate again"
        )
    verdict = _verdict_from_files(out, record)

    directory = verdict_directory(out)
    clear_done_marker(directory)
    atomic_write_text(directory / VERDICT_JSON_NAME, verdict.to_json())
    atomic_write_text(directory / VERDICT_MARKDOWN_NAME, verdict.to_markdown())
    logger.info("verdict %s written to %s", verdict.verdict, directory)
    reasons = verdict.refusal_reasons()
    if reasons:
        raise VerdictRefusedError(
            f"the {verdict.verdict} verdict was written to '{directory}', but its comparison "
            "cannot be used: " + "; ".join(reasons),
            verdict,
        )
    write_done_marker(
        directory,
        stage=STAGE_NAME,
        config_hash=record.config_hash,
        seed=record.seed,
        code_version=record.code_version,
        hardware_class=record.hardware_class,
        inputs=inputs,
    )
    return verdict


# SC-005: coverage values match within 0.5 percentage points, and median widths within 0.1 cm.
SEED_COVERAGE_TOLERANCE = 0.005
SEED_WIDTH_TOLERANCE_CM = 0.1
_RESULT_IDENTITY_COLUMNS = ("cell_id", "measurement")
_SEED_CHECK_COLUMNS = ("coverage", "median_width_cm")


class ReproductionDifferenceError(Exception):
    """Two outputs that must reproduce each other differ: sap verify and --seed-check exit 6.

    The message names the first difference. This is not a ValueError, so it never takes the exit
    code of a refused stage (4); the command maps it to exit code 6 on its own.
    """


def verify_output(out_directory: str | os.PathLike[str]) -> KillVerdict:
    """Recompute the verdict of an output directory and compare it with the stored verdict files.

    This is the check of ``sap verify`` (SC-003). The stored verdict/verdict.json and
    verdict/verdict.md must equal the recomputed text byte for byte. A missing stored file is a
    difference too. Returns the recomputed verdict; raises ReproductionDifferenceError (exit 6)
    that names the first difference. Inputs that cannot be read raise VerdictInputError first.
    """
    verdict = recompute(out_directory)
    directory = verdict_directory(out_directory)
    expected_texts = (
        (VERDICT_JSON_NAME, verdict.to_json()),
        (VERDICT_MARKDOWN_NAME, verdict.to_markdown()),
    )
    differences: list[str] = []
    for name, expected in expected_texts:
        path = directory / name
        if not path.is_file():
            differences.append(f"the stored file '{path}' is missing")
            continue
        stored = _read_text(path, "stored verdict file")
        if stored != expected:
            differences.append(
                f"{name} differs from the recomputed text, "
                f"{_first_text_difference(stored, expected)}"
            )
    _raise_on_differences(differences, "the stored verdict differs from the recomputed verdict")
    return verdict


def check_second_run(
    out_directory: str | os.PathLike[str], reference_directory: str | os.PathLike[str]
) -> None:
    """Compare a run with the reference output it must reproduce (sap run --seed-check, SC-005).

    data/manifest.csv and every shard under data/shards must be byte for byte equal to the
    reference. Each row of evaluate/results.csv must name the same cell and measurement, in the same
    order, and its coverage and median width must lie within SEED_COVERAGE_TOLERANCE and
    SEED_WIDTH_TOLERANCE_CM of the reference. Other columns are not compared. Any difference, or a
    missing file, raises ReproductionDifferenceError (exit 6) that names the first difference.
    """
    out = Path(out_directory)
    reference = Path(reference_directory)
    differences = _data_differences(out, reference) + _results_differences(out, reference)
    _raise_on_differences(
        differences, f"the run in '{out}' differs from the reference '{reference}'"
    )


def _raise_on_differences(differences: Sequence[str], subject: str) -> None:
    """Raise ReproductionDifferenceError that names the first difference, if there is any."""
    if not differences:
        return
    more = len(differences) - 1
    count = f" ({more} more difference{'s' if more > 1 else ''})" if more else ""
    raise ReproductionDifferenceError(f"{subject}: {differences[0]}{count}")


def _first_text_difference(stored: str, expected: str) -> str:
    """Name the first line at which two texts differ: its number, and both versions of it."""
    for number, (found, wanted) in enumerate(
        zip_longest(stored.splitlines(), expected.splitlines(), fillvalue="<no line>"), start=1
    ):
        if found != wanted:
            return f"line {number}: stored {found!r}, recomputed {wanted!r}"
    return "the two texts differ in their line endings or final newline"


def _data_differences(out: Path, reference: Path) -> list[str]:
    """Return the differences of the manifest and the shard files, compared byte for byte."""
    out_data = data_directory(out)
    reference_data = data_directory(reference)
    out_shards = _shard_names(out_data)
    reference_shards = _shard_names(reference_data)
    differences: list[str] = []
    if out_shards != reference_shards:
        differences.append(
            f"the shard files differ: {len(out_shards)} in the run, {len(reference_shards)} in the "
            "reference"
        )
    pairs = [(out_data / MANIFEST_NAME, reference_data / MANIFEST_NAME)]
    shared = sorted(set(out_shards) & set(reference_shards))
    pairs += [
        (out_data / SHARD_DIRECTORY_NAME / name, reference_data / SHARD_DIRECTORY_NAME / name)
        for name in shared
    ]
    for found, wanted in pairs:
        if not found.is_file() or not wanted.is_file():
            differences.append(f"'{found}' or '{wanted}' is missing")
        elif not filecmp.cmp(found, wanted, shallow=False):
            differences.append(f"'{found}' differs from '{wanted}' in its bytes")
    return differences


def _shard_names(data_folder: Path) -> list[str]:
    """Return the sorted names of the shard files of a data folder; none when it has no shards."""
    folder = data_folder / SHARD_DIRECTORY_NAME
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.glob("*.npz"))


def _results_differences(out: Path, reference: Path) -> list[str]:
    """Return the differences of evaluate/results.csv, row by row, within the SC-005 tolerances."""
    out_path = out / EVALUATE_STAGE / "results.csv"
    reference_path = reference / EVALUATE_STAGE / "results.csv"
    if not out_path.is_file() or not reference_path.is_file():
        return [f"'{out_path}' or '{reference_path}' is missing"]
    out_header, out_rows = _read_csv_rows(out_path)
    reference_header, reference_rows = _read_csv_rows(reference_path)
    required = _RESULT_IDENTITY_COLUMNS + _SEED_CHECK_COLUMNS
    if not all(column in out_header for column in required) or not all(
        column in reference_header for column in required
    ):
        return [
            f"'{out_path}' or '{reference_path}' lacks one of the columns {', '.join(required)}"
        ]
    if len(out_rows) != len(reference_rows):
        return [
            f"the results table has {len(out_rows)} rows, the reference has {len(reference_rows)}"
        ]
    differences: list[str] = []
    for line, (found, wanted) in enumerate(zip(out_rows, reference_rows, strict=True), start=2):
        name = f"row {line} ({found.get('cell_id')} {found.get('measurement')})"
        identity = tuple(found[column] for column in _RESULT_IDENTITY_COLUMNS)
        expected_identity = tuple(wanted[column] for column in _RESULT_IDENTITY_COLUMNS)
        if identity != expected_identity:
            differences.append(f"{name} names {identity}, the reference names {expected_identity}")
            continue
        for column, tolerance, unit in (
            ("coverage", SEED_COVERAGE_TOLERANCE, ""),
            ("median_width_cm", SEED_WIDTH_TOLERANCE_CM, " cm"),
        ):
            gap = _float_gap(found[column], wanted[column])
            # Written as "not within" so that a value that is not a number (NaN) also differs.
            if not gap <= tolerance:
                differences.append(
                    f"{name} column {column} is {found[column]}{unit}, the reference is "
                    f"{wanted[column]}{unit}; the tolerance is {tolerance}{unit}"
                )
    return differences


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return the header and the rows of a CSV file, read as text."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _float_gap(found: str, wanted: str) -> float:
    """Return the absolute gap of two numbers as text; NaN when either is not a finite number."""
    try:
        first, second = float(found), float(wanted)
    except ValueError:
        return math.nan
    if not (math.isfinite(first) and math.isfinite(second)):
        return math.nan
    return abs(first - second)


def _verdict_from_files(out: Path, record: RunRecord) -> KillVerdict:
    """Read the results table and the SC-004 check of an output directory; compute the verdict."""
    evaluate_directory = out / EVALUATE_STAGE
    cells = _read_result_cells(evaluate_directory / "results.csv", record)
    violations = _read_sc004_violations(evaluate_directory / "sc004.json", record)
    return compute_verdict(cells, sc004_violations=violations, record=record)


def _cell_identifier(views: int, noise_deg: float) -> str:
    """Return the cell name v<views>_n<noise>, for example v4_n0 (the form evaluate.py writes)."""
    return f"v{views}_n{noise_deg:g}"


def _index_cells(cells: Sequence[ResultCell]) -> dict[tuple[int, float, str], ResultCell]:
    """Return the rows by view count, noise level, and measurement, refusing a repeated row."""
    table: dict[tuple[int, float, str], ResultCell] = {}
    for cell in cells:
        key = (cell.views, cell.noise_deg, cell.measurement)
        if key in table:
            raise VerdictInputError(
                f"the results table lists {_cell_identifier(cell.views, cell.noise_deg)}:"
                f"{cell.measurement} more than once"
            )
        table[key] = cell
    return table


def _reported_row(
    table: Mapping[tuple[int, float, str], ResultCell], measurement: str, noise_deg: float
) -> ReportedRow:
    """Return the comparison row of one measurement at one noise level (reported only)."""
    one_view, four_view = COMPARE_VIEWS
    first = table.get((one_view, noise_deg, measurement))
    second = table.get((four_view, noise_deg, measurement))
    one_width = math.nan if first is None else first.median_width_cm
    four_width = math.nan if second is None else second.median_width_cm
    return ReportedRow(
        measurement=measurement,
        one_view_width_cm=one_width,
        four_view_width_cm=four_width,
        ratio=width_ratio(one_view_cm=one_width, four_view_cm=four_width),
        one_view_in_band=None if first is None else first.in_band,
        four_view_in_band=None if second is None else second.in_band,
    )


def _noise_rows(
    table: Mapping[tuple[int, float, str], ResultCell], noise_deg: float
) -> tuple[ReportedRow, ...]:
    """Return the comparison rows of every measurement at one noise level, or none."""
    if not any(key[1] == noise_deg for key in table):
        return ()
    return tuple(_reported_row(table, name, noise_deg) for name in EVALUATED_MEASUREMENTS)


def _listed(value: Any) -> Any:
    """Return a fixed value for a message, with tuples written as lists like the configuration."""
    return [_listed(item) for item in value] if isinstance(value, tuple) else value


def _same_value(found: Any, fixed: Any) -> bool:
    """Return True when a configured value equals a fixed one; True is not the number 1."""
    if isinstance(fixed, tuple):
        return (
            isinstance(found, list | tuple)
            and len(found) == len(fixed)
            and all(_same_value(left, right) for left, right in zip(found, fixed, strict=True))
        )
    if isinstance(fixed, str):
        return isinstance(found, str) and found == fixed
    return isinstance(found, int | float) and not isinstance(found, bool) and found == fixed


def _read_record(path: Path) -> RunRecord:
    """Read run_record.json, turning a defect in it into a VerdictInputError."""
    try:
        return read_run_record(path)
    except RunRecordError as error:
        raise VerdictInputError(str(error)) from error


def _read_text(path: Path, label: str) -> str:
    """Return the text of an input file, refusing one that is missing or unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise VerdictInputError(
            f"cannot find the {label} '{path}'; the stage that writes it has not run"
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise VerdictInputError(f"cannot read the {label} '{path}': {error}") from error


def _check_provenance(found: Mapping[str, Any], where: str, record: RunRecord) -> None:
    """Refuse a file or row whose seed, hash, code version, or hardware class differs."""
    for name, expected in (
        ("seed", record.seed),
        ("config_hash", record.config_hash),
        ("code_version", record.code_version),
        ("hardware_class", record.hardware_class),
    ):
        if name not in found:
            raise VerdictInputError(
                f"{where}: the field '{name}' is missing, but every file carries the run record "
                "fields (FR-025)"
            )
        value = found[name]
        if value != expected or type(value) is not type(expected):
            raise VerdictInputError(
                f"{where}: the field '{name}' is {value!r}, but the run record says {expected!r}; "
                "it comes from another run"
            )


def _text_field(row: Mapping[str, str], column: str, where: str) -> str:
    """Return one non-empty text field of a CSV row."""
    value = row[column].strip()
    if value == "":
        raise VerdictInputError(f"{where}: the column '{column}' is empty")
    return value


def _integer_field(row: Mapping[str, str], column: str, where: str) -> int:
    """Return one whole-number field of a CSV row."""
    text = _text_field(row, column, where)
    try:
        return int(text)
    except ValueError as error:
        raise VerdictInputError(
            f"{where}: the column '{column}' must be a whole number; got {text!r}"
        ) from error


def _number_field(row: Mapping[str, str], column: str, where: str, *, finite: bool) -> float:
    """Return one number field of a CSV row; nan and the infinities are allowed unless finite."""
    text = _text_field(row, column, where)
    try:
        value = float(text)
    except ValueError as error:
        raise VerdictInputError(
            f"{where}: the column '{column}' must be a number; got {text!r}"
        ) from error
    if finite and not math.isfinite(value):
        raise VerdictInputError(f"{where}: the column '{column}' must be finite; got {text!r}")
    return value


def _flag_field(row: Mapping[str, str], column: str, where: str) -> bool:
    """Return one true or false field of a CSV row, in any letter case."""
    text = _text_field(row, column, where)
    folded = text.lower()
    if folded not in ("true", "false"):
        raise VerdictInputError(
            f"{where}: the column '{column}' must be true or false; got {text!r}"
        )
    return folded == "true"


def _read_result_cells(path: Path, record: RunRecord) -> list[ResultCell]:
    """Read the rows of evaluate/results.csv that the verdict needs, after checking their run."""
    label = "results table"
    reader = csv.DictReader(io.StringIO(_read_text(path, label)))
    header = reader.fieldnames or []
    missing = [column for column in _READ_COLUMNS if column not in header]
    if missing:
        raise VerdictInputError(f"the {label} '{path}' lacks the column(s) {', '.join(missing)}")
    cells: list[ResultCell] = []
    try:
        for number, row in enumerate(reader, start=1):
            where = f"results row {number} ({row.get('cell_id')}, {row.get('measurement')})"
            if None in row or any(value is None for value in row.values()):
                raise VerdictInputError(f"{where}: the row does not have one value per column")
            _check_provenance(
                {
                    "seed": _integer_field(row, "seed", where),
                    "config_hash": _text_field(row, "config_hash", where),
                    "code_version": _text_field(row, "code_version", where),
                    "hardware_class": _text_field(row, "hardware_class", where),
                },
                where,
                record,
            )
            cells.append(
                ResultCell(
                    cell_id=_text_field(row, "cell_id", where),
                    views=_integer_field(row, "views", where),
                    noise_deg=_number_field(row, "noise_deg", where, finite=True),
                    measurement=_text_field(row, "measurement", where),
                    nominal_level=_number_field(row, "nominal_level", where, finite=True),
                    n_cal=_integer_field(row, "n_cal", where),
                    n_test=_integer_field(row, "n_test", where),
                    median_width_cm=_number_field(row, "median_width_cm", where, finite=False),
                    in_band=_flag_field(row, "in_band", where),
                )
            )
    except csv.Error as error:
        raise VerdictInputError(f"cannot parse the {label} '{path}': {error}") from error
    return cells


def _refuse_json_constant(name: str) -> Any:
    """Reject NaN and the infinities, which JSON does not allow."""
    raise ValueError(f"the JSON constant {name} is not a finite number")


def _read_sc004_violations(path: Path, record: RunRecord) -> int:
    """Return the violation count of evaluate/sc004.json, after checking its run."""
    label = "SC-004 check"
    text = _read_text(path, label)
    try:
        document = json.loads(text, parse_constant=_refuse_json_constant)
    except ValueError as error:
        raise VerdictInputError(f"the {label} '{path}' is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise VerdictInputError(f"the {label} '{path}' must be a JSON object")
    violations = document.get("violations")
    if isinstance(violations, bool) or not isinstance(violations, int) or violations < 0:
        raise VerdictInputError(
            f"the {label} '{path}': the field 'violations' must be a whole number of at least 0; "
            f"got {violations!r}"
        )
    _check_provenance(document, f"the {label} '{path}'", record)
    return violations


def _json_number(value: float) -> float | None:
    """Return a finite number for JSON, or None for nan and the infinities."""
    return value + 0.0 if math.isfinite(value) else None


def _decimals(value: float) -> str:
    """Format a ratio or a width with three decimals; nan and the infinities print as text."""
    return f"{value + 0.0:.3f}"


def _word(flag: bool) -> str:
    """Return a flag as the lowercase word of the verdict line."""
    return "true" if flag else "false"


def _width_text(width: float, flag: bool | None) -> str:
    """Return a width for a Markdown table cell; a cell missing from the table has no width."""
    return "missing" if flag is None else _decimals(width)


def _band_text(flag: bool | None) -> str:
    """Return an in-band flag for a Markdown table cell."""
    if flag is None:
        return "missing"
    return "yes" if flag else "**no (flagged)**"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Return the lines of a Markdown table."""

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    return [line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)]


def _comparison_headers(first_column: str) -> list[str]:
    """Return the header cells of a 1-view against 4-view comparison table."""
    one_view, four_view = COMPARE_VIEWS
    return [
        first_column,
        f"{one_view}-view median width (cm)",
        f"{four_view}-view median width (cm)",
        f"Ratio ({four_view}-view to {one_view}-view)",
        f"{one_view}-view cell in band",
        f"{four_view}-view cell in band",
    ]


def _comparison_row(row: ReportedRow) -> list[str]:
    """Return the table cells of one reported row."""
    return [
        row.measurement,
        _width_text(row.one_view_width_cm, row.one_view_in_band),
        _width_text(row.four_view_width_cm, row.four_view_in_band),
        _decimals(row.ratio),
        _band_text(row.one_view_in_band),
        _band_text(row.four_view_in_band),
    ]


def _sentence(reason: str) -> str:
    """Return a reason as a sentence that starts with a capital letter and ends with a period."""
    return reason[0].upper() + reason[1:] + "."


def _kill_reasons(verdict: KillVerdict) -> list[str]:
    """Return why the verdict is KILL, one sentence each, for verdict.md."""
    reasons: list[str] = []
    if not math.isfinite(verdict.median_ratio):
        reasons.append(
            f"The median of the four width ratios is {_decimals(verdict.median_ratio)}, so it "
            f"cannot be at most the threshold {THRESHOLD:.3f}."
        )
    elif verdict.median_ratio > THRESHOLD:
        reasons.append(
            f"The median of the four width ratios is {_decimals(verdict.median_ratio)}, above the "
            f"threshold {THRESHOLD:.3f}."
        )
    if verdict.out_of_band_cells:
        reasons.append(
            "These compared cells lie outside the coverage band, so a width comparison involving "
            "them is invalid (FR-013): " + ", ".join(verdict.out_of_band_cells) + "."
        )
    reasons.extend(_sentence(reason) for reason in verdict.refusal_reasons())
    return reasons


def _render_markdown(verdict: KillVerdict) -> str:
    """Return verdict.md: the verdict line, the comparison table, the rule, and the reported rows.

    The headings start at the third level, because sap report copies everything after the first line
    into results.md under its own Verdict heading.
    """
    one_view, four_view = COMPARE_VIEWS
    labels = (f"v{one_view}", f"v{four_view}")
    low_percent, high_percent = (f"{100 * edge:.0f}%" for edge in BAND)
    lines = [verdict.line, "", f"### Width comparison at {NOISE_DEG:g} degrees placement noise", ""]
    lines += _table(
        _comparison_headers("Circumference"),
        [
            [
                name,
                _width_text(
                    verdict.widths[labels[0]][name], verdict.in_band_flags[labels[0]][name]
                ),
                _width_text(
                    verdict.widths[labels[1]][name], verdict.in_band_flags[labels[1]][name]
                ),
                _decimals(verdict.ratios[name]),
                _band_text(verdict.in_band_flags[labels[0]][name]),
                _band_text(verdict.in_band_flags[labels[1]][name]),
            ]
            for name in MEASUREMENTS
        ],
    )
    median = f"Median of the four ratios: {_decimals(verdict.median_ratio)}"
    if math.isfinite(verdict.median_ratio):
        median += f" (full precision {verdict.median_ratio!r})"
    median += f". Threshold: {THRESHOLD:.3f}. The verdict compares the full precision values."
    if verdict.invalid_comparison:
        median += " The comparison is invalid, so this median is not a result."
    lines += ["", median, ""]
    if verdict.test_counts:
        lines += [
            "Basis: nominal level "
            + ", ".join(f"{100 * level:g}%" for level in verdict.nominal_levels)
            + ", calibration bodies (n_cal) "
            + ", ".join(str(count) for count in verdict.calibration_counts)
            + ", test bodies (n_test) "
            + ", ".join(str(count) for count in verdict.test_counts)
            + f", seed {verdict.seed}, taken from the compared cells of evaluate/results.csv.",
            "",
        ]
    lines += [
        f"Rule (FR-014): the verdict is PASS when the median ratio is at most {THRESHOLD:.3f}, all "
        f"eight compared cells ({one_view} and {four_view} views, chest, waist, hip, and thigh, "
        f"{NOISE_DEG:g} degrees) lie inside the coverage band of {low_percent} to {high_percent}, "
        "the comparison is valid, and evaluate/sc004.json counts no SC-004 violation. Otherwise it "
        "is KILL.",
        "",
    ]
    if verdict.verdict == "KILL":
        lines += ["### Why this verdict is KILL", ""]
        lines += [f"- {reason}" for reason in _kill_reasons(verdict)]
        lines += [""]
    lines += [
        "### Reported only (not part of the verdict)",
        "",
        "These rows are shown next to the verdict and never enter it (FR-014).",
        "",
        f"Height at {NOISE_DEG:g} degrees placement noise:",
        "",
        *_table(_comparison_headers("Measurement"), [_comparison_row(verdict.height)]),
        "",
    ]
    for noise in REPORTED_NOISE_DEG:
        rows = verdict.noise_rows[noise]
        lines += [f"#### {noise:g} degrees placement noise", ""]
        if rows:
            lines += [
                *_table(_comparison_headers("Measurement"), [_comparison_row(r) for r in rows])
            ]
        else:
            lines += [f"The results table holds no cell at {noise:g} degrees placement noise."]
        lines += [""]
    lines += [
        "### Provenance of this verdict",
        "",
        f"- Configuration hash: `{verdict.config_hash}`",
        f"- Seed: {verdict.seed}",
        f"- Code version: `{verdict.code_version}`",
        f"- Hardware class: {verdict.hardware_class}",
    ]
    return "\n".join(lines) + "\n"
