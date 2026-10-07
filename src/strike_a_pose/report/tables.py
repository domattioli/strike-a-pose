"""Markdown results tables for sap runs: results.md from the results CSV, verdict, and run record.

The layout follows contracts/artifacts.md (report/results.md) and FR-015. Every number is copied or
computed from the saved outputs: the cell rows from evaluate/results.csv, the verdict from
verdict/verdict.md, and the provenance from run_record.json. The verdict line is checked against
the pattern of contracts/cli.md before it is copied. Each row must carry the seed, configuration
hash, code version, and hardware class of the run record (FR-025, constitution Principle V). The
real-image rows of real/<dataset>/results.csv appear when those files exist, marked as reported
only (FR-020). This module implements no published method, so it cites no algorithm source.
"""

import csv
import io
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from strike_a_pose.checkpoint import atomic_write_text
from strike_a_pose.runrecord import STAGES, RunRecord, RunRecordError, read_run_record

__all__ = [
    "REAL_RESULT_COLUMNS",
    "RESULT_COLUMNS",
    "VERDICT_LINE_PATTERN",
    "ReportError",
    "render_results_markdown",
    "write_results_markdown",
]

# The columns of evaluate/results.csv with their kinds, in the order of contracts/artifacts.md.
_RESULT_KINDS: dict[str, str] = {
    "cell_id": "text",
    "views": "integer",
    "noise_deg": "number",
    "measurement": "text",
    "nominal_level": "number",
    "n_cal": "integer",
    "n_test": "integer",
    "coverage": "number",
    "median_width_cm": "number",
    "mae_cm": "number",
    "mean_signed_error_cm": "number",
    "clipped_count": "integer",
    "in_band": "flag",
    "q_hat": "number",
    "seed": "integer",
    "config_hash": "text",
    "code_version": "text",
    "hardware_class": "text",
}
# The columns of real/<dataset>/results.csv with their kinds, in the order of the same contract.
_REAL_KINDS: dict[str, str] = {
    "dataset": "text",
    "split": "text",
    "mask_source": "text",
    "cell_id": "text",
    "measurement": "text",
    "nominal_level": "number",
    "n_subjects": "integer",
    "n_skipped": "integer",
    "n_cal": "integer",
    "coverage": "number",
    "median_width_cm": "number",
    "mae_cm": "number",
    "mean_signed_error_cm": "number",
    "clipped_count": "integer",
    "q_hat": "number",
    "seed": "integer",
    "config_hash": "text",
    "code_version": "text",
    "hardware_class": "text",
}
RESULT_COLUMNS: tuple[str, ...] = tuple(_RESULT_KINDS)
REAL_RESULT_COLUMNS: tuple[str, ...] = tuple(_REAL_KINDS)

# The first line of verdict.md, in the fixed format of contracts/cli.md. The named groups capture
# the first 12 characters of the configuration hash and the seed, which must match the run record.
VERDICT_LINE_PATTERN = re.compile(
    r"VERDICT: (?:PASS|KILL) median_ratio=\S+ threshold=0\.700 cells_in_band=(?:true|false)"
    r" invalid=(?:true|false) sc004_violations=[0-9]+ \(chest=\S+ waist=\S+ hip=\S+ thigh=\S+\)"
    r" widths_v1_v4_cm=\(chest=\S+/\S+ waist=\S+/\S+ hip=\S+/\S+ thigh=\S+/\S+\)"
    r" config=(?P<config>[0-9a-f]{12}) seed=(?P<seed>[0-9]+)"
    r"(?:; comparison invalid for cells [a-z0-9_]+:[a-z]+(?:,[a-z0-9_]+:[a-z]+)*)?"
)

_CELL_HEADERS = (
    "Cell",
    "Measurement",
    "Nominal level",
    "Calibration bodies (n_cal)",
    "Test bodies (n_test)",
    "Seed",
    "Coverage",
    "In band",
    "Median width (cm)",
    "Width ratio to 1 view",
    "MAE (cm)",
    "Mean signed error (cm)",
    "Clipped intervals",
    "Calibration quantile (q_hat)",
)
_REAL_HEADERS = (
    "Dataset",
    "Split",
    "Mask source",
    "Cell",
    "Measurement",
    "Nominal level",
    "Subjects evaluated",
    "Subjects skipped",
    "Calibration bodies (n_cal)",
    "Seed",
    "Coverage",
    "Mean signed error (cm)",
    "Median width (cm)",
    "MAE (cm)",
    "Clipped intervals",
    "Calibration quantile (q_hat)",
)
_CELL_LEGEND = (
    "Cells are named `v<views>_n<noise>`: the view count, then the placement noise in degrees. "
    "Coverage is the share of test bodies whose interval holds the true measurement; compare it "
    "with the nominal level. A cell outside the tolerance band is flagged (FR-013). The width "
    "ratio divides the cell's median width by the median width of the 1-view cell at the same "
    "noise and measurement. MAE and the mean signed error are secondary metrics and follow the "
    "primary metrics."
)
_REAL_LEGEND = (
    "Reported only: these rows do not enter the kill verdict (FR-020). The mean signed error sits "
    "next to coverage because tape and surface measurements follow different protocols."
)


class ReportError(ValueError):
    """A report that cannot be written from its inputs. The message names the cause."""


def write_results_markdown(out_directory: str | os.PathLike[str]) -> Path:
    """Write report/results.md under the output directory, and return the path written.

    The inputs are evaluate/results.csv, verdict/verdict.md, and run_record.json, plus every
    real/<dataset>/results.csv that exists. Raises ReportError when an input is missing or fails a
    check, and names the file or row. The file is replaced atomically.
    """
    out = Path(out_directory)
    record = _read_record(out / "run_record.json")
    verdict_text = _read_text(out / "verdict" / "verdict.md", "verdict file")
    cell_rows = _read_table(out / "evaluate" / "results.csv", RESULT_COLUMNS, "results table")
    real_rows: list[dict[str, str]] = []
    for path in sorted((out / "real").glob("*/results.csv")):
        real_rows += _read_table(path, REAL_RESULT_COLUMNS, "real-image results table")
    text = render_results_markdown(
        verdict_text=verdict_text, cell_rows=cell_rows, record=record, real_rows=real_rows
    )
    return atomic_write_text(out / "report" / "results.md", text)


def render_results_markdown(
    *,
    verdict_text: str,
    cell_rows: Sequence[Mapping[str, Any]],
    record: RunRecord,
    real_rows: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Return the text of results.md for one run, built from the rows of its CSV tables.

    Raises ReportError when the first line of the verdict is not a verdict line, when the verdict or
    a row comes from another run, or when a row has a missing, non-numeric, or repeated value.
    """
    verdict_line, verdict_body = _verdict_parts(verdict_text, record)
    lines = [
        "# Kill-test results",
        "",
        "Generated by `sap report` from `evaluate/results.csv`, `verdict/verdict.md`, and "
        "`run_record.json`. The numbers are not edited by hand.",
        "",
        "## Verdict",
        "",
        "```text",
        verdict_line,
        "```",
        "",
    ]
    if verdict_body:
        lines += [verdict_body, ""]
    lines += ["## Results by cell", "", _CELL_LEGEND, "", *_cell_table(cell_rows, record), ""]
    if real_rows:
        lines += [
            "## Real-image results (reported only)",
            "",
            _REAL_LEGEND,
            "",
            *_real_table(real_rows, record),
            "",
        ]
    lines += _record_section(record)
    return "\n".join(lines) + "\n"


def _read_text(path: Path, label: str) -> str:
    """Return the UTF-8 text of a file, or raise ReportError that names it."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ReportError(
            f"cannot find the {label} '{path}'; the stage that writes it has not run"
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise ReportError(f"cannot read the {label} '{path}': {error}") from error


def _read_record(path: Path) -> RunRecord:
    """Return the run record at path. The message of a missing or invalid file names it."""
    try:
        return read_run_record(path)
    except RunRecordError as error:
        raise ReportError(str(error)) from error


def _read_table(path: Path, columns: Sequence[str], label: str) -> list[dict[str, str]]:
    """Return the data rows of a CSV table after checking that it has each named column."""
    reader = csv.DictReader(io.StringIO(_read_text(path, label)))
    header = reader.fieldnames or []
    missing = [column for column in columns if column not in header]
    if missing:
        raise ReportError(f"the {label} '{path}' lacks the column(s) {', '.join(missing)}")
    rows: list[dict[str, str]] = []
    try:
        for number, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ReportError(
                    f"data row {number} of the {label} '{path}' does not have one value per column"
                )
            rows.append(row)
    except csv.Error as error:
        raise ReportError(f"cannot parse the {label} '{path}': {error}") from error
    if not rows:
        raise ReportError(f"the {label} '{path}' has no data rows")
    return rows


def _verdict_parts(verdict_text: str, record: RunRecord) -> tuple[str, str]:
    """Return the verdict line and the rest of verdict.md, after checking the line and its run."""
    lines = verdict_text.splitlines()
    line = lines[0] if lines else ""
    match = VERDICT_LINE_PATTERN.fullmatch(line)
    if match is None:
        raise ReportError(
            "the first line of verdict.md is not a verdict line in the format of "
            f"contracts/cli.md: {line!r}"
        )
    config, seed = match.group("config"), int(match.group("seed"))
    if config != record.config_hash[:12] or seed != record.seed:
        raise ReportError(
            f"the verdict line names config {config} and seed {seed}, but the run record names "
            f"config {record.config_hash[:12]} and seed {record.seed}; the verdict comes from "
            "another run"
        )
    return line, "\n".join(lines[1:]).strip()


def _typed(row: Mapping[str, Any], kinds: Mapping[str, str], where: str) -> dict[str, Any]:
    """Convert the named columns of one row to their kinds: text, flag, integer, or number."""
    values: dict[str, Any] = {}
    for column, kind in kinds.items():
        if row.get(column) is None:
            raise ReportError(f"{where}: the column '{column}' is missing")
        text = str(row[column])
        if kind == "text":
            values[column] = text
        elif kind == "flag":
            folded = text.strip().lower()
            if folded not in ("true", "false"):
                raise ReportError(
                    f"{where}: the column '{column}' must be true or false; got {text!r}"
                )
            values[column] = folded == "true"
        else:
            parse = int if kind == "integer" else float
            noun = "a whole number" if kind == "integer" else "a number"
            try:
                values[column] = parse(text)
            except ValueError as error:
                raise ReportError(
                    f"{where}: the column '{column}' must be {noun}; got {text!r}"
                ) from error
    return values


def _check_provenance(values: Mapping[str, Any], where: str, record: RunRecord) -> None:
    """Refuse a row whose seed, hash, code version, or hardware class differs from the record."""
    for column, expected in (
        ("seed", record.seed),
        ("config_hash", record.config_hash),
        ("code_version", record.code_version),
        ("hardware_class", record.hardware_class),
    ):
        if values[column] != expected:
            raise ReportError(
                f"{where}: the column '{column}' is {values[column]!r}, but the run record says "
                f"{expected!r}; the row comes from another run"
            )


def _cell_table(rows: Sequence[Mapping[str, Any]], record: RunRecord) -> list[str]:
    """Return the Markdown table of the synthetic cells, each with its width ratio to 1 view."""
    cells: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for number, row in enumerate(rows, start=1):
        where = f"results row {number} ({row.get('cell_id')}, {row.get('measurement')})"
        cell = _typed(row, _RESULT_KINDS, where)
        _check_provenance(cell, where, record)
        key = (cell["cell_id"], cell["measurement"])
        if key in seen:
            raise ReportError(f"{where}: this cell and measurement appear in more than one row")
        seen.add(key)
        cells.append(cell)
    one_view_widths = {
        (cell["noise_deg"], cell["measurement"]): cell["median_width_cm"]
        for cell in cells
        if cell["views"] == 1
    }
    body = []
    for cell in cells:
        baseline = one_view_widths.get((cell["noise_deg"], cell["measurement"]))
        body.append(
            [
                cell["cell_id"],
                cell["measurement"],
                _percent(cell["nominal_level"], 0),
                str(cell["n_cal"]),
                str(cell["n_test"]),
                str(cell["seed"]),
                _percent(cell["coverage"], 1),
                "yes" if cell["in_band"] else "**no (flagged)**",
                _fixed(cell["median_width_cm"], 2),
                _ratio(cell["median_width_cm"], baseline),
                _fixed(cell["mae_cm"], 2),
                _fixed(cell["mean_signed_error_cm"], 2, signed=True),
                str(cell["clipped_count"]),
                _fixed(cell["q_hat"], 3),
            ]
        )
    return _table(_CELL_HEADERS, body)


def _real_table(rows: Sequence[Mapping[str, Any]], record: RunRecord) -> list[str]:
    """Return the Markdown table of the real-image rows, with mean signed error beside coverage."""
    body = []
    seen: set[tuple[str, ...]] = set()
    for number, row in enumerate(rows, start=1):
        where = (
            f"real-image row {number} ({row.get('dataset')}, {row.get('split')}, "
            f"{row.get('measurement')})"
        )
        values = _typed(row, _REAL_KINDS, where)
        _check_provenance(values, where, record)
        key = tuple(
            values[column]
            for column in ("dataset", "split", "mask_source", "cell_id", "measurement")
        )
        if key in seen:
            raise ReportError(f"{where}: this real-image row appears in more than one table")
        seen.add(key)
        body.append(
            [
                values["dataset"],
                values["split"],
                values["mask_source"],
                values["cell_id"],
                values["measurement"],
                _percent(values["nominal_level"], 0),
                str(values["n_subjects"]),
                str(values["n_skipped"]),
                str(values["n_cal"]),
                str(values["seed"]),
                _percent(values["coverage"], 1),
                _fixed(values["mean_signed_error_cm"], 2, signed=True),
                _fixed(values["median_width_cm"], 2),
                _fixed(values["mae_cm"], 2),
                str(values["clipped_count"]),
                _fixed(values["q_hat"], 3),
            ]
        )
    return _table(_REAL_HEADERS, body)


def _record_section(record: RunRecord) -> list[str]:
    """Return the run record section: provenance, library versions, and each stage timing."""
    rows = [
        ["Configuration hash", f"`{record.config_hash}`"],
        ["Seed", str(record.seed)],
        ["Code version", f"`{record.code_version}`"],
        ["Hardware class", record.hardware_class],
        ["Device", record.device_name],
    ]
    for name, version in record.versions.items():
        rows.append([f"Version: {name}", "not installed" if version is None else version])
    rows.append(["Started", record.started_at])
    rows.append(["Finished", record.finished_at or "not finished yet"])
    for stage in STAGES:
        seconds = record.timings[stage]
        timing = "not run" if seconds is None else _fixed(seconds, 2)
        rows.append([f"Timing: {stage} (seconds)", timing])
    return ["## Run record", "", *_table(("Field", "Value"), rows)]


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Return the lines of a Markdown table. Pipe characters inside a cell are escaped."""

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"

    return [line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)]


def _fixed(value: float, digits: int, signed: bool = False) -> str:
    """Format a number with a fixed count of decimals. nan and the infinities print as text."""
    if signed and math.isfinite(value):
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def _percent(fraction: float, digits: int) -> str:
    """Format a fraction as a percentage with a fixed count of decimals."""
    text = _fixed(100 * fraction, digits)
    return text + "%" if math.isfinite(fraction) else text


def _ratio(width: float, baseline: float | None) -> str:
    """Divide a width by its 1-view baseline to three decimals, or print n/a without a baseline."""
    if baseline is None or not math.isfinite(baseline) or baseline == 0:
        return "n/a"
    return _fixed(width / baseline, 3)
