"""Assert the shape of a tiny end-to-end run output directory (contracts/cli.md and artifacts.md).

Usage: python scripts/check_tiny_run.py <out-directory>

Exits 0 when every assertion holds. Exits 1 and names the failed assertion on stderr otherwise.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from strike_a_pose.report.tables import VERDICT_LINE_PATTERN

EXPECTED_RESULT_ROWS = 45
RUN_RECORD_KEYS = ("config_hash", "seed", "code_version", "hardware_class")
INVALID_SUFFIX = "; comparison invalid for cells "


class CheckFailure(Exception):
    """One assertion of the tiny-run check failed; the message names it."""


def _require(condition: bool, message: str) -> None:
    """Raise CheckFailure with the message when the condition does not hold."""
    if not condition:
        raise CheckFailure(message)


def _check_result_rows(out_directory: Path) -> None:
    """Assert that evaluate/results.csv holds exactly 45 data rows under a header row."""
    results_path = out_directory / "evaluate" / "results.csv"
    _require(results_path.is_file(), f"missing file evaluate/results.csv in {out_directory}")
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    data_rows = len(rows) - 1 if rows else 0
    _require(
        data_rows == EXPECTED_RESULT_ROWS,
        f"evaluate/results.csv has {data_rows} data rows, expected {EXPECTED_RESULT_ROWS}",
    )


def _check_verdict_line(out_directory: Path) -> None:
    """Assert that the first line of verdict/verdict.md is a verdict line, with the invalid suffix
    present exactly when cells_in_band=false."""
    verdict_path = out_directory / "verdict" / "verdict.md"
    _require(verdict_path.is_file(), f"missing file verdict/verdict.md in {out_directory}")
    lines = verdict_path.read_text(encoding="utf-8").splitlines()
    first_line = lines[0] if lines else ""
    _require(
        VERDICT_LINE_PATTERN.fullmatch(first_line) is not None,
        f"the first line of verdict/verdict.md is not a verdict line: {first_line!r}",
    )
    cells_in_band_true = " cells_in_band=true " in first_line
    has_suffix = INVALID_SUFFIX in first_line
    _require(
        has_suffix != cells_in_band_true,
        "the '; comparison invalid for cells' suffix must be present exactly when "
        f"cells_in_band=false, but the verdict line has suffix={has_suffix} and "
        f"cells_in_band={'true' if cells_in_band_true else 'false'}",
    )


def _check_run_record(out_directory: Path) -> None:
    """Assert that run_record.json is a JSON object holding the four run record fields."""
    record_path = out_directory / "run_record.json"
    _require(record_path.is_file(), f"missing file run_record.json in {out_directory}")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CheckFailure(f"run_record.json is not valid JSON: {error}") from error
    _require(isinstance(record, dict), "run_record.json does not hold a JSON object")
    for key in RUN_RECORD_KEYS:
        _require(key in record, f"run_record.json has no field '{key}'")


def check_tiny_run(out_directory: Path) -> None:
    """Run every assertion on the output directory; raise CheckFailure on the first failure."""
    _check_result_rows(out_directory)
    _check_verdict_line(out_directory)
    _check_run_record(out_directory)


def main(argv: list[str] | None = None) -> int:
    """Check the directory named on the command line; return the process exit code."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("check_tiny_run: usage: check_tiny_run.py <out-directory>", file=sys.stderr)
        return 1
    try:
        check_tiny_run(Path(arguments[0]))
    except CheckFailure as failure:
        print(f"check_tiny_run: FAILED: {failure}", file=sys.stderr)
        return 1
    print(f"check_tiny_run: OK: {arguments[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
