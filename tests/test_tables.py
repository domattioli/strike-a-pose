"""Smoke tests for report/tables.py: results.md from results.csv, verdict.md, and the run record."""

import csv
from pathlib import Path
from typing import Any

import pytest

from strike_a_pose.report.tables import (
    REAL_RESULT_COLUMNS,
    RESULT_COLUMNS,
    VERDICT_LINE_PATTERN,
    ReportError,
    render_results_markdown,
    write_results_markdown,
)
from strike_a_pose.runrecord import STAGES, RunRecord, write_run_record

CONFIG_HASH = "0123456789abcdef" * 4
CODE_VERSION = "0.1.0+abcdef012345"
SEED = 7
MEASUREMENTS = ("height", "chest", "waist", "hip", "thigh")
VERSIONS: dict[str, str | None] = {
    "python": "3.13.16",
    "numpy": "2.5.3",
    "torch": "2.14.1",
    "opencv": "4.14.0.94",
    "smplx": None,
    "sam2": None,
}
VERDICT_LINE = (
    "VERDICT: KILL median_ratio=0.816 threshold=0.700 cells_in_band=true invalid=false "
    "sc004_violations=0 (chest=0.800 waist=0.812 hip=0.820 thigh=0.830) "
    "widths_v1_v4_cm=(chest=10.000/8.000 waist=12.000/9.744 hip=13.000/10.660 "
    f"thigh=7.000/5.810) config={CONFIG_HASH[:12]} seed={SEED}"
)
VERDICT_TEXT = (
    VERDICT_LINE + "\n\n"
    "| Circumference | 1-view width (cm) | 4-view width (cm) | Ratio |\n"
    "| --- | --- | --- | --- |\n"
    "| chest | 10.000 | 8.000 | 0.800 |\n"
)


def _record(**changes: Any) -> RunRecord:
    """Return a run record for the test run, with two stage timings and one stage not yet run."""
    timings: dict[str, float | None] = dict.fromkeys(STAGES)
    timings["generate"] = 12.5
    timings["train"] = 3.25
    fields: dict[str, Any] = {
        "config_hash": CONFIG_HASH,
        "seed": SEED,
        "code_version": CODE_VERSION,
        "hardware_class": "cpu",
        "device_name": "cpu (test machine)",
        "versions": dict(VERSIONS),
        "started_at": "2026-10-07T12:00:00+00:00",
        "finished_at": "2026-10-07T12:30:00+00:00",
        "timings": timings,
    }
    fields.update(changes)
    return RunRecord(**fields)


def _cell_row(views: int, measurement: str, **changes: str) -> dict[str, str]:
    """Return one row of evaluate/results.csv. The 4-view width is 8 cm against 10 cm at 1 view."""
    row = {
        "cell_id": f"v{views}_n0",
        "views": str(views),
        "noise_deg": "0.0",
        "measurement": measurement,
        "nominal_level": "0.9",
        "n_cal": "2500",
        "n_test": "2500",
        "coverage": "0.9012",
        "median_width_cm": "10.0" if views == 1 else "8.0",
        "mae_cm": "0.50",
        "mean_signed_error_cm": "-0.10",
        "clipped_count": "0",
        "in_band": "true",
        "q_hat": "1.250",
        "seed": str(SEED),
        "config_hash": CONFIG_HASH,
        "code_version": CODE_VERSION,
        "hardware_class": "cpu",
    }
    row.update(changes)
    return row


def _cell_rows() -> list[dict[str, str]]:
    """Return the ten rows of a small run: 1 view and 4 views, five measurements each."""
    return [_cell_row(views, measurement) for views in (1, 4) for measurement in MEASUREMENTS]


def _real_row(**changes: str) -> dict[str, str]:
    """Return one row of real/bodym/results.csv."""
    row = {
        "dataset": "bodym",
        "split": "testA",
        "mask_source": "provided",
        "cell_id": "v2_n0",
        "measurement": "chest",
        "nominal_level": "0.9",
        "n_subjects": "1200",
        "n_skipped": "3",
        "n_cal": "2500",
        "coverage": "0.8500",
        "median_width_cm": "9.00",
        "mae_cm": "2.10",
        "mean_signed_error_cm": "1.25",
        "clipped_count": "0",
        "q_hat": "1.300",
        "seed": str(SEED),
        "config_hash": CONFIG_HASH,
        "code_version": CODE_VERSION,
        "hardware_class": "cpu",
    }
    row.update(changes)
    return row


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    """Write rows under the given header, creating the parent directory first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_run(out: Path, rows: list[dict[str, str]] | None = None) -> Path:
    """Write the inputs that sap report reads: run record, verdict, and the results table."""
    write_run_record(_record(), out / "run_record.json")
    (out / "verdict").mkdir(parents=True, exist_ok=True)
    (out / "verdict" / "verdict.md").write_text(VERDICT_TEXT, encoding="utf-8")
    cell_rows = _cell_rows() if rows is None else rows
    _write_csv(out / "evaluate" / "results.csv", cell_rows, list(RESULT_COLUMNS))
    return out


def _split_cells(line: str) -> list[str]:
    """Return the cells of one Markdown table line, without the outer pipes."""
    return [cell.strip() for cell in line.strip()[1:-1].split("|")]


def _table_after(text: str, first_header: str) -> list[dict[str, str]]:
    """Return the rows of the Markdown table whose header starts with first_header, by column."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {first_header} |"))
    headers = _split_cells(lines[start])
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(dict(zip(headers, _split_cells(line), strict=True)))
    return rows


def _render(rows: list[dict[str, str]] | None = None, **changes: Any) -> str:
    """Render results.md from in-memory rows, with the verdict and run record of the test run."""
    options: dict[str, Any] = {
        "verdict_text": VERDICT_TEXT,
        "cell_rows": _cell_rows() if rows is None else rows,
        "record": _record(),
    }
    options.update(changes)
    return render_results_markdown(**options)


def test_results_markdown_is_written_under_report_and_carries_the_verdict(output_dir: Path) -> None:
    out = _write_run(output_dir)
    path = write_results_markdown(out)
    assert path == out / "report" / "results.md"
    text = path.read_text(encoding="utf-8")
    assert VERDICT_LINE in text.splitlines()
    assert len(_table_after(text, "Cell")) == 10
    assert CONFIG_HASH in text
    assert CODE_VERSION in text
    assert sorted(path.parent.iterdir()) == [path]
    first_bytes = path.read_bytes()
    write_results_markdown(out)
    assert path.read_bytes() == first_bytes


def test_coverage_is_a_percentage_and_width_ratios_compare_with_the_one_view_cell() -> None:
    rows = {(row["Cell"], row["Measurement"]): row for row in _table_after(_render(), "Cell")}
    one_view = rows[("v1_n0", "chest")]
    four_view = rows[("v4_n0", "chest")]
    assert one_view["Coverage"] == "90.1%"
    assert one_view["Nominal level"] == "90%"
    assert one_view["Calibration bodies (n_cal)"] == "2500"
    assert one_view["Test bodies (n_test)"] == "2500"
    assert one_view["Seed"] == str(SEED)
    assert one_view["In band"] == "yes"
    assert one_view["Width ratio to 1 view"] == "1.000"
    assert four_view["Median width (cm)"] == "8.00"
    assert four_view["Width ratio to 1 view"] == "0.800"
    assert four_view["Mean signed error (cm)"] == "-0.10"


def test_a_cell_outside_the_tolerance_band_is_flagged() -> None:
    rows = _cell_rows()
    rows[0] = _cell_row(1, "height", coverage="0.8500", in_band="false")
    table = _table_after(_render(rows), "Cell")
    assert table[0]["In band"] == "**no (flagged)**"
    assert table[1]["In band"] == "yes"


def test_a_ratio_without_a_one_view_baseline_reads_not_applicable() -> None:
    rows = [_cell_row(4, measurement) for measurement in MEASUREMENTS]
    table = _table_after(_render(rows), "Cell")
    assert {row["Width ratio to 1 view"] for row in table} == {"n/a"}


def test_the_verdict_line_pattern_accepts_the_comparison_note_and_nothing_else() -> None:
    note = VERDICT_LINE.replace("cells_in_band=true", "cells_in_band=false")
    note += "; comparison invalid for cells v4_n0:waist,v1_n0:hip"
    assert VERDICT_LINE_PATTERN.fullmatch(VERDICT_LINE)
    assert VERDICT_LINE_PATTERN.fullmatch(note)
    assert VERDICT_LINE_PATTERN.fullmatch(VERDICT_LINE + " trailing") is None


@pytest.mark.parametrize(
    "verdict_text",
    [
        "VERDICT: KILL nonsense\n",
        VERDICT_LINE.replace("VERDICT:", "verdict:") + "\n",
        "",
    ],
)
def test_the_first_line_of_the_verdict_must_be_a_verdict_line(verdict_text: str) -> None:
    with pytest.raises(ReportError, match="not a verdict line"):
        _render(verdict_text=verdict_text)


def test_a_verdict_from_another_run_is_refused() -> None:
    other = VERDICT_TEXT.replace(f"config={CONFIG_HASH[:12]}", "config=fedcba987654")
    with pytest.raises(ReportError, match="another run"):
        _render(verdict_text=other)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("seed", "8"),
        ("config_hash", "f" * 64),
        ("code_version", "0.1.0+other"),
        ("hardware_class", "gpu"),
    ],
)
def test_a_cell_row_from_another_run_is_refused(column: str, value: str) -> None:
    rows = _cell_rows()
    rows[3][column] = value
    with pytest.raises(ReportError, match=f"'{column}'"):
        _render(rows)


def test_a_non_numeric_value_is_refused_and_named() -> None:
    rows = _cell_rows()
    rows[0]["coverage"] = "high"
    with pytest.raises(ReportError, match="'coverage' must be a number"):
        _render(rows)


def test_a_cell_and_measurement_listed_twice_is_refused() -> None:
    rows = [*_cell_rows(), _cell_row(1, "height")]
    with pytest.raises(ReportError, match="more than one row"):
        _render(rows)


def test_a_results_file_without_a_named_column_is_refused_by_the_write(output_dir: Path) -> None:
    out = _write_run(output_dir)
    columns = [column for column in RESULT_COLUMNS if column != "coverage"]
    rows = [{key: value for key, value in row.items() if key != "coverage"} for row in _cell_rows()]
    _write_csv(out / "evaluate" / "results.csv", rows, columns)
    with pytest.raises(ReportError, match="lacks the column\\(s\\) coverage"):
        write_results_markdown(out)
    assert not (out / "report" / "results.md").exists()


def test_a_results_file_without_data_rows_is_refused(output_dir: Path) -> None:
    out = _write_run(output_dir, rows=[])
    with pytest.raises(ReportError, match="no data rows"):
        write_results_markdown(out)


def test_a_missing_run_record_is_named_by_the_write(output_dir: Path) -> None:
    out = _write_run(output_dir)
    (out / "run_record.json").unlink()
    with pytest.raises(ReportError, match="run_record.json"):
        write_results_markdown(out)


def test_real_rows_appear_as_reported_only_when_their_files_exist(output_dir: Path) -> None:
    out = _write_run(output_dir)
    assert "Real-image results" not in write_results_markdown(out).read_text(encoding="utf-8")
    real_columns = list(REAL_RESULT_COLUMNS)
    _write_csv(out / "real" / "bodym" / "results.csv", [_real_row()], real_columns)
    text = write_results_markdown(out).read_text(encoding="utf-8")
    assert "## Real-image results (reported only)" in text
    real = _table_after(text, "Dataset")
    assert len(real) == 1
    assert real[0]["Coverage"] == "85.0%"
    assert real[0]["Mean signed error (cm)"] == "+1.25"
    assert real[0]["Subjects skipped"] == "3"


def test_a_real_row_from_another_run_is_refused() -> None:
    with pytest.raises(ReportError, match="'seed'"):
        render_results_markdown(
            verdict_text=VERDICT_TEXT,
            cell_rows=_cell_rows(),
            record=_record(),
            real_rows=[_real_row(seed="8")],
        )


def test_the_run_record_section_gives_timings_versions_and_unfinished_parts() -> None:
    text = _render(record=_record(finished_at=None))
    fields = {row["Field"]: row["Value"] for row in _table_after(text, "Field")}
    assert fields["Configuration hash"] == f"`{CONFIG_HASH}`"
    assert fields["Seed"] == str(SEED)
    assert fields["Code version"] == f"`{CODE_VERSION}`"
    assert fields["Hardware class"] == "cpu"
    assert fields["Version: numpy"] == "2.5.3"
    assert fields["Version: smplx"] == "not installed"
    assert fields["Finished"] == "not finished yet"
    assert fields["Timing: generate (seconds)"] == "12.50"
    assert fields["Timing: predict (seconds)"] == "not run"
