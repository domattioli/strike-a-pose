"""Smoke tests for verdict.py: the FR-014 rule, band misses, invalid comparisons, and recompute."""

import csv
import json
import math
import random
import re
import statistics
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from strike_a_pose.checkpoint import StageStatus, check_stage, read_done_marker, write_done_marker
from strike_a_pose.cli import main
from strike_a_pose.config import FIXED_KEYS, ConfigError, config_hash, load_config
from strike_a_pose.report.tables import RESULT_COLUMNS, VERDICT_LINE_PATTERN, write_results_markdown
from strike_a_pose.runrecord import RunRecord, start_run_record, write_run_record
from strike_a_pose.verdict import (
    BAND,
    COMPARE_VIEWS,
    EVALUATED_MEASUREMENTS,
    FIXED_CONFIGURATION,
    MEASUREMENTS,
    NOISE_DEG,
    REPORTED_NOISE_DEG,
    STAGE_REFUSED_EXIT_CODE,
    THRESHOLD,
    KillVerdict,
    ResultCell,
    VerdictError,
    VerdictInputError,
    VerdictRefusedError,
    check_fixed_configuration,
    compute_verdict,
    recompute,
    run_verdict,
    verdict_directory,
    width_ratio,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_CONTRACT = REPO_ROOT / "specs" / "001-kill-test-mvp" / "contracts" / "cli.md"

CIRCUMFERENCES = ("chest", "waist", "hip", "thigh")
ALL_MEASUREMENTS = ("height", *CIRCUMFERENCES)
VIEWS = (1, 2, 4)
NOISE = (0.0, 2.0, 5.0)
CODE_VERSION = "0.1.0+abcdef012345"
VERSIONS: dict[str, str | None] = {
    "python": "3.13.16",
    "numpy": "2.5.3",
    "torch": "2.14.1",
    "opencv": "4.14.0.94",
    "smplx": None,
    "sam2": None,
}
STARTED_AT = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)

# Median interval widths in cm at 0 degrees placement noise. The 1-view widths are the baseline.
# The PASS widths give the ratios 0.6, 0.625, 0.615, and 0.643 (median 0.620); the KILL widths give
# 0.9 for every circumference. Height has its own widths and never enters the verdict.
ONE_VIEW_CM = {"height": 6.0, "chest": 10.0, "waist": 12.0, "hip": 13.0, "thigh": 7.0}
PASS_FOUR_VIEW_CM = {"height": 4.5, "chest": 6.0, "waist": 7.5, "hip": 8.0, "thigh": 4.5}
KILL_FOUR_VIEW_CM = {"height": 5.4, "chest": 9.0, "waist": 10.8, "hip": 11.7, "thigh": 6.3}

VERDICT_FILES = ("verdict.json", "verdict.md")
# The keys of verdict.json in the order of contracts/artifacts.md.
VERDICT_JSON_KEYS = [
    "verdict",
    "median_ratio",
    "threshold",
    "cells_in_band",
    "out_of_band_cells",
    "invalid_comparison",
    "sc004_violations",
    "ratios",
    "widths",
    "noise_deg",
    "compare_views",
    "reported_only",
    "config_hash",
    "seed",
    "code_version",
    "hardware_class",
]


class Run(NamedTuple):
    """One output directory with the inputs of the verdict stage written into it."""

    config: dict[str, Any]
    out: Path
    record: RunRecord


@pytest.fixture
def config(tiny_config_path: Path, small_config: dict[str, object]) -> dict[str, Any]:
    """The tiny configuration with the test overrides of contracts/config.md, resolved."""
    overrides = [f"{key}={json.dumps(value)}" for key, value in small_config.items()]
    return load_config(tiny_config_path, overrides=overrides)


def cell_id(views: int, noise: float) -> str:
    """Return the cell name of contracts/artifacts.md, for example v4_n0."""
    return f"v{views}_n{noise:g}"


def cell_width(
    views: int,
    noise: float,
    name: str,
    one_view: Mapping[str, float] = ONE_VIEW_CM,
    four_view: Mapping[str, float] = PASS_FOUR_VIEW_CM,
) -> float:
    """Return the median width of a fixture cell: noise widens every interval by 5% per degree."""
    if views == 1:
        base = one_view[name]
    elif views == 4:
        base = four_view[name]
    else:
        base = (one_view[name] + four_view[name]) / 2
    return base * (1.0 + 0.05 * noise)


def result_rows(
    record: RunRecord,
    *,
    one_view: Mapping[str, float] = ONE_VIEW_CM,
    four_view: Mapping[str, float] = PASS_FOUR_VIEW_CM,
    views: tuple[int, ...] = VIEWS,
    noises: tuple[float, ...] = NOISE,
) -> list[dict[str, str]]:
    """Return the rows of evaluate/results.csv in the layout of contracts/artifacts.md.

    The order is cell order, then measurement order. Every row is in band and carries the provenance
    of the run record. A fraction, not a percentage, is written for the coverage.
    """
    return [
        {
            "cell_id": cell_id(count, noise),
            "views": str(count),
            "noise_deg": repr(float(noise)),
            "measurement": name,
            "nominal_level": "0.9",
            "n_cal": "2500",
            "n_test": "2500",
            "coverage": "0.9",
            "median_width_cm": repr(cell_width(count, noise, name, one_view, four_view)),
            "mae_cm": "0.5",
            "mean_signed_error_cm": "-0.1",
            "clipped_count": "0",
            "in_band": "true",
            "q_hat": "1.25",
            "seed": str(record.seed),
            "config_hash": record.config_hash,
            "code_version": record.code_version,
            "hardware_class": record.hardware_class,
        }
        for count in views
        for noise in noises
        for name in ALL_MEASUREMENTS
    ]


def find(rows: list[dict[str, str]], cell: str, name: str) -> dict[str, str]:
    """Return the row of one cell and measurement."""
    return next(row for row in rows if row["cell_id"] == cell and row["measurement"] == name)


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str] | None = None) -> None:
    """Write rows under the header of evaluate/results.csv (or the given header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns or list(RESULT_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sc004_document(record: RunRecord, violations: int = 0) -> dict[str, Any]:
    """Return the fields of evaluate/sc004.json in the layout of contracts/artifacts.md."""
    return {
        "n_bodies": 2000,
        "noise_levels": [0.0, 2.0, 5.0],
        "pairs": ["v2_vs_v1", "v4_vs_v2"],
        "violations": violations,
        "max_relative_excess": 0.0 if violations == 0 else 3e-05,
        "tolerance": 1e-05,
        "config_hash": record.config_hash,
        "seed": record.seed,
        "code_version": record.code_version,
        "hardware_class": record.hardware_class,
    }


def write_sc004(path: Path, document: dict[str, Any]) -> None:
    """Write evaluate/sc004.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def make_run(
    config: dict[str, Any],
    out: Path,
    *,
    edit: Callable[[list[dict[str, str]]], list[dict[str, str]]] | None = None,
    violations: int = 0,
    **row_options: Any,
) -> Run:
    """Write the inputs of the verdict stage: run record, results table, SC-004 check, and marker.

    ``edit`` receives the rows before they are written and returns the rows to write.
    """
    record = start_run_record(
        config,
        hardware_class="cpu",
        device_name="cpu (test machine)",
        started_at=STARTED_AT,
        code_version=CODE_VERSION,
        versions=VERSIONS,
    )
    write_run_record(record, out / "run_record.json")
    rows = result_rows(record, **row_options)
    if edit is not None:
        rows = edit(rows)
    write_csv(out / "evaluate" / "results.csv", rows)
    write_sc004(out / "evaluate" / "sc004.json", sc004_document(record, violations))
    write_done_marker(
        out / "evaluate",
        stage="evaluate",
        config_hash=record.config_hash,
        seed=record.seed,
        code_version=record.code_version,
        hardware_class=record.hardware_class,
        inputs={"calibrate": "b" * 64},
    )
    return Run(config, out, record)


def set_cells(
    rows: list[dict[str, str]], cell: str, names: tuple[str, ...], **changes: str
) -> list[dict[str, str]]:
    """Change columns of the rows of one cell, for the given measurements."""
    for name in names:
        find(rows, cell, name).update(changes)
    return rows


def read_document(out: Path) -> dict[str, Any]:
    """Read verdict/verdict.json, and refuse NaN and the infinities, which JSON does not allow."""

    def refuse(constant: str) -> Any:
        raise AssertionError(f"verdict.json holds the JSON constant {constant}")

    text = (verdict_directory(out) / "verdict.json").read_text(encoding="utf-8")
    return json.loads(text, parse_constant=refuse)


def read_markdown(out: Path) -> str:
    """Read verdict/verdict.md."""
    return (verdict_directory(out) / "verdict.md").read_text(encoding="utf-8")


def snapshot(out: Path) -> dict[str, bytes]:
    """Return the bytes of every file under the output directory, by relative path."""
    return {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }


def verdict_of(run: Run) -> KillVerdict:
    """Run the stage, and return the verdict whether it returned or refused with exit code 4."""
    try:
        return run_verdict(run.config, run.out)
    except VerdictRefusedError as refused:
        return refused.verdict


def contract_pattern() -> re.Pattern[str]:
    """Return the verdict-line regex that contracts/cli.md gives to scripts/check_tiny_run.py."""
    text = CLI_CONTRACT.read_text(encoding="utf-8")
    blocks = [
        block
        for block in re.findall(r"```text\n(.*?)\n```", text, flags=re.DOTALL)
        if block.startswith("^VERDICT: ")
    ]
    assert len(blocks) == 1, "contracts/cli.md must hold exactly one verdict-line regex block"
    return re.compile(blocks[0])


def markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    """Return the rows of the first Markdown table after a heading, by column header."""
    lines = text.splitlines()
    table = []
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("|"):
            table.append([cell.strip() for cell in line.strip()[1:-1].split("|")])
        elif table:
            break
    return [dict(zip(table[0], row, strict=True)) for row in table[2:]]


def expected_line(
    record: RunRecord, ratios: str, median: str, widths: str, *, band: str = "true", **options: str
) -> str:
    """Return the verdict line of a fixture run, written out from the format of the contract."""
    return (
        f"VERDICT: {options.get('verdict', 'PASS')} median_ratio={median} threshold=0.700 "
        f"cells_in_band={band} invalid={options.get('invalid', 'false')} "
        f"sc004_violations={options.get('violations', '0')} ({ratios}) "
        f"widths_v1_v4_cm=({widths}) config={record.config_hash[:12]} seed={record.seed}"
    )


PASS_RATIOS = "chest=0.600 waist=0.625 hip=0.615 thigh=0.643"
PASS_WIDTHS = "chest=10.000/6.000 waist=12.000/7.500 hip=13.000/8.000 thigh=7.000/4.500"
KILL_RATIOS = "chest=0.900 waist=0.900 hip=0.900 thigh=0.900"
KILL_WIDTHS = "chest=10.000/9.000 waist=12.000/10.800 hip=13.000/11.700 thigh=7.000/6.300"


def test_the_constants_state_the_fr014_rule() -> None:
    assert THRESHOLD == 0.70
    assert MEASUREMENTS == ("chest", "waist", "hip", "thigh")
    assert NOISE_DEG == 0.0
    assert COMPARE_VIEWS == (1, 4)
    assert BAND == (0.87, 0.93)
    assert EVALUATED_MEASUREMENTS == ALL_MEASUREMENTS
    assert REPORTED_NOISE_DEG == (2.0, 5.0)
    assert STAGE_REFUSED_EXIT_CODE == 4


def test_the_constants_equal_the_fixed_configuration_keys() -> None:
    held = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in FIXED_CONFIGURATION.items()
    }
    assert held == FIXED_KEYS


@pytest.mark.parametrize("name", ["tiny.yaml", "full.yaml"])
def test_the_shipped_configurations_pass_the_fixed_check(name: str) -> None:
    check_fixed_configuration(load_config(REPO_ROOT / "configs" / name))


def test_a_pass_fixture_gives_pass_and_writes_the_verdict_files(
    config: dict[str, Any], output_dir: Path
) -> None:
    run = make_run(config, output_dir)
    result = run_verdict(config, output_dir)
    ratios = {name: PASS_FOUR_VIEW_CM[name] / ONE_VIEW_CM[name] for name in CIRCUMFERENCES}
    line = expected_line(run.record, PASS_RATIOS, "0.620", PASS_WIDTHS)

    assert result.verdict == "PASS"
    assert result.exit_code == 0
    assert result.line == line
    assert read_markdown(output_dir).splitlines()[0] == line
    document = read_document(output_dir)
    assert list(document) == VERDICT_JSON_KEYS
    assert document["verdict"] == "PASS"
    assert document["median_ratio"] == pytest.approx(statistics.median(ratios.values()))
    assert document["threshold"] == 0.7
    assert document["cells_in_band"] is True
    assert document["out_of_band_cells"] == []
    assert document["invalid_comparison"] is False
    assert document["sc004_violations"] == 0
    assert document["ratios"] == ratios
    assert document["widths"] == {
        "v1": {name: ONE_VIEW_CM[name] for name in CIRCUMFERENCES},
        "v4": {name: PASS_FOUR_VIEW_CM[name] for name in CIRCUMFERENCES},
    }
    assert document["noise_deg"] == 0.0
    assert document["compare_views"] == [1, 4]
    assert list(document["reported_only"]) == ["height_ratio", "rows_2deg", "rows_5deg"]
    assert document["config_hash"] == run.record.config_hash
    assert document["seed"] == run.record.seed
    assert document["code_version"] == CODE_VERSION
    assert document["hardware_class"] == "cpu"


def test_the_stage_marker_follows_the_exit_code(config: dict[str, Any], output_dir: Path) -> None:
    run = make_run(config, output_dir)
    digest = config_hash(config)
    directory = verdict_directory(output_dir)
    run_verdict(config, output_dir)
    marker = read_done_marker(directory)
    assert marker is not None
    assert marker.stage == "verdict"
    assert marker.inputs == {"evaluate": digest}
    assert marker.code_version == CODE_VERSION
    assert marker.hardware_class == "cpu"
    status = check_stage(
        directory, stage="verdict", config_hash=digest, inputs={"evaluate": digest}
    )
    assert status.status is StageStatus.DONE

    make_run(run.config, output_dir, violations=1)
    with pytest.raises(VerdictRefusedError):
        run_verdict(config, output_dir)
    assert read_done_marker(directory) is None  # a refused verdict cannot look done


def test_running_the_stage_twice_gives_identical_verdict_files(
    config: dict[str, Any], output_dir: Path
) -> None:
    make_run(config, output_dir)
    run_verdict(config, output_dir)
    first = {name: (verdict_directory(output_dir) / name).read_bytes() for name in VERDICT_FILES}
    run_verdict(config, output_dir)
    second = {name: (verdict_directory(output_dir) / name).read_bytes() for name in VERDICT_FILES}
    assert first == second


def test_a_kill_fixture_above_the_threshold_is_a_kill_that_exits_zero(
    config: dict[str, Any], output_dir: Path
) -> None:
    run = make_run(config, output_dir, four_view=KILL_FOUR_VIEW_CM)
    result = run_verdict(config, output_dir)  # returns: the command exits 0

    assert result.verdict == "KILL"
    assert result.exit_code == 0
    assert result.cells_in_band is True
    assert result.invalid_comparison is False
    assert result.line == expected_line(
        run.record, KILL_RATIOS, "0.900", KILL_WIDTHS, verdict="KILL"
    )
    document = read_document(output_dir)
    assert document["verdict"] == "KILL"
    assert document["median_ratio"] == pytest.approx(0.9)
    assert read_done_marker(verdict_directory(output_dir)) is not None
    assert "above the threshold 0.700" in read_markdown(output_dir)


def test_the_threshold_is_inclusive_and_compares_the_unrounded_median(
    config: dict[str, Any], tmp_path: Path
) -> None:
    one_view = dict.fromkeys(ONE_VIEW_CM, 10.0)
    exactly = make_run(
        config, tmp_path / "exactly", one_view=one_view, four_view=dict.fromkeys(one_view, 7.0)
    )
    assert run_verdict(config, exactly.out).verdict == "PASS"

    above = make_run(
        config, tmp_path / "above", one_view=one_view, four_view=dict.fromkeys(one_view, 7.001)
    )
    result = run_verdict(config, above.out)
    assert result.verdict == "KILL"
    assert "median_ratio=0.700 " in result.line  # three decimals hide the excess; the files keep it
    assert read_document(above.out)["median_ratio"] == pytest.approx(0.7001)
    assert "0.7001" in read_markdown(above.out)


@pytest.mark.parametrize(
    ("four_view_widths", "expected_median", "expected_verdict"),
    [
        # Two ratios below 0.70 do not carry the median: the mean of 0.90 and 0.60 is 0.75.
        ({"chest": 5.0, "waist": 6.0, "hip": 9.0, "thigh": 9.5}, 0.75, "KILL"),
        # One ratio above 0.70 does not move the median: the mean of 0.60 and 0.70 is 0.65.
        ({"chest": 5.0, "waist": 6.0, "hip": 7.0, "thigh": 9.0}, 0.65, "PASS"),
    ],
    ids=["two-low-ratios-do-not-pass", "one-high-ratio-does-not-kill"],
)
def test_the_median_ratio_is_the_median_of_the_four_ratios(
    config: dict[str, Any],
    output_dir: Path,
    four_view_widths: dict[str, float],
    expected_median: float,
    expected_verdict: str,
) -> None:
    one_view = dict.fromkeys(ONE_VIEW_CM, 10.0)
    make_run(config, output_dir, one_view=one_view, four_view={"height": 5.0, **four_view_widths})
    result = run_verdict(config, output_dir)
    ratios = [four_view_widths[name] / 10.0 for name in CIRCUMFERENCES]
    assert statistics.median(ratios) == pytest.approx(expected_median)
    assert read_document(output_dir)["median_ratio"] == pytest.approx(expected_median)
    assert result.verdict == expected_verdict


def test_a_compared_cell_outside_the_band_is_a_legitimate_kill_that_exits_zero(
    config: dict[str, Any], output_dir: Path
) -> None:
    def edit(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        set_cells(rows, "v4_n0", ("waist",), in_band="false", coverage="0.85")
        return set_cells(rows, "v1_n0", ("hip",), in_band="false", coverage="0.95")

    run = make_run(config, output_dir, edit=edit)
    result = run_verdict(config, output_dir)  # returns: the command exits 0

    assert result.verdict == "KILL"
    assert result.exit_code == 0
    assert result.cells_in_band is False
    assert result.invalid_comparison is False
    assert result.median_ratio <= THRESHOLD  # the band alone makes this a KILL
    assert result.out_of_band_cells == ("v4_n0:waist", "v1_n0:hip")
    line = (
        expected_line(run.record, PASS_RATIOS, "0.620", PASS_WIDTHS, verdict="KILL", band="false")
        + "; comparison invalid for cells v4_n0:waist,v1_n0:hip"
    )
    assert result.line == line
    assert read_markdown(output_dir).splitlines()[0] == line
    document = read_document(output_dir)
    assert document["verdict"] == "KILL"
    assert document["cells_in_band"] is False
    assert document["out_of_band_cells"] == ["v4_n0:waist", "v1_n0:hip"]
    assert document["invalid_comparison"] is False
    assert read_done_marker(verdict_directory(output_dir)) is not None
    flags = {
        row["Circumference"]: (row["1-view cell in band"], row["4-view cell in band"])
        for row in markdown_table(
            read_markdown(output_dir), "### Width comparison at 0 degrees placement noise"
        )
    }
    assert flags == {
        "chest": ("yes", "yes"),
        "waist": ("yes", "**no (flagged)**"),
        "hip": ("**no (flagged)**", "yes"),
        "thigh": ("yes", "yes"),
    }


def test_the_comparison_note_is_present_exactly_when_cells_are_outside_the_band(
    config: dict[str, Any], tmp_path: Path
) -> None:
    clean = verdict_of(make_run(config, tmp_path / "clean"))
    assert "comparison invalid" not in clean.line
    for compared in (("v1_n0", "chest"), ("v4_n0", "thigh")):
        run = make_run(
            config,
            tmp_path / "_".join(compared),
            edit=lambda rows, compared=compared: set_cells(
                rows, compared[0], (compared[1],), in_band="false"
            ),
        )
        line = verdict_of(run).line
        assert line.endswith(f"; comparison invalid for cells {compared[0]}:{compared[1]}")
        assert "cells_in_band=false" in line


NON_FINITE_CASES = [
    pytest.param({("v1_n0", "waist"): "0.0"}, "waist", id="one-view-width-zero-gives-infinity"),
    pytest.param(
        {("v1_n0", "waist"): "0.0", ("v4_n0", "waist"): "0.0"}, "waist", id="both-widths-zero"
    ),
    pytest.param({("v4_n0", "hip"): "nan"}, "hip", id="four-view-width-not-a-number"),
    pytest.param({("v1_n0", "chest"): "inf"}, "chest", id="one-view-width-infinite"),
    pytest.param({("v4_n0", "thigh"): "inf"}, "thigh", id="four-view-width-infinite"),
    pytest.param({("v1_n0", "chest"): "-1.0"}, "chest", id="negative-width"),
]


@pytest.mark.parametrize(("widths", "named"), NON_FINITE_CASES)
def test_a_non_finite_ratio_gives_kill_written_before_exit_4(
    config: dict[str, Any], output_dir: Path, widths: dict[tuple[str, str], str], named: str
) -> None:
    def edit(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        for (cell, name), text in widths.items():
            find(rows, cell, name)["median_width_cm"] = text
        return rows

    make_run(config, output_dir, edit=edit)
    with pytest.raises(VerdictRefusedError) as refused:
        run_verdict(config, output_dir)

    assert refused.value.exit_code == 4
    assert f"width ratio for {named} is not finite" in str(refused.value)
    result = refused.value.verdict
    assert result.verdict == "KILL"
    assert result.invalid_comparison is True
    assert result.exit_code == 4
    document = read_document(output_dir)  # both files exist at the time of the exit
    assert document["verdict"] == "KILL"
    assert document["invalid_comparison"] is True
    assert document["ratios"][named] is None
    first_line = read_markdown(output_dir).splitlines()[0]
    assert first_line == result.line
    assert first_line.startswith("VERDICT: KILL ")
    assert " invalid=true " in first_line
    assert read_done_marker(verdict_directory(output_dir)) is None


MISSING_CASES = [
    pytest.param(
        lambda rows: [r for r in rows if (r["cell_id"], r["measurement"]) != ("v4_n0", "hip")],
        ["v4_n0:hip"],
        id="one-cell",
    ),
    pytest.param(
        lambda rows: [r for r in rows if r["views"] != "4"],
        [f"v{v}_n0:{n}" for n in CIRCUMFERENCES for v in (4,)],
        id="no-four-view-cells",
    ),
    pytest.param(
        lambda rows: [], [f"v{v}_n0:{n}" for n in CIRCUMFERENCES for v in (1, 4)], id="empty-table"
    ),
]


@pytest.mark.parametrize(("edit", "named"), MISSING_CASES)
def test_a_missing_compared_cell_gives_kill_with_invalid_comparison_written_before_exit_4(
    config: dict[str, Any], output_dir: Path, edit: Callable, named: list[str]
) -> None:
    make_run(config, output_dir, edit=edit)
    with pytest.raises(VerdictRefusedError) as refused:
        run_verdict(config, output_dir)

    assert refused.value.exit_code == 4
    assert ", ".join(named) in str(refused.value)
    document = read_document(output_dir)
    assert document["verdict"] == "KILL"
    assert document["invalid_comparison"] is True
    assert document["median_ratio"] is None
    assert document["out_of_band_cells"] == []  # a missing cell is not an out-of-band cell
    assert document["cells_in_band"] is True
    assert read_markdown(output_dir).splitlines()[0] == refused.value.verdict.line
    assert " invalid=true " in refused.value.verdict.line
    assert read_done_marker(verdict_directory(output_dir)) is None


def test_a_missing_cell_is_null_in_the_verdict_file(
    config: dict[str, Any], output_dir: Path
) -> None:
    make_run(
        config,
        output_dir,
        edit=lambda rows: [r for r in rows if (r["cell_id"], r["measurement"]) != ("v4_n0", "hip")],
    )
    with pytest.raises(VerdictRefusedError):
        run_verdict(config, output_dir)
    document = read_document(output_dir)
    assert document["widths"]["v4"]["hip"] is None
    assert document["widths"]["v1"]["hip"] == ONE_VIEW_CM["hip"]
    assert document["ratios"]["hip"] is None
    assert document["ratios"]["chest"] == PASS_FOUR_VIEW_CM["chest"] / ONE_VIEW_CM["chest"]
    assert "waist=0.625 hip=nan thigh=0.643" in read_markdown(output_dir).splitlines()[0]
    rows = markdown_table(
        read_markdown(output_dir), "### Width comparison at 0 degrees placement noise"
    )
    hip = next(row for row in rows if row["Circumference"] == "hip")
    assert hip["1-view median width (cm)"] == "13.000"
    assert hip["4-view median width (cm)"] == "missing"
    assert hip["Ratio (4-view to 1-view)"] == "nan"
    assert (hip["1-view cell in band"], hip["4-view cell in band"]) == ("yes", "missing")


def test_an_sc004_violation_gives_kill_and_never_leaves_a_pass_on_disk(
    config: dict[str, Any], output_dir: Path
) -> None:
    make_run(config, output_dir)
    assert run_verdict(config, output_dir).verdict == "PASS"
    assert read_document(output_dir)["verdict"] == "PASS"

    run = make_run(config, output_dir, violations=2)  # the same table, now with two violations
    with pytest.raises(VerdictRefusedError) as refused:
        run_verdict(config, output_dir)

    assert refused.value.exit_code == 4
    assert "the SC-004 check in evaluate/sc004.json counts 2 violation(s)" in str(refused.value)
    assert "the limit is 0" in str(refused.value)
    document = read_document(output_dir)
    assert document["verdict"] == "KILL"
    assert document["sc004_violations"] == 2
    assert document["invalid_comparison"] is False
    assert document["median_ratio"] <= THRESHOLD  # nothing else is wrong with the table
    line = expected_line(
        run.record, PASS_RATIOS, "0.620", PASS_WIDTHS, verdict="KILL", violations="2"
    )
    assert read_markdown(output_dir).splitlines()[0] == line
    assert read_done_marker(verdict_directory(output_dir)) is None


def test_an_invalid_comparison_and_an_sc004_violation_are_both_named(
    config: dict[str, Any], output_dir: Path
) -> None:
    make_run(
        config,
        output_dir,
        violations=1,
        edit=lambda rows: set_cells(rows, "v1_n0", ("hip",), median_width_cm="0.0"),
    )
    with pytest.raises(VerdictRefusedError, match=r"width ratio for hip is not finite.*SC-004"):
        run_verdict(config, output_dir)
    document = read_document(output_dir)
    assert document["invalid_comparison"] is True
    assert document["sc004_violations"] == 1


FIXED_VALUES = [
    pytest.param("verdict.threshold", 0.5, id="threshold"),
    pytest.param("verdict.measurements", ["chest", "waist", "hip"], id="circumferences"),
    pytest.param("verdict.noise_deg", 2.0, id="noise"),
    pytest.param("verdict.compare_views", [1, 2], id="views"),
    pytest.param("evaluate.band", [0.85, 0.95], id="band"),
    pytest.param("evaluate.measurements", ["height", "chest"], id="measurements"),
]


@pytest.mark.parametrize(("key", "value"), FIXED_VALUES)
def test_a_configuration_that_differs_from_fr014_is_refused_with_exit_2(
    config: dict[str, Any], output_dir: Path, key: str, value: Any
) -> None:
    make_run(config, output_dir)
    section, name = key.split(".")
    changed = {**config, section: {**config[section], name: value}}
    with pytest.raises(ConfigError) as refused:
        run_verdict(changed, output_dir)
    assert refused.value.key == key
    assert not verdict_directory(output_dir).exists()  # nothing was written


def test_a_missing_fixed_key_is_refused_with_exit_2(
    config: dict[str, Any], output_dir: Path
) -> None:
    make_run(config, output_dir)
    changed = {**config, "verdict": {"measurements": list(MEASUREMENTS)}}
    with pytest.raises(ConfigError, match="missing configuration key 'verdict.threshold'"):
        run_verdict(changed, output_dir)


def test_a_boolean_is_not_read_as_the_number_one() -> None:
    changed = {
        "evaluate": {"band": [0.87, 0.93], "measurements": list(ALL_MEASUREMENTS)},
        "verdict": {
            "threshold": 0.7,
            "measurements": list(MEASUREMENTS),
            "noise_deg": 0.0,
            "compare_views": [True, 4],
        },
    }
    with pytest.raises(ConfigError) as refused:
        check_fixed_configuration(changed)
    assert refused.value.key == "verdict.compare_views"


def test_an_overridden_threshold_is_refused_at_load_with_exit_2(
    tiny_config_path: Path, output_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ConfigError) as refused:
        load_config(tiny_config_path, overrides=["verdict.threshold=0.5"])
    assert refused.value.key == "verdict.threshold"

    code = main(
        ["verdict", "--config", str(tiny_config_path), "--out", str(output_dir)]
        + ["--set", "verdict.threshold=0.5"]
    )
    assert code == 2
    error = capsys.readouterr().err
    assert error.startswith("sap: error: ")
    assert "verdict.threshold" in error


@pytest.mark.parametrize("four_view", [PASS_FOUR_VIEW_CM, KILL_FOUR_VIEW_CM], ids=["pass", "kill"])
def test_height_and_the_two_and_five_degree_rows_never_change_the_verdict(
    config: dict[str, Any], tmp_path: Path, four_view: dict[str, float]
) -> None:
    baseline_run = make_run(config, tmp_path / "baseline", four_view=four_view)
    baseline = run_verdict(config, baseline_run.out)

    def disturb(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        kept = []
        for row in rows:
            if row["noise_deg"] == "5.0":
                continue  # the 5 degree rows are missing altogether
            if row["measurement"] == "height":
                row.update(median_width_cm="1000.0", in_band="false")
            if row["noise_deg"] == "2.0":
                row.update(median_width_cm="nan", in_band="false")
            kept.append(row)
        return kept

    disturbed_run = make_run(config, tmp_path / "disturbed", four_view=four_view, edit=disturb)
    disturbed = run_verdict(config, disturbed_run.out)  # returns: no exit 4, no band miss

    assert disturbed.exit_code == 0
    assert disturbed.line == baseline.line
    baseline_document = read_document(baseline_run.out)
    disturbed_document = read_document(disturbed_run.out)
    assert disturbed_document["reported_only"] != baseline_document["reported_only"]
    for document in (baseline_document, disturbed_document):
        del document["reported_only"]
    assert disturbed_document == baseline_document


def test_the_reported_only_part_holds_height_and_the_noise_rows(
    config: dict[str, Any], output_dir: Path
) -> None:
    make_run(config, output_dir)
    run_verdict(config, output_dir)
    reported = read_document(output_dir)["reported_only"]

    assert reported["height_ratio"] == PASS_FOUR_VIEW_CM["height"] / ONE_VIEW_CM["height"]
    for noise, key in ((2.0, "rows_2deg"), (5.0, "rows_5deg")):
        rows = reported[key]
        assert [row["measurement"] for row in rows] == list(ALL_MEASUREMENTS)
        for row in rows:
            name = row["measurement"]
            one = cell_width(1, noise, name)
            four = cell_width(4, noise, name)
            assert row == {
                "measurement": name,
                "width_v1_cm": one,
                "width_v4_cm": four,
                "ratio": four / one,
                "in_band_v1": True,
                "in_band_v4": True,
            }
    text = read_markdown(output_dir)
    rows = markdown_table(text, "#### 2 degrees placement noise")
    assert [row["Measurement"] for row in rows] == list(ALL_MEASUREMENTS)
    assert "Reported only (not part of the verdict)" in text


def test_a_table_with_only_the_compared_cells_reports_no_noise_rows(
    config: dict[str, Any], output_dir: Path
) -> None:
    make_run(config, output_dir, views=(1, 4), noises=(0.0,))  # the grid of the small_config
    result = run_verdict(config, output_dir)
    reported = read_document(output_dir)["reported_only"]
    assert result.verdict == "PASS"
    assert reported["rows_2deg"] == []
    assert reported["rows_5deg"] == []
    assert reported["height_ratio"] == PASS_FOUR_VIEW_CM["height"] / ONE_VIEW_CM["height"]
    assert "holds no cell at 2 degrees placement noise" in read_markdown(output_dir)


def test_every_verdict_line_matches_the_report_pattern_and_the_contract_regex(
    config: dict[str, Any], tmp_path: Path
) -> None:
    def band_miss(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return set_cells(rows, "v4_n0", ("waist", "hip"), in_band="false")

    def infinite(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return set_cells(rows, "v1_n0", ("chest",), median_width_cm="0.0")

    runs = {
        "pass": make_run(config, tmp_path / "pass"),
        "kill": make_run(config, tmp_path / "kill", four_view=KILL_FOUR_VIEW_CM),
        "band miss": make_run(config, tmp_path / "band", edit=band_miss),
        "invalid comparison": make_run(config, tmp_path / "invalid", edit=infinite),
        "sc004 violation": make_run(config, tmp_path / "sc004", violations=3),
        "missing cell": make_run(
            config, tmp_path / "missing", edit=lambda rows: [r for r in rows if r["views"] != "4"]
        ),
    }
    contract = contract_pattern()
    for label, run in runs.items():
        verdict = verdict_of(run)
        line = read_markdown(run.out).splitlines()[0]
        assert line == verdict.line, label
        assert VERDICT_LINE_PATTERN.fullmatch(line), f"{label}: {line}"
        assert contract.match(line), f"{label}: {line}"
        in_band = re.search(r"cells_in_band=(true|false)", line)
        assert in_band is not None
        assert ("; comparison invalid for cells " in line) == (in_band.group(1) == "false"), label
    assert "cells v4_n0:waist,v4_n0:hip" in verdict_of(runs["band miss"]).line


def test_the_verdict_markdown_puts_the_line_first_then_the_tables(
    config: dict[str, Any], output_dir: Path
) -> None:
    run = make_run(config, output_dir)
    run_verdict(config, output_dir)
    text = read_markdown(output_dir)
    lines = text.splitlines()

    assert text.endswith("\n")
    assert lines[0].startswith("VERDICT: PASS ")
    assert lines[1] == ""
    table = markdown_table(text, "### Width comparison at 0 degrees placement noise")
    assert [row["Circumference"] for row in table] == list(CIRCUMFERENCES)
    chest = table[0]
    assert chest["1-view median width (cm)"] == "10.000"
    assert chest["4-view median width (cm)"] == "6.000"
    assert chest["Ratio (4-view to 1-view)"] == "0.600"
    assert (chest["1-view cell in band"], chest["4-view cell in band"]) == ("yes", "yes")
    assert "Median of the four ratios: 0.620 (full precision 0.6201923076923077)." in text
    assert "Threshold: 0.700." in text
    assert (
        "Basis: nominal level 90%, calibration bodies (n_cal) 2500, test bodies (n_test) 2500, "
        f"seed {run.record.seed}, taken from the compared cells of evaluate/results.csv." in text
    )
    assert "coverage band of 87% to 93%" in text
    assert "Why this verdict is KILL" not in text
    assert f"- Configuration hash: `{run.record.config_hash}`" in text
    assert f"- Code version: `{CODE_VERSION}`" in text
    assert "- Hardware class: cpu" in text
    assert [line for line in lines if line.startswith("# ") or line.startswith("## ")] == []


def test_a_kill_explains_each_reason_in_the_markdown(
    config: dict[str, Any], output_dir: Path
) -> None:
    def edit(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        set_cells(rows, "v1_n0", ("hip",), in_band="false")
        return set_cells(rows, "v4_n0", ("chest",), median_width_cm="nan")

    make_run(config, output_dir, violations=2, edit=edit)
    with pytest.raises(VerdictRefusedError):
        run_verdict(config, output_dir)
    text = read_markdown(output_dir)
    reasons = text.split("### Why this verdict is KILL")[1].split("###")[0]
    assert "These compared cells lie outside the coverage band" in reasons
    assert "v1_n0:hip" in reasons
    assert "The width ratio for chest is not finite" in reasons
    assert "The SC-004 check in evaluate/sc004.json counts 2 violation(s)" in reasons
    assert "The median of the four width ratios is nan" in reasons
    assert "The comparison is invalid, so this median is not a result." in text


def test_the_report_stage_reads_the_verdict_markdown(
    config: dict[str, Any], tmp_path: Path
) -> None:
    for label, edit in (
        ("pass", None),
        ("band", lambda rows: set_cells(rows, "v4_n0", ("waist",), in_band="false")),
    ):
        run = make_run(config, tmp_path / label, edit=edit)
        result = run_verdict(config, run.out)
        report = write_results_markdown(run.out).read_text(encoding="utf-8")
        assert result.line in report.splitlines()
        assert "### Width comparison at 0 degrees placement noise" in report


def test_recompute_equals_the_stored_verdict_and_writes_nothing(
    config: dict[str, Any], output_dir: Path
) -> None:
    make_run(config, output_dir)
    run_verdict(config, output_dir)
    before = snapshot(output_dir)

    recomputed = recompute(output_dir)  # no configuration is needed

    assert snapshot(output_dir) == before
    assert recomputed.to_dict() == read_document(output_dir)
    assert recomputed.to_json() == (verdict_directory(output_dir) / "verdict.json").read_text(
        "utf-8"
    )
    assert recomputed.to_markdown() == read_markdown(output_dir)
    assert recomputed.line == read_markdown(output_dir).splitlines()[0]
    assert recomputed.exit_code == 0


def test_recompute_differs_from_the_stored_verdict_after_the_table_is_changed(
    config: dict[str, Any], output_dir: Path
) -> None:
    run = make_run(config, output_dir)
    run_verdict(config, output_dir)
    stored = read_document(output_dir)

    rows = result_rows(run.record, four_view=KILL_FOUR_VIEW_CM)
    write_csv(output_dir / "evaluate" / "results.csv", rows)  # a tampered table

    recomputed = recompute(output_dir)
    assert recomputed.verdict == "KILL"
    assert recomputed.to_dict() != stored


def test_recompute_returns_a_refused_verdict_without_raising_or_writing(
    config: dict[str, Any], output_dir: Path
) -> None:
    make_run(config, output_dir, violations=4)
    recomputed = recompute(output_dir)
    assert recomputed.verdict == "KILL"
    assert recomputed.exit_code == 4
    assert recomputed.refusal_reasons() == [
        "the SC-004 check in evaluate/sc004.json counts 4 violation(s), and the limit is 0"
    ]
    assert not verdict_directory(output_dir).exists()


def test_the_verdict_does_not_depend_on_the_order_of_the_rows(
    config: dict[str, Any], tmp_path: Path
) -> None:
    ordered = make_run(config, tmp_path / "ordered")

    def shuffle(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        random.Random(7).shuffle(rows)
        return rows

    shuffled = make_run(config, tmp_path / "shuffled", edit=shuffle)
    assert recompute(shuffled.out).to_dict() == recompute(ordered.out).to_dict()


@pytest.mark.parametrize(("flag", "expected"), [("True", True), ("FALSE", False), ("true", True)])
def test_the_band_flag_is_read_in_any_letter_case(
    config: dict[str, Any], output_dir: Path, flag: str, expected: bool
) -> None:
    make_run(
        config, output_dir, edit=lambda rows: set_cells(rows, "v1_n0", ("chest",), in_band=flag)
    )
    assert recompute(output_dir).in_band_flags["v1"]["chest"] is expected


def test_the_widths_are_not_rounded_in_the_files(config: dict[str, Any], output_dir: Path) -> None:
    one_view = {**ONE_VIEW_CM, "chest": 10.123456789}
    make_run(config, output_dir, one_view=one_view)
    run_verdict(config, output_dir)
    assert read_document(output_dir)["widths"]["v1"]["chest"] == 10.123456789
    assert "chest=10.123/6.000" in read_markdown(output_dir).splitlines()[0]


def corrupt_results(
    run: Run, mutate: Callable[[list[dict[str, str]]], list[dict[str, str]]]
) -> None:
    write_csv(run.out / "evaluate" / "results.csv", mutate(result_rows(run.record)))


INPUT_ERRORS = [
    pytest.param(
        lambda run: (run.out / "evaluate" / "results.csv").unlink(),
        "cannot find the results table",
        id="missing-results-table",
    ),
    pytest.param(
        lambda run: (run.out / "evaluate" / "sc004.json").unlink(),
        "cannot find the SC-004 check",
        id="missing-sc004-check",
    ),
    pytest.param(
        lambda run: (run.out / "run_record.json").unlink(),
        "run_record.json",
        id="missing-run-record",
    ),
    pytest.param(
        lambda run: (run.out / "evaluate" / "DONE.json").unlink(),
        "evaluate stage has no readable DONE.json",
        id="evaluate-not-done",
    ),
    pytest.param(
        lambda run: write_done_marker(
            run.out / "evaluate",
            stage="evaluate",
            config_hash="c" * 64,
            seed=run.record.seed,
            code_version=CODE_VERSION,
            hardware_class="cpu",
            inputs={},
        ),
        "was done under configuration",
        id="evaluate-done-for-another-configuration",
    ),
    pytest.param(
        lambda run: (run.out / "evaluate" / "sc004.json").write_text("{not json", "utf-8"),
        "is not valid JSON",
        id="sc004-check-not-json",
    ),
    pytest.param(
        lambda run: (run.out / "evaluate" / "sc004.json").write_text("[1, 2]", "utf-8"),
        "must be a JSON object",
        id="sc004-check-not-an-object",
    ),
    pytest.param(
        lambda run: (run.out / "evaluate" / "sc004.json").write_text(
            '{"violations": NaN}', "utf-8"
        ),
        "not a finite number",
        id="sc004-check-with-nan",
    ),
    pytest.param(
        lambda run: write_sc004(
            run.out / "evaluate" / "sc004.json", sc004_document(run.record, -1)
        ),
        "'violations' must be a whole number of at least 0",
        id="negative-violation-count",
    ),
    pytest.param(
        lambda run: write_sc004(
            run.out / "evaluate" / "sc004.json", {**sc004_document(run.record), "violations": True}
        ),
        "'violations' must be a whole number of at least 0",
        id="boolean-violation-count",
    ),
    pytest.param(
        lambda run: write_sc004(
            run.out / "evaluate" / "sc004.json", {**sc004_document(run.record), "seed": 99}
        ),
        "the field 'seed' is 99",
        id="sc004-check-of-another-run",
    ),
    pytest.param(
        lambda run: write_sc004(
            run.out / "evaluate" / "sc004.json",
            {k: v for k, v in sc004_document(run.record).items() if k != "code_version"},
        ),
        "the field 'code_version' is missing",
        id="sc004-check-without-provenance",
    ),
    pytest.param(
        lambda run: corrupt_results(run, lambda rows: [{**r, "seed": "99"} for r in rows]),
        "the field 'seed' is 99",
        id="results-of-another-seed",
    ),
    pytest.param(
        lambda run: corrupt_results(
            run, lambda rows: [{**r, "config_hash": "d" * 64} for r in rows]
        ),
        "the field 'config_hash'",
        id="results-of-another-configuration",
    ),
    pytest.param(
        lambda run: corrupt_results(
            run, lambda rows: [{**r, "code_version": "0.0.0"} for r in rows]
        ),
        "the field 'code_version' is '0.0.0'",
        id="results-of-another-code-version",
    ),
    pytest.param(
        lambda run: write_csv(
            run.out / "evaluate" / "results.csv",
            [{k: v for k, v in r.items() if k != "in_band"} for r in result_rows(run.record)],
            [c for c in RESULT_COLUMNS if c != "in_band"],
        ),
        "lacks the column(s) in_band",
        id="results-without-the-band-column",
    ),
    pytest.param(
        lambda run: corrupt_results(run, lambda rows: [{**rows[0], "views": "x"}, *rows[1:]]),
        "'views' must be a whole number",
        id="results-with-a-bad-view-count",
    ),
    pytest.param(
        lambda run: corrupt_results(
            run, lambda rows: [{**rows[0], "median_width_cm": "wide"}, *rows[1:]]
        ),
        "'median_width_cm' must be a number",
        id="results-with-a-bad-width",
    ),
    pytest.param(
        lambda run: corrupt_results(run, lambda rows: [{**rows[0], "in_band": "maybe"}, *rows[1:]]),
        "'in_band' must be true or false",
        id="results-with-a-bad-band-flag",
    ),
    pytest.param(
        lambda run: corrupt_results(run, lambda rows: [{**rows[0], "noise_deg": "nan"}, *rows[1:]]),
        "'noise_deg' must be finite",
        id="results-with-a-bad-noise-level",
    ),
    pytest.param(
        lambda run: corrupt_results(run, lambda rows: [{**rows[0], "n_test": ""}, *rows[1:]]),
        "'n_test' is empty",
        id="results-with-an-empty-count",
    ),
    pytest.param(
        lambda run: corrupt_results(run, lambda rows: [*rows, dict(rows[0])]),
        "v1_n0:height more than once",
        id="results-with-a-repeated-row",
    ),
    pytest.param(
        lambda run: (run.out / "evaluate" / "results.csv").write_text(
            ",".join(RESULT_COLUMNS) + "\nv1_n0,1\n", "utf-8"
        ),
        "does not have one value per column",
        id="results-with-a-short-row",
    ),
]


@pytest.mark.parametrize(("damage", "message"), INPUT_ERRORS)
def test_an_input_that_is_missing_or_from_another_run_is_refused_before_anything_is_written(
    config: dict[str, Any], output_dir: Path, damage: Callable[[Run], object], message: str
) -> None:
    run = make_run(config, output_dir)
    damage(run)
    with pytest.raises(VerdictInputError, match=re.escape(message)) as refused:
        run_verdict(config, output_dir)
    assert refused.value.exit_code == 4
    assert isinstance(refused.value, VerdictError)
    assert not verdict_directory(output_dir).exists()


def test_a_refused_input_leaves_the_earlier_verdict_as_it_was(
    config: dict[str, Any], output_dir: Path
) -> None:
    run = make_run(config, output_dir)
    run_verdict(config, output_dir)
    before = snapshot(verdict_directory(output_dir))
    (run.out / "evaluate" / "sc004.json").unlink()
    with pytest.raises(VerdictInputError):
        run_verdict(config, output_dir)
    assert snapshot(verdict_directory(output_dir)) == before


def test_a_run_record_of_another_configuration_is_refused(
    config: dict[str, Any], tiny_config_path: Path, output_dir: Path
) -> None:
    make_run(config, output_dir)
    other = load_config(tiny_config_path, overrides=["seed=2"])
    with pytest.raises(VerdictInputError, match="the verdict reads the outputs of one run only"):
        run_verdict(other, output_dir)
    assert not verdict_directory(output_dir).exists()


def test_recompute_refuses_the_same_inputs(config: dict[str, Any], output_dir: Path) -> None:
    run = make_run(config, output_dir)
    write_sc004(
        output_dir / "evaluate" / "sc004.json",
        {**sc004_document(run.record), "config_hash": "e" * 64},
    )
    with pytest.raises(VerdictInputError, match="comes from another run"):
        recompute(output_dir)


def test_the_width_ratio_follows_ieee_division_for_usable_widths() -> None:
    assert width_ratio(one_view_cm=10.0, four_view_cm=6.0) == 0.6
    assert width_ratio(one_view_cm=10.0, four_view_cm=0.0) == 0.0
    assert width_ratio(one_view_cm=0.0, four_view_cm=2.0) == float("inf")
    assert math.isnan(width_ratio(one_view_cm=0.0, four_view_cm=0.0))


@pytest.mark.parametrize(
    ("one_view", "four_view"),
    [
        (float("inf"), 5.0),
        (5.0, float("inf")),
        (float("nan"), 5.0),
        (5.0, float("nan")),
        (-1.0, 5.0),
        (5.0, -1.0),
    ],
)
def test_a_width_that_is_not_usable_gives_a_ratio_that_is_not_finite(
    one_view: float, four_view: float
) -> None:
    ratio = width_ratio(one_view_cm=one_view, four_view_cm=four_view)
    assert math.isnan(ratio)  # so a finite width over an infinite one never reads as the ratio 0


def test_compute_verdict_refuses_a_repeated_cell_and_a_bad_violation_count(
    config: dict[str, Any],
) -> None:
    record = start_run_record(
        config,
        hardware_class="cpu",
        device_name="cpu",
        code_version=CODE_VERSION,
        versions=VERSIONS,
    )
    cell = ResultCell("v1_n0", 1, 0.0, "chest", 0.9, 2500, 2500, 10.0, True)
    with pytest.raises(VerdictInputError, match="v1_n0:chest more than once"):
        compute_verdict([cell, cell], sc004_violations=0, record=record)
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError, match="violation count"):
            compute_verdict([cell], sc004_violations=bad, record=record)
