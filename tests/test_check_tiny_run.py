"""Tests for scripts/check_tiny_run.py: a fixture output directory passes, and broken ones fail."""

import csv
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_tiny_run.py"

VALID_VERDICT_LINE = (
    "VERDICT: KILL median_ratio=0.812 threshold=0.700 cells_in_band=true invalid=false "
    "sc004_violations=0 (chest=0.800 waist=0.810 hip=0.820 thigh=0.830) "
    "widths_v1_v4_cm=(chest=10.000/12.000 waist=20.000/24.000 hip=30.000/36.000 "
    "thigh=15.000/18.000) config=75a5c23d5818 seed=1"
)
OUT_OF_BAND_VERDICT_LINE = (
    "VERDICT: KILL median_ratio=0.812 threshold=0.700 cells_in_band=false invalid=false "
    "sc004_violations=0 (chest=0.800 waist=0.810 hip=0.820 thigh=0.830) "
    "widths_v1_v4_cm=(chest=10.000/12.000 waist=20.000/24.000 hip=30.000/36.000 "
    "thigh=15.000/18.000) config=75a5c23d5818 seed=1; "
    "comparison invalid for cells v4_n0:waist,v1_n0:hip"
)
RESULT_HEADER = ["cell_id", "views", "noise_deg", "measurement", "coverage", "in_band"]
RUN_RECORD = {
    "config_hash": "75a5c23d581887e75e117f2dfb609b8866f7e15ba661e0301c807d7f4993aac6",
    "seed": 1,
    "code_version": "0.1.0+test",
    "hardware_class": "cpu",
}


def _load_script() -> Callable[..., int]:
    """Import scripts/check_tiny_run.py by file path and return its main function."""
    spec = importlib.util.spec_from_file_location("check_tiny_run", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_tiny_run"] = module
    spec.loader.exec_module(module)
    return module.main


def _build_fixture(
    out_directory: Path,
    result_rows: int = 45,
    verdict_line: str = VALID_VERDICT_LINE,
    run_record: dict[str, object] | None = None,
) -> Path:
    """Write a tiny-run style output directory with the given row count, verdict line and record."""
    (out_directory / "evaluate").mkdir(parents=True)
    with (out_directory / "evaluate" / "results.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(RESULT_HEADER)
        for index in range(result_rows):
            writer.writerow([f"v1_n{index}", 1, 0, "chest", "0.900", "true"])
    (out_directory / "verdict").mkdir()
    (out_directory / "verdict" / "verdict.md").write_text(
        verdict_line + "\n\n| Cell | Ratio |\n", encoding="utf-8"
    )
    record = RUN_RECORD if run_record is None else run_record
    (out_directory / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
    return out_directory


def test_fixture_directory_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main = _load_script()
    out_directory = _build_fixture(tmp_path / "out")

    assert main([str(out_directory)]) == 0
    assert "check_tiny_run: OK" in capsys.readouterr().out


def test_out_of_band_verdict_with_suffix_passes(tmp_path: Path) -> None:
    main = _load_script()
    out_directory = _build_fixture(tmp_path / "out", verdict_line=OUT_OF_BAND_VERDICT_LINE)

    assert main([str(out_directory)]) == 0


def test_missing_result_row_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main = _load_script()
    out_directory = _build_fixture(tmp_path / "out", result_rows=44)

    assert main([str(out_directory)]) == 1
    error = capsys.readouterr().err
    assert "44 data rows, expected 45" in error


def test_malformed_verdict_line_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main = _load_script()
    broken = VALID_VERDICT_LINE.replace("threshold=0.700", "threshold=0.5")
    out_directory = _build_fixture(tmp_path / "out", verdict_line=broken)

    assert main([str(out_directory)]) == 1
    error = capsys.readouterr().err
    assert "is not a verdict line" in error


def test_suffix_without_out_of_band_cells_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main = _load_script()
    inconsistent = VALID_VERDICT_LINE + "; comparison invalid for cells v4_n0:waist"
    out_directory = _build_fixture(tmp_path / "out", verdict_line=inconsistent)

    assert main([str(out_directory)]) == 1
    assert "suffix must be present exactly when" in capsys.readouterr().err


def test_missing_suffix_when_out_of_band_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main = _load_script()
    inconsistent = OUT_OF_BAND_VERDICT_LINE.split("; comparison")[0]
    out_directory = _build_fixture(tmp_path / "out", verdict_line=inconsistent)

    assert main([str(out_directory)]) == 1
    assert "suffix must be present exactly when" in capsys.readouterr().err


def test_run_record_missing_field_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main = _load_script()
    record = {key: value for key, value in RUN_RECORD.items() if key != "hardware_class"}
    out_directory = _build_fixture(tmp_path / "out", run_record=record)

    assert main([str(out_directory)]) == 1
    assert "no field 'hardware_class'" in capsys.readouterr().err
