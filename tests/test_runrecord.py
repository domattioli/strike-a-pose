"""Smoke tests for runrecord.py: fields, hash and seed, versions, code version, timings, writes."""

import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from strike_a_pose import __version__, runrecord
from strike_a_pose.config import config_hash, load_config
from strike_a_pose.runrecord import (
    HARDWARE_CLASSES,
    STAGES,
    VERSION_NAMES,
    RunRecord,
    RunRecordError,
    current_code_version,
    current_library_versions,
    read_run_record,
    start_run_record,
    write_run_record,
)

FIELD_NAMES = [
    "config_hash",
    "seed",
    "code_version",
    "hardware_class",
    "device_name",
    "versions",
    "started_at",
    "finished_at",
    "timings",
]
OPENCV_NAMES = (
    "opencv-python-headless",
    "opencv-python",
    "opencv-contrib-python-headless",
    "opencv-contrib-python",
)
# Version text of the pinned environment (constraints.txt). The record copies what it is given.
FIXED_VERSIONS: dict[str, str | None] = {
    "python": "3.13.16",
    "numpy": "2.5.3",
    "torch": "2.14.1",
    "opencv": "4.14.0.94",
    "smplx": None,
    "sam2": None,
}
UTC_INSTANT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00")
# The code version is the bare package version, or the version with a 12-character commit and an
# optional dirty mark (see current_code_version).
CODE_VERSION = re.compile(rf"{re.escape(__version__)}(\+[0-9a-f]{{12}}(\.dirty)?)?")


@pytest.fixture
def resolved_config(tiny_config_path: Path, small_config: dict[str, object]) -> dict[str, Any]:
    """The tiny configuration with the test overrides of contracts/config.md, resolved."""
    overrides = [f"{key}={json.dumps(value)}" for key, value in small_config.items()]
    return load_config(tiny_config_path, overrides=overrides)


@pytest.fixture
def record(resolved_config: dict[str, Any]) -> RunRecord:
    """A complete record with fixed provenance, so no test depends on the checkout it runs in."""
    return start_run_record(
        resolved_config,
        hardware_class="cpu",
        device_name="cpu",
        started_at=datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc),
        code_version="0.1.0+0123456789ab",
        versions=FIXED_VERSIONS,
    )


class _FakeClock:
    """Stands in for the time module: each perf_counter call returns the next given reading."""

    def __init__(self, *readings: float) -> None:
        self._readings = iter(readings)

    def perf_counter(self) -> float:
        return next(self._readings)


def _git(directory: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=directory, capture_output=True, text=True, check=True
    )
    return completed.stdout


def _installed_version(names: tuple[str, ...]) -> str | None:
    """The test's own reading of the installed versions, to compare with the module."""
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return None


def test_fields_follow_the_contract(record: RunRecord) -> None:
    assert STAGES == (
        "generate",
        "train",
        "predict",
        "calibrate",
        "evaluate",
        "verdict",
        "report",
        "real_eval",
    )
    document = record.to_dict()
    assert list(document) == FIELD_NAMES
    assert list(document["versions"]) == list(VERSION_NAMES)
    assert list(document["timings"]) == list(STAGES)
    assert all(seconds is None for seconds in document["timings"].values())
    assert document["finished_at"] is None


def test_hash_and_seed_come_from_the_configuration(
    record: RunRecord, resolved_config: dict[str, Any]
) -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", record.config_hash)
    assert record.config_hash == config_hash(resolved_config)
    assert record.seed == resolved_config["seed"]
    changed = start_run_record(
        {**resolved_config, "seed": 2},
        hardware_class="cpu",
        device_name="cpu",
        code_version="0.1.0",
        versions=FIXED_VERSIONS,
    )
    assert changed.seed == 2
    assert changed.config_hash != record.config_hash


def test_hardware_class_must_be_cpu_or_gpu(resolved_config: dict[str, Any]) -> None:
    assert HARDWARE_CLASSES == ("cpu", "gpu")
    with pytest.raises(RunRecordError, match="hardware_class"):
        start_run_record(
            resolved_config,
            hardware_class="tpu",
            device_name="tpu",
            code_version="0.1.0",
            versions=FIXED_VERSIONS,
        )


def test_start_and_finish_times_are_utc_text(resolved_config: dict[str, Any]) -> None:
    early = datetime(2026, 10, 7, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    record = start_run_record(
        resolved_config,
        hardware_class="cpu",
        device_name="cpu",
        started_at=early,
        code_version="0.1.0",
        versions=FIXED_VERSIONS,
    )
    assert record.started_at == "2026-10-07T12:00:00+00:00"
    record.finish(datetime(2026, 10, 8, 1, 30, 15, 999999, tzinfo=timezone.utc))
    assert record.finished_at == "2026-10-08T01:30:15+00:00"
    record.finish()
    assert UTC_INSTANT.fullmatch(record.finished_at)


def test_naive_times_are_refused(resolved_config: dict[str, Any], record: RunRecord) -> None:
    with pytest.raises(RunRecordError, match="time zone"):
        start_run_record(
            resolved_config,
            hardware_class="cpu",
            device_name="cpu",
            started_at=datetime(2026, 10, 7, 12),
            code_version="0.1.0",
            versions=FIXED_VERSIONS,
        )
    with pytest.raises(RunRecordError, match="time zone"):
        record.finish(datetime(2026, 10, 8, 1, 30))


def test_library_versions_are_the_installed_distribution_versions() -> None:
    versions = current_library_versions()
    assert tuple(versions) == VERSION_NAMES
    assert versions["python"] == platform.python_version()
    assert versions["numpy"] == metadata.version("numpy")
    assert versions["torch"] == metadata.version("torch")
    assert versions["opencv"] == _installed_version(OPENCV_NAMES)
    assert versions["smplx"] == _installed_version(("smplx",))
    assert versions["sam2"] == _installed_version(("sam2",))


def test_a_missing_optional_library_reads_as_null(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = metadata.version

    def without_extras(name: str) -> str:
        if name in ("smplx", "sam2"):
            raise metadata.PackageNotFoundError(name)
        return installed(name)

    monkeypatch.setattr(metadata, "version", without_extras)
    versions = current_library_versions()
    assert versions["smplx"] is None
    assert versions["sam2"] is None
    assert versions["numpy"] == installed("numpy")


def test_code_version_of_this_checkout_has_the_expected_form() -> None:
    assert CODE_VERSION.fullmatch(current_code_version())


def test_code_version_outside_a_checkout_is_the_package_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    # Git searches upward from the package directory. The ceiling stops the search at tmp_path.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setattr(runrecord, "_PACKAGE_DIRECTORY", package)
    assert current_code_version() == __version__


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_code_version_names_the_commit_and_marks_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    package = checkout / "src" / "strike_a_pose"
    package.mkdir(parents=True)
    source = package / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(checkout, "init", "-q")
    _git(checkout, "add", "-A")
    _git(
        checkout,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "init",
    )
    commit = _git(checkout, "rev-parse", "--short=12", "HEAD").strip()
    monkeypatch.setattr(runrecord, "_PACKAGE_DIRECTORY", package)
    assert current_code_version() == f"{__version__}+{commit}"
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert current_code_version() == f"{__version__}+{commit}.dirty"


def test_stage_timings_sum_when_a_stage_is_timed_again(record: RunRecord) -> None:
    record.add_timing("train", 1.5)
    record.add_timing("train", 2.25)
    assert record.timings["train"] == 3.75
    assert record.timings["generate"] is None


@pytest.mark.parametrize("stage", ["real-eval", "bogus", ""])
def test_unknown_stage_is_refused(record: RunRecord, stage: str) -> None:
    with pytest.raises(RunRecordError, match="unknown stage"):
        record.add_timing(stage, 1.0)
    with pytest.raises(RunRecordError, match="unknown stage"):
        with record.time_stage(stage):
            pass


@pytest.mark.parametrize("seconds", [-0.5, float("nan"), float("inf"), True, "3", None])
def test_a_timing_must_be_finite_and_not_negative(record: RunRecord, seconds: Any) -> None:
    with pytest.raises(RunRecordError, match="timing of stage 'train'"):
        record.add_timing("train", seconds)
    assert record.timings["train"] is None


def test_time_stage_adds_the_elapsed_seconds(
    record: RunRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runrecord, "time", _FakeClock(10.0, 12.5))
    with record.time_stage("evaluate"):
        pass
    assert record.timings["evaluate"] == 2.5


def test_time_stage_records_nothing_when_the_block_raises(
    record: RunRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runrecord, "time", _FakeClock(1.0, 4.0))
    with pytest.raises(RuntimeError, match="simulated failure"):
        with record.time_stage("predict"):
            raise RuntimeError("simulated failure")
    assert record.timings["predict"] is None


def test_write_then_read_round_trips_and_creates_the_directory(
    record: RunRecord, output_dir: Path
) -> None:
    record.add_timing("generate", 0.5)
    record.finish(datetime(2026, 10, 7, 13, 0, tzinfo=timezone.utc))
    path = output_dir / "run_record.json"
    write_run_record(record, path)
    assert read_run_record(path) == record
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert list(json.loads(text)) == FIELD_NAMES
    assert [entry.name for entry in output_dir.iterdir()] == ["run_record.json"]


def test_a_resumed_record_adds_its_later_session_to_the_stage_timing(
    record: RunRecord, output_dir: Path
) -> None:
    record.add_timing("train", 1.0)
    path = output_dir / "run_record.json"
    write_run_record(record, path)
    resumed = read_run_record(path)
    resumed.add_timing("train", 2.0)
    write_run_record(resumed, path)
    assert read_run_record(path).timings["train"] == 3.0


def test_a_failed_rename_leaves_the_previous_record_whole(
    record: RunRecord, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run_record.json"
    write_run_record(record, path)
    previous = path.read_text(encoding="utf-8")
    record.add_timing("generate", 9.0)

    def failing_rename(source: str, target: str) -> None:
        raise OSError("simulated failure before the rename")

    monkeypatch.setattr(runrecord.os, "replace", failing_rename)
    with pytest.raises(OSError, match="simulated failure"):
        write_run_record(record, path)
    assert path.read_text(encoding="utf-8") == previous
    assert [entry.name for entry in tmp_path.iterdir()] == ["run_record.json"]


def test_write_refuses_a_record_that_would_not_read_back(record: RunRecord, tmp_path: Path) -> None:
    record.seed = -1
    path = tmp_path / "run_record.json"
    with pytest.raises(RunRecordError, match="seed"):
        write_run_record(record, path)
    assert not path.exists()


BROKEN_RECORDS = [
    pytest.param(lambda d: d.pop("device_name"), id="missing-field"),
    pytest.param(lambda d: d.update(extra="x"), id="unknown-field"),
    pytest.param(lambda d: d.update(hardware_class="tpu"), id="hardware-class"),
    pytest.param(lambda d: d.update(config_hash="abc"), id="short-config-hash"),
    pytest.param(lambda d: d.update(seed=-1), id="negative-seed"),
    pytest.param(lambda d: d.update(seed=True), id="boolean-seed"),
    pytest.param(lambda d: d.update(started_at="2026-10-07T12:00:00"), id="naive-start"),
    pytest.param(lambda d: d["timings"].pop("real_eval"), id="missing-stage"),
    pytest.param(lambda d: d["timings"].update(train=-2.0), id="negative-timing"),
    pytest.param(lambda d: d["versions"].pop("sam2"), id="missing-library"),
    pytest.param(lambda d: d["versions"].update(python=None), id="null-python"),
]


@pytest.mark.parametrize("break_field", BROKEN_RECORDS)
def test_read_refuses_a_record_that_breaks_a_field_rule(
    record: RunRecord, tmp_path: Path, break_field: Any
) -> None:
    document = record.to_dict()
    break_field(document)
    path = tmp_path / "run_record.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RunRecordError):
        read_run_record(path)


def test_read_refuses_non_finite_numbers(record: RunRecord, tmp_path: Path) -> None:
    text = json.dumps(record.to_dict()).replace('"generate": null', '"generate": NaN')
    path = tmp_path / "run_record.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RunRecordError, match="not valid JSON"):
        read_run_record(path)


def test_read_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RunRecordError, match="cannot read"):
        read_run_record(tmp_path / "absent.json")
