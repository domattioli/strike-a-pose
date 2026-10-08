"""Determinism tests: repeated generation, repeated CPU training, and the stored kill verdict."""

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from strike_a_pose.calibrate import calibrate_quantiles
from strike_a_pose.config import load_config
from strike_a_pose.data.generate import data_directory, generate_dataset
from strike_a_pose.data.manifest import MANIFEST_NAME, SHARD_DIRECTORY_NAME
from strike_a_pose.evaluate import run_evaluate
from strike_a_pose.predict import run_predict
from strike_a_pose.runrecord import start_run_record, write_run_record
from strike_a_pose.train import train_model
from strike_a_pose.verdict import (
    VERDICT_JSON_NAME,
    VerdictRefusedError,
    recompute,
    run_verdict,
    verdict_directory,
)

# SC-005: coverage values match within 0.5 percentage points, and median widths within 0.1 cm.
COVERAGE_TOLERANCE = 0.005
WIDTH_TOLERANCE_CM = 0.1


@pytest.fixture(scope="module", autouse=True)
def one_torch_thread():
    """Run on one thread, so these tests stay within 60 seconds on a busy machine."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


def resolved_config(tiny_config_path: Path, small_config: dict[str, Any]) -> dict[str, Any]:
    """Load configs/tiny.yaml with the small_config overrides, one --set per key."""
    return load_config(
        tiny_config_path, [f"{key}={json.dumps(value)}" for key, value in small_config.items()]
    )


def write_run_record_for(config: dict[str, Any], out: Path) -> None:
    """Write run_record.json, as the command does before its first stage."""
    out.mkdir(parents=True, exist_ok=True)
    record = start_run_record(config, hardware_class="cpu", device_name="cpu (test machine)")
    write_run_record(record, out / "run_record.json")


def generate(config: dict[str, Any], out: Path) -> None:
    """Write the run record and generate the bodies of one run."""
    write_run_record_for(config, out)
    generate_dataset(config, out)


def train_predict_calibrate_evaluate(config: dict[str, Any], out: Path) -> None:
    """Run the stages that follow generation, in the order of the command."""
    train_model(config, out)
    run_predict(config, out)
    calibrate_quantiles(config, out)
    run_evaluate(config, out)


def read_results(out: Path) -> list[dict[str, str]]:
    with (out / "evaluate" / "results.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_two_generations_give_identical_shards_and_manifest(
    tiny_config_path: Path, small_config: dict[str, Any], tmp_path: Path
) -> None:
    config = resolved_config(tiny_config_path, small_config)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(config, first)
    generate(config, second)

    first_data = data_directory(first)
    second_data = data_directory(second)
    shard_names = sorted(path.name for path in (first_data / SHARD_DIRECTORY_NAME).glob("*.npz"))
    assert len(shard_names) >= 2
    assert shard_names == sorted(
        path.name for path in (second_data / SHARD_DIRECTORY_NAME).glob("*.npz")
    )
    for name in shard_names:
        assert (first_data / SHARD_DIRECTORY_NAME / name).read_bytes() == (
            second_data / SHARD_DIRECTORY_NAME / name
        ).read_bytes(), f"shard {name} differs between two generations"
    assert (first_data / MANIFEST_NAME).read_bytes() == (second_data / MANIFEST_NAME).read_bytes()


def test_two_cpu_trainings_give_metrics_within_the_sc005_tolerance(
    tiny_config_path: Path, small_config: dict[str, Any], tmp_path: Path
) -> None:
    config = resolved_config(tiny_config_path, small_config)
    base = tmp_path / "base"
    generate(config, base)

    # Each training starts from its own copy of the generated data and run record.
    first = tmp_path / "train_a"
    second = tmp_path / "train_b"
    shutil.copytree(base, first)
    shutil.copytree(base, second)
    train_predict_calibrate_evaluate(config, first)
    train_predict_calibrate_evaluate(config, second)

    first_rows = read_results(first)
    second_rows = read_results(second)
    assert first_rows, "the results table has no rows"
    assert len(first_rows) == len(second_rows)
    for kept, again in zip(first_rows, second_rows, strict=True):
        assert (kept["cell_id"], kept["measurement"]) == (again["cell_id"], again["measurement"])
        coverage_gap = abs(float(kept["coverage"]) - float(again["coverage"]))
        assert coverage_gap <= COVERAGE_TOLERANCE, (
            f"coverage of {kept['cell_id']} {kept['measurement']} differs by {coverage_gap}"
        )
        width_gap = abs(float(kept["median_width_cm"]) - float(again["median_width_cm"]))
        assert width_gap <= WIDTH_TOLERANCE_CM, (
            f"median width of {kept['cell_id']} {kept['measurement']} differs by {width_gap} cm"
        )


def test_the_recomputed_verdict_equals_the_stored_verdict(
    tiny_config_path: Path, small_config: dict[str, Any], tmp_path: Path
) -> None:
    config = resolved_config(tiny_config_path, small_config)
    out = tmp_path / "run"
    generate(config, out)
    train_predict_calibrate_evaluate(config, out)

    # A refused comparison is still written before the error is raised, so the stored file exists.
    try:
        stored = run_verdict(config, out)
    except VerdictRefusedError as error:
        stored = error.verdict

    stored_text = (verdict_directory(out) / VERDICT_JSON_NAME).read_text(encoding="utf-8")
    recomputed = recompute(out)
    assert recomputed.verdict == stored.verdict
    assert recomputed.to_dict() == stored.to_dict()
    assert recomputed.to_json() == stored_text
