"""Tests for train.py: a two-epoch run on small data, resume equality, and the monitor slice."""

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch

from strike_a_pose.checkpoint import DONE_MARKER_NAME, TimeBudget, read_done_marker
from strike_a_pose.config import load_config
from strike_a_pose.data.dataset import ShapeDataset
from strike_a_pose.data.generate import data_directory, generate_dataset
from strike_a_pose.data.splits import DataSplit
from strike_a_pose.train import (
    HISTORY_COLUMNS,
    WARMUP_FRACTION,
    epoch_batches,
    learning_rate_factor,
    train_directory,
    train_model,
)

# Two epochs of small_config data. A batch of 8 bodies gives about eight steps per epoch, and a
# checkpoint every 3 steps puts checkpoints inside an epoch as well as at its end. The wider
# focal length keeps most bodies in frame, so most bodies are unflagged (tests/test_dataset.py).
TRAIN_OVERRIDES: dict[str, object] = {
    "train.epochs": 2,
    "train.batch_size": 8,
    "train.checkpoint_every": 3,
    "data.min_unflagged": 0,
    "camera.focal_px": 48,
}
RESUME_TOLERANCE = 1e-6


class Trained(NamedTuple):
    """A finished run: its configuration, its output folder, and its result."""

    config: dict[str, Any]
    out: Path
    result: Any


@pytest.fixture(scope="module", autouse=True)
def one_torch_thread():
    """Run on one thread, so these tests stay fast while other jobs share the machine."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Trained:
    """Generate small data and train two epochs once; the tests read the outputs or copy them."""
    from tests.conftest import SMALL_CONFIG_OVERRIDES, TINY_CONFIG

    merged = {**SMALL_CONFIG_OVERRIDES, **TRAIN_OVERRIDES}
    config = load_config(
        TINY_CONFIG, [f"{key}={json.dumps(value)}" for key, value in merged.items()]
    )
    out = tmp_path_factory.mktemp("trained")
    generate_dataset(config, out)
    result = train_model(config, out)
    return Trained(config, out, result)


def read_history(out: Path) -> list[dict[str, str]]:
    with (train_directory(out) / "history.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_run(source: Trained, destination: Path, keep_up_to_step: int) -> Path:
    """Copy a finished run and keep only its checkpoints up to a step, as an interrupted run."""
    shutil.copytree(source.out, destination)
    train_dir = train_directory(destination)
    steps_per_epoch = source.result.total_steps // 2
    for path in (train_dir / "checkpoints").glob("*.pt"):
        kind, _, number = path.stem.partition("_")
        step = int(number) if kind == "step" else (int(number) + 1) * steps_per_epoch
        if step > keep_up_to_step:
            path.unlink()
    for name in (DONE_MARKER_NAME, "model_final.pt", "history.csv"):
        (train_dir / name).unlink()
    return destination


def test_a_two_epoch_run_writes_every_output(trained):
    train_dir = train_directory(trained.out)
    result = trained.result
    assert result.completed and not result.skipped
    assert result.epochs_done == 2
    assert result.steps_done == result.total_steps
    assert (train_dir / "model_final.pt").is_file()
    assert (train_dir / "checkpoints" / "epoch_000.pt").is_file()
    assert (train_dir / "checkpoints" / "epoch_001.pt").is_file()
    assert (train_dir / "checkpoints" / "step_000003.pt").is_file()

    rows = read_history(trained.out)
    assert tuple(rows[0]) == HISTORY_COLUMNS
    assert [int(row["step"]) for row in rows] == list(range(1, result.total_steps + 1))
    assert all(math.isfinite(float(row["loss"])) for row in rows)
    # The monitor loss is logged on the last step of each epoch and nowhere else.
    monitored = [row for row in rows if row["monitor_loss"] != ""]
    assert [int(row["epoch"]) for row in monitored] == [0, 1]
    assert float(rows[-1]["loss"]) == pytest.approx(result.final_loss, abs=0)

    marker = read_done_marker(train_dir)
    assert marker is not None and marker.stage == "train"
    generate_marker = read_done_marker(data_directory(trained.out))
    assert marker.inputs == {"generate": generate_marker.config_hash}

    final = torch.load(train_dir / "model_final.pt", weights_only=True)
    assert final["step"] == result.total_steps
    assert all(torch.isfinite(value).all() for value in final["model"].values())


def test_the_learning_rate_warms_up_linearly_then_decays_by_cosine():
    total = 200
    warmup = int(WARMUP_FRACTION * total)
    assert warmup == 10
    factors = [learning_rate_factor(step, total) for step in range(total + 1)]
    assert factors[0] == pytest.approx(0.01)
    ramp = [factors[step + 1] - factors[step] for step in range(warmup)]
    assert all(delta == pytest.approx(ramp[0]) and delta > 0 for delta in ramp)
    assert factors[warmup] == pytest.approx(1.0) and max(factors) == pytest.approx(1.0)
    tail = factors[warmup:]
    assert all(later < earlier for earlier, later in zip(tail, tail[1:], strict=False))
    midpoint = warmup + (total - warmup) // 2
    assert factors[midpoint] == pytest.approx(0.01 + 0.99 * 0.5, abs=1e-3)
    assert factors[total] == pytest.approx(0.01)
    # A tiny run still warms up for one step.
    assert [learning_rate_factor(step, 4) for step in range(2)] == pytest.approx([0.01, 1.0])


def test_history_logs_the_warm_up_schedule(trained):
    peak = float(trained.config["train"]["lr"])
    total = trained.result.total_steps
    rates = [float(row["learning_rate"]) for row in read_history(trained.out)]
    print("learning rates:", rates[:3], "...", rates[-3:])
    expected = [peak * learning_rate_factor(step, total) for step in range(total)]
    assert rates == pytest.approx(expected, rel=1e-9)
    assert rates[0] == pytest.approx(0.01 * peak)
    assert max(rates) == pytest.approx(peak)
    assert rates[-1] < rates[total // 2] < peak


def test_resume_from_a_checkpoint_reaches_the_same_final_loss(trained, tmp_path):
    reference = trained.result
    # Step 3 lies inside the first epoch, so the resume starts in the middle of an epoch.
    mid_epoch = copy_run(trained, tmp_path / "mid", keep_up_to_step=3)
    resumed = train_model(trained.config, mid_epoch, resume=True)
    assert resumed.resumed_from_step == 3 and resumed.completed
    assert abs(resumed.final_loss - reference.final_loss) <= RESUME_TOLERANCE
    assert abs(resumed.final_monitor_loss - reference.final_monitor_loss) <= RESUME_TOLERANCE

    # The end of the first epoch is a checkpoint too, and the resume starts the second epoch.
    steps_per_epoch = reference.total_steps // 2
    epoch_end = copy_run(trained, tmp_path / "epoch", keep_up_to_step=steps_per_epoch)
    resumed = train_model(trained.config, epoch_end, resume=True)
    assert resumed.resumed_from_step == steps_per_epoch
    assert abs(resumed.final_loss - reference.final_loss) <= RESUME_TOLERANCE

    # The whole history matches, not only its last row.
    original = read_history(trained.out)
    for kept, again in zip(original, read_history(epoch_end), strict=True):
        assert abs(float(kept["loss"]) - float(again["loss"])) <= RESUME_TOLERANCE
    reference_state = torch.load(
        train_directory(trained.out) / "model_final.pt", weights_only=True
    )["model"]
    resumed_state = torch.load(train_directory(mid_epoch) / "model_final.pt", weights_only=True)[
        "model"
    ]
    for name, value in reference_state.items():
        assert torch.allclose(value, resumed_state[name], atol=RESUME_TOLERANCE, rtol=0)


def test_a_time_budget_stops_at_a_checkpoint_and_resume_finishes(trained, tmp_path):
    out = tmp_path / "budget"
    shutil.copytree(data_directory(trained.out), data_directory(out))
    # A budget of one microsecond has passed by the first checkpoint, so the stage stops there.
    stopped = train_model(trained.config, out, time_budget=TimeBudget(1e-6))
    assert not stopped.completed
    assert stopped.steps_done == 3
    assert not (train_directory(out) / DONE_MARKER_NAME).exists()
    assert not (train_directory(out) / "model_final.pt").exists()
    assert len(read_history(out)) == 3

    finished = train_model(trained.config, out, resume=True)
    assert finished.completed and finished.resumed_from_step == 3
    assert abs(finished.final_loss - trained.result.final_loss) <= RESUME_TOLERANCE

    again = train_model(trained.config, out, resume=True)
    assert again.skipped and again.completed
    assert abs(again.final_loss - trained.result.final_loss) <= RESUME_TOLERANCE


def test_the_sampler_yields_only_bodies_before_the_monitor_slice(trained):
    config = trained.config
    data = config["data"]
    split = DataSplit(data["n_train"], data["n_cal"], data["n_test"])
    assert split.n_monitor == math.ceil(0.05 * data["n_train"])
    limit = data["n_train"] - split.n_monitor

    dataset = ShapeDataset.for_training(config, data_directory(trained.out), split.sampler_range)
    body_ids = dataset.body_ids
    assert len(body_ids) > 0
    seen = []
    for epoch in range(config["train"]["epochs"]):
        for batch in epoch_batches(
            config["seed"], epoch, len(dataset), config["train"]["batch_size"]
        ):
            seen.extend(body_ids[position] for position in batch)
    assert seen and all(0 <= body_id < limit for body_id in seen)
    # Every body of the sampler range is drawn once per epoch.
    assert sorted(seen) == sorted(body_ids * config["train"]["epochs"])


def test_epoch_batches_depend_only_on_the_seed_and_the_epoch():
    first = epoch_batches(7, 0, 20, 8)
    assert first == epoch_batches(7, 0, 20, 8)
    assert first != epoch_batches(7, 1, 20, 8)
    assert first != epoch_batches(8, 0, 20, 8)
    assert sorted(position for batch in first for position in batch) == list(range(20))
    assert [len(batch) for batch in first] == [8, 8, 4]


def test_training_needs_the_data_stage(trained, tmp_path):
    with pytest.raises(FileNotFoundError, match="data stage is not done"):
        train_model(trained.config, tmp_path / "nothing")
