"""Smoke tests for checkpoint.py: DONE.json markers, staleness, atomic writes, and time budget."""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strike_a_pose.checkpoint import (
    DONE_MARKER_NAME,
    DoneMarker,
    StageStatus,
    TimeBudget,
    atomic_path,
    atomic_write_text,
    check_stage,
    clear_done_marker,
    input_hashes,
    parse_time_budget,
    read_done_marker,
    write_done_marker,
)
from strike_a_pose.config import config_hash, resolve_config

# Real configuration hashes (64 lowercase hexadecimal characters) for three seeds.
HASH_A = config_hash(resolve_config({"seed": 1}))
HASH_B = config_hash(resolve_config({"seed": 2}))
HASH_C = config_hash(resolve_config({"seed": 3}))

MARKER_FIELD_ORDER = [
    "stage",
    "config_hash",
    "seed",
    "code_version",
    "hardware_class",
    "inputs",
    "finished_at",
]


class FakeClock:
    """A clock that moves only when a test moves it, so budget tests never sleep."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _marker_fields(**changes: Any) -> dict[str, Any]:
    """Return the fields of a valid marker, with the given fields replaced."""
    fields: dict[str, Any] = {
        "stage": "train",
        "config_hash": HASH_A,
        "seed": 1,
        "code_version": "0.1.0",
        "hardware_class": "cpu",
        "inputs": {"generate": HASH_B},
        "finished_at": "2026-10-07T12:00:00+00:00",
    }
    fields.update(changes)
    return fields


def _write_marker(directory: Path, **overrides: Any) -> DoneMarker:
    """Write a valid marker into the directory, with the given fields replaced."""
    fields = _marker_fields(**overrides)
    fields.pop("finished_at")  # write_done_marker stamps the time itself
    return write_done_marker(directory, **fields)


def test_atomic_path_renames_the_file_only_after_the_block_ends(tmp_path):
    destination = tmp_path / "shard_0000.npz"
    with atomic_path(destination) as temporary:
        np.savez(temporary, values=np.arange(4))
        assert not destination.exists()
        assert temporary.name.endswith(".npz")
    with np.load(destination) as data:
        assert data["values"].tolist() == [0, 1, 2, 3]
    assert [path.name for path in tmp_path.iterdir()] == ["shard_0000.npz"]


def test_atomic_path_keeps_the_previous_file_when_the_block_fails(tmp_path):
    destination = tmp_path / "result.txt"
    destination.write_text("previous", encoding="utf-8")
    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_path(destination) as temporary:
            temporary.write_text("partial", encoding="utf-8")
            raise RuntimeError("interrupted")
    assert destination.read_text(encoding="utf-8") == "previous"
    assert [path.name for path in tmp_path.iterdir()] == ["result.txt"]


def test_atomic_path_refuses_a_block_that_writes_no_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        with atomic_path(tmp_path / "empty.bin"):
            pass
    assert list(tmp_path.iterdir()) == []


def test_atomic_path_creates_the_parent_directories(tmp_path):
    destination = tmp_path / "data" / "shards" / "shard_0001.npz"
    with atomic_path(destination) as temporary:
        np.savez(temporary, values=np.zeros(2))
    assert destination.is_file()


def test_atomic_write_text_writes_utf8_and_replaces_the_text(tmp_path):
    target = tmp_path / "notes.txt"
    atomic_write_text(target, "first\n")
    assert atomic_write_text(target, "second, with an accent: café\n") == target
    assert target.read_bytes() == "second, with an accent: café\n".encode()
    assert [path.name for path in tmp_path.iterdir()] == ["notes.txt"]


def test_write_done_marker_records_the_fields_in_contract_order(tmp_path):
    stage_directory = tmp_path / "train"
    written = _write_marker(stage_directory)
    data = json.loads((stage_directory / DONE_MARKER_NAME).read_text(encoding="utf-8"))
    assert list(data) == MARKER_FIELD_ORDER
    assert data["config_hash"] == HASH_A
    assert data["inputs"] == {"generate": HASH_B}
    assert datetime.fromisoformat(data["finished_at"]).utcoffset() == timedelta(0)
    assert read_done_marker(stage_directory) == written
    assert [path.name for path in stage_directory.iterdir()] == [DONE_MARKER_NAME]


def test_done_marker_lists_its_inputs_in_sorted_order(tmp_path):
    _write_marker(tmp_path, inputs={"train": HASH_C, "generate": HASH_B})
    data = json.loads((tmp_path / DONE_MARKER_NAME).read_text(encoding="utf-8"))
    assert list(data["inputs"]) == ["generate", "train"]


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"stage": ""}, "stage"),
        ({"config_hash": "abc"}, "config_hash"),
        ({"config_hash": HASH_A.upper()}, "config_hash"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"code_version": " "}, "code_version"),
        ({"hardware_class": ""}, "hardware_class"),
        ({"inputs": {"generate": "not-a-hash"}}, "inputs.generate"),
    ],
)
def test_a_marker_with_an_invalid_field_is_refused_and_writes_nothing(tmp_path, overrides, field):
    with pytest.raises(ValueError, match=field):
        _write_marker(tmp_path, **overrides)
    assert not (tmp_path / DONE_MARKER_NAME).exists()


@pytest.mark.parametrize("finished_at", ["yesterday", "2026-10-07T12:00:00"])
def test_a_marker_with_a_bad_finished_at_time_is_refused(finished_at):
    with pytest.raises(ValueError, match="finished_at"):
        DoneMarker(**_marker_fields(finished_at=finished_at))


def test_a_matching_marker_lets_the_stage_be_skipped(tmp_path):
    _write_marker(tmp_path)
    check = check_stage(tmp_path, stage="train", config_hash=HASH_A, inputs={"generate": HASH_B})
    assert check.status is StageStatus.DONE


def test_a_stage_without_a_directory_or_marker_must_run(tmp_path):
    check = check_stage(
        tmp_path / "train", stage="train", config_hash=HASH_A, inputs={"generate": HASH_B}
    )
    assert check.status is StageStatus.ABSENT
    assert DONE_MARKER_NAME in check.reason


UNREADABLE_MARKERS = [
    "{not json",
    "[]",
    json.dumps({key: value for key, value in _marker_fields().items() if key != "inputs"}),
    json.dumps({**_marker_fields(), "extra": 1}),
    json.dumps(_marker_fields(seed=True)),
    json.dumps(_marker_fields(finished_at="yesterday")),
    json.dumps(_marker_fields(config_hash="abc")),
]


@pytest.mark.parametrize("text", UNREADABLE_MARKERS)
def test_an_unreadable_or_invalid_marker_means_the_stage_must_run(tmp_path, text):
    (tmp_path / DONE_MARKER_NAME).write_text(text, encoding="utf-8")
    check = check_stage(tmp_path, stage="train", config_hash=HASH_A, inputs={"generate": HASH_B})
    assert check.status is StageStatus.ABSENT
    assert read_done_marker(tmp_path) is None


def test_a_different_configuration_hash_makes_the_stage_stale(tmp_path):
    _write_marker(tmp_path)
    check = check_stage(tmp_path, stage="train", config_hash=HASH_C, inputs={"generate": HASH_B})
    assert check.status is StageStatus.STALE
    assert "configuration hash" in check.reason


@pytest.mark.parametrize(
    ("inputs", "named"),
    [
        ({"generate": HASH_C}, "generate"),  # a changed input
        ({}, "generate"),  # a missing input
        ({"generate": HASH_B, "predict": HASH_C}, "predict"),  # an extra input
    ],
)
def test_a_changed_missing_or_extra_input_makes_the_stage_stale(tmp_path, inputs, named):
    _write_marker(tmp_path)
    check = check_stage(tmp_path, stage="train", config_hash=HASH_A, inputs=inputs)
    assert check.status is StageStatus.STALE
    assert named in check.reason


def test_a_marker_of_another_stage_does_not_satisfy_this_stage(tmp_path):
    _write_marker(tmp_path, stage="train")
    check = check_stage(tmp_path, stage="predict", config_hash=HASH_A, inputs={"generate": HASH_B})
    assert check.status is StageStatus.STALE
    assert "'train'" in check.reason


def test_clearing_the_marker_makes_the_stage_run_again(tmp_path):
    _write_marker(tmp_path)
    clear_done_marker(tmp_path)
    check = check_stage(tmp_path, stage="train", config_hash=HASH_A, inputs={"generate": HASH_B})
    assert check.status is StageStatus.ABSENT
    clear_done_marker(tmp_path)  # a second call finds nothing and does not fail


def test_a_stage_is_stale_when_one_of_its_upstream_markers_disappears(tmp_path):
    upstream = {"generate": tmp_path / "generate", "train": tmp_path / "train"}
    predict_directory = tmp_path / "predict"
    _write_marker(upstream["generate"], stage="generate", config_hash=HASH_B, inputs={})
    _write_marker(upstream["train"], stage="train", config_hash=HASH_A, inputs={"generate": HASH_B})
    _write_marker(
        predict_directory,
        stage="predict",
        config_hash=HASH_A,
        inputs={"generate": HASH_B, "train": HASH_A},
    )

    current = input_hashes(upstream)
    assert current == {"generate": HASH_B, "train": HASH_A}
    done = check_stage(predict_directory, stage="predict", config_hash=HASH_A, inputs=current)
    assert done.status is StageStatus.DONE

    clear_done_marker(upstream["train"])
    current = input_hashes(upstream)
    assert current == {"generate": HASH_B}
    stale = check_stage(predict_directory, stage="predict", config_hash=HASH_A, inputs=current)
    assert stale.status is StageStatus.STALE
    assert "train" in stale.reason


def test_a_time_budget_stops_only_when_the_next_unit_would_pass_it():
    clock = FakeClock()
    budget = TimeBudget(100.0, clock=clock)
    clock.now = 40.0
    assert budget.elapsed() == 40.0
    assert budget.should_stop(0.0) is False
    assert budget.should_stop(60.0) is False  # a unit that ends exactly at the budget
    assert budget.should_stop(60.5) is True
    clock.now = 99.9
    assert budget.should_stop(0.2) is True


def test_elapsed_time_counts_from_the_creation_of_the_budget():
    clock = FakeClock(5.0)
    budget = TimeBudget(10.0, clock=clock)
    clock.now = 12.0
    assert budget.elapsed() == 7.0
    assert budget.seconds == 10.0


def test_an_infinite_budget_never_stops_a_run():
    clock = FakeClock()
    budget = TimeBudget(math.inf, clock=clock)
    clock.now = 1e12
    assert budget.should_stop(1e9) is False


@pytest.mark.parametrize("seconds", [0.0, -5.0, math.nan, True])
def test_a_budget_must_be_a_positive_number_of_seconds(seconds):
    with pytest.raises(ValueError, match="positive"):
        TimeBudget(seconds, clock=FakeClock())


@pytest.mark.parametrize("unit_seconds", [-1.0, math.nan])
def test_a_unit_duration_must_be_zero_or_more_seconds(unit_seconds):
    budget = TimeBudget(10.0, clock=FakeClock())
    with pytest.raises(ValueError, match="zero or more"):
        budget.should_stop(unit_seconds)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("8.5h", 30600.0), ("2h", 7200.0), ("90m", 5400.0), ("0.5m", 30.0)],
)
def test_parse_time_budget_reads_hours_and_minutes(text, seconds):
    assert parse_time_budget(text) == seconds


@pytest.mark.parametrize("text", ["", "8.5", "8.5s", "h", "-1h", "1e3h", " 8h", "8 h", "0h", "0m"])
def test_parse_time_budget_refuses_any_other_form(text):
    with pytest.raises(ValueError, match="<hours>h or <minutes>m"):
        parse_time_budget(text)
