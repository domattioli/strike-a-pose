"""Stage checkpoints for sap runs: DONE.json markers, staleness checks, atomic writes, time budget.

The design follows specs/001-kill-test-mvp/research.md R9 and R10, contracts/artifacts.md, and the
stage state transitions of data-model.md. Every stage directory ends with a DONE.json marker that
holds the configuration hash and the configuration hashes of the markers the stage consumed. A
stage is stale when either hash differs from the current run. Files reach their final names by an
atomic rename (os.replace, Python standard library, documented at
https://docs.python.org/3/library/os.html#os.replace), so a partial file never carries a final
name. A run stops at a checkpoint when its next unit of work would pass the time budget (FR-029;
exit code 7 in contracts/cli.md). This module implements no published method, so it cites no
algorithm source.
"""

import json
import math
import os
import re
import secrets
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "DONE_MARKER_NAME",
    "DoneMarker",
    "StageCheck",
    "StageStatus",
    "TimeBudget",
    "atomic_path",
    "atomic_write_text",
    "check_stage",
    "clear_done_marker",
    "input_hashes",
    "parse_time_budget",
    "read_done_marker",
    "write_done_marker",
]

# The marker that ends every stage directory (FR-029, contracts/artifacts.md).
DONE_MARKER_NAME = "DONE.json"

# A SHA-256 hex digest as config_hash returns it: 64 lowercase hexadecimal characters.
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
# The --time-budget forms of contracts/cli.md: <hours>h or <minutes>m.
_BUDGET_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)([hm])")
_SECONDS_PER_UNIT = {"h": 3600.0, "m": 60.0}


@contextmanager
def atomic_path(destination: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield a temporary path beside destination, and rename it to destination when the block ends.

    The block must create the file at the yielded path, for example with numpy.savez or torch.save.
    The temporary name keeps the suffix of destination, so a writer that adds a suffix to a name
    that has none (numpy.savez adds .npz) writes to the name it was given. The file is flushed to
    disk before the rename, so a crash leaves either the old file or the whole new one. When the
    block raises, the temporary file is removed and destination keeps its previous content.
    """
    final = Path(destination)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(f".{final.stem}.{secrets.token_hex(8)}.temporary{final.suffix}")
    try:
        yield temporary
        _flush_file(temporary)
        os.replace(temporary, final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _flush_file(path: Path) -> None:
    """Write the file's data through to disk. Windows needs the file open for writing to flush."""
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(destination: str | os.PathLike[str], text: str) -> Path:
    """Write text to destination as UTF-8, atomically, and return the destination path."""
    final = Path(destination)
    with atomic_path(final) as temporary:
        temporary.write_text(text, encoding="utf-8")
    return final


class StageStatus(Enum):
    """What the DONE.json of a stage directory says about the current run."""

    ABSENT = "absent"  # no readable marker: the stage runs
    STALE = "stale"  # a marker exists, but its stage, configuration hash or input hashes differ
    DONE = "done"  # the marker matches the current run: --resume may skip the stage


@dataclass(frozen=True)
class StageCheck:
    """The status of one stage, with the reason for the progress log."""

    status: StageStatus
    reason: str


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"DONE.json field '{name}' must be non-empty text; got {value!r}")


def _require_hash(name: str, value: Any) -> None:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"DONE.json field '{name}' must be a SHA-256 hex digest of 64 lowercase characters; "
            f"got {value!r}"
        )


def _require_time(name: str, value: Any) -> None:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"DONE.json field '{name}' must be an ISO 8601 time; got {value!r}"
        ) from error
    if moment.tzinfo is None:
        raise ValueError(f"DONE.json field '{name}' must carry a time zone offset; got {value!r}")


@dataclass(frozen=True)
class DoneMarker:
    """The fields of one DONE.json, in the order of contracts/artifacts.md.

    inputs maps each upstream stage name to the configuration hash in that stage's own marker.
    Construction checks every field, so a marker read from disk is checked the same way as one
    about to be written.
    """

    stage: str
    config_hash: str
    seed: int
    code_version: str
    hardware_class: str
    inputs: dict[str, str]
    finished_at: str

    def __post_init__(self) -> None:
        _require_text("stage", self.stage)
        _require_hash("config_hash", self.config_hash)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(
                f"DONE.json field 'seed' must be a non-negative integer; got {self.seed!r}"
            )
        _require_text("code_version", self.code_version)
        _require_text("hardware_class", self.hardware_class)
        if not isinstance(self.inputs, Mapping):
            raise ValueError(f"DONE.json field 'inputs' must be a mapping; got {self.inputs!r}")
        for name, digest in self.inputs.items():
            _require_text("inputs key", name)
            _require_hash(f"inputs.{name}", digest)
        _require_time("finished_at", self.finished_at)

    def to_dict(self) -> dict[str, Any]:
        """Return the marker as a JSON-ready dict in the documented field order, inputs sorted."""
        return {
            "stage": self.stage,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "code_version": self.code_version,
            "hardware_class": self.hardware_class,
            "inputs": {name: self.inputs[name] for name in sorted(self.inputs)},
            "finished_at": self.finished_at,
        }


def write_done_marker(
    stage_directory: str | os.PathLike[str],
    *,
    stage: str,
    config_hash: str,
    seed: int,
    code_version: str,
    hardware_class: str,
    inputs: Mapping[str, str],
) -> DoneMarker:
    """Write DONE.json into the stage directory and return the marker that was recorded.

    Call this last, after every output of the stage is complete, because the marker says the stage
    is done. finished_at is the UTC time of the call. The write is atomic.
    """
    marker = DoneMarker(
        stage=stage,
        config_hash=config_hash,
        seed=seed,
        code_version=code_version,
        hardware_class=hardware_class,
        inputs=dict(inputs),
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    text = json.dumps(marker.to_dict(), indent=2) + "\n"
    atomic_write_text(Path(stage_directory) / DONE_MARKER_NAME, text)
    return marker


def read_done_marker(stage_directory: str | os.PathLike[str]) -> DoneMarker | None:
    """Return the marker in the stage directory, or None when it is missing, unreadable or invalid.

    A missing or unreadable marker is not an error here. The stage simply runs again.
    """
    path = Path(stage_directory) / DONE_MARKER_NAME
    try:
        fields = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(fields, dict):
        return None
    try:
        return DoneMarker(**fields)
    except (TypeError, ValueError):
        return None


def clear_done_marker(stage_directory: str | os.PathLike[str]) -> None:
    """Remove the marker before a stage writes its outputs, so an interrupted run cannot look done.

    Call this when a stage starts running, so that a crash midway leaves no marker behind.
    """
    (Path(stage_directory) / DONE_MARKER_NAME).unlink(missing_ok=True)


def input_hashes(stage_directories: Mapping[str, str | os.PathLike[str]]) -> dict[str, str]:
    """Return the configuration hash of each upstream stage's marker, keyed by stage name.

    An upstream stage without a readable marker is left out. check_stage then reports the
    downstream stage as stale, because the downstream marker recorded that input.
    """
    hashes: dict[str, str] = {}
    for name, directory in stage_directories.items():
        marker = read_done_marker(directory)
        if marker is not None:
            hashes[name] = marker.config_hash
    return hashes


def check_stage(
    stage_directory: str | os.PathLike[str],
    *,
    stage: str,
    config_hash: str,
    inputs: Mapping[str, str],
) -> StageCheck:
    """Decide whether a stage may be skipped on resume: its marker must match the current run.

    The marker must name this stage, carry the current configuration hash, and carry exactly the
    input hashes given (no more and no fewer). Anything else means the stage runs. ABSENT means
    there is no readable marker. STALE means a field differs (data-model.md, stage markers). The
    code version is recorded in the marker but is not compared, because data-model.md defines
    staleness by the configuration hash and the input hashes only.
    """
    marker = read_done_marker(stage_directory)
    if marker is None:
        return StageCheck(
            StageStatus.ABSENT,
            f"{DONE_MARKER_NAME} is missing or unreadable in {Path(stage_directory)}",
        )
    if marker.stage != stage:
        return StageCheck(
            StageStatus.STALE, f"the marker belongs to stage '{marker.stage}', not '{stage}'"
        )
    if marker.config_hash != config_hash:
        return StageCheck(StageStatus.STALE, "the configuration hash differs from the marker")
    expected = dict(inputs)
    if marker.inputs != expected:
        changed = sorted(
            name
            for name in set(marker.inputs) | set(expected)
            if marker.inputs.get(name) != expected.get(name)
        )
        return StageCheck(StageStatus.STALE, f"the input hashes differ for {', '.join(changed)}")
    return StageCheck(StageStatus.DONE, "the marker matches the configuration and its inputs")


class TimeBudget:
    """The wall-clock budget of one run, counted from the moment the budget is created.

    Before each unit of work (a shard, an epoch, a cell), a stage asks should_stop. When that unit
    would end after the budget does, the stage stops at its last checkpoint, and the run exits with
    code 7 so that a later session can resume (contracts/cli.md). A budget of math.inf never stops
    a run.
    """

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        """Start a budget of the given number of seconds, measured on the given clock."""
        if isinstance(seconds, bool) or math.isnan(seconds) or seconds <= 0:
            raise ValueError(f"a time budget must be a positive number of seconds; got {seconds!r}")
        self._seconds = float(seconds)
        self._clock = clock
        self._started = clock()

    @property
    def seconds(self) -> float:
        """The budget in seconds."""
        return self._seconds

    def elapsed(self) -> float:
        """Return the seconds since the budget was created."""
        return self._clock() - self._started

    def should_stop(self, next_unit_seconds: float) -> bool:
        """Return True when a unit of work of next_unit_seconds would end after the budget does.

        Pass the expected duration of that unit, for example the duration of the previous unit, or 0
        before any unit has run. A unit that ends exactly at the budget does not stop the run.
        """
        if math.isnan(next_unit_seconds) or next_unit_seconds < 0:
            raise ValueError(
                "the duration of a unit of work must be zero or more seconds; "
                f"got {next_unit_seconds!r}"
            )
        return self.elapsed() + next_unit_seconds > self._seconds


def parse_time_budget(text: str) -> float:
    """Return the seconds in a --time-budget value: hours as 8.5h, or minutes as 90m."""
    match = _BUDGET_PATTERN.fullmatch(text)
    if match is None or float(match.group(1)) <= 0:
        raise ValueError(
            "a time budget must be <hours>h or <minutes>m with a positive number, such as 8.5h "
            f"or 90m; got {text!r}"
        )
    amount, unit = match.groups()
    return float(amount) * _SECONDS_PER_UNIT[unit]
