"""Run record for sap runs: run_record.json fields, library versions, stage timings, atomic write.

The fields follow contracts/artifacts.md and the RunRecord entity of data-model.md. The record
names the configuration hash, seed, code version, hardware class, and library versions that
produced a run, so every committed table traces back to its environment (FR-025, constitution
Principle V, research R9). Each stage timing is written when its stage ends, and the file is
replaced atomically.
"""

import contextlib
import json
import math
import os
import platform
import re
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from strike_a_pose import __version__ as package_version
from strike_a_pose.config import config_hash as configuration_hash
from strike_a_pose.config import resolve_config

__all__ = [
    "HARDWARE_CLASSES",
    "STAGES",
    "VERSION_NAMES",
    "RunRecord",
    "RunRecordError",
    "current_code_version",
    "current_library_versions",
    "read_run_record",
    "start_run_record",
    "write_run_record",
]

# The stage names of run_record.json in run order. real_eval is the sap real-eval command.
STAGES: tuple[str, ...] = (
    "generate",
    "train",
    "predict",
    "calibrate",
    "evaluate",
    "verdict",
    "report",
    "real_eval",
)
# The hardware classes of FR-028 (device.py stores the same two values).
HARDWARE_CLASSES: tuple[str, ...] = ("cpu", "gpu")
# Python is always present. smplx and sam2 are optional extras, and read as null when not installed.
VERSION_NAMES: tuple[str, ...] = ("python", "numpy", "torch", "opencv", "smplx", "sam2")

# The fields of run_record.json in the order of contracts/artifacts.md.
_FIELD_NAMES: tuple[str, ...] = (
    "config_hash",
    "seed",
    "code_version",
    "hardware_class",
    "device_name",
    "versions",
    "started_at",
    "finished_at",
    "timings",
)
# The installed distribution names that provide each library. OpenCV ships under four names.
_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "numpy": ("numpy",),
    "torch": ("torch",),
    "opencv": (
        "opencv-python-headless",
        "opencv-python",
        "opencv-contrib-python-headless",
        "opencv-contrib-python",
    ),
    "smplx": ("smplx",),
    "sam2": ("sam2",),
}
_CONFIG_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
# The commit length in code_version. It matches the length the CLI prints for config_hash.
_COMMIT_LENGTH = 12
_GIT_TIMEOUT_SECONDS = 10.0
# The directory of this package's source. Git reads the checkout that contains it.
_PACKAGE_DIRECTORY = Path(__file__).resolve().parent


class RunRecordError(ValueError):
    """A run record that breaks the rules of its fields, or a file that cannot be read as one."""


def _empty_timings() -> dict[str, float | None]:
    """Return a timing map with every stage still unfinished."""
    return dict.fromkeys(STAGES)


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise RunRecordError(f"run record field '{name}' must be non-empty text; got {value!r}")
    return value


def _require_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunRecordError(
            f"run record field 'seed' must be a non-negative integer; got {value!r}"
        )
    return value


def _require_config_hash(value: Any) -> str:
    if not isinstance(value, str) or _CONFIG_HASH_PATTERN.fullmatch(value) is None:
        raise RunRecordError(
            "run record field 'config_hash' must be 64 lowercase hexadecimal characters; "
            f"got {value!r}"
        )
    return value


def _require_hardware_class(value: Any) -> str:
    if value not in HARDWARE_CLASSES:
        raise RunRecordError(
            f"run record field 'hardware_class' must be one of {', '.join(HARDWARE_CLASSES)}; "
            f"got {value!r}"
        )
    return value


def _parse_instant(text: str) -> datetime | None:
    """Parse ISO 8601 text, reading a trailing Z as UTC; return None when it does not parse."""
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _require_instant(name: str, value: Any) -> str:
    """Return a timestamp as text after checking that it parses and carries a time zone offset."""
    moment = _parse_instant(value) if isinstance(value, str) else None
    if moment is None or moment.utcoffset() is None:
        raise RunRecordError(
            f"run record field '{name}' must be an ISO 8601 timestamp with a time zone offset; "
            f"got {value!r}"
        )
    return value


def _format_instant(moment: datetime) -> str:
    """Return a timezone-aware datetime as UTC ISO 8601 text, to the second."""
    if moment.utcoffset() is None:
        raise RunRecordError("a run record time must include a time zone; got a naive datetime")
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _require_versions(value: Any) -> dict[str, str | None]:
    """Check the six library entries: python is text, and each other library is text or null."""
    if not isinstance(value, Mapping) or set(value) != set(VERSION_NAMES):
        raise RunRecordError(
            f"run record field 'versions' must hold exactly the keys {', '.join(VERSION_NAMES)}"
        )
    checked: dict[str, str | None] = {}
    for name in VERSION_NAMES:
        entry = value[name]
        if entry is None and name != "python":
            checked[name] = None
        elif isinstance(entry, str) and entry.strip() != "":
            checked[name] = entry
        else:
            allowed = "text" if name == "python" else "text or null"
            raise RunRecordError(
                f"run record field 'versions.{name}' must be {allowed}; got {entry!r}"
            )
    return checked


def _require_seconds(stage: str, value: Any) -> float:
    """Return a stage timing as a float: a finite number of seconds that is not negative."""
    message = (
        f"the timing of stage '{stage}' must be a finite number of seconds of at least 0; "
        f"got {value!r}"
    )
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunRecordError(message)
    try:
        seconds = float(value)
    except OverflowError as error:
        raise RunRecordError(message) from error
    if not math.isfinite(seconds) or seconds < 0:
        raise RunRecordError(message)
    return seconds


def _require_timings(value: Any) -> dict[str, float | None]:
    """Check the timing map: every stage of STAGES, each null or a valid number of seconds."""
    if not isinstance(value, Mapping) or set(value) != set(STAGES):
        raise RunRecordError(
            f"run record field 'timings' must hold exactly the stages {', '.join(STAGES)}"
        )
    return {
        stage: None if value[stage] is None else _require_seconds(stage, value[stage])
        for stage in STAGES
    }


def _check_stage(stage: Any) -> str:
    if stage not in STAGES:
        raise RunRecordError(f"unknown stage {stage!r}; the stages are {', '.join(STAGES)}")
    return stage


@dataclass
class RunRecord:
    """The fields of run_record.json (contracts/artifacts.md; RunRecord in data-model.md).

    Every field is checked when the record is built. A timing is null until its stage ends, and
    finished_at is null until the run ends, so a record written early in a run shows which stages it
    has completed.
    """

    config_hash: str
    seed: int
    code_version: str
    hardware_class: str
    device_name: str
    versions: dict[str, str | None]
    started_at: str
    finished_at: str | None = None
    timings: dict[str, float | None] = field(default_factory=_empty_timings)

    def __post_init__(self) -> None:
        self.config_hash = _require_config_hash(self.config_hash)
        self.seed = _require_seed(self.seed)
        self.code_version = _require_text("code_version", self.code_version)
        self.hardware_class = _require_hardware_class(self.hardware_class)
        self.device_name = _require_text("device_name", self.device_name)
        self.versions = _require_versions(self.versions)
        self.started_at = _require_instant("started_at", self.started_at)
        if self.finished_at is not None:
            self.finished_at = _require_instant("finished_at", self.finished_at)
        self.timings = _require_timings(self.timings)

    def add_timing(self, stage: str, seconds: float) -> None:
        """Add seconds to the timing of a stage.

        A stage timed more than once sums its timings. A stage that --resume finishes in a later
        session therefore reports its total compute time, when the caller continues the record it
        read back with read_run_record.
        """
        _check_stage(stage)
        duration = _require_seconds(stage, seconds)
        previous = self.timings[stage]
        self.timings[stage] = duration if previous is None else previous + duration

    @contextlib.contextmanager
    def time_stage(self, stage: str) -> Iterator[None]:
        """Time the block of one stage and add the elapsed seconds when the block ends normally.

        A block that raises records no timing, so a stage that fails keeps its null timing.
        """
        _check_stage(stage)
        started = time.perf_counter()
        yield
        self.add_timing(stage, time.perf_counter() - started)

    def finish(self, finished_at: datetime | None = None) -> None:
        """Record when the run ended, as UTC. Without an argument the current time is used."""
        self.finished_at = _format_instant(finished_at or datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-ready mapping, in the field order of the contract."""
        return {
            "config_hash": self.config_hash,
            "seed": self.seed,
            "code_version": self.code_version,
            "hardware_class": self.hardware_class,
            "device_name": self.device_name,
            "versions": dict(self.versions),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "timings": dict(self.timings),
        }

    @classmethod
    def from_dict(cls, document: Any) -> "RunRecord":
        """Build a record from a parsed run_record.json, refusing a missing or an unknown field."""
        if not isinstance(document, Mapping):
            raise RunRecordError("a run record must be a JSON object")
        missing = [name for name in _FIELD_NAMES if name not in document]
        unknown = sorted(str(name) for name in document if name not in _FIELD_NAMES)
        problems = []
        if missing:
            problems.append("missing " + ", ".join(missing))
        if unknown:
            problems.append("unknown " + ", ".join(unknown))
        if problems:
            raise RunRecordError("the run record has " + "; ".join(problems))
        return cls(**{name: document[name] for name in _FIELD_NAMES})


def start_run_record(
    config: Mapping[str, Any],
    *,
    hardware_class: str,
    device_name: str,
    started_at: datetime | None = None,
    code_version: str | None = None,
    versions: Mapping[str, str | None] | None = None,
) -> RunRecord:
    """Start the record of one run from its configuration, its hardware class, and its device name.

    The configuration is validated first, and the hash and seed come from it (research R9). The code
    version and the library versions are read from this environment unless the caller passes them.
    The start time defaults to the current time.
    """
    resolved = resolve_config(config)
    return RunRecord(
        config_hash=configuration_hash(resolved),
        seed=resolved["seed"],
        code_version=current_code_version() if code_version is None else code_version,
        hardware_class=hardware_class,
        device_name=device_name,
        versions=current_library_versions() if versions is None else dict(versions),
        started_at=_format_instant(started_at or datetime.now(timezone.utc)),
    )


def current_library_versions() -> dict[str, str | None]:
    """Return the Python version and the installed version of each library in VERSION_NAMES.

    A library that is not installed reads as None. The versions come from the installed package
    metadata, so no library module is imported.
    """
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in VERSION_NAMES[1:]:
        versions[name] = _installed_version(_DISTRIBUTIONS[name])
    return versions


def _installed_version(distributions: tuple[str, ...]) -> str | None:
    """Return the version of the first installed distribution in the tuple, or None."""
    for name in distributions:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return None


def current_code_version() -> str:
    """Return the package version, followed by the commit of the git checkout when git names one.

    The form is "<version>+<commit>", where the commit is its first 12 characters. ".dirty" is
    appended when the checkout has uncommitted or untracked changes, because the specification does
    not settle dirty trees (checklist CHK021), so the record names them. Outside a git checkout, or
    in a checkout with no commit yet, the form is the bare package version.
    """
    checkout = _checkout_state()
    return package_version if checkout is None else f"{package_version}+{checkout}"


def _git(arguments: list[str]) -> str | None:
    """Return the standard output of one git command run in the package directory, or None."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=_PACKAGE_DIRECTORY,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _checkout_state() -> str | None:
    """Return the commit of the checkout around this package, with ".dirty" when it has changes."""
    commit = _git(["rev-parse", f"--short={_COMMIT_LENGTH}", "HEAD"])
    if commit is None:
        return None
    status = _git(["status", "--porcelain"])
    # A status that cannot be read counts as dirty, because a clean tree cannot be shown.
    dirty = status is None or status.strip() != ""
    return commit.strip() + (".dirty" if dirty else "")


def write_run_record(record: RunRecord, path: str | Path) -> None:
    """Write run_record.json atomically, after checking that the record reads back as valid.

    A reader sees the previous file or the new one, never part of either. Call this when each stage
    ends as well as at the end of the run, so a session that stops early keeps the timings of the
    stages it finished. The parent directory is created when it is missing.
    """
    # Reading the document back refuses a record whose fields were changed after it was built.
    document = RunRecord.from_dict(record.to_dict()).to_dict()
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    _write_text_atomically(Path(path), text)


def _write_text_atomically(path: Path, text: str) -> None:
    """Write text to a temporary file beside path, then rename it over path.

    The rename replaces the file in one step on POSIX systems, so a partial file never takes the
    place of a complete one. A failed write removes its temporary file and leaves path unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    _sync_directory(path.parent)


def _sync_directory(directory: Path) -> None:
    """Flush a directory entry to disk where the platform allows it."""
    with contextlib.suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def read_run_record(path: str | Path) -> RunRecord:
    """Read and validate a run_record.json file. Raises RunRecordError for any defect."""
    location = Path(path)
    try:
        text = location.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        reason = getattr(error, "strerror", None) or error
        raise RunRecordError(f"cannot read the run record '{location}': {reason}") from error
    try:
        document = json.loads(text, parse_constant=_refuse_json_constant)
    except ValueError as error:
        raise RunRecordError(f"the run record '{location}' is not valid JSON: {error}") from error
    return RunRecord.from_dict(document)


def _refuse_json_constant(name: str) -> Any:
    """Reject NaN and the infinities, which JSON does not allow and no record field may hold."""
    raise ValueError(f"the JSON constant {name} is not a finite number")
