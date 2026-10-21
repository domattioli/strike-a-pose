"""Run-directory helpers shared by eval (003) and train (004) CLIs (constitution III).

Seeding, environment summary, git commit, run-dir creation, stable JSON/YAML writers.
"""

import hashlib
import json
import math
import os
import platform
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import yaml

from strike_a_pose.datagen.generate import git_commit

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SIG_DIGITS = 6


def seed_everything(seed: int) -> None:
    """Seed Python ``random``, NumPy's global RNG and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_info() -> dict:
    """``{"commit": sha | "unknown", "dirty": bool | None}`` of this checkout."""
    return git_commit()


def cpu_model() -> str | None:
    """The CPU model (first ``model name`` of /proc/cpuinfo, else ``platform.processor()``), so
    "same machine type" is checkable across runs (spec 012); None if unknown."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip() or None
    except OSError:
        pass
    return platform.processor() or None


def environment_summary() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pillow": PIL.__version__,
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model(),  # spec 012
        "torch_threads": torch.get_num_threads(),
    }


def checksums_sha256(dataset_dir: str | Path) -> str | None:
    """SHA-256 of a dataset's ``checksums.json`` (the per-image hashes, i.e. the pixels the
    manifest hashes do not cover; spec 012), or None if the file is missing."""
    p = Path(dataset_dir) / "checksums.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def utc_now() -> datetime:
    return datetime.now(UTC)


def make_run_dir(run_root: str | Path, run_name: str, now: datetime | None = None) -> Path:
    """Create ``<run_root>/<run_name>-<YYYYmmddTHHMMSSZ>``; raise FileExistsError if taken."""
    if not RUN_NAME_RE.match(run_name):
        raise ValueError(f"run_name must match {RUN_NAME_RE.pattern}, got {run_name!r}")
    stamp = (now or utc_now()).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    d = Path(run_root) / f"{run_name}-{stamp}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def round_sig(x: Any, digits: int = SIG_DIGITS) -> Any:
    """Recursively round floats to ``digits`` significant digits; NaN → None."""
    if isinstance(x, dict):
        return {str(k): round_sig(v, digits) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [round_sig(v, digits) for v in x]
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        v = float(x)
        if math.isnan(v):
            return None
        if math.isinf(v) or v == 0.0:
            return 0.0 if v == 0.0 else v
        r = round(v, digits - 1 - math.floor(math.log10(abs(v))))
        return 0.0 if r == 0 else r
    return x


def dumps_exact(obj: Any) -> str:
    """Stable JSON text: sig-digit rounding, sorted keys, indent 2, trailing newline."""
    return json.dumps(round_sig(obj), sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_json_exact(path: str | Path, obj: Any) -> None:
    Path(path).write_text(dumps_exact(obj))


def write_yaml(path: str | Path, obj: Any) -> None:
    Path(path).write_text(yaml.safe_dump(obj, sort_keys=True, default_flow_style=None))
