"""Round-trip config (YAML; spec 016 FR-001, data-model.md "RoundTripConfig")."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from strike_a_pose.runlog import RUN_NAME_RE

from .config import ConfigError


@dataclass(frozen=True)
class RoundTripConfig:
    sample_run: str  # 015 sample run dir (config.yaml, samples.json)
    probe_run: str  # 006 probe run dir of the same VAE (config.yaml, metrics.json)
    smpl_model_path: str = "assets/smpl/SMPL_NEUTRAL.npz"
    run_name: str = "roundtrip"
    run_root: str = "runs"
    seed: int = 0
    calibration_percentile: float = 5.0
    control: bool = True
    figure_samples: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kw: Any) -> "RoundTripConfig":
        return config_from_dict(
            {**self.to_dict(), **{k: v for k, v in kw.items() if v is not None}}
        )


_REQUIRED = ("sample_run", "probe_run")
_STRS = ("sample_run", "probe_run", "smpl_model_path", "run_name", "run_root")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def config_from_dict(data: Any) -> RoundTripConfig:
    if not isinstance(data, dict):
        raise ConfigError("config: expected a mapping")
    unknown = set(data) - {f.name for f in fields(RoundTripConfig)}
    if unknown:
        raise ConfigError(f"config: unknown keys {sorted(unknown)}")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ConfigError(f"config: missing required keys {missing}")
    d = dict(data)
    for key in _STRS:
        if key in d and (not isinstance(d[key], str) or not d[key]):
            raise ConfigError(f"{key}: must be a non-empty string")
    if "seed" in d and (not _is_int(d["seed"]) or d["seed"] < 0):
        raise ConfigError("seed: must be an integer >= 0")
    if "calibration_percentile" in d:
        p = d["calibration_percentile"]
        if (
            isinstance(p, bool)
            or not isinstance(p, int | float)
            or not math.isfinite(p)
            or not 0 < p <= 50
        ):
            raise ConfigError("calibration_percentile: must be a number in (0, 50]")
        d["calibration_percentile"] = float(p)
    if "control" in d and not isinstance(d["control"], bool):
        raise ConfigError("control: must be true or false")
    if "figure_samples" in d and (
        not _is_int(d["figure_samples"]) or not 1 <= d["figure_samples"] <= 32
    ):
        raise ConfigError("figure_samples: must be an integer in 1..32")
    cfg = RoundTripConfig(**d)
    if not RUN_NAME_RE.match(cfg.run_name):
        raise ConfigError(f"run_name: must match {RUN_NAME_RE.pattern}")
    return cfg


def load_config(path: str | Path) -> RoundTripConfig:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    return config_from_dict(data)
