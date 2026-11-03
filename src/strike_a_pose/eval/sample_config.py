"""Sampling config (YAML; spec 015 FR-008, data-model.md "SampleConfig")."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from strike_a_pose.runlog import RUN_NAME_RE

from .config import ConfigError


@dataclass(frozen=True)
class SampleConfig:
    dataset_dir: str
    split_file: str
    checkpoint: str  # the VAE
    reference_checkpoint: str | None = None  # a plain AE (lower reference bar)
    run_name: str = "sample"
    run_root: str = "runs"
    seed: int = 0
    n_samples: int = 1024
    n_recon: int = 1024
    recon_split: str = "val"  # split of the μ reconstructions (upper bar); "test" for the fixture
    batch_size: int = 256
    iou_threshold: float = 0.075
    figure_samples: int = 32  # 4 x 8 grid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kw: Any) -> "SampleConfig":
        return config_from_dict(
            {**self.to_dict(), **{k: v for k, v in kw.items() if v is not None}}
        )


_REQUIRED = ("dataset_dir", "split_file", "checkpoint")
_STRS = ("dataset_dir", "split_file", "checkpoint", "run_name", "run_root")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def config_from_dict(data: Any) -> SampleConfig:
    if not isinstance(data, dict):
        raise ConfigError("config: expected a mapping")
    unknown = set(data) - {f.name for f in fields(SampleConfig)}
    if unknown:
        raise ConfigError(f"config: unknown keys {sorted(unknown)}")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ConfigError(f"config: missing required keys {missing}")
    d = dict(data)
    for key in _STRS:
        if key in d and (not isinstance(d[key], str) or not d[key]):
            raise ConfigError(f"{key}: must be a non-empty string")
    ref = d.get("reference_checkpoint")
    if ref is not None and (not isinstance(ref, str) or not ref):
        raise ConfigError("reference_checkpoint: must be a non-empty string or null")
    if "seed" in d and (not _is_int(d["seed"]) or d["seed"] < 0):
        raise ConfigError("seed: must be an integer >= 0")
    for key in ("n_samples", "batch_size"):
        if key in d and (not _is_int(d[key]) or d[key] < 1):
            raise ConfigError(f"{key}: must be an integer >= 1")
    if "n_recon" in d and (not _is_int(d["n_recon"]) or d["n_recon"] < 0):
        raise ConfigError("n_recon: must be an integer >= 0")
    if "figure_samples" in d and (not _is_int(d["figure_samples"]) or
                                  not 1 <= d["figure_samples"] <= 64):  # fmt: skip
        raise ConfigError("figure_samples: must be an integer in 1..64")
    if "iou_threshold" in d:
        t = d["iou_threshold"]
        if (
            isinstance(t, bool)
            or not isinstance(t, int | float)
            or not math.isfinite(t)
            or not (0 < t < 1)
        ):
            raise ConfigError("iou_threshold: must be a number in (0, 1)")
        d["iou_threshold"] = float(t)
    if "recon_split" in d and d["recon_split"] not in ("val", "test"):
        raise ConfigError("recon_split: must be 'val' or 'test'")
    cfg = SampleConfig(**d)
    if not RUN_NAME_RE.match(cfg.run_name):
        raise ConfigError(f"run_name: must match {RUN_NAME_RE.pattern}")
    return cfg


def load_config(path: str | Path) -> SampleConfig:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    return config_from_dict(data)
