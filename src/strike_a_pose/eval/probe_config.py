"""Probe config (YAML; contract: specs/006-latent-probe/contracts/probe-run.md)."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from strike_a_pose.runlog import RUN_NAME_RE

from .config import ConfigError

TARGETS = ("joints_world", "joints_cam", "yaw", "smpl_pose")
PROBE_SPLITS = ("val", "test")


@dataclass(frozen=True)
class ProbeConfig:
    dataset_dir: str
    split_file: str
    checkpoint: str
    run_name: str = "probe"
    run_root: str = "runs"
    seed: int = 0
    cache_dir: str | None = "runs/probe-cache"
    batch_size: int = 256
    eval_splits: tuple[str, ...] = ("val", "test")
    targets: tuple[str, ...] = TARGETS
    ridge_lambdas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    mlp_hidden: tuple[int, ...] = (256, 256)
    mlp_epochs: int = 100
    mlp_batch_size: int = 256
    mlp_lr: float = 1e-3
    mlp_weight_decays: tuple[float, ...] = (0.0, 1e-4, 1e-3)
    pose_pca_ks: tuple[int, ...] = (8, 16)
    random_encoder_control: bool = True
    figure_split: str = "val"
    figure_samples: int = 6

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def with_overrides(self, **kw: Any) -> "ProbeConfig":
        return config_from_dict(
            {**self.to_dict(), **{k: v for k, v in kw.items() if v is not None}}
        )


_REQUIRED = ("dataset_dir", "split_file", "checkpoint")
_STRS = ("dataset_dir", "split_file", "checkpoint", "run_name", "run_root", "figure_split")
_POS_INTS = ("batch_size", "mlp_epochs", "mlp_batch_size", "figure_samples")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _list(d: dict, key: str, ok, what: str, unique: bool = True, cast=None) -> None:
    if key not in d:
        return
    v = d[key]
    if not isinstance(v, list | tuple) or not v or not all(map(ok, v)):
        raise ConfigError(f"{key}: must be a non-empty list of {what}")
    if unique and len(set(v)) != len(v):
        raise ConfigError(f"{key}: values must be unique")
    d[key] = tuple(cast(x) if cast else x for x in v)


def config_from_dict(data: Any) -> ProbeConfig:
    if not isinstance(data, dict):
        raise ConfigError("config: expected a mapping")
    unknown = set(data) - {f.name for f in fields(ProbeConfig)}
    if unknown:
        raise ConfigError(f"config: unknown keys {sorted(unknown)}")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ConfigError(f"config: missing required keys {missing}")
    d = dict(data)
    for key in _STRS:
        if key in d and (not isinstance(d[key], str) or not d[key]):
            raise ConfigError(f"{key}: must be a non-empty string")
    if d.get("cache_dir") is not None and (
        not isinstance(d["cache_dir"], str) or not d["cache_dir"]
    ):
        raise ConfigError("cache_dir: must be a non-empty string or null")
    if "seed" in d and not _is_int(d["seed"]):
        raise ConfigError("seed: must be an integer")
    for key in _POS_INTS:
        if key in d and (not _is_int(d[key]) or d[key] < 1):
            raise ConfigError(f"{key}: must be an integer >= 1")
    if "mlp_lr" in d:
        if not _is_num(d["mlp_lr"]) or d["mlp_lr"] <= 0:
            raise ConfigError("mlp_lr: must be a number > 0")
        d["mlp_lr"] = float(d["mlp_lr"])
    if "random_encoder_control" in d and not isinstance(d["random_encoder_control"], bool):
        raise ConfigError("random_encoder_control: must be true or false")
    _list(d, "eval_splits", lambda s: s in PROBE_SPLITS, f"split names from {list(PROBE_SPLITS)}")
    _list(d, "targets", lambda s: s in TARGETS, f"targets from {list(TARGETS)}")
    _list(d, "ridge_lambdas", lambda x: _is_num(x) and x > 0, "numbers > 0", cast=float)
    _list(d, "mlp_weight_decays", lambda x: _is_num(x) and x >= 0, "numbers >= 0", cast=float)
    _list(d, "mlp_hidden", lambda x: _is_int(x) and x >= 1, "integers >= 1", unique=False)
    _list(d, "pose_pca_ks", lambda x: _is_int(x) and x >= 1, "integers >= 1")
    cfg = ProbeConfig(**d)
    _check(cfg)
    return cfg


def _check(cfg: ProbeConfig) -> None:
    if not RUN_NAME_RE.match(cfg.run_name):
        raise ConfigError(f"run_name: must match {RUN_NAME_RE.pattern}")
    if "val" not in cfg.eval_splits:
        raise ConfigError("eval_splits: must contain 'val' (probe selection uses val)")
    if cfg.figure_split not in cfg.eval_splits:
        raise ConfigError(f"figure_split {cfg.figure_split!r} must be one of eval_splits")


def load_config(path: str | Path) -> ProbeConfig:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    return config_from_dict(data)
