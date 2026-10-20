"""Baseline eval config (YAML; contract: specs/003-baselines-and-metrics/contracts)."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from strike_a_pose.data.split import SPLITS
from strike_a_pose.runlog import RUN_NAME_RE


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineConfig:
    dataset_dir: str
    split_file: str
    run_name: str = "baselines"
    run_root: str = "runs"
    seed: int = 0
    eval_splits: tuple[str, ...] = ("val", "test")
    batch_size: int = 256
    image_pca_ks: tuple[int, ...] = (8, 16, 32, 64, 128)
    pose_pca_ks: tuple[int, ...] = (4, 8, 16, 32)
    iou_threshold: float = 0.075
    figure_split: str = "val"
    figure_samples: int = 6

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("eval_splits", "image_pca_ks", "pose_pca_ks"):
            d[key] = list(d[key])
        return d

    def with_overrides(self, **kw: Any) -> "BaselineConfig":
        return config_from_dict(
            {**self.to_dict(), **{k: v for k, v in kw.items() if v is not None}}
        )


_REQUIRED = ("dataset_dir", "split_file")
_INT_LISTS = ("image_pca_ks", "pose_pca_ks")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def config_from_dict(data: Any) -> BaselineConfig:
    if not isinstance(data, dict):
        raise ConfigError("config: expected a mapping")
    allowed = {f.name for f in fields(BaselineConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"config: unknown keys {sorted(unknown)}")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ConfigError(f"config: missing required keys {missing}")
    d = dict(data)
    for key in ("dataset_dir", "split_file", "run_name", "run_root", "figure_split"):
        if key in d and (not isinstance(d[key], str) or not d[key]):
            raise ConfigError(f"{key}: must be a non-empty string")
    for key in ("seed", "batch_size", "figure_samples"):
        if key in d and not _is_int(d[key]):
            raise ConfigError(f"{key}: must be an integer")
    for key in _INT_LISTS:
        if key in d:
            v = d[key]
            if not isinstance(v, list | tuple) or not v or not all(_is_int(k) for k in v):
                raise ConfigError(f"{key}: must be a non-empty list of integers")
            if any(k < 1 for k in v) or len(set(v)) != len(v):
                raise ConfigError(f"{key}: values must be unique and >= 1")
            d[key] = tuple(v)
    if "eval_splits" in d:
        v = d["eval_splits"]
        if not isinstance(v, list | tuple) or not v or not all(isinstance(s, str) for s in v):
            raise ConfigError("eval_splits: must be a non-empty list of split names")
        d["eval_splits"] = tuple(v)
    if "iou_threshold" in d:
        t = d["iou_threshold"]
        if isinstance(t, bool) or not isinstance(t, int | float):
            raise ConfigError("iou_threshold: must be a number")
        d["iou_threshold"] = float(t)
    cfg = BaselineConfig(**d)
    _check(cfg)
    return cfg


def _check(cfg: BaselineConfig) -> None:
    if not RUN_NAME_RE.match(cfg.run_name):
        raise ConfigError(f"run_name: must match {RUN_NAME_RE.pattern}")
    bad = [s for s in cfg.eval_splits if s not in SPLITS]
    if bad or len(set(cfg.eval_splits)) != len(cfg.eval_splits):
        raise ConfigError(f"eval_splits: unique names from {list(SPLITS)}, got {cfg.eval_splits}")
    if cfg.figure_split not in cfg.eval_splits:
        raise ConfigError(f"figure_split {cfg.figure_split!r} must be one of eval_splits")
    if cfg.batch_size < 1 or cfg.figure_samples < 1:
        raise ConfigError("batch_size and figure_samples must be >= 1")
    if not 0.0 < cfg.iou_threshold < 1.0:
        raise ConfigError("iou_threshold: must be in (0, 1)")


def load_config(path: str | Path) -> BaselineConfig:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    return config_from_dict(data)
