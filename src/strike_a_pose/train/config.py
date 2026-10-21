"""Train config (YAML; contract: specs/004-autoencoder-mvp/contracts/train-run.md)."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from strike_a_pose.data.split import SPLITS
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.models import DEFAULT_CHANNELS
from strike_a_pose.runlog import RUN_NAME_RE

LR_SCHEDULES = ("constant", "cosine")
WEIGHT_SELECTIONS = ("best_val", "final")  # spec 008: which weights a run scores
# Config keys added after runs existed, with the value a run directory whose config lacks
# the key was started under. Read on --resume only (new runs get the dataclass default),
# so run directories created before a key was added keep resuming.
LEGACY_VALUES: dict[str, Any] = {
    "select_weights": "final",  # spec 008: runs before it scored the final weights
    # spec 011 training variants: runs before them trained the plain AE (= the defaults)
    "mask_weight": 0.0,
    "variational": False,
    "beta": 0.0,
    "weight_decay": 0.0,
    "shift_px": 0,
    "dropout": 0.0,
    # spec 012 run guards: runs before them had none
    "early_stop": None,
    "abort_on_divergence": False,
    "divergence_ratio": 10.0,
}
MAX_SHIFT_PX = 32  # spec 011: "small" pixel shifts
DIVERGENCE_RATIO = 10.0  # spec 012 default; another value needs abort_on_divergence: true
EARLY_STOP_KEYS = ("patience", "min_epochs", "min_delta")


@dataclass(frozen=True)
class EarlyStop:
    """Early stop on a val plateau (spec 012): after an epoch with a finite val loss, from epoch
    ``min_epochs`` on, stop once ``patience`` epochs passed since the last epoch whose val loss
    was below the previous anchor by more than ``min_delta`` (train.guards.early_stop_state)."""

    patience: int
    min_epochs: int = 0
    min_delta: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: str
    split_file: str
    run_name: str = "ae"
    run_root: str = "runs"
    seed: int = 0
    latent_dim: int = 64
    channels: tuple[int, ...] = DEFAULT_CHANNELS
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    lr_schedule: str = "cosine"
    eval_splits: tuple[str, ...] = ("val", "test")
    select_weights: str = "best_val"
    context_pca_k: int | None = 64
    iou_threshold: float = 0.075
    figure_split: str = "val"
    figure_samples: int = 6
    # spec 011 training variants (the defaults are the plain AE, bit for bit)
    mask_weight: float = 0.0
    variational: bool = False
    beta: float = 0.0
    weight_decay: float = 0.0
    shift_px: int = 0
    dropout: float = 0.0
    # spec 012 run guards (off by default)
    early_stop: EarlyStop | None = None
    abort_on_divergence: bool = False
    divergence_ratio: float = DIVERGENCE_RATIO

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("channels", "eval_splits"):
            d[key] = list(d[key])
        return d

    def with_overrides(self, **kw: Any) -> "TrainConfig":
        return config_from_dict(
            {**self.to_dict(), **{k: v for k, v in kw.items() if v is not None}}
        )


_REQUIRED = ("dataset_dir", "split_file")
_STRS = (
    "dataset_dir",
    "split_file",
    "run_name",
    "run_root",
    "lr_schedule",
    "figure_split",
    "select_weights",
)
_POS_INTS = ("latent_dim", "epochs", "batch_size", "figure_samples")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def config_from_dict(data: Any) -> TrainConfig:
    if not isinstance(data, dict):
        raise ConfigError("config: expected a mapping")
    unknown = set(data) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ConfigError(f"config: unknown keys {sorted(unknown)}")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ConfigError(f"config: missing required keys {missing}")
    d = dict(data)
    for key in _STRS:
        if key in d and (not isinstance(d[key], str) or not d[key]):
            raise ConfigError(f"{key}: must be a non-empty string")
    if "seed" in d and not _is_int(d["seed"]):
        raise ConfigError("seed: must be an integer")
    for key in _POS_INTS:
        if key in d and (not _is_int(d[key]) or d[key] < 1):
            raise ConfigError(f"{key}: must be an integer >= 1")
    if "channels" in d:
        v = d["channels"]
        if not isinstance(v, list | tuple) or not v or not all(_is_int(c) and c >= 1 for c in v):
            raise ConfigError("channels: must be a non-empty list of integers >= 1")
        d["channels"] = tuple(v)
    numbers = ("lr", "iou_threshold", "mask_weight", "beta", "weight_decay", "dropout")
    for key in (*numbers, "divergence_ratio"):
        if key in d:
            if not _is_num(d[key]) or not math.isfinite(d[key]):
                raise ConfigError(f"{key}: must be a finite number")
            d[key] = float(d[key])
    if "shift_px" in d and not _is_int(d["shift_px"]):
        raise ConfigError("shift_px: must be an integer")
    if "variational" in d and not isinstance(d["variational"], bool):
        raise ConfigError("variational: must be true or false")
    if "abort_on_divergence" in d and not isinstance(d["abort_on_divergence"], bool):
        raise ConfigError("abort_on_divergence: must be true or false")
    if "early_stop" in d:
        d["early_stop"] = _early_stop(d["early_stop"])
    if "eval_splits" in d:
        v = d["eval_splits"]
        if not isinstance(v, list | tuple) or not v or not all(isinstance(s, str) for s in v):
            raise ConfigError("eval_splits: must be a non-empty list of split names")
        d["eval_splits"] = tuple(v)
    k = d.get("context_pca_k")
    if k is not None and (not _is_int(k) or k < 1):
        raise ConfigError("context_pca_k: must be an integer >= 1 or null")
    cfg = TrainConfig(**d)
    _check(cfg)
    return cfg


def _early_stop(v: Any) -> EarlyStop | None:
    """The ``early_stop`` block: null, or ``{patience, min_epochs, min_delta}`` (spec 012)."""
    if v is None or isinstance(v, EarlyStop):
        return v
    if not isinstance(v, dict):
        raise ConfigError("early_stop: must be a mapping {patience, min_epochs, min_delta} or null")
    unknown = set(v) - set(EARLY_STOP_KEYS)
    if unknown:
        raise ConfigError(f"early_stop: unknown keys {sorted(unknown)}")
    if "patience" not in v:
        raise ConfigError("early_stop.patience: required")
    patience, min_epochs, min_delta = v["patience"], v.get("min_epochs", 0), v.get("min_delta", 0.0)
    if not _is_int(patience) or patience < 1:
        raise ConfigError("early_stop.patience: must be an integer >= 1")
    if not _is_int(min_epochs) or min_epochs < 0:
        raise ConfigError("early_stop.min_epochs: must be an integer >= 0")
    if not _is_num(min_delta) or not math.isfinite(min_delta) or min_delta < 0:
        raise ConfigError("early_stop.min_delta: must be a finite number >= 0")
    return EarlyStop(patience, min_epochs, float(min_delta))


def _check(cfg: TrainConfig) -> None:
    if not RUN_NAME_RE.match(cfg.run_name):
        raise ConfigError(f"run_name: must match {RUN_NAME_RE.pattern}")
    if cfg.lr <= 0:
        raise ConfigError("lr: must be > 0")
    if cfg.lr_schedule not in LR_SCHEDULES:
        raise ConfigError(f"lr_schedule: one of {list(LR_SCHEDULES)}, got {cfg.lr_schedule!r}")
    allowed = [s for s in SPLITS if s != "train"]
    bad = [s for s in cfg.eval_splits if s not in allowed]
    if bad or len(set(cfg.eval_splits)) != len(cfg.eval_splits):
        raise ConfigError(f"eval_splits: unique names from {allowed}, got {list(cfg.eval_splits)}")
    if cfg.select_weights not in WEIGHT_SELECTIONS:
        raise ConfigError(
            f"select_weights: one of {list(WEIGHT_SELECTIONS)}, got {cfg.select_weights!r}"
        )
    if cfg.figure_split not in cfg.eval_splits:
        raise ConfigError(f"figure_split {cfg.figure_split!r} must be one of eval_splits")
    if not 0.0 < cfg.iou_threshold < 1.0:
        raise ConfigError("iou_threshold: must be in (0, 1)")
    for key in ("mask_weight", "beta", "weight_decay"):
        if getattr(cfg, key) < 0:
            raise ConfigError(f"{key}: must be >= 0")
    if cfg.variational and cfg.beta == 0:
        raise ConfigError("beta: must be > 0 with variational: true")
    if not cfg.variational and cfg.beta != 0:
        raise ConfigError("beta: needs variational: true")
    if not 0 <= cfg.shift_px <= MAX_SHIFT_PX:
        raise ConfigError(f"shift_px: must be in [0, {MAX_SHIFT_PX}]")
    if not 0.0 <= cfg.dropout < 1.0:
        raise ConfigError("dropout: must be in [0, 1)")
    if not cfg.divergence_ratio > 1.0:
        raise ConfigError("divergence_ratio: must be > 1")
    if not cfg.abort_on_divergence and cfg.divergence_ratio != DIVERGENCE_RATIO:
        raise ConfigError("divergence_ratio: needs abort_on_divergence: true")


def _read_yaml(path: str | Path) -> Any:
    try:
        return yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e


def load_config(path: str | Path) -> TrainConfig:
    return config_from_dict(_read_yaml(path))


def with_legacy_values(data: Any) -> Any:
    """``data`` with each :data:`LEGACY_VALUES` key it lacks filled in (present keys win;
    anything but a mapping is returned unchanged)."""
    return {**LEGACY_VALUES, **data} if isinstance(data, dict) else data


def load_run_config(path: str | Path) -> TrainConfig:
    """Config of an existing run directory (``--resume``): keys added after the run
    started take the value it was started under (:data:`LEGACY_VALUES`)."""
    return config_from_dict(with_legacy_values(_read_yaml(path)))
