# Data Model: 018

## `src/strike_a_pose/models/pose_head.py` (new)

- Constants: `POSE_TARGET = "joints_world"`, `POSE_JOINTS = 24`, `POSE_DIM = 72`,
  `POSE_HEAD_HIDDEN = (256, 256)`, `POSE_STD_FLOOR_M = 1e-6`.
- `root_relative(j: Tensor[n, 24, 3]) -> Tensor[n, 72]` = `(j - j[:, 0:1]).reshape(n, 72)`
  (dtype kept; same as `eval.probe._rel`).
- `class PoseHead(nn.Module)`: `__init__(latent_dim: int, hidden: tuple[int, ...] =
  POSE_HEAD_HIDDEN, d_out: int = POSE_DIM)`; `Linear(latent_dim, 256), ReLU, Linear(256, 256),
  ReLU, Linear(256, 72)` as `self.net`; `forward(z) -> [n, 72]`; `hparams() -> {"latent_dim",
  "hidden": list, "d_out"}`; `parameter_count() -> int` (64-d latent: 16,640 + 65,792 + 18,504 =
  100,936).
- `@dataclass(frozen=True) class PoseTargetStats`: `mean: Tensor[72]`, `scale: Tensor[72]`,
  `active: BoolTensor[72]` (float32 / bool).
  - `fit(y: Tensor[n, 72]) -> PoseTargetStats` (classmethod): float64 `mean`, population sd
    (`unbiased=False`); `active = sd > POSE_STD_FLOOR_M`; `scale = where(active, sd, 1.0)`; cast
    to float32. `ValueError("pose stats: empty")` on n = 0.
  - `standardize(y) -> (y - mean) / scale`; `unstandardize(s) -> s * scale + mean`, and for
    inactive coordinates returns exactly `mean`.
  - `to_dict() -> {"mean": Tensor, "scale": Tensor, "active": Tensor}`; `from_dict(d)`.
- `pose_loss(pred: Tensor[n, 72], target: Tensor[n, 72], active: BoolTensor[72]) -> Tensor` =
  `((pred - target)[:, active] ** 2).mean()` (both standardized).

## Training (`train/config.py`, `train/objective.py`, `train/trainer.py`)

- `TrainConfig.pose_weight: float = 0.0`; `LEGACY_VALUES["pose_weight"] = 0.0`; `_check`:
  `< 0` → `ConfigError("pose_weight: must be >= 0")`; `> 0 and variational` →
  `ConfigError("pose_weight: not with variational: true")`.
- `Objective.pose_weight: float = 0.0` (5th field; `from_config` passes it).
- `HISTORY_FIELDS` += `"train_pose"`; `_complete_row` adds `train_pose: None` when missing.
- `load_split(ds, batch_size, masks=False, joints=False)`: with `joints`, `a["joints_world"]`
  float32 `[n, 24, 3]` from `b["joints_3d_world"]`.
- `train_epoch(..., objective=None, pose: PoseBatch | None = None)` where
  `PoseBatch = NamedTuple(head: PoseHead, targets: Tensor[n, 72] (standardized, float32),
  active: BoolTensor[72])`; returns also `"train_pose"` (epoch mean of `pose_loss`, None if off).
- Checkpoint keys (both files): `pose_head` (state dict) , `pose_head_hparams`, `pose_stats`
  (`to_dict()`); plain runs write none of these keys.

## Metrics

- `metrics.json`: `pose_head: {"target": "joints_world", "hidden": [256, 256], "parameters":
  100936, "active_coordinates": int, "pose_weight": float}` (absent when off);
  `splits.<s>.pose_head: {"target": "joints_world", "mpjpe_mm": float}`;
  `units.pose_head_mpjpe_mm`: "mm, root-relative world-frame joint error of the jointly trained
  pose head (not a probe; spec 018)".

## Pose track (`src/strike_a_pose/eval/pose_track.py`, new)

- Constants `REAL_GAIN_SD = 2.0`, `POSE_TARGET_MM = 80.0`.
- `@dataclass(frozen=True) class PoseEntry`: `name: str`, `probe_mm: float`,
  `head_mm: float | None`, `mse: float`.
- `@dataclass(frozen=True) class PoseRank`: `name`, `probe_mm`, `head_mm`, `d_mse_rel`,
  `d_mm`, `z`, `real: bool`, `probe_ok: bool`, `head_ok: bool | None`.
- `seed_stats(values: list[float]) -> tuple[float, float]` (mean, sd ddof 1; `ValueError` if
  < 2 values or sd == 0).
- `rank(entries: list[PoseEntry], seeds: list[PoseEntry]) -> list[PoseRank]`: sorted by
  `probe_mm` ascending (ties by name); `d_mse_rel = mse / mean(seed mse) - 1`; `d_mm = probe_mm
  - mean`; `z = d_mm / sd`; `real = d_mm < -REAL_GAIN_SD * sd`; `probe_ok = probe_mm <=
  POSE_TARGET_MM`; `head_ok = None if head_mm is None else head_mm <= POSE_TARGET_MM`.
