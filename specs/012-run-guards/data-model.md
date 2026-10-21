# Data Model: Run Guards and Campaign Procedure

## Train config (additions)

| Key | Type | Default | Legacy value |
|---|---|---|---|
| `early_stop` | `null` \| `{patience: int ≥ 1, min_epochs: int ≥ 0 = 0, min_delta: float ≥ 0 = 0.0}` | `null` | `null` |
| `abort_on_divergence` | bool | `false` | `false` |
| `divergence_ratio` | float > 1 (≠ 10.0 only with the guard) | `10.0` | `10.0` |

Campaign value: `early_stop: {patience: 10, min_epochs: 20, min_delta: 0.0}`,
`abort_on_divergence: true`, `divergence_ratio: 10.0`.

## Early-stop state (`train.guards.early_stop_state`)

`{"anchor_epoch": int | null, "anchor_loss": float | null, "since": int | null,
"earliest_stop": int | null, "fired": null | {"epoch": int, "reason": str}}`; `since` = last
walked epoch − anchor epoch; `earliest_stop` = `max(anchor_epoch + patience, min_epochs)`.

## `metrics.json` (`train_version` 2, additive)

`training.stopped_early`: present iff `early_stop` is configured; `null` or
`{"epoch": e, "reason": "no val loss below 0.00216847 (epoch 34) by more than 0 for 10 epochs
(patience 10, min_epochs 20)"}`.

## `aborted.json` (new; instead of `metrics.json`)

```json
{"divergence_ratio": 10.0, "epoch": 7, "first_train_loss": 0.0352,
 "last_losses": [{"epoch": 5, "train_loss": 0.004, "val_loss": 0.003},
                 {"epoch": 6, "train_loss": 0.004, "val_loss": 0.003},
                 {"epoch": 7, "train_loss": "nan", "val_loss": "nan"}],
 "reason": "train loss nan at epoch 7 is not finite"}
```

## `lock` (new, transient)

`{"argv": [...], "boot_id": "…" | null, "pid": int, "started_utc": "…"}`; present while a
process owns the run dir; removed on exit; stale after a kill.

## `run_info.json` (003 / 004 / 006, additive)

`env.cpu_model` (str | null), `dataset.checksums_sha256` (str | null).

## Status script (`scripts/run_status.py --json`)

`{"run_dir", "run_name", "state", "epochs_done", "epochs_planned", "last_epoch_s",
"median_epoch_s", "minutes_since_activity", "stale_after_min", "best_val_epoch", "best_val_loss",
"last_val_loss", "val_vs_best", "early_stop", "divergence", "train_kl_epoch10",
"train_kl_last", "lock", "selection", "val_image_mse", "val_region_mse_mean", "aborted"}`;
`state` ∈ `running`, `stalled`, `looks dead`, `finished`, `stopped early`, `aborted`,
`no epoch yet`.
