# Data Model: 008 Best-Val Weight Selection

Normative contract: [`../004-autoencoder-mvp/contracts/train-run.md`](../004-autoencoder-mvp/contracts/train-run.md)
(Train v2). Probe side: [`../006-latent-probe/contracts/probe-run.md`](../006-latent-probe/contracts/probe-run.md)
(`checkpoint` row).

## Entities

**`select_weights`** (train config key, str): `best_val` (default) | `final`. Written into
the run's `config.yaml`. Legacy value for run directories whose `config.yaml` lacks it
(started before 008, read on `--resume` only): `final`.

**`LEGACY_VALUES`** (`train.config`): `{"select_weights": "final"}`; one entry per config
key added after runs existed.

**Selection** (`metrics.json` top-level `selection`, `train_version` 2):

| Key | Value |
|---|---|
| `select_weights` | the configured rule (`best_val` \| `final`) |
| `weights` | the applied weights (`best_val` \| `final`) |
| `selected_by` | `"val loss"` \| `"final (empty val)"` \| `"final (no finite val loss)"` \| `"config"` |
| `epoch` | epoch of the scored weights (1-based) |
| `val_loss` | val loss of that epoch; null when val is empty or has no finite loss |
| `checkpoint` | file holding the scored weights: `"best.pt"` \| `"checkpoint.pt"` |

| Rule | Val | `weights` | `selected_by` | `epoch` | `checkpoint` |
|---|---|---|---|---|---|
| `best_val` | has a finite loss | `best_val` | `val loss` | earliest minimum | `best.pt` |
| `best_val` | empty | `final` | `final (empty val)` | last | `checkpoint.pt` |
| `best_val` | samples, no finite loss | `final` | `final (no finite val loss)` | last | `checkpoint.pt` |
| `final` | any | `final` | `config` | last | `checkpoint.pt` |

**`best.pt`** (structure unchanged): `{"train_version": 2, "epoch", "val_loss",
"model_hparams", "model", "config"}`: weights of the earliest minimum-finite-val-loss
epoch so far; written after that epoch's `checkpoint.pt` whenever val has samples (under
`final` too); absent when val is empty or never finite. Scored file under `best_val`.

**`checkpoint.pt`** (structure unchanged, `train_version` 2): full resume state of the
last finished epoch; the scored file only when `final` applies.

## Resume: `best.pt` reconciliation (after the config / dataset / split checks)

| History best epoch `b` (checkpoint at `N`) | `best.pt` | Action |
|---|---|---|
| none (val empty or never finite) | absent | none |
| `b` | holds `b` (epoch, val loss, config after the legacy fill) | none |
| `b = N` | missing or another epoch / val loss | rewrite from the checkpoint's weights |
| `b < N`, rule `best_val` | missing or another epoch | refuse, exit 2 (weights of `b` lost) |
| `b < N`, rule `final` | missing or another epoch | none (informational file) |

## Evaluation

`weight_selection(history, rule)` → selection; under `best_val` the model's weights are
replaced by `best.pt`'s after the same match check (mismatch → exit 2, no
`metrics.json`); the 004 evaluation (`ae` entries, figure) then scores the model as
before. `training.best_val_epoch` / `best_val_loss` use the same finite-minimum rule.
