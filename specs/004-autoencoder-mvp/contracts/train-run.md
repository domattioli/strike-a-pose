# Contract: Train v2 (model API, train config, run directory)

Builds on the 003 eval contract (metrics, PCA, run-dir conventions).
Breaking changes bump `train_version` in `metrics.json`. v2 (slice 008
`best-val-selection`, 2026-10-20): the scored weights are chosen by `select_weights`
(default: the epoch with the lowest val loss, `best.pt`) and named in `metrics.json`
`selection`; v1 runs always scored the final weights.

## Model (`strike_a_pose.models.ConvAutoencoder`)

```python
ConvAutoencoder(image_size=(256, 192), latent_dim=64,
                channels=(16, 32, 64, 128, 256, 256))
model.encode(x: Tensor[B, 1, H, W]) -> Tensor[B, latent_dim]
model.decode(z: Tensor[B, latent_dim]) -> Tensor[B, 1, H, W]   # in [0, 1] (sigmoid)
model(x) -> decode(encode(x))
```

Encoder: per channel width `c`: `Conv2d(k3, s2, p1) → BatchNorm2d → LeakyReLU(0.2)`;
flatten → `Linear(c_last · H/2^n · W/2^n, latent_dim)`. Decoder: `Linear → reshape
[c_last, H/2^n, W/2^n]` → mirrored `ConvTranspose2d(k4, s2, p1) → BatchNorm2d →
LeakyReLU(0.2)` blocks → final `ConvTranspose2d(k4, s2, p1)` to 1 channel → sigmoid.
`H` and `W` must be divisible by `2^n` (`n = len(channels)`), else `ValueError`.
`load_checkpoint(path) -> (ConvAutoencoder, dict)` rebuilds the model in eval mode
from `checkpoint.pt` or `best.pt`.

## Train config (YAML, `configs/ae*.yaml`)

Flat mapping; unknown keys are an error.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `run_name` | str | `ae` | run dir prefix (`[A-Za-z0-9_.-]+`) |
| `run_root` | str | `runs` | parent of run dirs (git-ignored) |
| `seed` | int | `0` | Python / NumPy / torch seed + shuffle generator |
| `dataset_dir` | str | required | 001 dataset directory |
| `split_file` | str | required | frozen 002 split file |
| `latent_dim` | int > 0 | `64` | latent size |
| `channels` | list[int > 0] | `[16, 32, 64, 128, 256, 256]` | encoder widths (decoder mirrors) |
| `epochs` | int > 0 | `60` | passes over train |
| `batch_size` | int > 0 | `64` | train batch; also eval batch |
| `lr` | float > 0 | `0.001` | Adam learning rate |
| `lr_schedule` | `constant` \| `cosine` | `cosine` | cosine: per-step decay from `lr` to 0 |
| `eval_splits` | list[str] | `[val, test]` | splits scored after training (unique, not `train`) |
| `select_weights` | `best_val` \| `final` | `best_val` | scored weights (slice 008): `best_val` = the earliest epoch with the lowest finite val loss (`best.pt`), falling back to the final epoch when val is empty or has no finite loss; `final` = the last epoch |
| `context_pca_k` | int > 0 \| null | `64` | PCA baseline scored next to the AE (null: none) |
| `iou_threshold` | float in (0, 1) | `0.075` | silhouette threshold (003) |
| `figure_split` | str | `val` | split shown (must be in `eval_splits`) |
| `figure_samples` | int > 0 | `6` | figure rows (evenly spaced samples) |

The per-epoch val loss is computed when `val` has samples, whether or not `val`
is in `eval_splits`; it never changes training (epochs, lr, optimizer) and, under
`best_val`, selects the scored weights.

## CLI

```text
strike-a-pose-train --config PATH [--seed INT] [--run-root DIR]
strike-a-pose-train --resume RUN_DIR
```

`--resume` continues an interrupted run from its `checkpoint.pt` with the run dir's
own `config.yaml` (no overrides); the finished run equals an uninterrupted one on
the same machine/thread count. Exit 2: finished run (`metrics.json` exists), no
checkpoint, or checkpoint/config mismatch; exit 1: dataset manifest or split hash
differs from `run_info.json`. The manifest check uses the raw manifest hash
(`dataset.manifest_sha256`, SHA-256 of the `manifest.json` bytes, which include the
generating `code` commit), not the content hash the 002 split file uses: a dataset
regenerated at another commit does not resume, by design, because the manifest does
not hash the images (slice 007, 2026-10-20).

Config keys added after a run started are read, for that run dir only, with the value
the run was started under (`train.config.LEGACY_VALUES`; slice 008: `select_weights` →
`final`), on both sides of the checkpoint/config comparison: a run dir from before slice
008 resumes and scores its final weights. Before training, `best.pt` is reconciled with
the checkpointed history (slice 008): if the history's best epoch is the checkpoint's
own and `best.pt` is missing or holds another epoch (a kill between the `checkpoint.pt`
and `best.pt` writes), it is rewritten from the checkpoint; if the best epoch is older
and `best.pt` does not hold it, a `best_val` run exits 2 (those weights are lost; no
silent fallback to the final weights) and a `final` run continues.

Exit codes: 0 OK (prints run dir, per-epoch losses, the selection line — e.g. `scoring
best_val weights: epoch 34, val 0.00217 (val loss; best.pt); final epoch 60 val
0.00221` — and headline metrics); 1 dataset or split invalid; 2 config error (incl.
image size not divisible by `2^len(channels)`, a `best.pt` that does not hold the
selected epoch); 4 run dir exists. Inputs and config are checked before the run dir is
created.

## Run directory `<run_root>/<run_name>-<YYYYmmddTHHMMSSZ>/`

| File | Content |
|---|---|
| `config.yaml` | resolved config (overrides applied) |
| `run_info.json` | 003 fields (`argv`, `cwd`, `seed`, `git`, `env`, `dataset` (incl. `manifest_content_sha256`, slice 007), `split_file`, `started_utc`, `finished_utc`, `wall_s`) + `model_parameters`, `train_wall_s`, `figure`, `sessions` (one per process: `started_utc`, `start_epoch`, `argv`, `git`); `wall_s`/`train_wall_s` sum the sessions |
| `history.csv` | header `epoch,train_loss,val_loss,lr,epoch_s`; one row per epoch (1-based); `val_loss` empty when val is empty; `lr` = rate at the epoch's last step |
| `checkpoint.pt` | `{"model": state_dict, "model_hparams": {image_size, latent_dim, channels}, "optimizer": state_dict, "scheduler": state_dict \| null, "rng": {"shuffle", "torch", "numpy", "python"}, "history": [row, …], "elapsed_s": float, "train_wall_s": float, "epoch": int, "config": dict, "train_version": 2}`; resume state of the last finished epoch, written atomically after every epoch, before that epoch's `best.pt`; the scored file only when the final weights are scored (`selection.checkpoint`); never committed |
| `best.pt` | `{"model", "model_hparams", "epoch", "val_loss", "config", "train_version": 2}` of the earliest epoch with the lowest finite val loss so far (NaN never counts), written atomically after that epoch's `checkpoint.pt` whenever val has samples (also under `select_weights: final`, where it is informational); absent when val is empty or never finite; loads with `load_checkpoint`; the scored file under `best_val` (slice 008; additive since surrogate ruling 2026-10-20 #3); never committed. On resume the best is recomputed from the checkpointed full-precision history and the file reconciled (CLI). |
| `metrics.json` | below |
| `recon_<figure_split>.png` | 003 grid: columns `true`, `ae` (the scored weights), `mean_image`, `pca k=K` |

### `metrics.json` (`train_version` 2; image metrics as 003 `metrics_version` 1)

```json
{
  "train_version": 2,
  "metrics_version": 1,
  "dataset": {"name": "smpl_proc_v1", "manifest_sha256": "…", "split_file_sha256": "…"},
  "model": {"name": "ConvAutoencoder", "latent_dim": 64, "channels": [16, 32, 64, 128, 256, 256], "parameters": 3126897},
  "training": {"epochs": 60, "steps": 12000, "train_samples": 12800,
               "first_train_loss": 0.0, "final_train_loss": 0.0,
               "first_val_loss": 0.0, "final_val_loss": 0.0},
  "context": {"fit_split": "train", "pca_k": 64, "pca_k_effective": 64},
  "selection": {"select_weights": "best_val", "weights": "best_val", "selected_by": "val loss",
                "epoch": 34, "val_loss": 0.0, "checkpoint": "best.pt"},
  "units": {"image_mse": "…", "psnr_db": "…", "silhouette_iou": "…"},
  "splits": {
    "val": {"samples": 1600, "poses": 200,
            "image": {"ae": {"image_mse": 0.0, "psnr_db": 0.0, "silhouette_iou": 0.0},
                      "mean_image": {"…": 0.0},
                      "pca_k64": {"k": 64, "k_effective": 64, "…": 0.0}}},
    "test": {"…": "…"}
  }
}
```

`selection` (slice 008) names the scored weights: `select_weights` (the configured
rule), `weights` (applied: `best_val` | `final`), `selected_by` (`"val loss"` |
`"final (empty val)"` | `"final (no finite val loss)"` | `"config"`), `epoch`, `val_loss`
(that epoch's val loss; null when val is empty or has no finite loss) and `checkpoint`
(`"best.pt"` | `"checkpoint.pt"`, the file holding them). Every `splits.*.image.ae`
entry and the figure's `ae` column come from those weights; test is scored once, for
them. Only the selected weights are scored, so their gain over the final weights is
known on val only, the split that selected them (`selection.val_loss` vs
`training.final_val_loss`): compare best-val numbers with final-weight numbers (e.g. the
published 004 run, `train_version` 1) with that caveat. A `train_version` 1 file has no
`selection` and scored the final weights.

`training` also holds `best_val_epoch` / `best_val_loss` (the same finite-minimum rule;
null without a finite val loss; additive, surrogate ruling 2026-10-20 #3; `run_info.json`
gains `best_val_epoch` and `best_checkpoint`). `first_*` = epoch 1 values. Empty
splits: `{"samples": 0, "poses": 0}`; null where no val. Floats: 6 significant digits,
NaN → null, sorted keys (003 writer). Deterministic for the same inputs, code and
machine/thread count.
