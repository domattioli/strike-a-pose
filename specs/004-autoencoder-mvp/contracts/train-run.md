# Contract: Train v2 (model API, train config, run directory)

Builds on the 003 eval contract (metrics, PCA, run-dir conventions).
Breaking changes bump `train_version` in `metrics.json`. v2 (slice 008
`best-val-selection`, 2026-10-20): the scored weights are chosen by `select_weights`
(default: the epoch with the lowest val loss, `best.pt`) and named in `metrics.json`
`selection`; v1 runs always scored the final weights. Slice 011 `training-variants`
(2026-10-21) adds optional training variants (config keys below; their defaults are the plain
AE, bit for bit) with additive records; `train_version` stays 2. Slice 012 `run-guards`
(2026-10-21) adds opt-in run guards (early stop, divergence abort), a run-dir lock and two
`run_info.json` provenance fields; additive, `train_version` stays 2.

## Model (`strike_a_pose.models.ConvAutoencoder`)

```python
ConvAutoencoder(image_size=(256, 192), latent_dim=64,
                channels=(16, 32, 64, 128, 256, 256), variational=False)
model.encode(x: Tensor[B, 1, H, W]) -> Tensor[B, latent_dim]   # mu for a VAE
model.encode_dist(x) -> (mu, logvar | None)                    # slice 011
model.decode(z: Tensor[B, latent_dim]) -> Tensor[B, 1, H, W]   # in [0, 1] (sigmoid)
model(x) -> decode(encode(x))                                  # deterministic
```

Encoder: per channel width `c`: `Conv2d(k3, s2, p1) → BatchNorm2d → LeakyReLU(0.2)`;
flatten → `Linear(c_last · H/2^n · W/2^n, latent_dim)`. Decoder: `Linear → reshape
[c_last, H/2^n, W/2^n]` → mirrored `ConvTranspose2d(k4, s2, p1) → BatchNorm2d →
LeakyReLU(0.2)` blocks → final `ConvTranspose2d(k4, s2, p1)` to 1 channel → sigmoid.
`H` and `W` must be divisible by `2^n` (`n = len(channels)`), else `ValueError`.
`load_checkpoint(path) -> (ConvAutoencoder, dict)` rebuilds the model in eval mode
from `checkpoint.pt` or `best.pt`.

VAE (slice 011, `variational=True`): a `logvar = Linear(flat, latent_dim)` head on the
flattened bottleneck (the input of the encoder's last `Linear`), created after encoder and
decoder, so every other parameter initializes as for the plain AE at the same seed; state
keys `logvar.weight`, `logvar.bias`; 3,323,569 parameters at the defaults (plain 3,126,897).
`hparams()` carries `variational`; a checkpoint without it loads as the plain AE. Code from
before slice 011 cannot load a VAE checkpoint.

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
| `lr` | float > 0 | `0.001` | learning rate (AdamW since slice 011, equal to Adam at `weight_decay: 0`) |
| `lr_schedule` | `constant` \| `cosine` | `cosine` | cosine: per-step decay from `lr` to 0 |
| `eval_splits` | list[str] | `[val, test]` | splits scored after training (unique, not `train`) |
| `select_weights` | `best_val` \| `final` | `best_val` | scored weights (slice 008): `best_val` = the earliest epoch with the lowest finite val loss (`best.pt`), falling back to the final epoch when val is empty or has no finite loss; `final` = the last epoch |
| `context_pca_k` | int > 0 \| null | `64` | PCA baseline scored next to the AE (null: none) |
| `iou_threshold` | float in (0, 1) | `0.075` | silhouette threshold (003) |
| `figure_split` | str | `val` | split shown (must be in `eval_splits`) |
| `figure_samples` | int > 0 | `6` | figure rows (evenly spaced samples) |
| `mask_weight` | float ≥ 0 | `0.0` | slice 011: α of the mask-weighted MSE (objective below); 0 = plain MSE |
| `variational` | bool | `false` | slice 011: VAE (`logvar` head, reparameterized training, KL term) |
| `beta` | float ≥ 0 | `0.0` | slice 011: KL weight; > 0 iff `variational` |
| `weight_decay` | float ≥ 0 | `0.0` | slice 011: AdamW decoupled weight decay (every parameter, one group) |
| `shift_px` | int in [0, 32] | `0` | slice 011: each train image shifted by its own integer offset in `[-k, k]²`, zero fill |
| `dropout` | float in [0, 1) | `0.0` | slice 011: dropout on the latent in training (after sampling) |
| `early_stop` | null \| `{patience: int ≥ 1, min_epochs: int ≥ 0 (0), min_delta: float ≥ 0 (0.0)}` | null | slice 012: stop on a val plateau (below) |
| `abort_on_divergence` | bool | `false` | slice 012: abort a diverging run (below; exit 5) |
| `divergence_ratio` | float > 1 | `10.0` | slice 012: train-loss ratio to epoch 1 that counts as divergence; another value needs `abort_on_divergence: true` |

The per-epoch val loss is computed when `val` has samples, whether or not `val`
is in `eval_splits`; it never changes training (epochs, lr, optimizer) and, under
`best_val`, selects the scored weights.

The slice 011 keys default to the pre-011 computation and carry that value in
`train.config.LEGACY_VALUES`, so run directories started before them resume; so do the slice
012 keys (guards off).

### Run guards (slice 012)

Both read only the per-epoch history (full precision, as checkpointed), after the epoch's
history row, `checkpoint.pt`, `best.pt` and log line, and on `--resume` from the checkpointed
history before training, so a resumed run decides as an uninterrupted one. Neither touches
the optimizer, the lr schedule (cosine over the configured `epochs`) or any RNG.

- **Early stop** (`early_stop`): an epoch with a finite val loss improves iff there is no anchor
  yet or its val loss is below the anchor by more than `min_delta` (the anchor moves to it).
  After an epoch e with a finite val loss the run stops iff `e ≥ min_epochs` and `e − anchor
  epoch ≥ patience`; NaN epochs count as epochs without improvement; an empty val never stops
  (log `early stop inactive: val is empty`). The run is then scored as usual: the selection
  (`select_weights`) and the evaluation are unchanged, and a stopped run is a prefix of the full
  run. Log `early stop after epoch e/E: <reason>`.
- **Divergence** (`abort_on_divergence`), first match at the last epoch e: its train loss is not
  finite; `e > 3` and its train loss `> divergence_ratio ×` epoch 1's; the val losses of
  epochs `e − 2 … e` are all non-finite (an empty val never counts). Checked before the early
  stop. The run writes `aborted.json`, no `metrics.json` or figure, logs `aborted after epoch e:
  <reason>` and exits 5.

### Training objective (slice 011)

Per train batch: `x` in `[0, 1]` (after the shift), `z = mu` (plain) or
`mu + exp(logvar / 2)·eps` (VAE), then dropout, `y = decode(z)`, `m = [x > 0]` (the
silhouette), `P = H·W`. `recon = mse_loss(y, x)` when `mask_weight` is 0, else
`sum(w·(y − x)²) / sum(w)` with `w = 1 + mask_weight·m` (sums over the batch). VAE:
`KL = mean_b ½·sum_j (mu² + exp(logvar) − 1 − logvar)` (nats per image),
`loss = recon + beta·KL / P`; otherwise `loss = recon`. Shift offsets, `eps` and dropout
masks come from the global torch RNG (saved per epoch, so a resume stays exact), each only
when its knob is on. The val loss, the selection, the metrics and the figure use the
eval-mode `model(x) = decode(mu)` with the plain MSE, whatever the variant.

## CLI

```text
strike-a-pose-train --config PATH [--seed INT] [--run-root DIR]
strike-a-pose-train --resume RUN_DIR
```

`--resume` continues an interrupted run from its `checkpoint.pt` with the run dir's
own `config.yaml` (no overrides); the finished run equals an uninterrupted one on
the same machine/thread count. Exit 2: finished run (`metrics.json` exists), no
checkpoint, or checkpoint/config mismatch; exit 1: dataset manifest or split hash
differs from `run_info.json`; exit 5: the run was aborted by the divergence guard
(`aborted.json`, slice 012; never resumable: start a new run). Both a new run and `--resume`
hold `run_dir/lock` while they run (slice 012) and exit 2 when it names a live
trainer process of this boot (command line containing `strike-a-pose-train` or
`strike_a_pose.train`, the `python -m` form: amended 2026-10-21 by slice 014); a lock left by a killed process is stale and
replaced. The manifest check uses the raw manifest hash
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
selected epoch, a run dir locked by a live process); 4 run dir exists; 5 aborted by the
divergence guard, or `--resume` of such a run (slice 012). Inputs and config are checked before
the run dir is created.

## Run directory `<run_root>/<run_name>-<YYYYmmddTHHMMSSZ>/`

| File | Content |
|---|---|
| `config.yaml` | resolved config (overrides applied) |
| `run_info.json` | 003 fields (`argv`, `cwd`, `seed`, `git`, `env`, `dataset` (incl. `manifest_content_sha256`, slice 007), `split_file`, `started_utc`, `finished_utc`, `wall_s`) + `model_parameters`, `train_wall_s`, `figure`, `sessions` (one per process: `started_utc`, `start_epoch`, `argv`, `git`); `wall_s`/`train_wall_s` sum the sessions |
| `history.csv` | header `epoch,train_loss,val_loss,lr,epoch_s,train_recon,train_kl` (the last two since slice 011; runs before it have the first five); one row per epoch (1-based); `train_loss` = epoch mean of the objective, `train_recon` = of its reconstruction term (= `train_loss` unless `variational`), `train_kl` = mean KL in nats per image (empty unless `variational`); `val_loss` empty when val is empty; `lr` = rate at the epoch's last step |
| `checkpoint.pt` | `{"model": state_dict, "model_hparams": {image_size, latent_dim, channels, variational}, "optimizer": state_dict, "scheduler": state_dict \| null, "rng": {"shuffle", "torch", "numpy", "python"}, "history": [row, …], "elapsed_s": float, "train_wall_s": float, "epoch": int, "config": dict, "train_version": 2}`; resume state of the last finished epoch, written atomically after every epoch, before that epoch's `best.pt`; the scored file only when the final weights are scored (`selection.checkpoint`); never committed |
| `best.pt` | `{"model", "model_hparams", "epoch", "val_loss", "config", "train_version": 2}` of the earliest epoch with the lowest finite val loss so far (NaN never counts), written atomically after that epoch's `checkpoint.pt` whenever val has samples (also under `select_weights: final`, where it is informational); absent when val is empty or never finite; loads with `load_checkpoint`; the scored file under `best_val` (slice 008; additive since surrogate ruling 2026-10-20 #3); never committed. On resume the best is recomputed from the checkpointed full-precision history and the file reconciled (CLI). |
| `metrics.json` | below |
| `aborted.json` | slice 012, instead of `metrics.json` when the divergence guard ends the run: `{"epoch", "reason", "divergence_ratio", "first_train_loss", "last_losses": [{"epoch", "train_loss", "val_loss"}, … up to 3]}`; non-finite losses as the strings `"nan"`, `"inf"`, `"-inf"` |
| `lock` | slice 012, transient: `{"pid", "boot_id", "argv", "started_utc"}` of the process running on the run dir; removed on exit, left stale by a kill |
| `recon_<figure_split>.png` | 003 grid: columns `true`, `ae` (the scored weights), `mean_image`, `pca k=K` |

### `metrics.json` (`train_version` 2; image metrics as 003 `metrics_version` 1)

```json
{
  "train_version": 2,
  "metrics_version": 1,
  "dataset": {"name": "smpl_proc_v1", "manifest_sha256": "…", "split_file_sha256": "…"},
  "model": {"name": "ConvAutoencoder", "latent_dim": 64, "channels": [16, 32, 64, 128, 256, 256], "parameters": 3126897, "variational": false},
  "training": {"epochs": 60, "steps": 12000, "train_samples": 12800,
               "first_train_loss": 0.0, "final_train_loss": 0.0,
               "first_val_loss": 0.0, "final_val_loss": 0.0,
               "objective": {"mask_weight": 0.0, "beta": 0.0, "weight_decay": 0.0, "shift_px": 0, "dropout": 0.0}},
  "context": {"fit_split": "train", "pca_k": 64, "pca_k_effective": 64},
  "selection": {"select_weights": "best_val", "weights": "best_val", "selected_by": "val loss",
                "epoch": 34, "val_loss": 0.0, "checkpoint": "best.pt"},
  "regions": {"feet": {"half_size_px": 8, "joints": ["left_ankle", "right_ankle", "left_foot", "right_foot"]},
              "hands": {"half_size_px": 10, "joints": ["left_hand", "right_hand"]},
              "head": {"half_size_px": 14, "joints": ["head"]}},
  "units": {"image_mse": "…", "psnr_db": "…", "silhouette_iou": "…", "regions": "…"},
  "splits": {
    "val": {"samples": 1600, "poses": 200,
            "image": {"ae": {"image_mse": 0.0, "psnr_db": 0.0, "silhouette_iou": 0.0,
                             "regions": {"feet": {"image_mse": 0.0, "samples": 1600},
                                         "hands": {"image_mse": 0.0, "samples": 1600},
                                         "head": {"image_mse": 0.0, "samples": 1600}}},
                      "mean_image": {"…": 0.0, "regions": {"…": {"image_mse": 0.0, "samples": 1600}}},
                      "pca_k64": {"k": 64, "k_effective": 64, "…": 0.0, "regions": {"…": "…"}}}},
    "test": {"…": "…"}
  }
}
```

Per-region image metrics (slice 009, 2026-10-21; additive, so `train_version` stays 2 and
`metrics_version` 1): every image-model entry (`ae`, `mean_image`, `pca_k<K>`) of a split
with samples carries `regions` (`hands`, `feet`, `head`: `image_mse`, `samples`), with the
top-level `regions` definitions and `units.regions`, exactly as in the 003 contract
(`score_images` with labels, the same code for every image model). They are scored in the
evaluation after the selected weights are loaded, from the same predictions as the headline
metrics, so the `ae` regions are those of the weights `selection` names. For this the scored
splits (val when it is in `eval_splits`, every other eval split) keep their 2D keypoints and
visibility in memory next to the masks (≈ 0.2 KB per sample); no config key, file, log line
or figure changes, and a run directory started before slice 009 resumes unchanged and
reports regions when it finishes.

Training variants (slice 011, 2026-10-21; additive): `model.variational` and
`training.objective` (`mask_weight`, `beta`, `weight_decay`, `shift_px`, `dropout`).
`first_train_loss` / `final_train_loss` are the configured objective and compare across runs
only at equal objective keys; every image metric is the plain one of `decode(mu)`.

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

`training.stopped_early` (slice 012) is present iff `early_stop` is configured: null (the run
trained all its epochs) or `{"epoch", "reason"}`; `training.epochs` / `steps` stay the configured
values, `final_*` are the last trained epoch's. `run_info.json` gains `env.cpu_model` and
`dataset.checksums_sha256` (SHA-256 of the dataset's `checksums.json`, i.e. the pixels; slice
012).

`training` also holds `best_val_epoch` / `best_val_loss` (the same finite-minimum rule;
null without a finite val loss; additive, surrogate ruling 2026-10-20 #3; `run_info.json`
gains `best_val_epoch` and `best_checkpoint`). `first_*` = epoch 1 values. Empty
splits: `{"samples": 0, "poses": 0}`; null where no val. Floats: 6 significant digits,
NaN → null, sorted keys (003 writer). Deterministic for the same inputs, code and
machine/thread count.
