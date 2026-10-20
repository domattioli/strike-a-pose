# Contract: Train v1 (model API, train config, run directory)

Builds on the 003 eval contract (metrics, PCA, run-dir conventions).
Breaking changes bump `train_version` in `metrics.json`.

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
from `checkpoint.pt`.

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
| `context_pca_k` | int > 0 \| null | `64` | PCA baseline scored next to the AE (null: none) |
| `iou_threshold` | float in (0, 1) | `0.075` | silhouette threshold (003) |
| `figure_split` | str | `val` | split shown (must be in `eval_splits`) |
| `figure_samples` | int > 0 | `6` | figure rows (evenly spaced samples) |

The per-epoch val loss is computed when `val` has samples, whether or not `val`
is in `eval_splits`.

## CLI

```text
strike-a-pose-train --config PATH [--seed INT] [--run-root DIR]
```

Exit codes: 0 OK (prints run dir, final losses, headline metrics); 1 dataset or split
invalid; 2 config error (incl. image size not divisible by `2^len(channels)`);
4 run dir exists. Inputs and config are checked before the run dir is created.

## Run directory `<run_root>/<run_name>-<YYYYmmddTHHMMSSZ>/`

| File | Content |
|---|---|
| `config.yaml` | resolved config (overrides applied) |
| `run_info.json` | 003 fields (`argv`, `cwd`, `seed`, `git`, `env`, `dataset`, `split_file`, `started_utc`, `finished_utc`, `wall_s`) + `model_parameters`, `train_wall_s`, `figure` |
| `history.csv` | header `epoch,train_loss,val_loss,lr,epoch_s`; one row per epoch (1-based); `val_loss` empty when val is empty; `lr` = rate at the epoch's last step |
| `checkpoint.pt` | `{"model": state_dict, "model_hparams": {image_size, latent_dim, channels}, "optimizer": state_dict, "epoch": int, "config": dict, "train_version": 1}` (never committed) |
| `metrics.json` | below |
| `recon_<figure_split>.png` | 003 grid: columns `true`, `ae`, `mean_image`, `pca k=K` |

### `metrics.json` (`train_version` 1; image metrics as 003 `metrics_version` 1)

```json
{
  "train_version": 1,
  "metrics_version": 1,
  "dataset": {"name": "smpl_proc_v1", "manifest_sha256": "…", "split_file_sha256": "…"},
  "model": {"name": "ConvAutoencoder", "latent_dim": 64, "channels": [16, 32, 64, 128, 256, 256], "parameters": 3126897},
  "training": {"epochs": 60, "steps": 12000, "train_samples": 12800,
               "first_train_loss": 0.0, "final_train_loss": 0.0,
               "first_val_loss": 0.0, "final_val_loss": 0.0},
  "context": {"fit_split": "train", "pca_k": 64, "pca_k_effective": 64},
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

`first_*` = epoch 1 values. Empty splits: `{"samples": 0, "poses": 0}`; null where
no val. Floats: 6 significant digits, sorted keys (003 writer). Deterministic for
the same inputs, code and machine/thread count.
