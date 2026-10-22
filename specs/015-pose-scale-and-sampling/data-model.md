# Data model: slice 015

## SampleConfig (`eval/sample_config.py`, YAML)

| key | type | default | rule |
|---|---|---|---|
| dataset_dir | str | required | |
| split_file | str | required | |
| checkpoint | str | required | VAE checkpoint (`model_hparams.variational` true) |
| reference_checkpoint | str \| null | null | plain AE, same image size and latent_dim |
| run_name | str | `sample` | `RUN_NAME_RE` |
| run_root | str | `runs` | |
| seed | int | 0 | ≥ 0 |
| n_samples | int | 1024 | > 0 |
| n_recon | int | 1024 | ≥ 0, capped at the val size |
| batch_size | int | 256 | > 0 |
| iou_threshold | float | 0.075 | (0, 1) |
| figure_samples | int | 16 | 1..64 |

Unknown keys → `ConfigError`. `with_overrides(seed, run_root, checkpoint)` like ProbeConfig.

## samples.json

```
{"metrics_version": 1,
 "checkpoint": {"path", "sha256"}, "reference_checkpoint": {"path", "sha256"} | null,
 "dataset": {"name", "manifest_sha256", "split_file_sha256"},
 "criteria": {"iou_threshold", "largest_fraction": 0.95, "area_percentiles": [1, 99],
              "area_bounds_px": [lo, hi], "train_images": N},
 "sets": {"vae_prior": S, "ae_prior": S | null, "vae_recon_val": S}}
S = {"n", "valid_fraction", "fail_empty", "fail_area", "fail_components",
     "nn_mse_mean", "nn_mse_median", "foreground_px_mean"}
```

## Promotion verdict (`eval/promotion.py`)

`Measures(mse: float, mpjpe_mm: float, train_kl: float | None = None)`;
`verdict(variant, r0, r0b, variational=False) -> Verdict` with fields `label` (one of
`"WIN-pixel"`, `"WIN-latent"`, `"DROPPED"`, `"REPORTED"`), `d_mse_rel`, `d_mpjpe_mm`,
`seed_gap_rel`, `probe_gap_mm`, `reason` (str).

## run_info.json addition

`peak_rss_mb`: float, `resource.getrusage(RUSAGE_SELF).ru_maxrss / 1024` (Linux KiB → MiB),
rounded to 0.1, written with `finished_utc` (train, baseline, probe, sample).
