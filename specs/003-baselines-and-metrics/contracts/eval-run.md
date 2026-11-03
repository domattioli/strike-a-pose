# Contract: Eval v1 (metrics API, baseline config, run directory)

Consumed by slice 004 (AE eval reuses the metric functions and run-directory
helpers; its numbers are reported next to `metrics.json` from this slice).
Breaking changes bump `metrics_version` in `metrics.json`.

## Metrics API (`strike_a_pose.eval.metrics`)

All functions take NumPy arrays or torch tensors with a leading batch dim and
return a `[B]` array of the same type (NumPy in → NumPy out; `region_mask` returns a
`[B, H, W]` mask of the keypoints' type). Internally torch.

| Function | Inputs | Per-sample value | Units |
|---|---|---|---|
| `image_mse(pred, target)` | `[B, ...]` same shape, images in `[0, 1]` | mean of `(pred − target)²` over all non-batch elements | intensity² |
| `psnr(pred, target, max_val=1.0)` | as above | `10·log10(max_val² / max(mse, 1e-10))` (cap 100 dB at `max_val = 1`) | dB |
| `silhouette_iou(pred, mask, threshold=0.075)` | `pred [B, ...]` float, `mask [B, ...]` bool | `|(pred > thr) ∧ mask| / |(pred > thr) ∨ mask|`; 1.0 when the union is empty | ratio |
| `mpjpe(pred, target, root=0)` | `[B, J, 3]` | mean over joints of `‖(p − p_root) − (t − t_root)‖₂` | input units (metres → ×1000 for mm) |
| `keypoint_error_2d(pred, target, visible=None)` | `[B, J, 2]`, `visible [B, J]` bool | mean over (visible) joints of `‖p − t‖₂`; NaN if no visible joint | pixels |
| `project_points(points, K, R, t)` | world `[B, J, 3]`, `K [B,3,3]`, `R [B,3,3]`, `t [B,3]` | `[B, J, 2]`: `x_c = R x + t`, `u = K₀₀ x_c/z_c + K₀₂`, `v = K₁₁ y_c/z_c + K₁₂` | pixels |
| `region_mask(keypoints, visible, image_size, joints, half_size)` (slice 009) | `keypoints [B, J, 2]` (u, v) px, `visible [B, J]` bool, `image_size` (H, W), joint indices, `half_size` h ≥ 0 | bool `[B, H, W]`: pixel (r, c) is in iff some listed joint is visible with `\|r − ⌊v⌋\| ≤ h` and `\|c − ⌊u⌋\| ≤ h` (union of (2h + 1)² squares on the pixels holding the joints, clipped to the image; invisible joints add nothing) | — |
| `masked_mse(pred, target, mask)` (slice 009) | `[B, ...]` same shape, `mask` bool | mean of `(pred − target)²` over the mask's pixels; NaN where the mask is empty | intensity² |

Constants: `IOU_THRESHOLD = 0.075`, `PSNR_MSE_FLOOR = 1e-10`, `ROOT_JOINT = 0`;
`REGIONS` (slice 009) = name → (SMPL-24 joint indices, half-size h px): `hands` ((22, 23),
10), `feet` ((7, 8, 10, 11), 8), `head` ((15,), 14) — sized for the 001 camera ring (3.5 m,
focal 320 px, 256 × 192); a dataset with another camera distance or body shape revisits
them (009 spec FR-001).

## Scoring (`strike_a_pose.eval.report`)

`score_images(pred_fn, images, masks, batch_size, threshold=0.075, *, labels=None)`
scores flattened rows `images [n, H*W]` (bool `masks [n, H*W]`); `pred_fn` is called once
per batch and returns predicted rows. Result: the means over rows of the per-image
`image_mse`, `psnr_db`, `silhouette_iou`. With `labels = (keypoints [n, J, 2], visible
[n, J], (H, W))` of the same rows (slice 009) it adds `regions`: per region of `REGIONS`,
`image_mse` = float64 mean of the per-sample `masked_mse` over the samples whose region is
non-empty and `samples` = their count (`image_mse` NaN when 0); masks are built per batch
from the labels; the headline values are bit-identical with and without labels. Both the
003 baselines and the 004 trainer score every image model through it.
`region_definitions()` returns the `metrics.json` top-level `regions` block.

## PCA (`strike_a_pose.eval.pca`)

`fit_pca(x: Tensor [n, d], k: int, rel_tol=1e-6) -> PCA` — exact (Gram or
covariance eigendecomposition, float64), keeps the top `min(k, rank)` components
where rank counts eigenvalues `> rel_tol × λ_max`. `PCA` fields: `mean [d]`,
`components [k_eff, d]` (orthonormal rows), `variances [k_eff]`,
`total_variance`, `n_samples`, `numerical_rank`; property `n_components`
(= `k_eff` = `min(k, numerical_rank)`). Methods: `reconstruct(x, k=None)` (uses the first
`min(k, k_eff)` components), `explained_variance_ratio(k)`, `k_effective(k)`.

## Baseline config (YAML, `configs/baselines*.yaml`)

Flat mapping; unknown keys are an error.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `run_name` | str | `baselines` | run dir prefix (`[A-Za-z0-9_.-]+`) |
| `run_root` | str | `runs` | parent of run dirs (git-ignored) |
| `seed` | int | `0` | Python / NumPy / torch seed (logged) |
| `dataset_dir` | str | required | 001 dataset directory |
| `split_file` | str | required | frozen 002 split file for that dataset |
| `eval_splits` | list[str] | `[val, test]` | splits to score (subset of train/val/test, unique) |
| `batch_size` | int > 0 | `256` | loading / scoring batch size |
| `image_pca_ks` | list[int > 0] | `[8, 16, 32, 64, 128]` | image PCA sizes (one fit at max k); may be `[]` (slice 010 FR-014): no image PCA, `fit.image_pca` null, no `pca_k<K>` image entries or figure columns |
| `pose_pca_ks` | list[int > 0] | `[4, 8, 16, 32]` | pose PCA sizes |
| `iou_threshold` | float in (0, 1) | `0.075` | silhouette threshold |
| `figure_split` | str | `val` | split shown in the figure (must be in `eval_splits`) |
| `figure_samples` | int > 0 | `6` | rows in the figure (evenly spaced samples) |

## CLI

```text
strike-a-pose-eval baseline --config PATH [--seed INT] [--run-root DIR]
```

Overrides replace the config values and are written into the resolved
`config.yaml`. Exit codes: 0 OK (prints run dir + headline metrics); 1 dataset
or split invalid (`DatasetError`); 2 config error; 4 run dir already exists.
Inputs are checked before the run dir is created (no dir on exit 1/2).

## Run directory `<run_root>/<run_name>-<YYYYmmddTHHMMSSZ>/`

| File | Content |
|---|---|
| `config.yaml` | resolved config (all keys, overrides applied) |
| `run_info.json` | `argv`, `cwd`, `seed`, `git` (`commit`, `dirty`), `env` (python, platform, numpy, torch, pillow, cpu_count, `cpu_model` (slice 012), torch_threads), `dataset` (`dir`, `manifest_sha256` = SHA-256 of the raw `manifest.json` bytes, `manifest_content_sha256` = manifest content hash of the 002 split-file contract, added by slice 007, 2026-10-20; `checksums_sha256` = SHA-256 of the dataset's `checksums.json`, i.e. its pixels, null if absent, added by slice 012, 2026-10-21), `split_file` (`path`, `sha256`), `started_utc`, `finished_utc`, `wall_s` |
| `metrics.json` | below; deterministic bytes for the same inputs + code |
| `recon_<figure_split>.png` | grid: header row of labels, then one row per sample: true, mean image, PCA k for each k |

### `metrics.json` (`metrics_version` 1)

```json
{
  "metrics_version": 1,
  "dataset": {"name": "smpl_proc_v1", "manifest_sha256": "…", "split_file_sha256": "…"},
  "fit": {
    "split": "train", "samples": 12800, "poses": 1600,
    "image_pca": {"numerical_rank": 12799, "n_components": 128, "explained_variance_ratio": {"8": 0.5, "…": 0}},
    "pose_pca": {"numerical_rank": 69, "n_components": 32, "explained_variance_ratio": {"4": 0.5, "…": 0}}
  },
  "regions": {
    "feet": {"half_size_px": 8, "joints": ["left_ankle", "right_ankle", "left_foot", "right_foot"]},
    "hands": {"half_size_px": 10, "joints": ["left_hand", "right_hand"]},
    "head": {"half_size_px": 14, "joints": ["head"]}
  },
  "units": {"image_mse": "…", "psnr_db": "…", "silhouette_iou": "…", "mpjpe_mm": "…", "kp2d_px": "…", "regions": "…"},
  "splits": {
    "val": {
      "samples": 1600, "poses": 200,
      "image": {
        "mean_image": {"image_mse": 0.0, "psnr_db": 0.0, "silhouette_iou": 0.0,
                       "regions": {"feet": {"image_mse": 0.0, "samples": 1600},
                                   "hands": {"image_mse": 0.0, "samples": 1600},
                                   "head": {"image_mse": 0.0, "samples": 1600}}},
        "pca_k64": {"k": 64, "k_effective": 64, "image_mse": 0.0, "psnr_db": 0.0, "silhouette_iou": 0.0,
                    "regions": {"…": {"image_mse": 0.0, "samples": 1600}}}
      },
      "pose": {
        "mean_pose": {"mpjpe_mm": 0.0, "kp2d_px": 0.0, "kp2d_samples": 1600},
        "pca_k8": {"k": 8, "k_effective": 8, "mpjpe_mm": 0.0, "kp2d_px": 0.0, "kp2d_samples": 1600}
      }
    },
    "test": {"…": "…"}
  }
}
```

Split entries with `samples: 0` have no `image` / `pose` keys. Floats are
rounded to 6 significant digits; NaN is written as `null`. Keys sorted, indent 2.

Per-region image metrics (slice 009, 2026-10-21; additive, so `metrics_version` stays 1:
no existing key changes meaning): every image entry of a split with samples carries
`regions` — per region, `image_mse` (mean over the split's samples whose region is
non-empty of the per-sample MSE over all the region's pixels, body and background; null
when none) and `samples` (their count). The top-level `regions` block records the
definitions the values were computed with (joints by SMPL-24 name, `half_size_px`), and
`units.regions` the unit and aggregation. Region values compare models within a region:
each region has its own share of body and background pixels, so they are not compared
with each other or with the whole-image MSE. Pose entries are unchanged.
