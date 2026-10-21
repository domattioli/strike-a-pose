# Contract: Probe v1 (probe config, latent cache, run directory)

Builds on the 003 eval contract (metrics, PCA, run-dir conventions) and reads 004
checkpoints (`contracts/train-run.md` of slice 004). Breaking changes bump
`probe_version` in `metrics.json`.

## Probe config (YAML, `configs/probe*.yaml`)

Flat mapping; unknown keys are an error.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `run_name` | str | `probe` | run dir prefix (`[A-Za-z0-9_.-]+`) |
| `run_root` | str | `runs` | parent of run dirs (git-ignored) |
| `seed` | int | `0` | Python / NumPy / torch seed; MLP init + shuffling; random-encoder init |
| `dataset_dir` | str | required | 001 dataset directory |
| `split_file` | str | required | frozen 002 split file |
| `checkpoint` | str | required | 004 checkpoint whose encoder is probed; any 004 `checkpoint.pt` or `best.pt` loads. Probe the train run's scored weights, the file its `metrics.json` names in `selection.checkpoint` (slice 008): `best.pt` by default when val has samples, else `checkpoint.pt` |
| `cache_dir` | str \| null | `runs/probe-cache` | latent cache directory (null: no cache) |
| `batch_size` | int > 0 | `256` | loading / encoding batch |
| `eval_splits` | list[str] | `[val, test]` | splits reported; must contain `val` (selection) |
| `targets` | list[str] | `[joints_world, joints_cam, yaw, smpl_pose]` | unique subset of these (slice 010: `yaw` is always the ring index, also with `camera_jitter`; `joints_cam` is in the sample's own camera frame) |
| `ridge_lambdas` | list[float > 0] | `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10]` | ridge grid (penalty `λ·n` on standardized latents) |
| `mlp_hidden` | list[int > 0] | `[256, 256]` | MLP hidden widths (ReLU) |
| `mlp_epochs` | int > 0 | `100` | MLP passes over train |
| `mlp_batch_size` | int > 0 | `256` | MLP batch |
| `mlp_lr` | float > 0 | `0.001` | Adam lr (cosine decay to 0 per step) |
| `mlp_weight_decays` | list[float ≥ 0] | `[0.0, 1e-4, 1e-3]` | MLP grid (Adam weight decay) |
| `pose_pca_ks` | list[int > 0] | `[8, 16]` | pose-PCA baselines per joint frame |
| `random_encoder_control` | bool | `true` | also probe an untrained, seeded encoder |
| `figure_split` | str | `val` | split shown in the figures (in `eval_splits`) |
| `figure_samples` | int > 0 | `6` | skeleton figure rows |

## CLI

```text
strike-a-pose-eval probe --config PATH [--seed INT] [--run-root DIR] [--checkpoint PATH]
```

Exit codes: 0 OK (prints run dir + headline numbers); 1 dataset or split invalid;
2 config error (incl. missing checkpoint, checkpoint image size ≠ dataset image
size); 4 run dir exists. Inputs are checked before the run dir is created.

## Latent cache

`<cache_dir>/<ckpt12>-<manifest12>-<split12>.pt` (first 12 hex digits of each
SHA-256). `manifest` is the raw manifest hash (SHA-256 of the `manifest.json` bytes,
which include the generating `code` commit), not the content hash the 002 split file
uses: a dataset regenerated at another commit misses the cache, by design, because
the manifest does not hash the images (slice 007, 2026-10-20).
Content: `{"probe_version": 1, "checkpoint_sha256", "manifest_sha256",
"split_sha256", "seed", "encoders": [names], "splits": {split: {"latents": {encoder:
float32 [n, latent_dim]}, "joints_world" [n,24,3], "joints_cam" [n,24,3], "kp2d"
[n,24,2], "visible" [n,24], "K", "R", "t", "camera_index" [n], "pose" [n,72],
"pose_index" [n]}}}` (float64 labels). Reused only if all three hashes, the seed
and the encoder set match; otherwise recomputed and overwritten. Never committed.

## Run directory `<run_root>/<run_name>-<YYYYmmddTHHMMSSZ>/`

| File | Content |
|---|---|
| `config.yaml` | resolved config (overrides applied) |
| `run_info.json` | 003 fields (`argv`, `cwd`, `seed`, `git`, `env`, `dataset` (incl. `manifest_content_sha256`, slice 007), `split_file`, `started_utc`, `finished_utc`, `wall_s`) + `checkpoint` (`path`, `sha256`, `epoch`, `train_run_git` from the checkpoint's run dir if present), `cache` (`path`, `hit`), `encode_s`, `figures` |
| `metrics.json` | below |
| `skeleton_<figure_split>.png` | rows = evenly spaced samples; columns: true, ridge, mlp (trained encoder, `joints_world`) drawn on the input image (true skeleton blue, prediction orange) |
| `yaw_confusion_<figure_split>.png` | 8×8 count grids (rows true, columns predicted) for the trained-encoder ridge and MLP yaw probes |

VAE checkpoints (slice 011 `training-variants`, 2026-10-21): a `variational` 004 checkpoint
loads like any other; its encoder is probed through `encode(x)` = μ (deterministic, no
sampling), and the random-encoder control stays a plain seeded encoder (the same weights as
an untrained VAE's encoder at that seed). `encoder.variational` (additive; `probe_version`
stays 1) records which kind was probed.

### `metrics.json` (`probe_version` 1)

```json
{
  "probe_version": 1,
  "metrics_version": 1,
  "dataset": {"name": "…", "manifest_sha256": "…", "split_file_sha256": "…"},
  "encoder": {"checkpoint_sha256": "…", "epoch": 60, "latent_dim": 64, "variational": false},
  "fit": {"split": "train", "samples": 12800, "poses": 1600},
  "units": {"mpjpe_mm": "…", "kp2d_px": "…", "accuracy": "…", "angle_deg": "…"},
  "classes": ["Front_0", "…"],
  "selection": {
    "ridge": {"joints_world": {"value": 0.01, "selected_by": "val mpjpe_mm", "val": {"0.0001": 0.0, "…": 0.0}}},
    "mlp": {"…": "…"}, "random_ridge": {"…": "…"}, "random_mlp": {"…": "…"}
  },
  "splits": {
    "val": {
      "samples": 1600, "poses": 200,
      "targets": {
        "joints_world": {"ridge": {"mpjpe_mm": 0.0, "kp2d_px": 0.0}, "mlp": {}, "random_ridge": {}, "random_mlp": {},
                         "mean_pose": {}, "pca_k8": {"k": 8, "k_effective": 8, "mpjpe_mm": 0.0, "kp2d_px": 0.0}},
        "joints_cam": {"…": "…"},
        "yaw": {"ridge": {"accuracy": 0.0, "confusion": [[0]], "errors_by_offset_deg": {"45": 0, "90": 0, "135": 0, "180": 0}},
                "…": "…", "majority": {"…": "…"}},
        "smpl_pose": {"ridge": {"angle_deg": 0.0}, "…": "…", "mean_pose": {"angle_deg": 0.0}}
      }
    },
    "test": {"…": "…"}
  }
}
```

Probe names: `ridge`, `mlp` (trained encoder), `random_ridge`, `random_mlp`
(random-encoder control, only when enabled). `selected_by` is `"val <metric>"` or
`"default (empty val)"`. Empty splits: `{"samples": 0, "poses": 0}`. Floats: 6
significant digits, sorted keys, NaN → null (003 writer). Deterministic bytes for
the same inputs, code and machine / thread count.
