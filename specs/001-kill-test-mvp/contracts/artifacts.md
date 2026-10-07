<!-- provenance: author=domattioli model=claude-fable-5-1 effort=high date=2026-10-07 skill=speckit-plan repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
# Contract: output artifacts

**Branch**: `001-kill-test-mvp` | **Date**: 2026-10-07 | **Plan**: [../plan.md](../plan.md) | **Entities**: [../data-model.md](../data-model.md)

Everything lives under the `--out` directory, outside version control. Every CSV has a header row; every JSON file carries the run record fields `config_hash`, `seed`, `code_version`, `hardware_class` (FR-025). Each stage directory ends with `DONE.json` (FR-029). Files marked "committed" are copied to `specs/001-kill-test-mvp/results/` for the gate decision; nothing else is committed.

```text
<out>/
├── run_record.json                      # RunRecord (committed)
├── data/
│   ├── manifest.csv                     # one row per body
│   ├── summary.json                     # counts: bodies, rejections, flags per shard
│   ├── shards/shard_0000.npz ...        # silhouettes and per-body arrays
│   └── DONE.json
├── train/
│   ├── checkpoints/epoch_000.pt ...     # model, optimizer, scheduler, RNG states, epoch, step
│   ├── model_final.pt
│   ├── history.csv                      # step, epoch, loss terms, learning rate, wall time
│   └── DONE.json
├── predict/
│   ├── cal_v1_n0.npz ... test_v4_n5.npz # per split and cell
│   └── DONE.json
├── calibrate/
│   ├── quantiles.csv
│   └── DONE.json
├── evaluate/
│   ├── results.csv                      # ResultCell rows (committed)
│   ├── per_sample/test_v1_n0.csv ...    # intervals per body (not committed)
│   └── DONE.json
├── verdict/
│   ├── verdict.json                     # KillVerdict (committed)
│   └── verdict.md                       # verdict line plus the ratio table (committed)
├── report/
│   ├── results.md                       # human-readable table with run record (committed)
│   ├── coverage_vs_views.png            # one subplot per measurement, one series per noise level (committed)
│   └── width_vs_views.png               # same layout (committed)
└── real/
    ├── bodym/results.csv                # aggregate rows for testA and testB (committed)
    ├── bodym/subjects.csv               # per subject (not committed)
    ├── ssp3d/results.csv                # aggregate rows per mask source (committed)
    └── ssp3d/subjects.csv               # per subject (not committed)
```

## Schemas

### `data/manifest.csv`
`body_id, shard, split, pose_source, pose_rejections, flags, betas_0..betas_9, pose_root_0..2, pose_body_0..62, cam{i}_azimuth_deg, cam{i}_distance_m, cam{i}_height_m, cam{i}_noise_axis_x, cam{i}_noise_axis_y, cam{i}_noise_axis_z` for `i` in 0..3, then `height_cm, chest_cm, waist_cm, hip_cm, thigh_cm`. Floats are written with `repr` precision so two runs compare byte for byte (FR-024).

### `data/shards/shard_NNNN.npz`
`body_id (n,) int64`, `masks (n, 4, H*W/8) uint8` bit-packed, `K (3, 3) float64`, `R_true (n, 4, 3, 3) float64`, `t_true (n, 4, 3) float64`, `noise_axis (n, 4, 3) float64`, `betas (n, 10) float64`, `pose_root (n, 3)`, `pose_body (n, 63)`, `measurements (n, 5) float64`, `flags (n,) int64` bit field. Written to a temporary name and renamed, so a partial shard never exists.

### `predict/<split>_<cell>.npz`
`body_id (n,)`, `m_true (n, 5)`, `m_median (n, 5)`, `spread (n, 5)`, `latent_var_mean (n,)` (mean fused variance over latent dimensions, for SC-004), `samples (n, K, 5) float32` (kept for recomputation), `views`, `noise_deg`.

### `calibrate/quantiles.csv`
`cell_id, views, noise_deg, measurement, n_cal, alpha, q_hat, spread_floor_cm`.

### `evaluate/results.csv` (committed)
`cell_id, views, noise_deg, measurement, nominal_level, n_cal, n_test, coverage, median_width_cm, mae_cm, mean_signed_error_cm, clipped_count, in_band, q_hat, seed, config_hash`. 45 rows in cell order then measurement order.

### `verdict/verdict.json` (committed)
`verdict, median_ratio, threshold, cells_in_band, ratios {chest, waist, hip, thigh}, widths {v1: {...}, v4: {...}}, noise_deg, compare_views, reported_only {height_ratio, rows_2deg, rows_5deg}, config_hash, seed, code_version, hardware_class`.

### `real/<dataset>/results.csv` (committed)
`dataset, split, mask_source, cell_id, measurement, nominal_level, n_subjects, n_skipped, coverage, median_width_cm, mae_cm, mean_signed_error_cm, clipped_count, q_hat, seed, config_hash`.

### `real/<dataset>/subjects.csv` (not committed)
`dataset, split, subject_id, mask_source, mask_status, m_true_*, m_median_*, lower_*, upper_*, covered_*` for the five measurements.

### `run_record.json` (committed)
`config_hash, seed, code_version, hardware_class, device_name, versions {python, numpy, torch, opencv, smplx, sam2}, started_at, finished_at, timings {generate, train, predict, calibrate, evaluate, verdict, report, real_eval}`.

### `DONE.json`
`stage, config_hash, inputs {stage_name: config_hash_of_that_marker}, finished_at, code_version`. A stage is stale when its own `config_hash` or any input hash differs from the current run (data-model state transitions).
