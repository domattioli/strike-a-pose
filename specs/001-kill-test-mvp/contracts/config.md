<!-- provenance: author=domattioli model=claude-opus-5-5 effort=max date=2026-10-08 skill=speckit-implement repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
# Contract: configuration schema

**Branch**: `001-kill-test-mvp` | **Date**: 2026-10-07 | **Plan**: [../plan.md](../plan.md)

YAML, loaded by `config.py`. Unknown keys are an error (exit 2). The canonical dump (sorted keys, resolved defaults) is hashed with SHA-256 into `config_hash`. Two files ship: `configs/tiny.yaml` (CPU smoke, FR-027) and `configs/full.yaml` (Kaggle run). Columns: key, type, tiny value, full value, rule.

## Top level and assets

| Key | Type | tiny | full | Rule |
|---|---|---|---|---|
| `seed` | int | 1 | 20261007 | one integer seed for the whole run (FR-024) |
| `device` | enum auto, cpu, cuda | cpu | auto | FR-028 |
| `assets.root` | path or null | null | null | null means `SAP_ASSET_ROOT` |

## body

| Key | Type | tiny | full | Rule |
|---|---|---|---|---|
| `body.model` | enum standin, smplx | standin | smplx | smplx needs `<root>/smplx/SMPLX_NEUTRAL.npz` |
| `body.n_betas` | int | 10 | 10 | shape coefficients used |
| `body.beta_clip` | float | 3.0 | 3.0 | standard normal clipped to plus or minus this value |

## pose (FR-001)

| Key | Type | tiny | full | Rule |
|---|---|---|---|---|
| `pose.source` | enum limits, amass | limits | amass | amass needs `<root>/amass/<subset>/**/*.npz` |
| `pose.amass_subsets` | list of str | [] | ["ACCAD", "CMU"] | folder names under `<root>/amass/`; operator's choice |
| `pose.limits_deg` | map joint group to float | table | table | spine 35, neck 50, head 50, collar 20, shoulder 150, elbow 150, wrist 60, hip 120, knee 150, ankle 45 |
| `pose.capsule_overlap_cm` | float | 1.0 | 1.0 | self-intersection tolerance |
| `pose.max_rejections` | int | 100 | 100 | per body; exceeded is a stage failure (exit 4) |

## camera (FR-002, FR-005)

| Key | Type | tiny | full | Rule |
|---|---|---|---|---|
| `camera.n_cameras` | int | 4 | 4 | cameras per rig, always rendered |
| `camera.image_size` | int | 64 | 128 | square mask |
| `camera.focal_px` | float | 64 | 128 | pinhole focal length in pixels |
| `camera.distance_m` | [min, max] | [2.5, 4.0] | [2.5, 4.0] | uniform |
| `camera.height_m` | [min, max] | [0.8, 1.8] | [0.8, 1.8] | uniform |
| `camera.min_separation_deg` | float | 20 | 20 | azimuths of one rig; rejection sampling |
| `camera.lookat_jitter_m` | float | 0.05 | 0.05 | per axis, uniform |

## data (FR-003, FR-024)

| Key | Type | tiny | full | Rule |
|---|---|---|---|---|
| `data.n_train` | int | 256 | 20000 | contiguous split ranges |
| `data.n_cal` | int | 64 | 2500 | at least `calibrate.min_cal`; 2,500 generated so that at least `data.min_unflagged` (2,000) remain after flagged bodies are excluded (SC-002) |
| `data.n_test` | int | 64 | 2500 | 2,500 generated so that at least `data.min_unflagged` (2,000) remain after flagged bodies are excluded (SC-002) |
| `data.min_unflagged` | int | 32 | 2000 | generation exits 4 when fewer unflagged calibration or test bodies remain, naming both counts (SC-002) |
| `data.shard_size` | int | 64 | 500 | bodies per shard; resume unit |

## measure (FR-004)

| Key | Type | tiny | full | Rule |
|---|---|---|---|---|
| `measure.step_cm` | float | 0.5 | 0.5 | search step of the research R6 slice searches, for ground truth and predictions alike; 1.0 is a research R13 mitigation, and a change re-runs generation |

## model and train (FR-006 to FR-009, FR-029)

| Key | Type | tiny | full | Rule |
|---|---|---|---|---|
| `model.latent_dim` | int | 8 | 16 | |
| `model.channels` | list of int | [8, 16, 32, 32, 32] | [32, 64, 128, 256, 256] | one entry per strided block |
| `model.camera_embed_dim` | int | 16 | 64 | |
| `train.epochs` | int | 2 | 30 | |
| `train.batch_size` | int | 32 | 128 | |
| `train.lr` | float | 1e-3 | 3e-4 | AdamW |
| `train.kl_weight` | float | 1e-3 | 1e-3 | after warm-up over the first 20% of steps |
| `train.views_train` | list of int | [1, 2, 3, 4] | [1, 2, 3, 4] | uniform draw per sample |
| `train.noise_train_deg` | [min, max] | [0, 5] | [0, 5] | uniform per view |
| `train.checkpoint_every` | int steps | 10 | 500 | plus every epoch end |

## predict, calibrate, evaluate, verdict (FR-010 to FR-014)

| Key | Type | tiny | full | Rule |
|---|---|---|---|---|
| `predict.n_samples` | int | 8 | 32 | latent samples per body; 16 is a research R13 mitigation |
| `predict.measure_mode` | enum exact, linearized | exact | exact | `exact`: every latent sample's mesh is measured; `linearized`: a first-order expansion of the measurement map around each body's median sample, 11 mesh evaluations per body and cell; a research R13 mitigation; ground truth always uses `exact` |
| `calibrate.alpha` | float | 0.10 | 0.10 | nominal miscoverage |
| `calibrate.min_cal` | int | 32 | 200 | refusal threshold (FR-011) |
| `calibrate.spread_floor_cm` | float | 0.001 | 0.001 | guards division by zero only; it must sit far below every model spread, or it pins interval widths and hides the effect of the number of views (implementation finding on the tiny run) |
| `evaluate.views` | list of int | [1, 2, 4] | [1, 2, 4] | cells |
| `evaluate.noise_deg` | list of float | [0, 2, 5] | [0, 2, 5] | cells |
| `evaluate.band` | [low, high] | [0.87, 0.93] | [0.87, 0.93] | fixed by FR-014; inclusive, compared as integers (`100 * covered >= 87 * n_test` and `100 * covered <= 93 * n_test`); any other value is a configuration error (exit 2) |
| `evaluate.measurements` | list | [height, chest, waist, hip, thigh] | same | fixed order |
| `verdict.threshold` | float | 0.70 | 0.70 | fixed by FR-014; any other value is a configuration error (exit 2) |
| `verdict.measurements` | list | [chest, waist, hip, thigh] | same | fixed by FR-014; any other value is a configuration error (exit 2) |
| `verdict.noise_deg` | float | 0 | 0 | fixed by FR-014; any other value is a configuration error (exit 2) |
| `verdict.compare_views` | [low, high] | [1, 4] | [1, 4] | fixed by FR-014; any other value is a configuration error (exit 2) |

## real (FR-017 to FR-020)

| Key | Type | tiny | full | Rule |
|---|---|---|---|---|
| `real.bodym.path` | path or null | null | `<root>/bodym` | splits testA, testB |
| `real.bodym.side_azimuth_deg` | float | 90 | 90 | sign confirmed at implementation |
| `real.ssp3d.path` | path or null | null | `<root>/ssp3d` | |
| `real.ssp3d.mask_source` | enum provided, sam2 | provided | provided | |
| `real.sam2.checkpoint` | path or null | null | `<root>/sam2/sam2.1_hiera_base_plus.pt` | extra `[real]` |
| `real.nominal_camera.distance_m` | float | 3.0 | 3.0 | |
| `real.nominal_camera.height_m` | float | 1.2 | 1.2 | |
| `real.nominal_camera.lookat_height_m` | float | 0.9 | 0.9 | look-at point above the floor, about the bounding-box center of an average upright body, matching the synthetic rigs |
| `real.mask_min_area_fraction` | float | 0.02 | 0.02 | usability rule |
| `real.mask_min_component_fraction` | float | 0.90 | 0.90 | usability rule |

## Overrides

`--set key.path=value` on any subcommand overrides one key. It is repeatable, one key per `--set`, and the value is parsed as YAML, so `--set evaluate.views=[1,4]` sets a list. The override enters the canonical dump, so it changes `config_hash`. An override of a key fixed by FR-014 (`verdict.*`, `evaluate.band`) is a configuration error (exit 2): the constants live in `verdict.py`, and the configuration only restates them.

## Test overrides (`small_config`)

Every pytest test that runs a pipeline stage uses `configs/tiny.yaml` with these overrides, provided by the `small_config` fixture of `tests/conftest.py`, so it finishes within 60 seconds (constitution Principle VI). No FR-014 fixed key is touched.

| Key | Value | Reason |
|---|---|---|
| `data.n_train`, `data.n_cal`, `data.n_test` | 64, 32, 32 | smaller splits |
| `data.shard_size` | 32 | four shards, so resume is exercised |
| `data.min_unflagged`, `calibrate.min_cal` | 16, 16 | a few flagged bodies cannot stop generation or calibration; 16 keeps the 90% quantile index inside the calibration set |
| `train.epochs` | 1 | |
| `predict.n_samples` | 4 | |
| `camera.image_size`, `camera.focal_px` | 32, 32 | the focal length moves with the image size, so the field of view stays 53 degrees and bodies stay in frame |
| `evaluate.views`, `evaluate.noise_deg` | [1, 4], [0] | only the two compared cells, so the verdict still computes |
