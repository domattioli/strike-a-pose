<!-- provenance: author=domattioli model=claude-fable-5-1 effort=high date=2026-10-07 skill=speckit-plan repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
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
| `data.n_cal` | int | 64 | 2000 | at least `calibrate.min_cal` |
| `data.n_test` | int | 64 | 2000 | |
| `data.shard_size` | int | 64 | 500 | bodies per shard; resume unit |

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
| `predict.n_samples` | int | 8 | 32 | latent samples per body |
| `calibrate.alpha` | float | 0.10 | 0.10 | nominal miscoverage |
| `calibrate.min_cal` | int | 32 | 200 | refusal threshold (FR-011) |
| `calibrate.spread_floor_cm` | float | 0.1 | 0.1 | |
| `evaluate.views` | list of int | [1, 2, 4] | [1, 2, 4] | cells |
| `evaluate.noise_deg` | list of float | [0, 2, 5] | [0, 2, 5] | cells |
| `evaluate.band` | [low, high] | [0.87, 0.93] | [0.87, 0.93] | inclusive (FR-013) |
| `evaluate.measurements` | list | [height, chest, waist, hip, thigh] | same | fixed order |
| `verdict.threshold` | float | 0.70 | 0.70 | FR-014 |
| `verdict.measurements` | list | [chest, waist, hip, thigh] | same | circumferences entering the median ratio |
| `verdict.noise_deg` | float | 0 | 0 | |
| `verdict.compare_views` | [low, high] | [1, 4] | [1, 4] | |

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
| `real.mask_min_area_fraction` | float | 0.02 | 0.02 | usability rule |
| `real.mask_min_component_fraction` | float | 0.90 | 0.90 | usability rule |

## Overrides

`--set key.path=value` on any subcommand overrides one key; the override enters the canonical dump, so it changes `config_hash`.
