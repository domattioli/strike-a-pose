# Data Model: 010 Dataset Realism Options

Extends [001 data-model](../001-dataset-generation/data-model.md) and the
[layout contract v1](../001-dataset-generation/contracts/dataset-layout.md); `schema_version`
stays 1 (additive, optional keys; today's datasets byte-identical).

## Generation config (`strike_a_pose.datagen.config.GenerationConfig`)

| Field | Type | Default | Notes |
|---|---|---|---|
| `ranges_version` | `predecessor-v1-nohands` \| `predecessor-v2-armsdown` | v1 | procedural source only (other value with `amass` → error) |
| `body_shape` | null \| `{std, clip}` | null | `std` > 0 (1.0), `clip` > 0 (2.0); needs `num_betas` ≥ 1 |
| `camera_jitter` | null \| `{height_m, distance_m, pitch_deg, margin_px, max_attempts}` | null | ranges `[lo, hi]`, lo ≤ hi: height within ±1 m ([-0.3, 0.3]), distance 0 ≤ lo, hi ≤ 3 m ([0.0, 0.5]), pitch within ±30° ([0.0, 5.0], positive = down); `margin_px` int ≥ 0 (2); `max_attempts` int ≥ 1 (16) |
| `num_poses` | int ≥ 1 | 2000 | existing; the more-poses option |

A present block means "on"; omitted keys inside it take the defaults. `to_dict()` (=
`manifest.config`) leaves out each of the three fields while it holds its default and writes
enabled blocks resolved (floats; ranges as lists). Rule for every later generator field.

## Ranges tables (`strike_a_pose.datagen.procedural`)

`RANGES_TABLES[version] = (ranges, offsets)`. `predecessor-v1-nohands`: `RANGES_DEG`, no
offsets. `predecessor-v2-armsdown`: v1 with z of 13 / 14 `[-10, 10)` and of 16 / 17 `[-20, 20)`;
offsets z −65° (16) and +65° (17), added after sparsification. Manifest
`inputs.pose_source.ranges_deg[*]` gains `offset_deg` `[x, y, z]` only for tables with offsets.

## Labels (`samples/<id>.json`)

| Key | Change |
|---|---|
| `pose_source.euler_deg` | includes the table's offsets |
| `betas` | the pose's random betas when `body_shape` is on (same for its views), else zeros |
| `camera.R`, `camera.t` | the used camera (jittered or ring) |
| `camera.name`, `yaw_deg`, `K`, `width`, `height` | the ring camera's |
| `camera.jitter` | only with `camera_jitter`: `{height_m, distance_m, pitch_deg, attempts, fallback}`; deltas of the used camera (zeros on fallback) |

Manifest `cameras` stays the ring (the loader's `camera_index` and the probe's yaw classes).

## Random streams

| Stream | Seeding | Draws |
|---|---|---|
| poses | `default_rng(seed)` (unchanged) | per pose: 69 bounded integers, k, kept set |
| body shape | `default_rng([seed, 1, pose_index])` | `num_betas` normals |
| camera jitter | `default_rng([seed, 2, pose_index, ring_index])` | per attempt: δh, δd, δp uniforms |

## Dataset series (surrogate ruling Q1: chain)

| Dataset | = predecessor + | Compared with |
|---|---|---|
| `smpl_proc_v2` | `ranges_version: predecessor-v2-armsdown` | `smpl_proc_v1` (controlled: same draws except 4 angles) |
| `smpl_proc_v3` | `body_shape: {std: 1.0, clip: 2.0}` | `smpl_proc_v2` |
| `smpl_proc_v4` | `camera_jitter` (defaults above) | `smpl_proc_v3` |
| `smpl_proc_v5` | `num_poses: 4000` | `smpl_proc_v4` |

Configs: `configs/<dataset>/{generate,baselines,ae,probe}.yaml`; splits
`splits/<dataset>.json` (`pose-permutation-v1`, seed 0, 0.8 / 0.1 / 0.1) written by the run
procedure; `smpl_proc_v5`: `image_pca_ks: []`, `context_pca_k: null` (FR-014).

## 003 baseline config (FR-014)

`image_pca_ks` may be `[]`: no image PCA, `metrics.json` `fit.image_pca` null, no `pca_k<K>`
image entries or figure columns.
