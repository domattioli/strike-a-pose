# Contract: Dataset Layout v1

Consumed by slice 002. Breaking changes bump `schema_version`.

```text
<output_root>/<dataset_name>/
├── manifest.json      # identity artifact (no timestamps/host info)
├── checksums.json     # {"<relative path>": "<sha256>"} for every .png
├── run_info.json      # started/finished UTC, python/numpy/pillow versions, platform, argv
└── samples/
    ├── 00000_Front_0.png        # RGB uint8, H×W×3, grey body on black
    ├── 00000_Front_0_mask.png   # L uint8, 255 = body, 0 = background
    └── 00000_Front_0.json       # labels
```

## Labels JSON (`samples/<id>.json`)

All floats rounded to 6 decimals. Arrays are nested lists.

| Key | Shape / type | Meaning |
|---|---|---|
| `schema_version` | int | 1 |
| `sample_id` | str | |
| `pose_index` | int | |
| `camera_name` | str | |
| `pose_source` | {`type`: "procedural", `ranges_version`: str, `euler_deg`: [24][3]} or {`type`: "amass", `file`: str, `frame`: int, `root_orient`: [3]} | pose origin (amended 2026-10-20); `euler_deg` includes the table's offsets (slice 010) |
| `pose` | [72] | SMPL axis-angle, root zeroed |
| `betas` | [num_betas] | zeros, or the pose's random betas with `body_shape` (same for its views; slice 010) |
| `joints_3d_world` | [24][3] | metres, world frame |
| `joints_3d_cam` | [24][3] | metres, camera frame (OpenCV) |
| `keypoints_2d` | [24][2] | pixels (u=x right, v=y down) |
| `visibility` | [24] bool | in front and inside image |
| `camera` | {`name`, `K` [3][3], `R` [3][3], `t` [3], `width`, `height`, `yaw_deg`} (+ `jitter`) | `R`, `t`: the sample's camera; `name`, `K`, size, `yaw_deg`: its ring camera; `jitter` {`height_m`, `distance_m`, `pitch_deg`, `attempts`, `fallback`} only with `camera_jitter` (slice 010) |

Invariant (SC-003): `project(K, R·X_w + t)` equals `keypoints_2d` within 1 px
and `joints_3d_cam = R·joints_3d_world + t` within 1e-4 m.

## manifest.json

JSON, sorted keys, indent 2, trailing newline. Fields per data-model.md.

Slice 010 (2026-10-21): `config` leaves out the generator fields added after layout v1
(`ranges_version`, `body_shape`, `camera_jitter`) while they hold their defaults, so earlier
datasets are byte-identical; `cameras` is always the ring; `inputs.pose_source.ranges_deg[*]`
carries `offset_deg` for tables with offsets. All additions are optional keys:
`schema_version` stays 1.

Frozen defaults (slice 012, 2026-10-21): a manifest `config` without one of those fields means
the value in `datagen.config.ADDED_FIELD_DEFAULTS` (`ranges_version: predecessor-v1-nohands`,
`body_shape: null`, `camera_jitter: null`). These values are frozen for layout v1: changing one
would silently change the meaning of every existing v1 manifest, so it requires a new
`schema_version` (pinned literally by `tests/test_realism.py`).

## CLI (`strike-a-pose-data`)

```text
strike-a-pose-data generate --config PATH --seed INT [--overwrite] [--output-root PATH]
strike-a-pose-data validate DATASET_DIR
strike-a-pose-data make-capsule-body [--out PATH] [--fixture DIR] [--overwrite]
```

- `make-capsule-body` (slice 013): writes the interim capsule body (default
  `assets/synthetic/capsule_v1.npz`; skeleton from the fixture dir, default
  `tests/fixtures/smpl_fixture`). Exit codes: 0 written or unchanged (same bytes); 1 the built
  file's SHA-256 differs from the pinned one (nothing written); 3 fixture labels missing; 4 the
  file exists with other bytes and `--overwrite` not given. `smpl_model_path` may name any
  SMPL-format body; `inputs.smpl_model` then records that file's name and SHA-256 (no schema
  change).

- `generate` exit codes: 0 ok; 2 config error; 3 missing SMPL (or, for `pose_source: amass`, AMASS) assets
  (message names the missing file and points to `docs/data-assets.md`;
  nothing written); 4 output exists and `--overwrite` not given.
- `--overwrite` deletes the existing dataset dir first (restart cleanly).
- `--output-root` overrides `output_root` (used for the fixture).
- `validate`: exit 0 if the dataset satisfies this contract (complete, all
  files present, label SHA-256 match, label schema/shapes, SC-003 invariant);
  else exit 1 listing problems.
