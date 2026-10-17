# Data Model: 001 Dataset Generation

Schema version: `1`. Normative contract: [`contracts/dataset-layout.md`](contracts/dataset-layout.md).

## Skeleton (`strike_a_pose.skeleton`)

`SMPL24_JOINT_NAMES` (order is the keypoint index):
0 pelvis, 1 left_hip, 2 right_hip, 3 spine1, 4 left_knee, 5 right_knee,
6 spine2, 7 left_ankle, 8 right_ankle, 9 spine3, 10 left_foot, 11 right_foot,
12 neck, 13 left_collar, 14 right_collar, 15 head, 16 left_shoulder,
17 right_shoulder, 18 left_elbow, 19 right_elbow, 20 left_wrist,
21 right_wrist, 22 left_hand, 23 right_hand.

`SMPL24_PARENTS = (-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21)`.
Skeleton name: `"smpl24"`.

## Config (`strike_a_pose.config.GenerationConfig`, YAML)

| Field | Type | Default | Notes |
|---|---|---|---|
| `dataset_name` | str | required | `[A-Za-z0-9_-]+` |
| `output_root` | path | `data/generated` | dataset written to `<output_root>/<dataset_name>` |
| `smpl_model_path` | path | `assets/smpl/SMPL_NEUTRAL.npz` | `.npz` or `.pkl` |
| `amass_root` | path | `assets/amass` | |
| `amass_subsets` | list[str] | required, ≥1 | sub-collection dir names under `amass_root` |
| `num_poses` | int ≥1 | 2000 | |
| `frame_stride` | int ≥1 | 30 | |
| `num_betas` | int | 10 | betas all zero (Q4) |
| `image.height` / `image.width` | int | 256 / 192 | |
| `camera_ring.count` | int ≥1 | 8 | |
| `camera_ring.distance_m` | float >0 | 3.5 | |
| `camera_ring.height_m` | float | 0.0 | |
| `camera_ring.focal_px` | float >0 | 320.0 | |

Unknown keys → error. Seed is NOT in the config; it is a required CLI flag
(FR-001) and recorded in the manifest.

## Camera

`name: str`, `K: 3×3`, `R: 3×3`, `t: 3`, `width`, `height`, `yaw_deg`.

## PoseRecord

`index` (0..N-1, draw order), `source_file` (POSIX path relative to
`amass_root`), `frame` (int), `pose` (72 floats, axis-angle, root zeroed),
`source_root_orient` (3 floats, original).

## Sample

`sample_id = f"{pose_index:05d}_{camera_name}"`; files
`samples/<id>.png`, `samples/<id>_mask.png`, `samples/<id>.json`.

## Manifest

See contract. Key fields: `schema_version`, `dataset_name`, `complete`,
`seed`, `config`, `code` {`commit`, `dirty`}, `inputs` {`smpl_model`
{`filename`, `sha256`}, `amass` [{`file`, `sha256`}] (used files only, sorted)},
`skeleton` {`name`, `joints`, `parents`}, `cameras` [Camera], `counts`
{`poses`, `cameras`, `samples`}, `samples` [{`id`, `image`, `mask`, `labels`,
`labels_sha256`}] (pose-major, camera ring order).

## State transitions

`(absent)` → `incomplete` (manifest written with `complete: false`, empty
`samples`, before rendering) → `complete` (final manifest atomically replaced
via temp file + rename). A directory whose manifest says `complete: false` is
never valid input to later slices.
