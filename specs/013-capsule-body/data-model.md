# Data Model: 013 Capsule Body

## Capsule body file (`assets/synthetic/capsule_v1.npz`, never committed)

| Key | Shape, dtype | Content |
|---|---|---|
| `v_template` | [5952, 3] float64 | 28 capsules (in `SEGMENTS` order) then 24 joint markers; 1e-6 m |
| `shapedirs` | [5952, 3, 10] float64 | 0 size (0.04·v), 1 limb thickness (0.10·radial), 2 torso width (0.08·r1 offset), 3–9 zero |
| `posedirs` | [5952, 3, 207] float32 | zeros |
| `J_regressor` | [24, 5952] float64 | one-hot on the markers |
| `weights` | [5952, 24] float64 | one-hot: a segment's skin joint; a marker's own joint |
| `kintree_table` | [2, 24] int64 | [[2³²−1, SMPL parents 1..23], [0..23]] |
| `f` | [11744, 3] int64 | capsule triangles (consistent winding); markers in no face |

Identity: `CAPSULE_V1_SHA256`; a dataset's manifest records it under `inputs.smpl_model`.
Version policy: any body change is `capsule_v2` (new pin, new datasets).

## Segment table (`capsule.SEGMENTS`)

`(skin joint, (joint, offset A), (joint, offset B), r1, r2)`: spec FR-002's table; 6 trunk /
head segments + 11 per side.

## Capsule dataset series (stage 3)

`capsule_proc_v1` and the independent single-change datasets `capsule_proc_v2a`–`v2d`, twins of
the SMPL datasets; results under `docs/results/013-capsule/`.
