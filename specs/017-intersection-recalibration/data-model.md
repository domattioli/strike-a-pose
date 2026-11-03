# Data Model: 017

`datagen/intersection.py` (NumPy):

- Constants: `SHOULDER_CLEARANCE_M = 0.05`, `MIN_INSIDE_FRACTION = 0.004`,
  `SHOULDER_JOINTS = (16, 17)`, `SPINE3 = 9`; unchanged `TORSO_JOINTS`, `ARM_JOINTS`,
  `MARGIN_M`, `MAX_INTERSECTING_FRACTION`.
- `min_inside_vertices(arm_vertices: int) -> int` = `max(1, math.ceil(MIN_INSIDE_FRACTION *
  arm_vertices))`.
- `TorsoVolume(rest_verts, rest_joints, weights, margin_m=MARGIN_M,
  shoulder_clearance_m=SHOULDER_CLEARANCE_M)`; attributes as 014 plus `min_inside: int`,
  `spine3_half_x: float`.
- `count_arm_torso_intersections(rest_verts, rest_joints, posed_verts, weights, margin_m=MARGIN_M,
  shoulder_clearance_m=SHOULDER_CLEARANCE_M) -> int` (raw count).
- `Survey` gains `min_inside: int`, `spine3_half_x: float`; `intersecting = sum(c >= min_inside)`.
- `survey(body, poses, betas, margin_m=MARGIN_M, shoulder_clearance_m=SHOULDER_CLEARANCE_M)`.

Script JSON report: 014 keys plus `check_version` "017", `shoulder_clearance_m`,
`spine3_half_x_m` (6 decimals), `min_inside_vertices`; `count_median` / `count_max` /
`worst_poses` over poses with count ≥ `min_inside_vertices`.
