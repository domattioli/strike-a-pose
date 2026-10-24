# Tasks: Recalibrated Limb–Torso Intersection Check

Input: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md). Rules for every implementer: tests first, run them red and note
the red result in your report; then code; the batch ends green with
`uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked pytest -q` (whole suite). Never `uv sync`, never add a dependency, never
`git add -A`, never commit or push (the lead commits). Never commit or copy
`assets/smpl/SMPL_NEUTRAL.npz`. A training queue is running (`data/grill_queue.sh`): do not kill
it; the only real-SMPL runs allowed are the skip-if-absent test of T011 and T013 (each < 1 min,
< 1 GB). Slice 016 is implemented in parallel: do not touch `specs/016-*`, `src/strike_a_pose/eval/`,
`configs/roundtrip*`. Python 3.11 syntax. Touch only the files your batch lists. House style:
docstrings cite "spec 017 FR-xxx".

## Stage 0 — design (lead)

- [x] T001 spec.md (surrogate rulings 2026-10-24), plan.md, research.md, data-model.md, tasks.md.

## Batch B1 — recalibrated check (FR-001–FR-004)

Files: `src/strike_a_pose/datagen/intersection.py`, `scripts/intersection_check.py`,
`tests/test_intersection.py`.

- [ ] T010 Tests (red first), edits to `tests/test_intersection.py`:
  - Update every call to the new signatures (data-model.md): `_count` builds
    `rest, rest_j = body.forward(np.zeros(72), ZERO_BETAS)` and calls
    `count_arm_torso_intersections(rest, rest_j, verts, body.weights, margin)`; same in
    `test_rigid_motion_…` and `test_degenerate_torso_is_refused` (`TorsoVolume(rest, rest_j,
    body.weights, margin_m=0.07)`, `TorsoVolume(rest, rest_j, w)`); the spy in
    `test_script_poses_are_the_dataset_poses` becomes
    `def spy(body, poses, betas, margin_m, shoulder_clearance_m)` and forwards both.
  - `test_survey_counts_and_fraction`: add `assert s.min_inside == 1`.
  - `test_min_inside_vertices()`: `min_inside_vertices(64) == 1`, `(2559) == 11`,
    `(250) == 1`, `(251) == 2`, `(0) == 1`.
  - `test_spine3_is_capped_at_the_shoulders(body)`: rest/rest_j of the widened body;
    `v = TorsoVolume(rest, rest_j, body.weights)`; `v.spine3_half_x == pytest.approx(0.13)`
    (synthetic J16 x 0.18, J17 x −0.18; uncapped 0.14); with `shoulder_clearance_m=0.0` →
    `0.14` (min with the AABB half); with `shoulder_clearance_m=0.18` → `ValueError` matching
    "half-extent"; the 9-segment's `half[0]` equals `spine3_half_x`; other segments' halves
    equal those of `shoulder_clearance_m=0.0`.
  - `test_cap_only_removes_hits(body)`: for the 400 v1 poses of seed 0 (`sample_procedural_poses(
    400, 0)`), raw count with clearance 0.05 ≤ raw count with clearance 0.0, per pose.
  - `test_script_reports_and_gates`: add `r["check_version"] == "017"`,
    `r["shoulder_clearance_m"] == 0.05`, `r["min_inside_vertices"] == 1`,
    `r["spine3_half_x_m"] == 0.13`; `--shoulder-clearance-m 0.0` gives
    `r["spine3_half_x_m"] == 0.14`.
  - `test_script_refusals`: replace the `huge` body (the cap no longer lets spine3 swallow the
    shoulders) by `_arm_in_pelvis(synthetic_smpl_npz)`: a helper that loads the npz, sets the 8
    `v_template` rows with `weights[:, 22] == 1.0` (left hand cube) to the mean of the rows with
    `weights[:, 0] == 1.0`, saves as `armin_<name>`; expect exit 2 and "INVALID".
  - `test_real_smpl_calibration()` with
    `pytest.mark.skipif(not (ROOT / "assets/smpl/SMPL_NEUTRAL.npz").exists(), reason="licensed
    SMPL asset absent")`: load the body; helper `_euler_pose({joint: (x, y, z) degrees})` via
    `euler_xyz_to_matrix(np.radians(...))` → `matrix_to_axis_angle` into `pose[3j:3j+3]`
    (both from `strike_a_pose.datagen.procedural`); raw counts with betas 0:
    rest == 0; arms down 60° `{16: (0, 0, -60), 17: (0, 0, 60)}` == 0;
    arms forward `{16: (0, -90, 0), 17: (0, 90, 0)}` == 0; hands on hips
    `{16: (0,0,-60), 17: (0,0,60), 18: (0,-100,0), 19: (0,100,0)}` == 0;
    left forearm through chest `{16: (0,-90,0), 18: (0,-150,0)}` ≥ 500;
    left arm through belly `{16: (0,0,-90), 18: (0,-30,0)}` ≥ 100;
    both crossed `{16: (0,-80,0), 17: (0,80,0), 18: (0,-150,0), 19: (0,150,0)}` ≥ 1000
    (measured 0 / 0 / 0 / 0 / 641 / 188 / 1521). Then `survey` of
    `sample_procedural_poses(2000, 0)` with betas 0: `s.min_inside == 11`,
    `s.arm_vertices == 2559`, `abs(s.intersecting - 35) <= 2`,
    `s.spine3_half_x == pytest.approx(0.1238, abs=5e-4)`. Measured ≈ 12 s.
- [ ] T011 `src/strike_a_pose/datagen/intersection.py`:
  - Constants `SPINE3 = 9`, `SHOULDER_JOINTS = (16, 17)`, `SHOULDER_CLEARANCE_M = 0.05`
    (comment: "≈ SMPL upper-arm radius; spec 017 FR-001"), `MIN_INSIDE_FRACTION = 0.004`.
  - `def min_inside_vertices(arm_vertices: int) -> int: return max(1, math.ceil(
    MIN_INSIDE_FRACTION * arm_vertices))`.
  - `TorsoVolume.__init__(self, rest_verts, rest_joints, weights, margin_m=MARGIN_M,
    shoulder_clearance_m=SHOULDER_CLEARANCE_M)`: as 014, except for `j == SPINE3`:
    `cap = abs(rest_joints[16, 0] - rest_joints[17, 0]) / 2.0 - shoulder_clearance_m`;
    `half[0] = min(half[0], cap)` *before* the `(half <= 0).any()` check (same message
    "torso segment 9: half-extent <= margin …" is fine; must contain "half-extent").
    Store `self.spine3_half_x = float(half[0])` of segment 9,
    `self.min_inside = min_inside_vertices(len(self.arm_index))`,
    `self.shoulder_clearance_m`.
  - `count_arm_torso_intersections(rest_verts, rest_joints, posed_verts, weights,
    margin_m=MARGIN_M, shoulder_clearance_m=SHOULDER_CLEARANCE_M) -> int`.
  - `Survey`: add fields `min_inside: int`, `spine3_half_x: float` (after `torso_vertices`);
    `intersecting = sum(c >= self.min_inside for c in self.counts)`.
  - `survey(body, poses, betas, margin_m=MARGIN_M, shoulder_clearance_m=SHOULDER_CLEARANCE_M)`:
    `rest, rest_j = body.forward(np.zeros(72), b)`; pass both. Module docstring: add the cap and
    the per-pose minimum, "spec 017 FR-001, FR-002 (014's box reached past the shoulders on
    real SMPL: 80 % of v1 poses flagged by armpit skin)".
- [ ] T012 `scripts/intersection_check.py`: import `SHOULDER_CLEARANCE_M`; flag
  `--shoulder-clearance-m` (float, default `SHOULDER_CLEARANCE_M`, must be ≥ 0 else
  `ConfigError`); pass to `survey`; `hits` = poses with `c >= s.min_inside`; report adds
  `"check_version": "017"`, `"shoulder_clearance_m": a.shoulder_clearance_m`,
  `"spine3_half_x_m": round(s.spine3_half_x, 6)`, `"min_inside_vertices": s.min_inside`;
  `_print` line 3 appends `; spine3 half-width {spine3_half_x_m:.3f} m (shoulder clearance
  {shoulder_clearance_m:.3f} m); a pose intersects at >= {min_inside_vertices} arm vertices`.
  Docstring: usage line gains `[--shoulder-clearance-m 0.05]`; the "FAIL when" sentence says
  "poses with at least `ceil(0.4 %)` of the arm vertices inside (spec 017)".
- [ ] T013 Verification run (real SMPL, after green): `uv run --locked python
  scripts/intersection_check.py --config configs/smpl_proc_v2d/generate.yaml --seed 0` →
  expect `89 / 4000 = 2.23 %` (±2 poses), PASS, exit 0. Paste the output into the report.

## Batch B2 — docs (FR-005)

Files: `specs/014-realism-ablations/spec.md`, `docs/HANDOFF.md`.

- [ ] T020 014 spec: under FR-004 add one line "Superseded in part by spec 017 (2026-10-24):
  spine3 capped at the shoulders − 5 cm, ≥ 0.4 % of arm vertices per pose; real-SMPL rates in
  017 research R4."
- [ ] T021 `docs/HANDOFF.md` lines ≈ 88–90 (intersection description) and the v2a gate comment
  (≈ 212): state the 017 rule and "real SMPL: v1 1.75 %, v2d 2.23 %, v3 2.24 %, armsdown-v1
  23 % (FAIL)". No other HANDOFF edits.

## Stage 9 — lead

- [ ] T090 Review, commit, roadmap entry (done at design time; update with the T013 output).

## Analyze (lead, 2026-10-24)

- Coverage: FR-001 T010/T011; FR-002 T010/T011; FR-003 T010/T012; FR-004 T010 (real test);
  FR-005 T020/T021; SC-001 T010 (v1) + T013 (v2d); v3 / armsdown rates are research R4 only
  (≈ 1 min each; not re-run to spare the queue). SC-002 T010 + suite.
- Consistency: signature order `(rest_verts, rest_joints, …)` identical in data-model, T010,
  T011; constants match spec Q2/Q3; 11 = ceil(0.004 × 2559) = ceil(10.236).
- Risk: existing wide-torso tests: the cap (0.13 m) shrinks only spine3 x and only removes hits
  (T010 `test_cap_only_removes_hits`); `test_arm_driven_into_the_torso_is_counted` drives the
  elbow cube to x ≈ 0, unaffected. If any existing 014 assertion fails, report it; do not
  change constants.
- No overlap with 016's files. No `assets/` file staged.
