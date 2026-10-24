# Feature Specification: Recalibrated Limb–Torso Intersection Check

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `017-intersection-recalibration`)
**Created**: 2026-10-24
**Status**: Approved at surrogate scope; operator decision 2026-10-24 ("yes" to recalibrating) as
approval basis (clarification batch 1 answered by surrogate ruling 2026-10-24, below)
**Input**: The slice-014 check (`datagen/intersection.py`, `scripts/intersection_check.py`) flags
≈ 79 % of the v1 procedural poses on real SMPL and was made diagnostic-only. Find out whether the
check is wrong (a) or the pose table really interpenetrates (b); recalibrate against a principled
reference; report the corrected flag rate on v1 / v2d / v3; name the pose-table change only if (b).

Settled context (measured at `a219e7a`, real `SMPL_NEUTRAL.npz` SHA-256 `f09c8ece…`, betas 0,
seed 0, CPU, ≤ 1 GB; scripts and figures in the session scratchpad, not committed): see
[research.md](research.md) R1–R4.

- **Diagnosis: (a), the check is wrong.** On 500 v1 poses, 403 / 404 flagged poses are hit
  *only* in the spine3 segment (joint 9), and the hit vertices are dominated by joints 16 / 17
  (upper arm at the shoulder); only 3.2 % of poses have any forearm / hand (18–23) vertex inside.
  Cause: SMPL's joint 9 dominates the upper chest out to the armpits, so the spine3 box's
  half-width (0.195 m after the 1 cm margin) is *wider* than the shoulder joints' half-span
  (0.1735 m). Any arm lowering pushes the armpit skin, which is part of the arm, inside that
  cylinder: arms down 60° (a normal standing pose) → 57 vertices "inside"; arms forward 90° → 97.
  Renders of typical flagged poses show a few red armpit vertices on plainly clean bodies.
- **Residual (b) is small.** After the fix, ≈ 2 % of v1 poses still flag; renders show forearms /
  hands genuinely sunk into chest, belly or back under strong spine twist (spine1 y ±75°) with a
  folded elbow (the worst, pose 1407, 614 vertices). Below the 5 % gate: no pose-table change.

## Clarifications

### Session 2026-10-24 (batch 1, surrogate rulings; operator may overrule)

- Q1 Fix: (a) cap the spine3 segment's lateral half-extent at the rest shoulder half-span minus
  a 5 cm shoulder clearance, keeping all arm vertices (upper arm included); (b) test only
  forearm / hand vertices (18–23); (c) drop arm vertices within 10 cm of a shoulder joint.
  **Ruling (a)**: it removes the cause (a torso box reaching past the shoulder) instead of
  ignoring the upper arm; an upper arm driven through the chest is still caught. (b) passes the
  safe controls too but misses upper-arm penetration; (c) still flags arms-down 60° (24 vertices)
  and arms-forward (10).
- Q2 Clearance value: **5 cm** (≈ SMPL upper-arm radius at the shoulder). Sensitivity on 2,000 v1
  poses: 0 cm 18.5 %, 3 cm 3.1 %, 5 cm 2.1 %, 8 cm 1.8 %: the plateau starts at 5 cm.
- Q3 Per-pose rule: a pose intersects iff at least `ceil(0.004 × arm_vertices)` arm vertices are
  inside (SMPL: 11 of 2,559; synthetic cube body: 1 of 64), so 1–10 vertices of side contact /
  skinning blend do not count. **Ruling: yes** (v1: 2.15 % → 1.75 %).
- Q4 Gate status: **diagnostic → gate again** at the unchanged 5 % (`MAX_INTERSECTING_FRACTION`),
  since v1 / v2d / v3 pass with margin and `armsdown-v1` (≈ 23 %) fails as 014 intended.
- Q5 Pose table: **unchanged** (diagnosis (a); residual 1.8–2.2 % < 5 %). If the operator wants
  the residual gone anyway, the option would be narrowing spine1 y (±75° → ±45°) — not designed
  here, operator-gated.

### Session 2026-10-24 (operator rulings)

- Operator ruling 1: the recalibrated check IS a hard 5 % gate again for realism datasets (v2a
  etc.), confirming Q4 (`MAX_INTERSECTING_FRACTION` unchanged, exit 1 on FAIL).
- Operator ruling 2: no pose-table change (spine1 y range stays), confirming Q5.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A check that means something on real SMPL (Priority: P1)

The researcher runs `scripts/intersection_check.py` on a procedural config with the real body and
gets a fraction that separates clean poses from interpenetrating ones: the rest pose and known
safe poses count 0, obviously crossed limbs count hundreds of vertices.

**Independent Test**: synthetic-body tests (CI) plus a real-SMPL calibration test skipped when
`assets/smpl/SMPL_NEUTRAL.npz` is absent.

**Acceptance Scenarios**:

1. **Given** the real body, **When** the controls of FR-004 are counted, **Then** safe ones are
   0 and crossed ones exceed their minimums.
2. **Given** `configs/smpl_proc_v2d/generate.yaml --seed 0`, **When** the script runs, **Then**
   it prints 89 / 4000 = 2.23 %: PASS (exit 0).
3. **Given** an `armsdown-v1` config (v2a), **Then** FAIL (exit 1), ≈ 23 %.

### Edge Cases

- A body whose capped spine3 half-width is ≤ 0 → `ValueError` "half-extent" (as 014 degeneracy).
- Rest pose with any vertex inside → INVALID (exit 2), unchanged.
- Synthetic cube body: the cap (0.18 − 0.05 = 0.13 m) only binds on the widened torso (0.14 m);
  existing 014 tests keep their assertions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** `TorsoVolume` takes the rest joints; spine3 (joint 9) half-x =
  `min(aabb_half_x − margin_m, |J16_x − J17_x| / 2 − SHOULDER_CLEARANCE_M)` with
  `SHOULDER_CLEARANCE_M = 0.05`; other segments unchanged; arm set unchanged (16–23).
- **FR-002** A pose intersects iff its inside count ≥ `min_inside_vertices(arm_vertices) =
  max(1, ceil(MIN_INSIDE_FRACTION × arm_vertices))`, `MIN_INSIDE_FRACTION = 0.004`.
  `Survey.intersecting` / `fraction` use it; `counts` stay raw.
- **FR-003** Script report gains `shoulder_clearance_m`, `spine3_half_x_m`,
  `min_inside_vertices`, `check_version: "017"`; `worst_poses` lists only intersecting poses;
  new flag `--shoulder-clearance-m` (default 0.05); gate default 5 % unchanged.
- **FR-004** Reference controls (real SMPL, betas 0; Euler xyz degrees via
  `euler_xyz_to_matrix` → `matrix_to_axis_angle`): must be 0 inside — rest; arms down 60°
  (16 z −60, 17 z +60); arms forward 90° (16 y −90, 17 y +90); hands on hips (arms down 60° +
  18 y −100, 19 y +100). Must intersect — left forearm through chest (16 y −90, 18 y −150) ≥ 500;
  left arm through belly (16 z −90, 18 y −30) ≥ 100; both crossed deep (16 y −80, 17 y +80,
  18 y −150, 19 y +150) ≥ 1000.
- **FR-005** Docs: 014 spec gets a "superseded by 017" note on FR-004; HANDOFF's intersection
  lines state the recalibrated rule and rates; `scripts/intersection_check.py` docstring updated.

### Key Entities

`TorsoVolume`, `Survey` (data-model.md).

## Success Criteria *(mandatory)*

- **SC-001** Real SMPL, seed 0, betas 0: v1 (2,000) 35 = 1.75 %; v2d (4,000) 89 = 2.23 %;
  v3 (8,000) 179 = 2.24 %; `armsdown-v1` (2,000) 460 = 23.0 % (014 rule: 80.3 / 81.1 / 80.7 /
  100 %). Tolerance for the implementer's rerun: ±2 poses per set (report, don't tune).
- **SC-002** FR-004 controls hold; full suite green; nothing under `assets/` committed.
