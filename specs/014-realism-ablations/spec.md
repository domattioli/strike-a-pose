# Feature Specification: Realism Ablations (amendment to 010)

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `014-realism-ablations`)
**Created**: 2026-10-21
**Status**: Approved by operator decision 2026-10-21 (every requirement below follows from it;
no clarification batch). The open item is the winners' dataset (FR-008), pending.
**Input**: Operator decisions 2026-10-21, relayed by the orchestrator: (1) the realism datasets
are independent single-change datasets, each compared with v1 (no cumulative chain):
`smpl_proc_v2a` = v1 + arms-down table, `v2b` = v1 + body shape, `v2c` = v1 + camera jitter,
`v2d` = v1 + 4,000 poses (+ `context_pca_k: null`, `image_pca_ks: []`); (2) the arms-down
table `predecessor-v2-armsdown` is renamed `armsdown-v1` (it is not the predecessor's table)
and its confound is documented; (3) an automatic limb–torso intersection count with a
threshold replaces "visual check only": v2a's split is not frozen if more than 5 % of its poses
intersect; the visual check stays as an additional step; (4) the campaign procedure (HANDOFF
and 012 `tasks.md`, identical), the 010 documents, README, roadmap and HANDOFF are updated; the
winners' dataset is "pending operator decision; default v2a". Slice 013 (capsule body builder)
is not in this tree; nothing here depends on it.

Settled context (checked at `e131cc7`):

- **Configs.** `configs/smpl_proc_v{2,3,4,5}/{generate,baselines,ae,probe}.yaml` form 010's
  cumulative chain (010 FR-010, surrogate ruling Q1 (a)). Their `ae.yaml` lack the 012 guard
  keys (`early_stop`, `abort_on_divergence`, `divergence_ratio`), which `configs/ae.yaml` sets;
  `tests/test_realism.py::test_series_run_configs_match_v1` exempts them (012 T012). No dataset,
  split or run of any of them exists (`splits/` holds `smpl_fixture.json` and
  `smpl_proc_v1.json`; nothing under `docs/results/` or `data/generated/` names the table).
- **Table.** `datagen.procedural.ARMSDOWN_VERSION = "predecessor-v2-armsdown"`: the v1 table
  with shoulder z (16, 17) narrowed `[-50, 50)` → `[-20, 20)`, collar z (13, 14) `[-30, 30)` →
  `[-10, 10)`, and offsets z −65° (16) / +65° (17) after sparsification (010 FR-003). The name
  says "predecessor", but the predecessor never had this table; the narrowing is a lead default
  of 010 (clarification 6), not part of the operator's "arms down".
- **Intersection.** 010 has no collision model: its edge case says "Arm–body interpenetration
  is not detected"; FR-003 bounds it by the table and a visual check of 16 poses with real SMPL.
  010 clarification 6 measured a joint-level proxy on the fixture skeleton (elbow / wrist / hand
  within a box around the torso axis): 0.5 % / 0.1 % (v1, L / R) vs 1.8 % / 1.0 % (the chosen
  table) over 2,000 poses; no mesh was used.
- **Body model.** `SMPLModel.forward(pose, betas) → (verts, joints)`; `weights` (V × 24) are
  the skinning weights. The synthetic test body (`tests/conftest.py`) is one 6-cm cube per joint
  on a rough T-pose skeleton, each cube's 8 vertices weighted 1 to its joint; its torso (cubes at
  joints 0, 3, 6, 9) is far thinner than a human torso.
- **Memory / code for v2d.** The 012 batched mean image is in place (peak ≈ 3 GB AE with
  `context_pca_k: null`, ≈ 6 GB baselines with `image_pca_ks: []`; 012 research R4).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Four independent realism datasets (Priority: P1)

The researcher generates v2a, v2b, v2c and v2d from committed configs; each differs from
`smpl_proc_v1` in exactly one change, so each result is compared with R0 and attributes one
effect.

**Independent Test**: each `generate.yaml` differs from `configs/default.yaml` in
`dataset_name` and one key; each run config differs from its v1 counterpart only in names and
paths (v2d: plus the two PCA keys); each AE config carries the 012 guards; each generates a
valid dataset with the synthetic body.

### User Story 2 - An honest table name (Priority: P1)

`ranges_version: armsdown-v1` selects the arms-down table; the old name is rejected.

**Independent Test**: config accepts `armsdown-v1`, rejects `predecessor-v2-armsdown`; the
manifest and labels name `armsdown-v1`; the table itself is unchanged.

### User Story 3 - Automatic limb–torso intersection count (Priority: P1)

Before v2a's split is frozen, `scripts/intersection_check.py --config
configs/smpl_proc_v2a/generate.yaml --seed 0` poses the dataset's own poses with the real body
model and reports the fraction of poses with an arm inside the torso; above 5 % it fails and
the split is not frozen.

**Independent Test**: synthetic body: rest pose → 0; a pose that drives the left elbow into the
torso → > 0, its mirror for the right arm → > 0; a rigid motion of the whole body leaves the
count unchanged; the script on a synthetic config prints the rate, passes / fails against
`--threshold`, writes nothing, and gives the documented exit codes.

### Edge Cases

- AMASS configs → the script refuses (exit 2): only the procedural table is checked.
- A config with `body_shape`: each pose is checked with its own betas (the dataset's stream), its
  rest shape with the same betas.
- The rest pose of the body model itself intersects (the approximation does not fit that body)
  → the check is invalid: exit 2, nothing reported as pass or fail.
- `--num-poses N` smaller than the config's: the first N poses of the dataset (prefix, 010
  FR-006).
- A torso segment whose rest half-extent does not exceed the margin, or with fewer than 3
  vertices → error (exit 2).
- Arm–thigh contact is not counted (thighs are not torso); arm–head contact neither.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (configs): `configs/smpl_proc_v{2,3,4,5}/` MUST be replaced by
  `configs/smpl_proc_v2{a,b,c,d}/{generate,baselines,ae,probe}.yaml`:

  | Dataset | `generate.yaml` = `configs/default.yaml` + | Run configs |
  |---|---|---|
  | `smpl_proc_v2a` | `ranges_version: armsdown-v1` | copies of v1's |
  | `smpl_proc_v2b` | `body_shape: {std: 1.0, clip: 2.0}` | copies of v1's |
  | `smpl_proc_v2c` | `camera_jitter: {height_m: [-0.3, 0.3], distance_m: [0.0, 0.5], pitch_deg: [0.0, 5.0], margin_px: 2, max_attempts: 16}` | copies of v1's |
  | `smpl_proc_v2d` | `num_poses: 4000` | + `image_pca_ks: []` (baselines), `context_pca_k: null` (AE) |

  Run configs are copies of `configs/baselines.yaml`, `ae.yaml` (including its 011 variant keys
  at their defaults and its 012 guards) and `probe.yaml`, with `run_name`, `dataset_dir`,
  `split_file` (and the probe's placeholder `checkpoint`) set for the dataset. Seed 0; splits
  `pose-permutation-v1`, seed 0, default fractions (v2a / v2b / v2c get v1's pose lists; v2d
  3,200 / 400 / 400).
- **FR-002** (config test): the 010 series tests MUST be rewritten for "each v2x differs from v1
  only in dataset name / paths and its one key": generation configs against
  `configs/default.yaml`; run configs against `configs/{baselines,ae,probe}.yaml` with no guard
  exemption; the set of `configs/smpl_proc_*` directories is exactly v2a–v2d.
- **FR-003** (rename): `ARMSDOWN_VERSION` MUST become `armsdown-v1`; the table (ranges and
  offsets) is unchanged; `predecessor-v2-armsdown` is no longer accepted (config error, exit 2).
  No alias: no dataset, split or result with the old name exists (checked: `grep`).
- **FR-004** (intersection function): `datagen/intersection.py` (NumPy only) MUST count, for a
  posed body, the arm vertices inside the torso volume:
  - arm vertices: dominant skinning weight (argmax over joints) in 16–23 (upper arm, forearm,
    hand, fingers of both sides); torso segments: dominant joint 0, 3, 6, 9 (pelvis, spine1–3);
  - each torso segment's volume is the elliptic cylinder inscribed in the axis-aligned bounding
    box of its rest-pose vertices (axis along y, semi-axes along x and z, height the box's y
    extent), each half-extent shrunk by `margin_m` (default 0.01 m);
  - per segment, the posed arm vertices are mapped into the segment's rest frame by the inverse
    of the best-fit rigid motion of its vertices (rest → posed, Kabsch / orthogonal Procrustes,
    proper rotation);
  - an arm vertex counts once if it lies inside any segment's volume.

  Superseded in part by spec 017 (2026-10-24): spine3 capped at the shoulders − 5 cm, ≥ 0.4 % of
  arm vertices per pose; real-SMPL rates in 017 research R4.
  Rationale (research R1): human torso cross-sections are close to ellipses wider than deep;
  inscribing the ellipse in the box excludes the box corners; per-segment rigid frames follow
  spine bending; Kabsch uses only vertices and skinning weights (no model internals, works for
  any LBS body); the margin ignores grazing contact and the skinning blend at segment borders.
  The test is a lower bound in the chest-to-armpit region (the inscribed ellipse is smaller than
  the true section) and ignores thighs and head.
- **FR-005** (survey): one function MUST pose a list of poses (with per-pose betas) and return the
  per-pose counts, the rest-pose count and the fraction of poses with a count ≥ 1.
- **FR-006** (script): `scripts/intersection_check.py --config CFG --seed S [--num-poses N]
  [--smpl-model PATH] [--threshold 0.05] [--margin-m 0.01] [--json]` (argparse; read-only) MUST
  sample the config's procedural poses exactly as `generate` does (`sample_procedural_poses(N,
  S, ranges_version)`, betas from the dataset's stream when `body_shape` is set), and print: the
  config, table, seed, number of poses, body model file and SHA-256, the joint sets and margin,
  the rest-pose count, poses with an intersection / N and the fraction, PASS or FAIL against the
  threshold, the median and maximum count over intersecting poses and the five worst pose
  indices. Exit 0 pass; 1 fail (fraction > threshold); 2 config error, AMASS source, invalid
  check (rest pose intersects, degenerate segment); 3 missing body model.
- **FR-007** (procedure): the v2a step MUST run the script with threshold 5 % after `validate`
  and before `split`; FAIL → v2a's split is not frozen, the dataset is not used, the outcome is
  recorded, the step stops and the campaign continues with v2b (amending the table needs an
  attended session and a new version string, e.g. `armsdown-v2`). PASS → the visual check
  (`Front_0`, `Right_90` renders of poses 0–15, recorded in 001 research R9 with the script's
  rate) as an additional step, then the split.
- **FR-008** (campaign): the campaign procedure in `docs/HANDOFF.md` and
  `specs/012-run-guards/tasks.md` MUST stay identical and describe v2a–v2d as independent, each
  compared with R0, with their committed configs, the v2a intersection gate, v2d's batched-mean
  and no-PCA notes, and the W step as "pending operator decision: which of v2a–v2d (or v1 only)
  the winning variants run on; default if no answer: v2a".
- **FR-009** (documentation, same slice): see *Documentation scope*.
- **FR-010** (tests, written first; pytest, synthetic body, no SMPL): (a) configs (FR-002);
  (b) rename (accept / reject, manifest name); (c) intersection: rest pose 0; a constructed left
  and right arm-into-torso pose > 0; whole-body rigid motion invariance; margin monotonicity;
  degenerate segment error; Kabsch recovers a known rotation; (d) survey and script on synthetic
  configs: rate printed, `--json` fields, exit 0 / 1 by threshold, exit 2 for AMASS, exit 3 for
  a missing model, nothing written; (e) a widened-torso synthetic body: the arms-down table's
  rate exceeds v1's.
- **FR-011** (constraints): no new dependency; Python 3.11-compatible; `uv run --locked` only;
  `docs/results/`, `splits/`, `pyproject.toml`, `uv.lock`, `.github/` untouched; generator
  output for every config without `ranges_version: armsdown-v1` byte-unchanged (no other code
  path changes); the CLI unchanged.

### Documentation scope (FR-009)

- 010 `spec.md`: dated notes "amended 2026-10-21 by slice 014" at the status, US1, US5, the
  interpenetration edge case, FR-001, FR-003 (rename + confound), FR-010 (independent datasets,
  replacing the chain), the run procedure, clarifications 2 and 6 and batch Q1 (original text
  kept); 010 `tasks.md` Follow-up R rewritten for v2a–v2d and the gate (dated); 010 `data-model.md`
  and `research.md` R1: dated notes.
- 001 `spec.md` FR-002 note, `data-model.md` row, `research.md` R9: the new name.
- `README.md`, `specs/roadmap.md` (#15 line, 010 row, changelog entry 2026-10-21 appended),
  `docs/HANDOFF.md` (state, sign-off list, next work, campaign procedure).
- Not edited: `docs/results/`, 012 `spec.md`, 001–013 `plan.md`.

## Success Criteria *(mandatory)*

- **SC-001**: ruff check, ruff format --check and the full pytest green; pytest grows by ≤ 10 s.
- **SC-002**: every FR-010 case passes; the script reports for `configs/default.yaml` and
  `configs/smpl_proc_v2a/generate.yaml` on the synthetic body (rates recorded in `tasks.md`,
  with the caveat that the synthetic torso is not a human torso).
- **SC-003**: the campaign procedure copies are identical (script check) and their shell blocks
  pass `bash -n`.
- "Done" does not depend on SMPL, a generated dataset or a run.

## Assumptions

- The 5 % threshold is the operator's; the count's absolute level on real SMPL is unknown until
  the v2a step runs it (the joint-level proxy of 010 clarification 6 predicts ≈ 1–2 % per arm).
- The SMPL skinning weights make 16–23 dominant on the arms and 0 / 3 / 6 / 9 on the torso
  (SMPL's part segmentation); the script prints the vertex counts per set so a session can see
  a surprise.

## Clarifications

### Session 2026-10-21 (operator decisions; lead defaults marked)

1. Q: Chain or independent datasets? → A: independent v2a–v2d, each vs v1 (operator; replaces
   010 surrogate ruling Q1 (a)).
2. Q: Table name? → A: `armsdown-v1`, no alias (operator; nothing was generated with the old
   name).
3. Q: The confound? → A: documented, not removed: v2a changes two things relative to v1 — the arms
   are lowered (the intended change) and shoulder / collar z vary less (±20° / ±10° instead of
   ±50° / ±30°), i.e. less pose variety; and lowered arms overlap the torso in more views. A v2a
   effect is the sum of the three (operator: document).
4. Q: Intersection test? → A: FR-004 (per-segment inscribed elliptic cylinders in Kabsch-fitted
   rest frames, 1 cm margin). Considered: the convex hull of the torso vertices (needs SciPy or
   a hand-written hull; fills the waist and, when the spine bends, the space beside the torso,
   counting arm vertices that are outside the body); a nearest-torso-vertex normal sign test (wrong on coarse meshes such as the synthetic
   cubes, where a point just outside a face is "behind" its nearest corner's normal); spherical
   capsules on the spine (one radius cannot fit a torso 1.7× wider than deep). *(lead default)*
5. Q: What counts as an intersecting pose? → A: at least one arm vertex inside (after the 1 cm
   margin); the per-pose counts are printed so a near-threshold result can be judged.
   *(lead default)*
6. Q: Script or CLI subcommand? → A: `scripts/intersection_check.py` (the 001 CLI contract stays
   unchanged, as `scripts/run_status.py`). *(lead default)*
7. Q: Gate failure? → A: v2a not split, step stopped and reported, the campaign continues with
   v2b (a table amendment is code; check-ins do not write code). *(lead default)*
8. Q: Winners' dataset? → A: pending operator decision; default if no answer: v2a (operator).
