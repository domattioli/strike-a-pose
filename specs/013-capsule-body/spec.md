# Feature Specification: Capsule Body (interim license-free body model)

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `013-capsule-body`)
**Created**: 2026-10-21
**Status**: Approved at surrogate scope (surrogate rulings 2026-10-21; operator sign-off
pending). Rulings: Q1 mirror the campaign with independent single-change datasets, Q2 (a),
Q3 (b). Staged implementation: the body builder first (branch `wip-013`); configs and the
README / roadmap / HANDOFF edits (FR-008 to FR-010) after slice 012 and the 010 amendments land.
**Input**: Operator decision 2026-10-21, relayed by the orchestrator: the licensed SMPL model is
still unavailable (the private repo `domattioli/smpl-assets` is empty), so the project gets an
interim, license-free body, "a capsule humanoid built on SMPL's own skeleton", so that the
whole pipeline and the compute campaign (generation, split, baselines, AE with the 008 / 011 /
012 options, probe, region metrics, the 010 realism options) can run now on a dataset
`capsule_proc_v1`. Its results are labelled as infrastructure tests, not README headline
numbers. When SMPL arrives, the real campaign runs unchanged.
Orchestrator brief: spec and one batch of ≤ 3 questions first; the body is an SMPL-format `.npz`
the unchanged generator loads; rest skeleton recovered from the committed fixture labels; one
capsule per bone; deterministic build with a pinned SHA-256; a CLI subcommand; configs for
`capsule_proc_v1`; render sanity tests; no new dependency; constitution VI; CI stays fast.
Slice 012 (`run-guards`) is being implemented in the same working tree.

Settled context (checked at `e98134e` with read-only experiments in the locked environment,
numpy 2.5.4, Python 3.13; the prototype body lives in the session scratchpad, not in the repo):

- **Body interface** (`datagen/smpl.py`). `SMPLModel.load` reads `v_template [V,3]`,
  `shapedirs [V,3,S]`, `posedirs [V,3,207]`, `J_regressor [24,V]`, `weights [V,24]`,
  `kintree_table [2,24]`, `f [F,3]` from an `.npz` (`allow_pickle=False`), casts them to
  float64 / int64 and checks only that the regressor has 24 rows and the tree 24 columns:
  nothing requires regressor vertices to lie on faces or weights to be smooth. `forward`:
  `v_shaped = v_template + shapedirs[..., :nb] @ betas[:nb]`, `j_rest = J_regressor @
  v_shaped`, pose blend `posedirs @ (R[1:] − I)`, rigid chain `G_i = G_p · [R_i | j_rest[i] −
  j_rest[p]]`, LBS. Joint i's rotation moves the vertices weighted to i, i.e. the bones from i
  to its children: a bone p → c must be skinned to **p**. `num_shape_components = S`; 010
  refuses `num_betas > S` when `body_shape` is set, and every config uses `num_betas: 10`.
- **Generator** (`datagen/generate.py`). Loads the body from `smpl_model_path`; the manifest
  records `inputs.smpl_model = {filename, sha256 of the file bytes}`; nothing else is
  body-specific (no code names `smpl_proc` or `SMPL_NEUTRAL` except the config default). A
  missing file exits 3 pointing at `docs/data-assets.md`. Split files (007) pin the manifest
  *content*, which includes that digest: a body file that is not byte-reproducible would make a
  regenerated capsule dataset fail its own split.
- **Rasterizer** (`datagen/render.py`). Coverage uses barycentrics divided by the signed area
  (either winding), shading `|n · view|` (either orientation): winding and normals are
  irrelevant; interpenetrating closed parts are resolved by the z-buffer. Cost: faces are
  sorted by bounding-box area and processed in chunks of 4,096 as arrays of (faces in chunk) ×
  (largest box in chunk), so a few long faces make every render slow (measured below).
- **Rest skeleton** (fixture). Undoing the chain on the committed labels
  (`j_rest[0] = J[0]`, `j_rest[i] = j_rest[p] + G_pᵀ (J[i] − J[p])`, `G` from the label's
  `pose`) gives the same 24 rest joints for both fixture poses (max difference 2.7 µm; the mean,
  rounded to 1 µm, is within 1.6 µm of each). T-pose, metres, +Y up, facing +Z: pelvis
  (−0.001795, −0.223333, 0.028219); hips (±0.07, −0.315); knees y −0.69; ankles y −1.09;
  foot joints (balls) y −1.144, z +0.09; neck y 0.288; **head joint y 0.353** (the brief's
  "≈ 0.44" is where the head sphere's centre goes, not the joint); shoulders (±0.17, 0.226);
  elbows x ±0.43; wrists ±0.68; hand joints ±0.77. Left / right mirror differences up to 7 mm
  are SMPL's own (kept).
- **Prototype** (28 capsules on that skeleton, rigid skinning, 24 marker vertices; the FR-002
  table). Unsubdivided capsules (rings only in the caps; 2,096 vertices) render at **370 ms**
  per 256 × 192 sample (SMPL: ≈ 18 ms) because of the long cylinder faces; with cylinder rings
  ≤ 0.03 m apart, 16 segments and 4 cap rings: 5,952 vertices, 11,744 faces, longest edge
  0.073 m, render 19 ms, `forward` 3.4 ms, 30 ms per sample end to end through
  `generate_dataset` (≈ 8 min for 16,000 samples). On the 16 fixture samples (same pose and
  camera, β = 0): joints equal the labels to 1.1 µm; every keypoint lies inside the capsule
  silhouette (24 / 24 in all 16); IoU with the fixture's real-SMPL mask 0.73–0.82 (mean 0.77);
  body area 7.45 % of the pixels (SMPL 7.46 %; per view 0.97–1.06×); margins top / bottom / left / right ≥ 44 / 2 /
  14 / 17 px (SMPL 44 / 8 / 16 / 19). Ring cameras, 2,000 procedural poses each: no joint ever
  leaves the image at β = 0; samples with some vertex outside the frame 0.9 % (v1 table) /
  0.7 % (arms-down), 2.6 % / 2.2 % with `body_shape` (2 samples with a joint outside); a
  16-pose dataset validates (SC-003 invariant) with no empty mask.
- **Same poses, cameras, split, joints.** A copy of `configs/default.yaml` with only the dataset
  name and body path changed draws exactly `smpl_proc_v1`'s poses (seed 0, same table) and
  cameras; `make_split` depends only on the pose indices and the seed (a 2,000-pose dataset,
  seed 0, gives `splits/smpl_proc_v1.json`'s pose lists exactly); at β = 0 the joint labels
  equal SMPL's up to the µm-level skeleton error. Pose-level numbers (003 mean pose / pose PCA,
  006 oracle references) therefore reproduce the published `smpl_proc_v1` ones; image-level
  numbers do not compare.
- **File determinism.** `np.savez` (uncompressed) is byte-deterministic: zip entries are dated
  1980-01-01, keys keep their order, and the stdlib zip container is identical under Python
  3.11, 3.12 and 3.13 (checked). `np.savez_compressed` depends on the zlib build (system vs
  python-build-standalone) and is avoided. CI's 3.11 leg resolves numpy 2.4.6, the 3.13 leg
  2.5.4: float results may differ in the last bit (libm / SIMD), so values are rounded before
  writing. `posedirs` zeros `[V,3,207]` cost 30 MB at V ≈ 6,000 in float64, 15 MB in float32.
- **Slice 012** (in progress here): adds guard keys to `configs/ae.yaml`, `ae_{mask,vae,reg}.yaml`,
  `configs/smpl_proc_v{2..5}/ae.yaml`, a new `configs/ae_vae_beta0.01.yaml`, a "Campaign
  procedure" section in `docs/HANDOFF.md`, `scripts/run_status.py`, and freezes
  `datagen.config.ADDED_FIELD_DEFAULTS`. 013 adds no generator field and copies the configs as
  they stand after 012.

**License position** (lead assessment, not legal advice; surrogate ruling Q3 (b): the skeleton is
derived at build time from the committed fixture, no SMPL-derived table enters `src/`; **the
operator, as the SMPL licensee, should confirm this reading**). The capsule body uses one SMPL-derived quantity: the 24 rest-joint
positions of the neutral mean body (72 numbers, `J_regressor · v_template`). The repository
already publishes them implicitly: every committed fixture label carries the posed joints and
the pose (derived renders and labels, approved by the operator in 001 Q8), and ten lines of
code recover them. No SMPL array (template vertices, faces, blend shapes, regressor, skinning
weights) is read, copied or approximated; the mesh, radii, skinning, regressor and shape
directions are authored here. The SMPL model license restricts the model and software
(non-commercial research use; no reproduction, modification or redistribution of the model);
72 joint coordinates are model output of the same kind as the committed labels, so the table
stands on the same footing as the fixture. The generated `.npz` is never committed
(`/assets/` is git-ignored) and is named and documented as not SMPL.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build the interim body (Priority: P1)

The operator runs one command and gets `assets/synthetic/capsule_v1.npz`, byte-identical on
every machine with the locked environment, which the unchanged generator loads as its body.

**Why this priority**: operator decision; every other story needs the file.

**Independent Test**: build twice in memory, compare bytes and the pinned SHA-256; load with
`SMPLModel`; CLI exit codes.

**Acceptance Scenarios**:

1. **Given** a clean checkout, **When** `strike-a-pose-data make-capsule-body` runs, **Then** the
   file is written, its SHA-256 equals the digest pinned in the code, and both are printed.
2. **Given** the file already exists with the same bytes, **When** the command reruns, **Then**
   it exits 0 ("unchanged"); with other bytes it exits 4 unless `--overwrite`.
3. **Given** the body, **When** it is posed with any pose at β = 0, **Then** its joints are the
   forward kinematics of the SMPL rest skeleton (the fixture poses: the labels' joints within
   1e-5 m), and each capsule moves rigidly with its joint.

### User Story 2 - A capsule dataset through the unchanged pipeline (Priority: P1)

The operator generates, validates and splits `capsule_proc_v1` and runs baselines, AE and probe
from `configs/capsule_proc_v1/`, with no code path that differs from the SMPL campaign's.

**Why this priority**: the operator's purpose (the pipeline and campaign run now).

**Independent Test**: generation on the in-memory body (small image) → `validate_dataset` = [],
non-empty masks, keypoints in frame; full-size fixture renders against the real-SMPL masks.

**Acceptance Scenarios**:

1. **Given** the body and `configs/capsule_proc_v1/generate.yaml`, **When** generated with seed
   0, **Then** the dataset validates (SC-003 projection invariant), `inputs.smpl_model` names
   `capsule_v1.npz` with the pinned digest, and its poses and cameras are `smpl_proc_v1`'s.
2. **Given** that dataset, **When** split with seed 0, **Then** the pose lists equal
   `splits/smpl_proc_v1.json`'s.
3. **Given** its 003 baselines, **Then** the pose rows (mean pose, pose PCA) reproduce the
   published `smpl_proc_v1` pose numbers at the README's precision (SC-005).

### User Story 3 - Body shape does something (Priority: P2)

With `body_shape` on (the 010 chain from v3), the capsule's betas change its size, limb
thickness and torso width, and bone lengths follow the size component.

**Why this priority**: the 010 shape option must be exercised, not silently inert.

**Independent Test**: unit checks of the three shape components on the built arrays.

**Acceptance Scenarios**:

1. **Given** β0 = ±2, **Then** every vertex and joint scales by 1 ± 0.08 about the world origin.
2. **Given** β1 or β2 ≠ 0, **Then** limb radii or torso width change and no joint moves.

### User Story 4 - The campaign runs now, labelled as infrastructure (Priority: P2)

A session (or the 012 heartbeat check-in) runs the capsule campaign by the 012 procedure with a
substitution table; its records say "infrastructure test, capsule body" wherever they appear.

**Why this priority**: operator decision; the campaign rehearses the 012 guards, rules and
check-ins on full-scale runs before the SMPL campaign.

**Independent Test / Acceptance**: the twin configs load and differ from their SMPL twins only
in the substituted keys (test); the procedure lists commands, order, cost, records and the
SMPL-arrival rule.

### User Story 5 - SMPL arrives: nothing to change (Priority: P1)

When `SMPL_NEUTRAL.npz` becomes available, the SMPL campaign runs from F1 exactly as 012
documents; no SMPL config, split, result or code path was changed by 013.

**Why this priority**: operator decision ("the real campaign runs unchanged").

**Independent Test**: SMPL configs, fixture, `splits/`, `docs/results/` byte-unchanged; the
generator, trainer and evaluators are untouched apart from one CLI subcommand.

### Edge Cases

- `--out` exists: same bytes → exit 0 "unchanged"; other bytes → exit 4 unless `--overwrite`.
- The build's digest differs from the pin (a numpy / platform difference): exit 1, nothing
  written, the message names both digests and the numpy / Python versions.
- `capsule_v1.npz` missing at `generate`: exit 3 as for SMPL; `docs/data-assets.md` (the
  message's pointer) gains the build command.
- `num_betas` > 10 with `body_shape`: the existing 010 config error.
- Crops at the ring cameras (0.7–2.6 % of samples have a vertex outside; joints only with
  `body_shape`): kept and flagged by `visibility`, as with SMPL; `camera_jitter` rejects
  cropped candidates as before.
- Interpenetrating capsules (joints, arms down against the torso): accepted; the z-buffer draws
  the visible surface of the union.
- A future fixture regeneration with the same SMPL file and other poses recovers the same
  skeleton to µm (rounding may still flip a last digit): the pinned digest test reports any
  change, and a changed body is a new version (FR-005).
- SMPL arrives mid-campaign: lead default 13 (the SMPL campaign preempts).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (rest skeleton, surrogate ruling Q3 (b)): `src/strike_a_pose/datagen/capsule.py`
  MUST derive the 24 rest-joint positions at build time from the committed fixture labels
  (`tests/fixtures/smpl_fixture/samples/*.json`, the single source; no SMPL-derived literal in
  `src/`): per pose, the recovery above from its labels (all views of a pose MUST carry the same
  `joints_3d_world` and `pose`, else an error); the mean over the poses in ascending pose index,
  rounded to 1e-6 m, not symmetrized. The fixture directory is a builder argument (CLI
  `--fixture`, default `tests/fixtures/smpl_fixture`); a missing or unreadable fixture is a
  missing-asset error (exit 3). Tests MUST check that the poses agree within 5 µm and that the
  result has the documented landmarks (pelvis, ankles, head joint).
- **FR-002** (segments and mesh): the v1 body MUST be 28 capsules. A capsule runs from point A to
  point B (joint positions, or a joint plus a fixed rest-frame offset for the head, hand and foot
  segments), has an elliptical section with semi-axes `(r1, r2)` (`r2` along world +Z made
  perpendicular to the axis, or world +Y when the axis's z component is ≥ 0.9 in magnitude;
  `r1` along their cross product) and hemi-ellipsoidal caps centred on A and B with axial radius `max(r1, r2)`;
  16 vertices per ring, 4 latitude rings per cap plus a pole, cylinder rings at most 0.03 m
  apart; each capsule a closed 2-manifold with consistent winding (hygiene; the renderer does
  not need it). The table (lead default 6; tunable once, before the digest is pinned, to meet
  SC-003; left / right mirrored by joint):

  | Segment(s) | Skin joint | From → to | `r1 × r2` (m) |
  |---|---|---|---|
  | pelvis → hips | 0 | 0 → 1, 0 → 2 | 0.085 |
  | lower / mid / upper spine | 0, 3, 6 | 0 → 3, 3 → 6, 6 → 9 | 0.13 × 0.10, 0.13 × 0.10, 0.135 × 0.10 |
  | chest → neck, chest → collars | 9 | 9 → 12, 9 → 13, 9 → 14 | 0.06; 0.08 |
  | collars → shoulders | 13, 14 | 13 → 16, 14 → 17 | 0.06 |
  | neck | 12 | 12 → 15 | 0.055 |
  | head | 15 | head joint + (0, 0.07, 0.02) → + (0, 0.08, 0.02) | 0.095 (crown ≈ +0.53 m) |
  | upper arms, forearms | 16 / 17, 18 / 19 | 16 → 18, 18 → 20 (and right) | 0.05; 0.04 |
  | palms | 20, 21 | 20 → 22, 21 → 23 | 0.018 × 0.04 (flat) |
  | fingers | 22, 23 | hand joint → + (±0.08, −0.005, 0) | 0.015 × 0.04 |
  | thighs, shins | 1 / 2, 4 / 5 | 1 → 4, 4 → 7 (and right) | 0.075; 0.05 |
  | heel → ball | 7, 8 | ankle + (0, −0.06, −0.06) → foot joint + (0, −0.02, 0) | 0.04 × 0.03 |
  | toes | 10, 11 | foot joint + (0, −0.02, 0) → + (0, −0.03, 0.08) | 0.04 × 0.025 |

  Measured with this table: 5,952 vertices (24 markers included), 11,744 faces, soles ≈ −1.20 m
  and crown ≈ +0.53 m (010's estimates of SMPL's: −1.20 / +0.52).
- **FR-003** (skinning, regressor, pose blend, tree): every vertex of a segment MUST have weight
  1 on the segment's skin joint (the proximal joint p of a bone p → c; the leaf joint for head,
  fingers and toes; the ankle for heel → ball); no blending. Capsule caps centred on a shared
  joint stay concentric under any rotation, so the union is closed at every joint angle;
  blended LBS weights would only pinch the capsules. 24 marker vertices sit at the rest joints,
  belong to no face and have weight 1 on their own joint; `J_regressor` is one-hot on them
  (rows sum to 1), so `J_regressor @ v_template` is the FR-001 table. `posedirs` zeros
  `[V, 3, 207]`; `kintree_table = [[4294967295, parents 1..23], [0..23]]`.
- **FR-004** (shape components): `shapedirs` `[V, 3, 10]`: component 0 (size) = `0.04 · v` for
  every vertex and marker (scaling about the world origin, the ring's look-at height: β0 = ±2 →
  ±8 %, 1.59–1.86 m tall, thigh 0.347–0.407 m); component 1 (limb thickness) = `0.10 · (v − a)`
  for every segment but the pelvis, spine, chest, neck and head ones (skin joints 0, 3, 6, 9,
  12, 15), `a` the closest point of the segment axis
  (±20 % radius at ±2); component 2 (torso width) = `0.08 · ((v − a) · e1) e1` for the pelvis
  and spine segments (skin joints 0, 3, 6), `e1` the `r1` direction (±16 % width); components 3–9 zero (lead
  default 7). Components 1–2 move no marker, hence no joint.
- **FR-005** (file format, determinism): the writer MUST emit, with `np.savez` (uncompressed),
  the seven arrays in the order `v_template, shapedirs, posedirs, J_regressor, weights,
  kintree_table, f`; float arrays rounded to 1e-6 with −0.0 normalized to +0.0; dtypes fixed
  (float64; `posedirs` float32 zeros, exact; int64 for `f` and `kintree_table`); ≈ 19 MB. The
  module MUST pin `CAPSULE_V1_SHA256` (the file digest in the locked environment); any change
  to the body after that is a new version (`capsule_v2`, new datasets), never an edit of v1.
- **FR-006** (CLI): `strike-a-pose-data make-capsule-body [--out PATH] [--overwrite]`, default
  `--out assets/synthetic/capsule_v1.npz` (the path the configs name): builds in memory, compares
  the digest with the pin (mismatch: exit 1, nothing written), creates parent directories,
  writes, prints the path and digest. Existing file: identical bytes → exit 0 "unchanged";
  other bytes → exit 4 unless `--overwrite`. Not chosen: a `builtin:capsule_v1` value of
  `smpl_model_path` resolved by the generator (a generator change; provenance by code version
  instead of a file digest in the manifest).
- **FR-007** (generator untouched): `generate.py`, `config.py`, `smpl.py`, `render.py`,
  `cameras.py`, the dataset layout and its `schema_version` MUST stay as they are: a capsule
  dataset is a generation config whose `smpl_model_path` names the capsule file.
- **FR-008** (configs, surrogate ruling Q1; written last, after 012 and the 010 amendments
  land): twin configs under `configs/capsule_proc_<id>/`: for `capsule_proc_v1`, the generate
  config (twin of `configs/default.yaml`) and the run configs of the SMPL campaign on v1 (R0,
  R0b seed 1, R1 mask-weighted MSE, R2 VAE β 0.01, R3 regularization, baselines, probe, as the
  SMPL configs then name them); for the independent single-change datasets `capsule_proc_v2a`
  (arms-down table), `v2b` (body shape), `v2c` (camera jitter), `v2d` (4,000 poses), the twins
  of the corresponding SMPL dataset configs (generate, baselines, AE, probe). The final file
  list follows the SMPL configs as they exist after 012 and the 010 amendments. A twin
  differs from its SMPL twin only in `dataset_name` and `smpl_model_path:
  assets/synthetic/capsule_v1.npz` (generate) or in `run_name`, `dataset_dir`, `split_file`
  and the probe's placeholder `checkpoint` (`smpl_proc_vN` → `capsule_proc_vN` in each), with
  a header comment "infrastructure test: capsule body, not SMPL (spec 013)". Twins are written
  after 012 lands (they carry its guard keys).
- **FR-009** (labelling, per batch Q2, lead proposal (a)): capsule results MUST be recorded under
  `docs/results/013-capsule/`: a `README.md` that opens with "Infrastructure tests on the
  interim capsule body (spec 013); not SMPL; image-level numbers are not comparable with any
  SMPL result", a numbers table and the run list, and one subdirectory per run with the usual
  copies (`config.yaml`, `metrics.json`, `run_info.json`, figures). The main README carries no
  capsule number: a status-table row, a Usage paragraph and one pointer line under Results.
  Campaign-log lines (012) for capsule steps are marked `capsule`.
- **FR-010** (procedure): the capsule campaign procedure (section below) MUST be documented in
  this slice's `tasks.md` and, identically, as a subsection of 012's "Campaign procedure" in
  `docs/HANDOFF.md`: the 012 procedure with the substitution table, F1 skipped, the capsule R0
  as the control of capsule runs, and the SMPL-arrival rule.
- **FR-011** (tests, written first; pytest, CPU, the v1 body built in memory, no reduced
  variant: build < 0.1 s, ≈ 20 ms per full-size render):
  (a) skeleton: FR-001 re-derivation; names / parents = `skeleton.py`; pelvis, ankle, head rows;
  (b) arrays: shapes and dtypes, V and F pinned (prototype 5,952 / 11,744), faces in range, markers in
  no face, `J_regressor` one-hot on the markers, `weights` one-hot rows, `posedirs` zero,
  `SMPLModel` loads it, `num_shape_components` 10;
  (c) zero pose: `forward(0, 0)` gives `v_template` and the table;
  (d) kinematics: the fixture poses' joints equal the labels within 1e-5 m; under seeded random
  poses every capsule moves rigidly (pairwise distances kept) and its cap centres land on the
  posed joints;
  (e) mesh: each capsule a closed 2-manifold (every edge in exactly two of its faces); longest
  edge ≤ 0.08 m (the render-cost bound);
  (f) shape: FR-004 values; joints unchanged by β1, β2; components 3–9 zero;
  (g) determinism: two builds byte-equal; the written file's SHA-256 equals `CAPSULE_V1_SHA256`
  (both CI legs); CLI: writes and prints, "unchanged" on rerun, exit 4 without `--overwrite`,
  overwrite works, exit 1 on a mismatching pin (pin monkeypatched), parent dirs created;
  (h) render sanity, full size (256 × 192, ring of 3.5 m, focal 320): the 16 fixture
  (pose, camera) pairs: non-empty masks, silhouette ≥ 2 px inside every border, every keypoint
  pixel inside the capsule mask, IoU with the fixture's SMPL mask ≥ 0.70 and body area within
  ±15 % of it, each; the first two
  procedural poses of seed 0 for both tables at the 8 ring cameras: non-empty masks, all
  keypoints inside the image;
  (i) end to end: `generate_dataset` on a capsule file (2 poses, 64 × 48, focal 80, as
  `small_cfg`), with and without `body_shape`: `validate_dataset` = [] (SC-003 invariant),
  `inputs.smpl_model` = the file's name and digest;
  (j) configs: every file under `configs/capsule_proc_v*/` loads with its loader and differs
  from its SMPL twin exactly as FR-008 states; the capsule generate configs generate on the
  capsule body (2 poses, small image) and validate.
  Existing tests unchanged.
- **FR-012** (constraints): no new dependency (`pyproject.toml`, `uv.lock` untouched); NumPy
  only; Python 3.11-compatible; `uv run --locked`; `.github/workflows/ci.yml` unchanged; no
  generated body, dataset or run committed (`/assets/`, `/data/`, `/runs/` ignored).

### Documentation scope

- `docs/data-assets.md`: section "Interim capsule body (no license needed)": what it is (and is
  not), the build command and digest, the configs, results = infrastructure tests.
- `specs/001-dataset-generation/contracts/dataset-layout.md`: CLI section gains
  `make-capsule-body` and its exit codes; one note: `smpl_model_path` may name any SMPL-format
  body, `inputs.smpl_model` then records it (no schema change).
- `specs/001-dataset-generation/spec.md`: dated note "amended 2026-10-21 by slice 013" on FR-002
  ("the neutral SMPL model" stays the body of every reported result; the capsule body serves
  infrastructure runs); original text kept.
- `README.md`: status row; Usage: build command, `configs/capsule_proc_v*/`, procedure pointer;
  Results: one pointer line to `docs/results/013-capsule/` (FR-009).
- `specs/roadmap.md`: slice row, status line, changelog entry 2026-10-21 (operator decision);
  SMPL access stays an open item.
- `docs/HANDOFF.md`: state (013), the capsule subsection of the campaign procedure (FR-010),
  next work (the capsule campaign can start without SMPL).
- Not edited: SMPL configs, `splits/`, the fixture, `docs/results/` outside `013-capsule/`,
  001–012 `plan.md` / `tasks.md`.

### Key Entities

- **Capsule body v1**: the generated `.npz` (SMPL format), identified by `CAPSULE_V1_SHA256`.
- **Rest skeleton table**: 24 × 3 SMPL-derived joint positions (FR-001).
- **Segment table**: the 28 capsules (FR-002), with skin joints and shape roles (FR-004).
- **Capsule dataset series**: `capsule_proc_v1` … (Q1), twins of `smpl_proc_v1` …
- **Infrastructure record**: `docs/results/013-capsule/` and capsule campaign-log lines.

## Success Criteria *(mandatory)*

- **SC-001**: ruff, ruff format, pytest and the CI smoke steps green locally and in CI (3.11 and
  3.13), including the digest pin on both legs.
- **SC-002**: pytest grows by ≤ 10 s on 4 vCPU.
- **SC-003**: fidelity on the fixture (FR-011 (d), (h)): joints within 1e-5 m of the labels, every
  keypoint inside the capsule silhouette, IoU ≥ 0.70 per view against the real-SMPL masks,
  silhouettes ≥ 2 px inside the frame, body area per view within ±15 % of SMPL's (prototype
  0.97–1.06×).
- **SC-004** (verification): in this container the full `capsule_proc_v1` builds, generates
  (≤ 15 min; ≈ 8 min expected) and validates, and its seed-0 split's pose lists equal
  `splits/smpl_proc_v1.json`'s; nothing of it is committed by the slice.
- **SC-005** (first campaign check, ≈ 3 min, may run in verification): the 003 pose rows on
  `capsule_proc_v1` reproduce the published `smpl_proc_v1` ones at the README's precision (mean
  pose 253.4 / 250.3 mm, PCA k = 8 94.0 / 93.6, k = 16 34.2 / 34.6; 2D px likewise).
- **SC-006**: SMPL configs, fixture, `splits/` and `docs/results/` byte-unchanged; only
  `datagen/cli.py` gains code outside the new module.

## Assumptions

- Capsule numbers are infrastructure tests: image-level metrics (MSE, PSNR, IoU, regions, AE,
  probe) are not comparable with SMPL's; pose-level numbers at β = 0 are (same joints). Faceted
  flat shading and mitten hands are accepted.
- numpy 2.4.6 (3.11) and 2.5.4 (3.13) write the same bytes for the same rounded arrays (the .npy
  v1.0 header and the zip container are stable; the pin test on both legs verifies). If they do
  not, the pin moves to a content digest (each array's dtype, shape and bytes) on both legs and
  the procedure builds the body in the locked 3.13 environment only.
- Slice 012 lands first; its rulings (Q1–Q3) apply to the capsule campaign unchanged.
- Out of scope: pose correctives (`posedirs`), realistic shape spaces, textures, a committed
  capsule fixture, CI smoke steps for the capsule, any change to SMPL reporting.

## Capsule campaign procedure (follow-up C; no SMPL needed; not part of "done")

Prerequisites: 012 and 013 landed; locked environment; the 012 heartbeat, status script and
decision rules.

1. `uv run --locked strike-a-pose-data make-capsule-body` (< 5 s; digest checked).
2. `strike-a-pose-data generate --config configs/capsule_proc_v1/generate.yaml --seed 0`
   (≈ 8 min), `validate`, `split --seed 0 --out splits/capsule_proc_v1.json`; check that its
   `poses` equal `splits/smpl_proc_v1.json`'s; commit the split.
3. Baselines (`configs/capsule_proc_v1/baselines.yaml`, ≈ 3 min) and the SC-005 check; copies to
   `docs/results/013-capsule/`.
4. The 012 campaign procedure from the re-baseline on, with these substitutions:

   | SMPL campaign (012) | Capsule campaign |
   |---|---|
   | F1 | skipped (no published capsule dataset to reproduce) |
   | `configs/default.yaml` | `configs/capsule_proc_v1/generate.yaml` |
   | `configs/<run>.yaml` (v1: baselines, ae, probe, variants) | `configs/capsule_proc_v1/<run>.yaml` |
   | SMPL single-change dataset configs (`v2a`–`v2d`) | `configs/capsule_proc_v2{a,b,c,d}/*.yaml` |
   | `splits/smpl_proc_<id>.json` | `splits/capsule_proc_<id>.json` |
   | R0 (`ae-smpl_proc_v1`), the control | capsule R0 (`ae-capsule_proc_v1`), control of capsule runs only |
   | `docs/results/<slice>-<run>/` | `docs/results/013-capsule/<run>/` |
   | arms-down visual check with real SMPL (010) | the same check on the capsule images |

5. SMPL arrival preempts (lead default 13). Capsule outcomes may lead to code fixes (through a
   slice, tests first), never to changes of the SMPL campaign's configs, thresholds or order.

Run order (surrogate ruling Q1): v1 R0 → R0b (seed 1) → R1 → R2 → R3 → v2a → v2b → v2c → v2d;
the four v2 datasets are independent single-change datasets (each compared with v1), not a
chain. Cost ≈ 012's estimate without F1 (≈ 8 min generation per 2,000-pose dataset instead
of 5).

## Clarifications

### Session 2026-10-21 (lead defaults; the batch below is pending)

1. Q: Delivery? → A: an explicit file from a CLI subcommand (FR-006); the manifest's file
   digest is the provenance, the generator is untouched. *(lead default)*
2. Q: Where does the skeleton live? → A: derived at build time from the committed fixture
   (FR-001; superseded lead default "literal table", see Q3 below). Consequence: the CLI reads
   `tests/fixtures/` (not packaged; run from a checkout), and regenerating the fixture would
   change the body, which the pinned digest test then reports.
3. Q: Skinning? → A: rigid, on the bone's proximal joint (FR-003). The brief's "one-hot on the
   bone's child joint" would turn a bone about its far end with the child's rotation (the
   thigh would swing when the knee bends). *(lead default)*
4. Q: Regressor? → A: 24 marker vertices in no face, one-hot rows: exact joints, no constraint
   in `SMPLModel`, harmless for the jitter in-frame check (markers are inside the body).
   *(lead default)*
5. Q: Resolution? → A: 16 × 4 rings, cylinder rings ≤ 0.03 m (≈ 6,000 vertices, ≈ 11,700
   faces): SMPL-like vertex count and render cost (19 vs 18 ms), so capsule timings predict the
   SMPL campaign's. The brief's 2–4 k vertices render at 25 ms (12 × 3, 0.04 m) and at 370 ms
   without cylinder rings. *(lead default)*
6. Q: Radii and offsets? → A: the FR-002 table (prototype: IoU 0.73–0.82 against the SMPL
   fixture masks, same body area); tunable once before the pin. Elliptical sections for the
   spine and the flat hands and feet, circular elsewhere. *(lead default)*
7. Q: Shape components 3–9? → A: zero: every config keeps `num_betas: 10`, so capsule configs
   and label shapes equal the SMPL campaign's; their betas are drawn, recorded and inert
   (documented; no consumer reads betas). Considered: S = 3 with `num_betas: 3` in the capsule
   configs; seven more directions (limb lengths, shoulder / hip width, head size).
   *(lead default)*
8. Q: Determinism? → A: FR-005 (rounding, −0.0, uncompressed npz, float32 zero `posedirs`,
   pinned digest checked by the CLI and a test). *(lead default)*
9. Q: Config directory? → A: `configs/capsule_proc_vN/` (FR-008). *(lead default)*
10. Q: Seeds? → A: those of the SMPL campaign (0): identical poses, cameras, split and β = 0
    joints, hence the SC-005 cross-check. *(lead default)*
11. Q: Tests' body? → A: the v1 body itself; no low-resolution variant (≈ 2 s of renders).
    *(lead default)*
12. Q: Figure? → A: optional: one committed PNG in `docs/results/013-capsule/` (capsule vs SMPL
    silhouettes, fixture pose 0, 8 views) made by a snippet in `tasks.md`; not tested.
    *(lead default)*
13. Q: SMPL arrives mid-campaign? → A: the attended session that installs SMPL stops the capsule
    campaign (kills its running step; the run dir stays resumable), records it in the campaign
    log and starts F1; capsule steps resume only if the operator asks. *(lead default)*
14. Q: Body changes later? → A: new version (`capsule_v2`, new pin, new dataset names); v1 is
    never edited once pinned. *(lead default)*

### Clarification batch for the orchestrator (2026-10-21, ruled)

- **Q1 — Which capsule datasets and runs?** → **Ruling**: mirror the campaign with the
  operator's 2026-10-21 independent single-change datasets (FR-008, run order in the procedure).
  *(surrogate ruling 2026-10-21)* Options as put: (a) The full 012 campaign mirrored (FR-008 as
  written): `capsule_proc_v1` with the variants and the chain `capsule_proc_v2` → `v5`; 24
  twin configs; ≈ 18–20 h CPU; every 012 step, guard and decision rule rehearsed. (b)
  `capsule_proc_v1` with the variants plus `capsule_proc_v5` only (all four 010 options and the
  4,000-pose memory settings in one dataset); 12 configs; ≈ 10 h; the chain rules not
  exercised. (c) `capsule_proc_v1` only (generate, baselines, AE, probe: the brief's list); 4
  configs; ≈ 1.7 h; variants and chain wait for SMPL. **Lead proposal: (a)**: it is the
  operator's "the compute campaign … can run now", the CPU is idle until SMPL arrives, and
  rehearsing the 012 rules on full-scale runs is the point; the copies are mechanical and
  test-pinned.
- **Q2 — Where do capsule numbers appear?** → **Ruling (a)**. *(surrogate ruling 2026-10-21)*
  Options as put: (a) `docs/results/013-capsule/` only (labelled README
  with a numbers table, per-run copies); the main README gets a status row, Usage and one
  pointer line, no capsule number (FR-009). (b) As (a) plus a labelled "Infrastructure runs
  (capsule body)" subsection with the key numbers in the main README, after Results. (c)
  Campaign log only; no copies. **Lead proposal: (a)**: the strictest reading of "not README
  headline numbers"; copies keep constitution V ("claims cite the run") and survive a lost
  container.
- **Q3 — Confirm the license position (operator as SMPL licensee).** → **Ruling (b)**: derive at
  build time; operator confirmation of the license reading pending. *(surrogate ruling
  2026-10-21)* Options as put: (a) Commit the 72-number
  rest skeleton as a literal table in `src/` (FR-001; same exposure as the committed fixture
  labels). (b) Keep no literal: derive it at build time from the committed fixture labels (no
  new copy in the source; the body then depends on the fixture never being regenerated and on
  `tests/` being present). (c) No SMPL-derived number: an anthropometric skeleton (joint
  labels, arms-down geometry and the SC-005 cross-check are lost). **Lead proposal: (a)**: the
  data is already public in the repo, and it keeps the body stable and the code packaged.
