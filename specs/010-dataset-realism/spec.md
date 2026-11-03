# Feature Specification: Dataset Realism Options

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `010-dataset-realism`)
**Created**: 2026-10-21
**Status**: Implemented 2026-10-21 (verification: [tasks.md](tasks.md) T018–T019; datasets
and runs: its run procedure, needs SMPL). Approved at surrogate scope (surrogate rulings
2026-10-21; operator sign-off pending).
*(Amended 2026-10-21 by slice 014, operator decisions 2026-10-21 ([014 spec](../014-realism-ablations/spec.md)): the datasets are independent
single-change datasets `smpl_proc_v2a`–`v2d`, each compared with `smpl_proc_v1`, replacing the
cumulative chain of surrogate ruling Q1 (FR-010); the arms-down table is renamed `armsdown-v1`
everywhere below (FR-003); an automatic limb–torso intersection count gates the v2a split
(run procedure). Original text kept below; where it conflicts, the notes win.)*
Approval basis: the operator's #15 decision quoted below; the surrogate ruled (a) on Q1–Q3
and (b) on Q4 and accepted every lead default.
**Input**: Operator answer to roadmap open question #15 ("Dataset realism: arms-down poses,
random betas (body shape), camera variation (height, distance, pitch, non-ring yaw)?"),
chosen 2026-10-20 from the options in `docs/HANDOFF.md` ("Operator decisions requested",
#15) and relayed by the orchestrator: "All four dataset-realism changes, each as its own
dataset version and experiment, in this order: (a) arms-down poses first, (b) random body
shape (betas), (c) camera ring jitter (small height/distance/pitch jitter around the 8
named cameras with an in-frame check; no free cameras), (d) more poses (config only). Each
becomes a config-selectable generator option whose default reproduces today's dataset byte
for byte; new datasets get their own config under `configs/`, a frozen split under
`splits/` and, later, baselines/AE/probe runs."
Decision (a) **reopens the 001 operator answer "keep arms-out limitation"** (2026-10-20,
roadmap changelog; 001 research R9: "shoulder z is ±50° around the SMPL T-pose, so arms are
never lowered to the sides"): the default table keeps that limitation (today's dataset is
unchanged); the new table lifts it.
Orchestrator brief (Fable surrogate scope): spec and batched clarifications first; "done" is
code + tests + docs + configs + a documented run procedure, not numbers (no SMPL model,
dataset or checkpoint in this container; the private asset repo `domattioli/smpl-assets`
exists but is empty); slice 009 was implemented concurrently in the same working tree.

Settled context (operator decision above; code facts and measurements checked in this
session at `702d2f5` with read-only experiments in the locked environment; `datagen`, `data`,
`configs` and their tests are unchanged through `7625437`, where slice 009 landed):

- **Generator.** `datagen.generate.generate_dataset` samples every pose first
  (`procedural.sample_procedural_poses(n, seed)`: one `numpy.random.default_rng(seed)`
  stream; per pose 69 integer angles in `[-A, B)` from `RANGES_DEG`, then the k ≥ 5
  sparsification), poses the body with `betas = zeros(num_betas)`, and renders each pose from
  the fixed ring `cameras.camera_ring` (8 cameras, distance 3.5 m, height 0, look-at
  `(0, 0, 0)`, focal 320 px, 256 × 192). Labels carry `pose_source`, `pose`, `betas`, joints,
  keypoints, visibility and `camera` (`name, K, R, t, width, height, yaw_deg`); the manifest
  carries `config = cfg.to_dict()` (`dataclasses.asdict`), `inputs.pose_source` (version and
  full ranges table) and `cameras` (the ring). The global `random` / `np.random` seeds are set
  but nothing draws from them.
- **Manifest identity (slice 007).** The split files pin the manifest *content* (manifest
  without `code`), which includes `config`, `inputs`, `cameras` and every label's
  `labels_sha256`. Follow-up F1 (needs SMPL) regenerates `smpl_proc_v1` at the current commit
  and requires content `9000c915…` (the `406b68f` dataset behind the published 003 / 004 / 006
  numbers). Any change to a v1 manifest's `config`, `inputs`, `cameras` or labels would make
  that check fail. The committed fixture manifest's `config` has exactly 11 keys.
- **Draw structure (measured).** Narrowing a non-zero range in the table (shoulder z
  `[-50, 50)` → `[-20, 20)`, collar z `[-30, 30)` → `[-10, 10)`) leaves every other drawn
  angle and the sparsification pattern identical (2,000 poses at seeds 0, 3 and 7; locked
  numpy 2.5.4). The first 2,000 poses of `sample_procedural_poses(4000, 0)` equal
  `sample_procedural_poses(2000, 0)`. `sample_procedural_poses(2, 3)` reproduces the
  committed fixture's `euler_deg` exactly and its `pose` to the labels' 6 decimals.
- **Arm kinematics (fixture skeleton).** The rest skeleton recovered from the two fixture
  poses (agreement 2.7 µm) is a T-pose with world-aligned joint-local axes: pelvis
  (−0.002, −0.223, 0.028), shoulders (±0.17, 0.226), elbows x ±0.43, wrists ±0.68, hand
  joints ±0.77, hip joints (±0.07, −0.31); upper arm 0.261 m, forearm 0.249 m, wrist → hand
  0.086 m. Euler order is `R = Rz·Ry·Rx` (z outermost), so adding an angle to the shoulder's
  z value rotates the whole sampled shoulder rotation about the parent's z axis (the
  frontal-plane normal); lowering is **negative z at the left shoulder (16), positive z at
  the right shoulder (17)**. Forward kinematics, left wrist (x, y) in m at a lowering of
  45° (0.53, −0.14) (A-pose); 65° (0.38, −0.24), hand at hip-joint height; 85° (0.21, −0.28),
  beside the thigh; 90° (0.17, −0.28), wrist x = shoulder x; 110° (0.00, −0.25), through the
  torso. Elbow flexion (y) then swings the forearm forward (65° plus elbow y −90° puts the
  wrist at z = +0.21 m, in front of the body). The predecessor's elbow z range `[-35, 5)` is
  the same for both elbows (not mirrored; R9), so it bends the left forearm toward the thigh
  and the right one outward and up.
- **Body shape.** `SMPLModel.forward` computes `v_shaped = v_template + shapedirs[..., :nb]
  @ betas[:nb]` with `nb = min(len(betas), shape components)` (extra betas are silently
  ignored) and the joints from `J_regressor @ v_shaped`, so betas change bone lengths and the
  root position. The synthetic test body model (`tests/conftest.py`) has zero `shapedirs`.
- **Framing.** 91.4 px per metre at the look-at depth; a pitch of 1° moves the image content
  5.6 px; the T-pose foot joints project to v ≈ 235 of 256. The 16 fixture silhouettes keep
  at least 44 px to the top, **8 px** to the bottom, 16 px left and 19 px right (bottom 20–23
  px for fixture pose 0, 8–13 px for pose 1): the margin at the ring camera is pose-dependent
  and can be small. A level camera raised by 0.1 m moves the body down 9 px; a camera raised
  or lowered by 0.3 m but aimed at the look-at point moves the T-pose soles by ≤ 4 px and
  the crown by ≤ 1 px (soles at y ≈ −1.20 m, crown ≈ +0.52 m: estimates, no mesh here);
  0.5 m farther shrinks the body to 87.5 %.
- **Consumers.** The 002 loader takes `camera_index` from the position of the label's
  camera *name* in the manifest `cameras` and reads `K, R, t, yaw_deg` from the label (other
  `camera` keys are ignored), and of the manifest `config` only `image`; `check_labels`
  checks the top-level label keys and the SC-003 projection invariant with the label's own
  `K, R, t`. The 003 pose baselines are fitted and scored in the world frame, their 2D score
  projects with each sample's `K, R, t`; the 006 probe's yaw target is `camera_index` and its
  `joints_cam` target and 2D projections use each sample's camera.
- **Memory.** The 003 image PCA fits all train images: 12,800 for a 2,000-pose dataset
  (≈ 10 GB peak, ran on the 15 GB dev machine); 25,600 for 4,000 poses (data 5.0 GB + centred
  copy 5.0 GB + float64 Gram 5.2 GB + eigenvectors 5.2 GB + `eigh` working memory: an
  estimated 25–30 GB). The 003 config rejects `image_pca_ks: []`; the 004 config accepts
  `context_pca_k: null`.
- **Environment.** No SMPL model, dataset or checkpoint here; 4 vCPU, 15 GB RAM, Python 3.13;
  CI runs Python 3.11 and 3.13. Timings of record: `smpl_proc_v1` generation 290 s (351 MB);
  003 baselines 182 s; AE 84 min CPU (60 epochs; 2 h 45 min calendar with two container
  restarts); probe 7.5 min.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Arms-down poses (Priority: P1)

*(Amended 2026-10-21 by slice 014: read `armsdown-v1` for `predecessor-v2-armsdown` in this story.)*

The researcher sets `ranges_version: predecessor-v2-armsdown` in a generation config and
gets a dataset whose arms hang at the sides instead of around the T-pose, while every other
joint angle is drawn exactly as with today's table.

**Why this priority**: operator decision #15 (a), first in the operator's order. Arms near
the T-pose are the most visible unrealism of `smpl_proc_v1` (001 R9) and keep limbs away from
the torso, the overlap case the 004 reconstructions handle worst ("crossing limbs merge").

**Independent Test**: synthetic body model, procedural source: generate with the v2 table;
check the shoulder and collar angles in the labels, compare every other angle with a v1-table
generation at the same seed, read the manifest's table, validate.

**Acceptance Scenarios**:

1. **Given** `ranges_version: predecessor-v2-armsdown`, **When** a dataset is generated,
   **Then** each pose's left shoulder z angle is −65° plus a value that is 0 (sparsified) or
   in `[-20, 20)`, its right shoulder z angle +65° plus such a value, both collar z angles are
   0 or in `[-10, 10)`, every other angle is 0 or in its v1 range, and `pose` is the
   axis-angle of these offset-included Euler angles.
2. **Given** the same seed with the v1 and the v2 table, **When** poses are sampled, **Then**
   every angle except the four changed (shoulder and collar z) is identical and so is the
   sparsification pattern (FR-003).
3. **Given** a v2-table dataset, **When** its manifest and labels are read, **Then** they name
   `predecessor-v2-armsdown`, and every `inputs.pose_source.ranges_deg` entry carries the
   joint's `offset_deg`.

---

### User Story 2 - Random body shape (Priority: P1)

The researcher sets `body_shape: {std: 1.0, clip: 2.0}` and every pose gets its own random
body (SMPL betas), shared by its camera views.

**Why this priority**: operator decision #15 (b). Every `smpl_proc_v1` sample shows the same
mean-shape body (001 Q4: "random shapes later").

**Independent Test**: synthetic body model whose `shapedirs` scale the body: betas within
`[-clip, clip]`, equal across a pose's views, different across poses, bone lengths changed;
with the zero-`shapedirs` model the images and joints equal a mean-shape generation; poses
identical to a generation without `body_shape`.

**Acceptance Scenarios**:

1. **Given** `body_shape` set, **When** a dataset is generated, **Then** each label's `betas`
   holds `num_betas` values in `[-clip, clip]`, identical for all views of its pose, and the
   body (vertices, joints) is posed with them.
2. **Given** the same config with and without `body_shape`, **When** both are generated,
   **Then** `pose`, `pose_source` and the pose order are identical (separate random stream).
3. **Given** `body_shape` set and a body model with fewer shape components than `num_betas`,
   **When** generation starts, **Then** it fails before writing anything (exit 2).

---

### User Story 3 - Camera ring jitter with an in-frame check (Priority: P1)

The researcher sets `camera_jitter` and each sample is rendered from a camera near its ring
camera (height, distance, pitch) that keeps the whole body in the image, or from the ring
camera itself when no such camera is found.

**Why this priority**: operator decision #15 (c). Today every view of every pose comes from
one of 8 exact extrinsics; the operator asked for small variation around them, no free
cameras, and no cropping.

**Independent Test**: synthetic body model: every accepted sample's silhouette keeps the
margin; name, yaw, `K` and size equal the ring camera's; impossible ranges make every sample
fall back, and then the labels equal a generation without jitter except for the `jitter`
block; zero-width ranges at 0 reproduce the ring cameras; the 002 loader gives the ring
`camera_index`.

**Acceptance Scenarios**:

1. **Given** `camera_jitter` set, **When** a sample is generated, **Then** its camera keeps the
   ring camera's name, yaw, intrinsics and image size, sits at the ring yaw with distance
   d + δd and height h + δh, is aimed at the ring's look-at point and tilted down by δp
   (deltas within the configured ranges; FR-005), and the label's
   `camera.jitter` records δh, δd, δp, the number of attempts and `fallback: false`.
2. **Given** an accepted camera, **Then** every vertex of that sample's body projects inside
   the image with a margin of `margin_px` pixels, so the silhouette never touches the border.
3. **Given** no candidate accepted within `max_attempts`, **Then** the ring camera is used and
   the label records zero deltas, `attempts = max_attempts` and `fallback: true`.
4. **Given** a jittered dataset, **When** it is validated and loaded with the 002 loader,
   **Then** the 001 validator passes and each item has the ring camera's `camera_index` and
   `yaw_deg` and the sample's own `K, R, t`; the manifest `cameras` list is the ring exactly as
   without jitter.

---

### User Story 4 - Today's dataset stays byte-identical (Priority: P1)

A config that sets none of the new options generates exactly what it generates today, so the
published 003 / 004 / 006 numbers keep their provenance and F1 still works.

**Why this priority**: operator constraint ("default reproduces today's dataset byte for
byte"); the 007 content hash makes any serialization change visible.

**Independent Test**: the pins of FR-009 against the committed fixture; explicit default
values versus absent keys; a same-machine before/after generation of synthetic datasets.

**Acceptance Scenarios**:

1. **Given** a config without the new fields, or with them set to their defaults explicitly,
   **When** a dataset is generated, **Then** `manifest.json`, `checksums.json`, every label and
   every image are byte-identical to the pre-010 generator's output for the same config, seed,
   inputs and machine.
2. **Given** `configs/fixture.yaml`, **Then** its `to_dict()` equals the committed fixture
   manifest's `config`; the default table's poses for seed 3 equal the fixture's; the
   manifest `inputs.pose_source` and `cameras` built for it equal the fixture's.
3. **Given** F1 run after this slice with the recorded SMPL file, **Then** the provenance check
   can still match `9000c915…` (no 010 change enters a v1 manifest).

---

### User Story 5 - The dataset series: configs, more poses, run procedure (Priority: P2)

*(Amended 2026-10-21 by slice 014: "its predecessor" reads `smpl_proc_v1` for every new dataset; the
run procedure also lists the intersection gate.)*

The researcher generates the new datasets in the operator's order from committed configs,
freezes each split, and runs the baselines, the AE and the probe with committed run configs,
following a documented procedure with time and disk estimates.

**Why this priority**: operator decision #15 ("each as its own dataset version and
experiment"; (d) "more poses, config only"). The runs need the SMPL model, so this slice
delivers everything up to the first command that needs it.

**Independent Test**: the committed configs load with their loaders; each generation config
differs from its predecessor by exactly one documented change; with `num_poses` doubled the
first poses' samples are byte-identical; every series config, pointed at the synthetic body
model with fewer poses and smaller images, generates a valid dataset.

**Acceptance Scenarios**:

1. **Given** the committed series configs (FR-010), **Then** each generation config differs
   from its predecessor's in `dataset_name` and exactly one documented option, and each run
   config differs from its `smpl_proc_v1` counterpart only in names and paths (and, for
   `smpl_proc_v5`, `image_pca_ks: []` and `context_pca_k: null`, FR-014).
2. **Given** `num_poses: 4000` and otherwise the same config, **Then** the samples of poses
   0–1999 are byte-identical to those of the 2,000-pose dataset (labels and images).
3. **Given** the run procedure, **Then** it lists the order (F1 → `smpl_proc_v1` re-baseline →
   each new dataset), the commands, the expected time and disk use, the arms-down visual check
   and what gets committed.

### Edge Cases

- `ranges_version` other than the default with `pose_source: amass` → config error (the table
  applies to the procedural source only). `body_shape` and `camera_jitter` apply to AMASS
  poses too (`pose_index` = draw order).
- Unknown table version, unknown keys in `body_shape` / `camera_jitter`, a range that is not
  two finite numbers with lo ≤ hi, `std` or `clip` ≤ 0, a height delta beyond ±1 m, a
  distance delta below 0 or above 3 m, a pitch beyond ±30°, `margin_px` < 0, `max_attempts`
  < 1, `body_shape` with `num_betas` 0 → config error, exit 2, nothing written (the bounds
  keep every candidate a sane, upright camera; the defaults are far inside them).
- `body_shape: {}` → enabled with the default `std` and `clip` (serialized resolved);
  `body_shape: null` and `camera_jitter: null` ≡ absent.
- Clipping puts mass at ±clip (≈ 4.6 % of components at std 1, clip 2); kept as is.
- `body_shape` without `camera_jitter` checks no framing: a larger body can lose a few pixels
  at the ring camera's border, like an extreme pose today (visibility flags joints outside the
  image; 001 edge case).
- Sparsified arm angles: about half of the shoulder z values are sparsified to 0, so about
  half of the arms sit exactly at ∓65° (as half sit exactly in the T-pose in `smpl_proc_v1`).
- Arm–body interpenetration is not detected (no collision model); FR-003 bounds it by the
  table and the run procedure's visual check with the real mesh.
  *(Amended 2026-10-21 by slice 014: still not detected during generation, but counted afterwards:
  `scripts/intersection_check.py` counts arm vertices inside per-segment torso volumes and
  fails the v2a split above 5 % of poses; the visual check stays as an additional step.)*
- Self-occlusion: visibility still ignores it (001 R6); with arms down, more joints are hidden
  behind the torso in side views but stay "visible" (documented limitation, now more frequent).
- A pose that the ring camera itself crops: jitter may find an in-frame candidate (accepted)
  or fall back (cropped exactly as today, flagged by visibility). An accepted camera never
  crops; the margin is checked on the sample's own posed and shaped vertices.
- `camera_jitter` with every range `[0, 0]` → the ring cameras exactly (R, t bit-identical);
  labels carry the `jitter` block (zeros, 1 attempt).
- A ring `count` other than 8 works with jitter (names `Cam<i>_<deg>` as today).
- Labels round floats to 6 decimals as today; the generator uses the unrounded draws (as for
  `pose`); the SC-003 invariant holds with the rounded `R, t` (1e-4 m, 1 px tolerances).
- Rejection makes the accepted camera distribution pose-dependent (a pose near the frame edge
  gets the smaller deltas); the acceptance rate is reported with each jittered dataset.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (config surface): `datagen.config.GenerationConfig` MUST gain three optional
  fields, strictly validated like the existing ones (unknown keys, types and ranges → config
  error, exit 2):

  | Field | Value | Default (= today) |
  |---|---|---|
  | `ranges_version` | `predecessor-v1-nohands` \| `predecessor-v2-armsdown` (procedural source only) | `predecessor-v1-nohands` |
  | `body_shape` | `null` or `{std: float > 0 (1.0), clip: float > 0 (2.0)}` | `null` (mean shape) |
  | `camera_jitter` | `null` or `{height_m: [lo, hi] within ±1 ([-0.3, 0.3]), distance_m: [lo ≥ 0, hi ≤ 3] ([0.0, 0.5]), pitch_deg: [lo, hi] within ±30 ([0.0, 5.0]), margin_px: int ≥ 0 (2), max_attempts: int ≥ 1 (16)}` | `null` (ring cameras) |

  A present block means "on"; its omitted keys take the values in parentheses. Numbers are
  normalized (ranges, `std`, `clip` as floats) so equal settings serialize identically.
  `num_poses` (existing) is the more-poses option.
  *(Amended 2026-10-21 by slice 014: the second `ranges_version` value is `armsdown-v1`; `predecessor-v2-armsdown` is
  rejected, no alias, since no dataset was generated with it.)*
- **FR-002** (serialization): `to_dict()`, and so `manifest.config`, MUST omit each of the three
  fields while it holds its default (absent, `null`, or the default version string), and
  serialize it fully resolved otherwise. A config that uses none of them serializes to exactly
  today's 11 keys and values. The rule applies to every generator field added later and is
  stated in the 001 data model.
- **FR-003** (arms-down table): `datagen.procedural` MUST keep `predecessor-v1-nohands`
  unchanged and add `predecessor-v2-armsdown` = the v1 table with three changes:
  shoulder z (16, 17) narrowed to `[-20, 20)`; collar z (13, 14) narrowed to `[-10, 10)`; an
  offset column, z −65° at the left shoulder (16) and +65° at the right shoulder (17), zero
  elsewhere. Offsets are added after sparsification (a sparsified shoulder z becomes the
  offset), enter `euler_deg` in the labels and `pose` through the unchanged Euler → axis-angle
  conversion; hands stay zero. Every other value is drawn as with the v1 table (the draw
  structure in *Settled context*; tested). The manifest's `inputs.pose_source` records the
  version and, for a table with offsets, `offset_deg` `[x, y, z]` in every `ranges_deg`
  entry; a table without offsets writes today's entries. Rationale and the measured arm
  statistics: clarification 6. The table is frozen only after the visual check of the run
  procedure (step R2).
  *(Amended 2026-10-21 by slice 014, operator decision 2026-10-21: the table is named `armsdown-v1` (it is not the
  predecessor's table). Confound, documented and kept: v2a differs from v1 in three ways at
  once — the arms are lowered (the intended change); shoulder z and collar z vary less (±20° /
  ±10° instead of ±50° / ±30°: less pose variety); and lowered arms overlap the torso in more
  views (self-occlusion, contact). A v2a vs v1 difference is their sum. The table is frozen
  only if `scripts/intersection_check.py` reports at most 5 % of v2a's poses with an arm
  vertex inside the torso (014 FR-004 – FR-007) and the visual check agrees.)*
- **FR-004** (body shape): with `body_shape` set, each pose MUST get `num_betas` betas, each
  N(0, std²) clipped to `[-clip, clip]`, drawn from a stream of its own seeded by (seed, pose
  index); all views of the pose share them; the label `betas` holds them and the body model
  is posed with them. Generation MUST refuse, before writing anything (exit 2), a `num_betas`
  larger than the body model's shape components while `body_shape` is set. Without
  `body_shape`, betas stay zeros.
- **FR-005** (camera jitter): with `camera_jitter` set, each sample's camera MUST be chosen as
  follows. Candidates are drawn in a fixed order from a stream seeded by (seed, pose index,
  ring index): δh, δd, δp uniform in their ranges. Candidate: the ring camera's yaw θ, centre
  `(−(d+δd)·sin θ, h+δh, (d+δd)·cos θ)`, aimed at the ring's look-at point `(0, h, 0)`, then
  tilted down by δp about its own horizontal axis (no roll; surrogate ruling Q2). It is accepted iff
  every vertex of the sample's posed (and shaped) body is in front of it and projects to
  `margin_px ≤ u < W − margin_px`, `margin_px ≤ v < H − margin_px`. The first accepted
  candidate is used; after `max_attempts` rejections the ring camera is used. The sample's
  projection, render, keypoints, visibility and label `camera.R`, `camera.t` come from the
  used camera; `camera.name`, `yaw_deg`, `K`, `width`, `height` are the ring camera's; the
  label `camera` gains `jitter: {height_m, distance_m, pitch_deg, attempts, fallback}`
  (deltas of the used camera, zeros on fallback). The manifest `cameras` stays the ring. No
  yaw, roll or focal jitter (operator: "no free cameras").
- **FR-006** (more poses): `num_poses` MUST stay a plain config value; the pose stream is
  sequential and the shape and jitter streams are keyed by pose (and ring) index, so, for any
  option set, the samples of the first N poses of a larger dataset are byte-identical to those
  of the N-pose dataset with the same other settings.
- **FR-007** (random streams): the pose stream MUST stay `default_rng(seed)` with today's draw
  order; the shape and jitter streams MUST be independent of it and of each other (seeded
  from the seed, a stream tag and the indices, e.g. through NumPy's `SeedSequence`), so
  switching one option never changes another option's draws, and a rejected camera candidate
  never shifts another sample's draws.
- **FR-008** (unchanged surfaces): label top-level keys (`LABEL_KEYS`), manifest top-level
  keys (`MANIFEST_KEYS`), `schema_version` 1, the sample-id format, the CLI and its exit
  codes, `validate`, the 002 loader and sample dict, and the 003 / 004 / 006 code MUST stay
  unchanged (except FR-014). The additions are optional nested keys
  (`camera.jitter`, `ranges_deg[*].offset_deg`) and optional `config` keys, written only when
  an option is on.
- **FR-009** (today's dataset): with no option on, the generator's output MUST be
  byte-identical to the pre-010 generator's for the same config, seed, inputs and machine.
  Pinned in CI by: `load_config("configs/fixture.yaml").to_dict()` equals the committed
  fixture manifest's `config`; the default table's poses for (2, seed 3) equal the fixture's
  `euler_deg` and, at 6 decimals, its `pose`; the manifest `inputs.pose_source` and `cameras`
  built for the fixture config equal the fixture's; label key sets and `camera` keys equal
  the fixture's; a synthetic generation with the new fields given explicitly at their defaults
  is byte-identical to one without them. Checked once on the dev machine (tasks verification
  record): synthetic datasets generated with today's configs at the last commit before 010's
  first source change and after this slice are byte-identical (manifest, labels, checksums,
  images). F1 with SMPL is the end-to-end check.
- **FR-010** (dataset series): the slice MUST commit one config directory per new dataset,
  `configs/<dataset>/` with `generate.yaml`, `baselines.yaml`, `ae.yaml` and `probe.yaml`.
  Names, contents and order: a cumulative chain (surrogate ruling Q1), each dataset its
  predecessor plus one change:

  | Dataset | `generate.yaml` = predecessor + | Predecessor |
  |---|---|---|
  | `smpl_proc_v2` | `ranges_version: predecessor-v2-armsdown` | `smpl_proc_v1` (`configs/default.yaml`) |
  | `smpl_proc_v3` | `body_shape: {std: 1.0, clip: 2.0}` | `smpl_proc_v2` |
  | `smpl_proc_v4` | `camera_jitter: {height_m: [-0.3, 0.3], distance_m: [0.0, 0.5], pitch_deg: [0.0, 5.0], margin_px: 2, max_attempts: 16}` | `smpl_proc_v3` |
  | `smpl_proc_v5` | `num_poses: 4000` | `smpl_proc_v4` |

  Generation seed 0 for all (as `smpl_proc_v1`). The run configs are copies of
  `configs/baselines.yaml`, `ae.yaml` and `probe.yaml` with `run_name`, `dataset_dir`,
  `split_file` (and the probe's placeholder `checkpoint`) set for the dataset; for
  `smpl_proc_v5` also `image_pca_ks: []` (baselines) and `context_pca_k: null` (AE), FR-014.
  Splits `splits/<dataset>.json` are written by the run procedure (they need the generated
  dataset): `pose-permutation-v1`, seed 0, default fractions (surrogate ruling Q3: also for
  `smpl_proc_v5`, 3,200 / 400 / 400), which gives the 2,000-pose datasets exactly the pose
  lists of `splits/smpl_proc_v1.json`. Comparisons: `smpl_proc_v2` against `smpl_proc_v1` is
  the controlled one (the same draws except the four narrowed and offset angles, FR-003);
  `smpl_proc_v3`, `v4` and `v5` are each compared with their predecessor (one change each;
  `v5` against `v4` on different test poses, same distribution).
  *(Amended 2026-10-21 by slice 014, operator decision 2026-10-21, replacing the chain: four independent
  single-change datasets, each `smpl_proc_v1` + one key and each compared with `smpl_proc_v1`
  (the R0 re-baseline of the campaign procedure):

  | Dataset | `generate.yaml` = `configs/default.yaml` + | Run configs |
  |---|---|---|
  | `smpl_proc_v2a` | `ranges_version: armsdown-v1` | v1's, names and paths |
  | `smpl_proc_v2b` | `body_shape: {std: 1.0, clip: 2.0}` | v1's, names and paths |
  | `smpl_proc_v2c` | `camera_jitter` (the defaults of FR-001) | v1's, names and paths |
  | `smpl_proc_v2d` | `num_poses: 4000` | + `image_pca_ks: []`, `context_pca_k: null` |

  The AE configs carry the 012 guards as `configs/ae.yaml`. v2a / v2b / v2c get the pose lists
  of `splits/smpl_proc_v1.json`; v2d 3,200 / 400 / 400 (different test poses than v1's, same
  distribution). No dataset combines the changes; a combined dataset is a later decision.)*
- **FR-011** (tests, written first; pytest, synthetic body model, no SMPL):
  (a) config: defaults; each field's validation errors (edge cases); `null` ≡ absent; `{}`
  resolved; number normalization;
  (b) serialization: the FR-009 pins; enabled options serialize resolved; explicit defaults
  generate byte-identical datasets;
  (c) arms-down: offsets and narrowed ranges in the labels (generalized range check
  `v == off or lo + off ≤ v < hi + off`); v1 / v2 draws identical except the four changed
  values; manifest table with `offset_deg`; v1 table byte-identical to today's manifest entries;
  (d) body shape: with a `shapedirs`-bearing synthetic variant, betas within `[-clip, clip]`,
  equal across views, different across poses, bone lengths changed; with the zero-`shapedirs`
  model, images and joints equal the mean-shape generation; poses identical with and without;
  `num_betas` above the model's components refused before writing;
  (e) camera jitter: name, yaw, `K`, size unchanged; deltas within ranges; accepted samples
  keep the margin (mask and vertex check); ranges that force rejections exercise the retry
  loop; impossible ranges → every sample falls back and the labels equal the no-jitter labels
  apart from `camera.jitter`; zero ranges → R, t bit-identical to the ring; manifest `cameras`
  unchanged; the 002 loader's `camera_index` and `yaw_deg` are the ring's;
  (f) every option alone and all together: `validate_dataset == []`, byte-reproducible for the
  same config and seed, label top-level keys = `LABEL_KEYS`, `camera` keys = today's (+
  `jitter` when on);
  (g) prefix property (FR-006) with all options on;
  (h) series configs: each loads; consecutive generation configs differ by the documented
  key; each, with the synthetic model, 2 poses and 64 × 48 images, generates a valid dataset;
  run configs load with the 003 / 004 / 006 loaders and differ from v1's only in the listed
  keys;
  (i) every existing test stays green unchanged, except where a test enumerates config fields
  or lists invalid 003 configs (`image_pca_ks: []` becomes valid);
  (j) 003 with `image_pca_ks: []`: a fixture baseline run writes the mean image and the pose
  baselines, no `pca_k*` entry, `fit.image_pca` null, a two-column figure.
- **FR-012** (documentation, same slice): see *Documentation scope*.
- **FR-014** (003 without image PCA; surrogate ruling Q4 (b)): the 003 baseline config MUST
  accept `image_pca_ks: []`: no image PCA is fitted, `metrics.json` `fit.image_pca` is null,
  no `pca_k<K>` image entries or figure columns are written, and the mean image and the pose
  baselines are scored as today; non-empty lists behave exactly as before. `smpl_proc_v5`
  uses it (25,600 train images: image PCA needs an estimated 25–30 GB, i.e. a ≥ 32 GB
  machine; the dev machine has 15 GB), and its AE config uses `context_pca_k: null`.
- **FR-013** (constraints): no new dependency (`pyproject.toml`, `uv.lock` untouched); Python
  3.11-compatible; `.github/workflows/ci.yml` unchanged; no dataset, split file or run result
  produced or committed in this slice; `docs/results/`, `splits/`, the committed fixture and
  the v1 configs (`configs/default.yaml`, `fixture.yaml`, `baselines.yaml`, `ae.yaml`,
  `probe.yaml`) byte-unchanged; nothing under `data/`, `runs/` or `assets/` committed.

### Documentation scope (FR-012)

- `specs/001-dataset-generation/spec.md`: dated notes "amended 2026-10-21 by slice 010" on
  FR-002 (table versions; body shape no longer always mean shape), FR-003 (optional jitter
  around the ring), the out-of-view edge case, Q4, the Q9 update ("keep mean shape and the
  existing camera ring": still the defaults) and the reopened "keep arms-out limitation"
  answer; original text kept.
- `specs/001-dataset-generation/contracts/dataset-layout.md`: label rows `pose_source`
  (offset-included `euler_deg`), `betas`, `camera` (used camera's `R, t`; ring `name`,
  `yaw_deg`; optional `jitter`); manifest paragraph (omitted default fields, `cameras` = the
  ring, optional `offset_deg`); `schema_version` stays 1 (additive optional keys; v1 datasets
  byte-identical).
- `specs/001-dataset-generation/data-model.md`: config rows of FR-001, the FR-002 rule, the
  random streams (FR-007).
- `specs/001-dataset-generation/research.md`: dated notes in R5 (jittered camera, in-frame
  check) and R9 (v2 table, kinematics, the visual check to be recorded by step R2; "arms are
  never lowered" holds for v1 only), each pointing to 010's research.
- `specs/002-data-pipeline/contracts/sample-contract.md`: `betas` (zeros unless `body_shape`),
  `camera.R` / `camera.t` (the sample's camera), `camera.yaw_deg` (ring yaw).
- `specs/003-baselines-and-metrics/research.md` R2: dated note (with jitter `R` varies per
  sample; world-frame scoring and per-sample 2D projection are unaffected).
- `specs/006-latent-probe/contracts/probe-run.md`: one note (yaw target = ring index;
  `joints_cam` in the sample's camera frame).
- `specs/009-region-metrics/spec.md` FR-001: dated note (squares unchanged for the 010
  datasets; clarification 21).
- `README.md`: status table (010 row); Usage: the three options, the series configs and the
  run procedure pointer; the 4,000-pose memory note; Results unchanged.
- `specs/roadmap.md`: #15 answered (operator, 2026-10-20) and implemented by 010; post-MVP
  table row; status and "Out of MVP" lines; changelog entry 2026-10-21, appended at the end.
- `docs/HANDOFF.md`: state (010 implemented, approval status); next work = F1, then the run
  procedure below; #15 removed from "Operator decisions requested"; the clause "any generator
  change (#15) must land after F1" becomes "generation waits for F1 and a same-code v1
  re-baseline".
- `specs/003-baselines-and-metrics/contracts/eval-run.md` and `spec.md`: dated notes for
  FR-014 (`image_pca_ks` may be empty; `fit.image_pca` null).
- Not edited: `docs/results/`; 001–009 `plan.md` / `tasks.md`; `.github/workflows/ci.yml`;
  `docs/data-assets.md`.

### Key Entities

- **Ranges table version**: name, per-joint `[-A, B)` ranges per axis, optional offsets.
- **Body shape setting**: `std`, `clip`; per pose, the betas (label `betas`).
- **Camera jitter setting**: ranges, margin, attempts; per sample, the used camera and its
  jitter record (label `camera`, `camera.jitter`).
- **Dataset series**: names, config directories, split files and runs of FR-010.

## Success Criteria *(mandatory)*

- **SC-001**: ruff, ruff format, pytest and the CI smoke commands green locally and, after the
  orchestrator's push, on CI (Python 3.11 + 3.13); pytest grows by ≤ 10 s locally.
- **SC-002**: FR-009 holds: the pins pass in CI, and the same-machine before/after comparison
  of synthetic datasets is byte-identical (recorded in the tasks verification record).
- **SC-003**: each option alone, all options together and each series config (synthetic body
  model, reduced size) produce a dataset that passes `validate`, is byte-identical when
  regenerated with the same config and seed, and satisfies the acceptance scenarios of US1–US3
  and US5 scenario 2.
- **SC-004**: the 002 / 003 / 004 / 006 tests pass, edited only as FR-011 (i) allows; a
  jittered, shaped synthetic dataset loads with the 002 loader with ring `camera_index` and
  `yaw_deg`.
- **SC-005**: `docs/results/`, `splits/`, the fixture and the v1 configs are byte-unchanged; no
  document reports numbers for the new datasets.
- "Done" does not depend on the SMPL model, a generated dataset or a training run.

## Assumptions

- The SMPL file recorded in the fixture manifest (`f408141b…`) arrives in a later session; F1
  runs before any realism dataset is generated.
- The locked NumPy keeps the draw structure (FR-003 test); a NumPy change would change every
  dataset anyway.
- The joint-level arm statistics (clarification 6) predict the real mesh well enough to fix the
  table; the visual check (R2) confirms or amends it before the `smpl_proc_v2` split is frozen.
- Runs use the dev machine (4 vCPU, 15 GB RAM); image PCA for `smpl_proc_v5` would need a
  ≥ 32 GB machine and is not part of the procedure (FR-014).
- The realism runs use the training and evaluation code of the `smpl_proc_v1` re-baseline run
  (best-val selection, 009 region metrics); a code change in between is recorded with the
  results.

## Run procedure (follow-up R; needs SMPL; not part of "done")

R1. Prerequisites: `assets/smpl/SMPL_NEUTRAL.npz` with the recorded SHA-256; F1 done
(`docs/HANDOFF.md`). If F1's provenance check fails after 010 landed, regenerate at a pre-010
commit (e.g. `702d2f5`) before concluding: the same failure there → environment; success there
→ a 010 regression, fixed before anything else. Then the `smpl_proc_v1` re-baseline with the
same code as the series (003 baselines, AE `configs/ae.yaml`, probe): the reference row for
`smpl_proc_v2`.

*(Amended 2026-10-21 by slice 014: the runs follow the campaign procedure in `docs/HANDOFF.md` (steps v2a – v2d, each
compared with R0); `DS` below reads `smpl_proc_v2a`, `v2b`, `v2c`, `v2d`; v2a's split waits for
the intersection gate, `scripts/intersection_check.py --config configs/smpl_proc_v2a/generate.yaml
--seed 0` (FAIL above 5 % of poses: no split), and then the visual check below; the time table's
`smpl_proc_v5` column is v2d's.)*

R2. Per dataset, in order (`DS` = `smpl_proc_v2`, `smpl_proc_v3`, `smpl_proc_v4`,
`smpl_proc_v5`):

```bash
uv sync --locked && mkdir -p runs
uv run --locked strike-a-pose-data generate --config configs/$DS/generate.yaml --seed 0
uv run --locked strike-a-pose-data validate data/generated/$DS
# smpl_proc_v2 only: visual check before the split (below)
uv run --locked strike-a-pose-data split data/generated/$DS --seed 0 --out splits/$DS.json
uv run --locked strike-a-pose-eval baseline --config configs/$DS/baselines.yaml  # v5: no PCA
setsid nohup uv run --locked strike-a-pose-train --config configs/$DS/ae.yaml \
    > runs/train-$DS.log 2>&1 &        # after a restart: strike-a-pose-train --resume RUN_DIR
uv run --locked strike-a-pose-eval probe --config configs/$DS/probe.yaml \
    --checkpoint runs/ae-$DS-<stamp>/best.pt
```

Visual check (`smpl_proc_v2`): inspect the `Front_0` and `Right_90` renders of poses 0–15;
record arm placement and the number of renders with visible arm–body intersection in 001
research R9 (as the v1 check was). If intersections are frequent, delete the dataset, amend
the table (bump the version string if a dataset with it was ever split or published) and
regenerate. Jittered datasets (`v4`, `v5`): report the fallback count and mean attempts from
the labels (`camera.jitter`). Results go to `docs/results/{003-baselines,004-ae,006-probe}-$DS/`
and one README comparison table (one row per dataset, `smpl_proc_v1` re-baseline first); split
files are committed with them.

| Step (4 vCPU) | `smpl_proc_v2` / `v3` / `v4` each | `smpl_proc_v5` |
|---|---|---|
| generate | ≈ 5 min, ≈ 351 MB | ≈ 10 min, ≈ 700 MB |
| validate + split | ≈ 1 min | ≈ 2 min |
| 003 baselines | ≈ 3 min | ≈ 3 min (no image PCA) |
| AE, 60 epochs | ≈ 85 min CPU (2–3 h calendar with restarts) | ≈ 170 min CPU |
| probe | ≈ 8 min | ≈ 16 min |
| total | ≈ 100 min CPU | ≈ 200 min CPU |

Series ≈ 8.5 h CPU after the ≈ 1.6 h `smpl_proc_v1` re-baseline; plan about twice that in
calendar time (container restarts), i.e. several 3-hour session windows.

## Notes for the plan (specify session, 2026-10-21)

Facts found while specifying (not requirements; checked at `702d2f5`):

1. **First step, before any source edit:** generate synthetic datasets with today's configs
   (`proc_cfg`-style procedural and `small_cfg`-style AMASS, as in `tests/test_procedural.py`
   and `tests/conftest.py`) into a directory outside the repository; they are the SC-002
   "before" reference.
2. **Touch points.** `datagen/config.py` (fields, validation, `to_dict` omission);
   `datagen/procedural.py` (table registry keyed by version, offsets, `ranges_table(version)`;
   keep `RANGES_VERSION` / `RANGES_DEG` as the v1 names the tests import);
   `datagen/cameras.py` (jittered camera from the ring camera and the deltas, in-frame test on
   projected vertices); `datagen/generate.py` (table version, per-pose betas, per-sample
   camera, the `jitter` block, the `num_betas` check); `datagen/smpl.py` (expose the number of
   shape components). Tests: `test_config.py`, `test_procedural.py`, `test_cameras.py`,
   `test_generate.py`, a `shapedirs`-bearing variant next to `make_synthetic_smpl` in
   `tests/conftest.py`. Slice 009 (landed at `7625437`) touched only `eval/`, `train/`,
   their tests, contracts and docs; 010's README / roadmap / HANDOFF edits build on its
   versions.
3. **Zero deltas must give the ring camera bit for bit**: build the jittered camera with the
   same `look_at` call as `camera_ring` (centre and target) so `[0, 0]` ranges reproduce `R, t`
   exactly (FR-011 (e)).
4. **Cost.** The in-frame test projects 6,890 vertices per candidate (well under 1 ms) against
   ≈ 18 ms of rendering per sample (290 s / 16,000); acceptance is expected near 100 % with the
   default ranges (fixture margins), so generation time is unchanged within noise.
5. **Tests that enumerate config fields**: none found today; `test_bad_configs` gains cases.

## Clarifications

### Session 2026-10-21 (autonomous run, Fable surrogate scope)

Defaults chosen by the lead within the surrogate's scope are marked *(lead default)*; the
four questions sent to the surrogate in one batch carry *(surrogate ruling 2026-10-21)*. The
surrogate accepted every lead default, naming in particular: a present block means "on",
omit-when-default serialization pinned against the fixture manifest, the ∓65° table with
shoulder z `[-20, 20)` and collar z `[-10, 10)` (the departure from the predecessor's collar
range stays flagged; the real-SMPL visual check is a step of the run procedure before the
`smpl_proc_v2` split is frozen), separate random streams for betas and jitter, per-dataset
config directories, the 009 region squares unchanged with the cross-dataset caveat, and the
HANDOFF's "#15 after F1" clause replaced by "generation waits for F1 and a same-code v1
re-baseline". Every answer is provisional and overturnable by the operator.

1. Q: Slice number and slug? → A: `010-dataset-realism` (next free number); one slice for the
   four generator options; the datasets and runs follow its run procedure. *(lead default)*
2. Q: One dataset per change: cumulative chain or independent variants, and their names?
   → A: a cumulative chain `smpl_proc_v2` (arms-down) → `v3` (+ shape) → `v4` (+ jitter) →
   `v5` (+ 4,000 poses); each step changes one thing, so consecutive comparisons attribute
   each change, and `v5` is the full-realism dataset (batch Q1, option (a)). *(surrogate
   ruling 2026-10-21)* *(Amended 2026-10-21 by slice 014: overturned by the operator, 2026-10-21: independent
   `smpl_proc_v2a` – `v2d`, option (b) of batch Q1; FR-010 note.)*
3. Q: Config shape? → A: FR-001: three optional fields; a present block means "on" (no
   `enabled` flag, so a disabled block with non-default numbers cannot change the manifest of
   an identical dataset); defaults inside the blocks; numbers normalized. Considered: the
   memo's `{random: false, …}` / `{enabled: false, …}` blocks. *(lead default)*
4. Q: How do new fields stay out of v1 manifests? → A: FR-002, omit-when-default in
   `to_dict()`, pinned against the committed fixture manifest (FR-009); the same rule for the
   new optional label and `inputs` keys. *(lead default)*
5. Q: May 010 land before F1? → A: yes: no 010 change enters a v1 manifest (FR-002, FR-009);
   the generation of new datasets waits for F1 and the `smpl_proc_v1` re-baseline (R1). The
   HANDOFF's "after F1" constraint was about manifest keys, which FR-002 removes.
   *(lead default)*
6. Q: Arms-down table? → A: FR-003: offset ∓65° on the shoulder z axis, shoulder z narrowed to
   `[-20, 20)`, collar z to `[-10, 10)`, offsets after sparsification, hands zero, every other
   predecessor value (incl. the non-mirrored elbow z) kept; version `predecessor-v2-armsdown`.
   Basis: forward kinematics on the fixture skeleton over 2,000 sampled poses (seed 0), with
   arm abduction measured in the torso (spine3) frame (0° = straight down, 90° = T-pose,
   negative = toward the midline) and a torso-overlap proxy (an elbow, wrist or hand joint
   within 0.14 m of the torso axis laterally, ±0.12 m front-back, between the hips and the
   shoulders):

   | Table | Abduction median (1st pct L / R) | Proxy L / R |
   |---|---|---|
   | v1 (T-pose ±50°) | 87° | 0.5 % / 0.1 % |
   | ∓70°, collar ±30 (memo) | 17° (−25° / −20°) | 9.6 % / 5.9 % |
   | ∓65°, collar ±30 | 22° (−20° / −15°) | 6.4 % / 3.1 % |
   | ∓60°, collar ±30 | 27° (−15° / −10°) | 4.2 % / 1.9 % |
   | **∓65°, collar ±10 (chosen)** | 22° (−3° / −1°) | 1.8 % / 1.0 % |

   The collar's z rotation adds directly to the arm's elevation (same axis), so with the arms
   lowered its ±30° crosses the arm into the torso; narrowing it to ±10° removes most crossings
   while keeping arms close to the body. Considered: the memo's ∓70° with the collar unchanged
   (most intersections); mirroring the elbow z ranges (a further deviation from the
   predecessor, not asked for). The table is confirmed or amended by the visual check (R2).
   *(lead default)* *(Amended 2026-10-21 by slice 014: version `armsdown-v1`; the narrowing is a confound of the
   v2a comparison (FR-003 note); the automatic intersection count of 014 gates the split.)*
7. Q: Does a table change shift the other draws? → A: no (measured; *Settled context*): the
   `smpl_proc_v2` poses equal `smpl_proc_v1`'s except the four narrowed and offset angles,
   which makes v1 → v2 a controlled comparison; pinned by a test (FR-011 (c)). *(lead default)*
8. Q: Body-shape distribution? → A: per pose, N(0, 1) per component clipped at ±2 (the common
   choice for SMPL shape coefficients; the clip excludes extreme bodies, and uniform sampling
   over ±2 would over-weight them), `num_betas` 10 as configured today, shared by the pose's
   views (the split unit is the pose). *(lead default)*
9. Q: Betas beyond the model's shape components? → A: refused before writing (FR-004): the
   model silently ignores them while the labels would record them. *(lead default)*
10. Q: Camera jitter parametrization? → A: move the centre (δh ±0.3 m, δd 0–0.5 m), aim at
    the ring's look-at point, then tilt down 0–5°, with the projected-vertex in-frame check
    and the recorded ring-camera fallback (batch Q2, option (a); FR-005). *(surrogate ruling
    2026-10-21)*
11. Q: Jitter ranges, check, fallback? → A: δh `[-0.3, 0.3]` m, δd `[0, 0.5]` m (never
    closer than the ring), δp `[0, 5]°` down only (the ring camera's bottom margin can be 8 px;
    an upward tilt of 2° would remove 11 px), margin 2 px, 16 attempts; check on the projected
    mesh vertices (the extreme joints lie 5–15 px inside the soles, crown and fingertips; the
    silhouette lies inside the vertices' bounding box); fallback to the ring camera (= today),
    recorded. *(lead default)*
12. Q: Per-sample records? → A: label `camera.jitter` (FR-005); acceptance statistics computed
    from the labels in the run procedure, no new manifest or `run_info.json` field. *(lead
    default)*
13. Q: Manifest `cameras`, loader, probe? → A: `cameras` stays the ring (the loader's
    `camera_index` and the probe's yaw classes stay exact: yaw is never jittered); the sample
    dict is unchanged (the `jitter` block stays in the label files). *(lead default)*
14. Q: How many more poses? → A: `num_poses: 4000` (2×; the HANDOFF estimate the operator
    chose from); generation ≈ 10 min, AE ≈ 170 min. *(lead default)*
15. Q: Split of the 4,000-pose dataset? → A: the standard permutation split, config only, as
    the operator said (batch Q3, option (a)). *(surrogate ruling 2026-10-21)*
16. Q: Image PCA for the 4,000-pose dataset? → A: allow `image_pca_ks: []` in the 003 config
    so the dataset still gets a logged baseline run (mean image, pose baselines); its AE run
    uses `context_pca_k: null`; the memory estimate and the ≥ 32 GB need for image PCA are
    documented (batch Q4, option (b); FR-014). *(surrogate ruling 2026-10-21)*
17. Q: Config layout? → A: one directory per new dataset with `generate.yaml`,
    `baselines.yaml`, `ae.yaml`, `probe.yaml` (16 files; flat `<kind>_<dataset>.yaml` names
    would put 16 near-duplicates next to v1's); v1 keeps its files. Run configs committed now
    (validated by tests) so the run procedure needs no edits. *(lead default)*
18. Q: Training and evaluation settings for the series? → A: identical to v1's committed
    configs (seed 0, 60 epochs, `select_weights: best_val`, probe grid), so consecutive
    datasets differ only in the data; for `smpl_proc_v5` the 2× data at 60 epochs also means
    2× optimizer steps (stated with its results). *(lead default)*
19. Q: Schema version? → A: stays 1: additions are optional nested keys and optional config
    keys; every reader ignores them and v1 datasets are byte-identical. *(lead default)*
20. Q: Verification without SMPL? → A: FR-009 pins (CI), the same-machine before/after
    comparison (tasks), F1 (with SMPL), and every option on the synthetic body model. *(lead
    default)*
21. Q: The 009 region squares (fixed at the ring's 3.5 m and the mean shape)? → A: unchanged:
    region numbers compare models within a dataset; across datasets with the stated caveat
    (jitter up to 12.5 % smaller bodies, shapes a few % larger or smaller). *(lead default)*
22. Q: Out of scope? → A: free or yaw-jittered cameras, roll, focal jitter, AMASS, textures,
    backgrounds, occlusion-aware visibility, a collision check, a new committed fixture,
    generating datasets or splits, any run, and training or evaluation changes (except
    FR-014). *(lead default)*
23. Q: Dates? → A: the session clock, 2026-10-21; the roadmap changelog entry is appended at
    the end. *(lead default)*

### Clarification batch for the surrogate (sent and ruled 2026-10-21)

- **Q1 — The four datasets: cumulative chain or independent variants?** (a) Cumulative chain
  in the operator's order: `smpl_proc_v2` = v1 + arms-down, `smpl_proc_v3` = v2 + random
  shape, `smpl_proc_v4` = v3 + camera jitter, `smpl_proc_v5` = v4 + 4,000 poses. Consecutive
  versions differ by exactly one change, so each experiment measures that change on top of
  the earlier ones; the last dataset carries all four and is the one later work (#16
  variants) would use; the operator's order matters. Integer versions say "successor"; the
  memo's letters would suggest siblings. (b) Independent variants of v1: `smpl_proc_v2a`
  (arms-down), `v2b` (shape), `v2c` (jitter), `v2d` (4,000 poses), each compared with v1:
  effects in isolation, no interactions; no dataset combines the changes (later work needs a
  fifth one). Same code and run cost (four datasets, ≈ 8.5 h CPU). **Lead proposal: (a)**:
  "dataset versions … in this order" reads as a sequence, and realism is cumulative.
  **Ruling (surrogate, 2026-10-21): (a)**; `smpl_proc_v2` vs `smpl_proc_v1` is the controlled
  comparison (same draws except the four narrowed angles), `v3`+ are compared with their
  predecessor.
  *(Amended 2026-10-21 by slice 014: the operator chose (b), 2026-10-21, with the names `smpl_proc_v2a` – `v2d`.)*
- **Q2 — Camera jitter parametrization (FR-005).** (a) Aim at the ring's look-at point: the
  centre moves by δh `[-0.3, 0.3]` m and δd `[0, 0.5]` m, the camera aims at `(0, h, 0)` and
  then tilts down by δp `[0, 5]°`. Height changes the viewing elevation (±4.9°) while the body
  stays centred (soles move ≤ 4 px), distance scales it (≥ 87.5 %), the tilt shifts it up
  (≤ 28 px): each parameter has one visible effect, and the elevation adds real viewpoint
  variety. (b) Level translation plus absolute pitch (the memo): the camera moves by δh
  `[-0.15, 0.10]` m (limited by the feet margin) and δd `[0, 0.5]` m and pitches δp `[0, 5]°`
  down from horizontal: height and pitch both mostly translate the body vertically (9 px per
  0.1 m, 5.6 px per degree) and the elevation changes by ≤ 2.5°, so the variation is close to
  an image shift-and-scale augmentation. Same code size. **Lead proposal: (a)**: it matches
  the "look-at the body" wording of the option the operator chose, varies the viewpoint
  rather than the crop, and leaves the in-frame check mostly slack (less rejection bias).
  **Ruling (surrogate, 2026-10-21): (a)**, with the projected-vertex in-frame check and the
  ring-camera fallback recorded in the labels.
- **Q3 — Split of the 4,000-pose dataset.** (a) The standard split of all 4,000 poses
  (`pose-permutation-v1`, seed 0, 0.8 / 0.1 / 0.1 → 3,200 / 400 / 400): config only, as
  decided; its test set is a different sample of the same distribution than the 2,000-pose
  datasets', so the comparison carries test-set sampling noise (estimated a few % of the MSE,
  against the 2× train/val gap of the 004 run that more poses should narrow). (b) A nested
  split: keep the predecessor's val and test pose lists (poses 0–1999, byte-identical
  samples) and put all 2,000 added poses into train (3,600 / 200 / 200): a paired comparison on
  identical test images, but a small split-extension feature in slice 002's module, CLI and
  contract (new algorithm name, base-split reference, tests). **Lead proposal: (a)**: the
  operator asked for "config only", and the expected effect is larger than the sampling
  noise; (b) can follow if the result is ambiguous.
  **Ruling (surrogate, 2026-10-21): (a)**, config only, as the operator said.
- **Q4 — Image-PCA baseline for the 4,000-pose dataset** (25,600 train images; an estimated
  25–30 GB peak vs 15 GB on the dev machine). (a) None on the dev machine: its `ae.yaml` sets
  `context_pca_k: null` (the AE is reported next to the mean image); its `baselines.yaml` is
  committed but run only on a machine with ≥ 32 GB RAM; the pose baselines come with the probe
  (mean pose, pose PCA). No code. (b) Allow `image_pca_ks: []` in 003 so the baseline run
  (mean image, mean pose, pose PCA) fits the dev machine: ≈ 10 lines, tests and a 003 contract
  note. (c) Fit the image PCA on a fixed train subset of 12,800 images (v1's train size) in
  003 and 004: a PCA-64 column for every dataset at equal fit size; new keys in both configs
  (+ a `LEGACY_VALUES` entry), the most code. **Lead proposal: (a)**: constitution V is met by
  the mean image and the probe's pose baselines, and the operator asked for "config only".
  **Ruling (surrogate, 2026-10-21): (b)**: about 10 lines and a test so the large dataset
  still gets a logged baseline run; the AE run uses `context_pca_k: null`; the memory
  estimate and the ≥ 32 GB need for image PCA are documented.
