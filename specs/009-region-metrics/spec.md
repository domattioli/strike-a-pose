# Feature Specification: Per-Region Image Metrics

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `009-region-metrics`)
**Created**: 2026-10-20
**Status**: Implemented 2026-10-21 (verification: [tasks.md](tasks.md) T015). Approved at
surrogate scope (surrogate rulings 2026-10-20; operator sign-off pending).
Approval basis: the operator's choice of option (i) for roadmap open question #16
(2026-10-20), quoted below; the surrogate ruled (a) on the three batch questions and
accepted every lead default.
**Input**: Operator answer to roadmap open question #16 ("Training improvements / VAE
(blur, lost extremities, mild overfitting in 004)?"), 2026-10-20: **option (i)**, put to
the operator as "per-region MSE metrics (hands / feet / head box; no face joint in
SMPL-24) — metrics only, lands before the re-baseline run" (`docs/HANDOFF.md`, "Operator
decisions requested") and relayed by the orchestrator as: "Per-region image metrics first:
hands, feet and a head box (SMPL-24 has no face joint) MSE from the 2D keypoints as extra
metrics. No training change; lands before the re-baseline run so its numbers come with
that run." This takes up the candidate that surrogate ruling 2026-10-20 #4 (approved by
the operator 2026-10-21) deferred; the rest of ruling #4 stands: the headline image
metrics stay MSE and silhouette IoU @ 0.075, and every existing number keeps its value.
Orchestrator brief (Fable surrogate scope): spec and batched clarifications first; "done"
must not depend on a full-dataset run (no SMPL model, dataset or checkpoint in the
container); the next slice (training variants: mask-weighted MSE, VAE, regularization)
keeps every variant's predictor behind `predict_rows` feeding `score_images`, so the region
scoring sits in that seam and applies to every image model alike.

Settled context (operator decision above; code facts and fixture measurements checked in
this session at `5382075`):

- **One scoring seam.** `eval/report.py` `score_images(pred_fn, images, masks, batch_size,
  threshold)` takes flattened rows `[n, H*W]` (it does not know `H, W`), calls `pred_fn`
  once per batch and returns the float64 means of the per-image `image_mse`, `psnr_db` and
  `silhouette_iou`. 003 `eval/baselines.py` `score_split` calls it for the mean image and
  every PCA k; 004 `train/trainer.py` `_train_and_evaluate` calls it, after
  `load_selected_weights` (008), for `ae` (`predict_rows`), `mean_image` and `pca_k<K>`.
  The 006 probe does not score images. The CLI output lines (`eval/cli.py`, the trainer's
  `_headline`) read only the headline keys.
- **Labels at the seam.** 003 `load_arrays` already holds `kp2d` (float64 `[n, 24, 2]`) and
  `visible` (`[n, 24]`) for every scored split. The 004 trainer's `load_split` does
  **not**: it keeps uint8 images, `pose_index` and, for scored splits, bool masks (val is
  loaded once, before training, and held for the whole run). This slice must keep the 2D
  keypoints and visibility of scored splits there (≈ 0.2 KB per sample).
- **Keypoint and pixel conventions.** 2D keypoints are pixels, (u right, v down), origin
  top-left (002 sample contract). The renderer samples pixel (r, c) at its centre
  (c + 0.5, r + 0.5), so the pixel holding keypoint (u, v) is row ⌊v⌋, column ⌊u⌋.
  `visibility` = in front of the camera and inside `[0, W) × [0, H)` (001 research R6);
  self-occlusion is not modelled, so a hand hidden behind the torso is "visible".
- **Joints** (`datagen.skeleton.SMPL24_JOINT_NAMES`): ankles 7 / 8, feet 10 / 11 (ball of
  the foot), neck 12, head 15, wrists 20 / 21, hands 22 / 23 (knuckles); no face,
  fingertip, toe-tip or head-top joint. Mean-shape bone lengths (fixture): neck–head
  8.3 cm, wrist–hand 8.6 cm, ankle–foot 13.4 cm.
- **Camera scale.** Fixed level ring of 8 cameras at 3.5 m, focal 320 px, 256 × 192 images:
  91.4 px per metre at the ring's centre; the fixture's region joints lie 2.75–4.25 m from
  the camera (75–116 px/m, ±25 %). The projected pelvis–neck length of the same body ranges
  12.5–43 px across the fixture's views and poses (foreshortening), so it is no reliable
  scale.
- **Fixture** (`tests/fixtures/smpl_fixture`: 2 poses × 8 views; split train 1 / val 0 /
  test 1 pose): all 24 joints visible in all 16 samples; the body covers 7.5 % of an image.
  With the FR-001 regions, body pixels fill on average 50 % of a sample's hands region
  (min 23 %), 39 % of its feet region (min 32 %) and 70 % of its head region (min 53 %).
  Fixture test split, region MSE (all region pixels) next to the whole-image MSE:

  | Predictor | whole image | hands | feet | head |
  |---|---|---|---|---|
  | mean image (fixture train) | 0.0190 | 0.122 (6×) | 0.121 (6×) | 0.193 (10×) |
  | true image, Gaussian blur σ = 1 px | 0.0006 | 0.0080 (14×) | 0.0072 (13×) | 0.0049 (9×) |
  | true image, Gaussian blur σ = 2 px | 0.0015 | 0.0167 (11×) | 0.0196 (13×) | 0.0117 (8×) |

  Blur, the 004 failure mode, costs the thin extremities most. Region values are compared
  across models within a region, not with the whole image or with each other (each region
  has its own share of body and background pixels).
- **Tests and runs.** The 003 and 004 fixture runs report identical `mean_image` and
  `pca_k4` entries at `5382075`. `tests/test_train_selection.py` asserts that
  `metrics.json`'s whole `ae` entry equals a `score_images` re-score (`_rescore`), so it
  must pass the labels once entries gain regions.
- **Writer and versions.** `runlog.write_json_exact`: 6 significant digits, NaN → null,
  sorted keys. `metrics_version` 1 (003 contract: "breaking changes bump"); `train_version`
  2 (008; 008 plan: "`train_version` stays 2 for additive fields").
- **Environment.** No SMPL model, dataset or checkpoint in this container; pytest 267
  passed in 31.7 s at `5382075` (4 vCPU, Python 3.13); CI runs Python 3.11 and 3.13. A
  plain torch prototype of the region scoring (square masks + masked MSE, three regions)
  takes 2.2 s per model per 1,600-image split on 4 vCPU.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Extremity error of every model in a training run (Priority: P1)

The researcher trains the AE with `configs/ae.yaml`. Next to the whole-image metrics, the
run reports the reconstruction error inside three image regions — both hands, both feet,
the head — for the scored AE weights and for the mean image and PCA refit in the same run,
on every eval split.

**Why this priority**: this is the operator's decision (#16 (i)). The 004 reconstructions
lose fingers, faces and feet (README, Results), which the whole-image MSE, dominated by
background pixels, hardly shows; the next slice compares training variants on exactly
these regions.

**Independent Test**: fixture train run: every image model on test carries the three
regions with a finite `image_mse` in [0, 1] and `samples` 8, equal to a recomputation from
the scored weights with the library functions.

**Acceptance Scenarios**:

1. **Given** a train run and a scored split with samples, **When** the run finishes,
   **Then** `splits.<split>.image.<model>.regions.<region>` holds `image_mse` (mean over
   the split's samples whose region is non-empty) and `samples` (their count) for each
   region and for `ae`, `mean_image` and `pca_k<K>`; the top-level `regions` block names
   each region's joints and square size, and `units.regions` states the unit and the
   aggregation.
2. **Given** `select_weights: best_val` (008), **When** the run scores, **Then** the `ae`
   regions come from the same selected weights and the same predictions as the headline
   `ae` metrics.
3. **Given** the same config, data and code, **When** the run is repeated, or interrupted
   and resumed, **Then** `metrics.json` is byte-identical (004 FR-010, 008 FR-006).

### User Story 2 - Extremity error of the baselines (Priority: P1)

`strike-a-pose-eval baseline` reports the same regions for the mean image and every PCA k
on every eval split, so the PCA sweep shows how extremity error falls with k next to the
whole-image error.

**Why this priority**: constitution V (every model next to trivial baselines); baseline
and model numbers are only comparable if both CLIs score regions through the same code.

**Independent Test**: fixture baseline run; the 003 and 004 fixture runs report identical
`mean_image` and `pca_k4` entries, regions included.

**Acceptance Scenarios**:

1. **Given** a baseline run, **When** it scores a split with samples, **Then** every image
   baseline entry carries `regions` as in US1; pose entries are unchanged.
2. **Given** one dataset and split, **When** the 003 CLI and the 004 trainer score the mean
   image and PCA with the same k, **Then** their entries are identical, regions included.

### User Story 3 - Existing numbers, files and runs unchanged (Priority: P1)

Every number the project reports today keeps its value; the region metrics only add keys.

**Why this priority**: operator constraint (headline metrics and every existing number
byte-identical); the published 003 / 004 / 006 numbers and the re-baseline comparison rely
on it.

**Independent Test**: fixture smoke runs at `5382075` and after this slice.

**Acceptance Scenarios**:

1. **Given** the fixture baseline and train smoke runs before and after this slice, **When**
   both `metrics.json` are parsed and the added keys removed (top-level `regions`,
   `units.regions`, `regions` in each image-model entry), **Then** the objects are equal;
   the reconstruction figures, resolved `config.yaml` and `history.csv` (except `epoch_s`)
   are identical, and the CLI output lines are unchanged.
2. **Given** a run directory started before this slice, **When** `--resume` finishes it,
   **Then** it resumes as before (no config key is added) and its `metrics.json` carries
   the regions.
3. **Given** a reader of `metrics_version` 1 / `train_version` 2 files, **When** it reads a
   file written after this slice, **Then** every key it knows has its old meaning.

### User Story 4 - Region masks and masked MSE as library functions (Priority: P2)

A researcher (notebook, later slice) builds a region's pixel mask for a batch from its 2D
keypoints and visibility, and scores any image pair inside it, from NumPy arrays or torch
tensors alike, like the other 003 metric functions.

**Why this priority**: constitution I and IV (logic in the package, unit-tested); the
scoring of US1 and US2 is built from these functions.

**Independent Test**: hand-computed cases on NumPy and torch inputs.

**Acceptance Scenarios**:

1. **Given** a visible joint at (u, v) inside the image, **When** the mask of a region with
   half-size h is built, **Then** it is the (2h + 1) × (2h + 1) square of pixels centred
   on row ⌊v⌋, column ⌊u⌋.
2. **Given** joints near the image border, overlapping squares, or joints not visible,
   **When** the mask is built, **Then** squares are clipped to the image, overlapping
   pixels count once, and invisible joints add no pixel.
3. **Given** a mask and two image batches, **When** the masked MSE runs, **Then** each
   sample's value is the mean squared error over its mask's pixels, NaN where the mask is
   empty, returned as `[B]` of the input's array type.

### Edge Cases

- Region joint not visible (outside the image; behind the camera cannot happen on the
  ring): dropped; the region is the union of the remaining visible joints' squares. All of
  a region's joints invisible → the region is empty for that sample, which is left out of
  that region's mean and not counted in its `samples`; other regions are unaffected.
- Every sample of a split has an empty region → `image_mse` null, `samples` 0.
- A joint just outside the image (not visible) whose square would reach into the image:
  not scored (the label's visibility decides; clarification 5).
- Square partly outside the image: clipped; the sample's value is the mean over its
  in-image pixels.
- Overlapping squares within a region (hands together; ankle and foot of one foot): one
  union, each pixel once. Overlapping regions (a hand at the face): regions are
  independent; a pixel can count in two regions.
- A hand or foot hidden behind the body (no self-occlusion in the labels): its square
  shows the occluder and is scored as it is, for every model alike (documented limitation).
- Body parts closer to the camera than 3.5 m look up to ≈ 25 % larger than FR-001
  assumes, so fingertips or the crown can fall outside the square (documented; the FR-001
  squares are fixed in pixels).
- Keypoints held as float32 (004 trainer) or float64 (003): ⌊·⌋ of a float32 value and of
  its exact float64 conversion are equal, so both paths build the same masks.
- Keypoint on a pixel edge (u = 10.0) → column 10, matching the renderer's pixel footprint
  [c, c + 1).
- Empty split → `{"samples": 0, "poses": 0}` as today (no `image`, hence no regions).
- `val` not in the train config's `eval_splits` → not scored, so neither its masks nor its
  keypoints are kept (as today for the masks).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (regions): three regions MUST be defined in code (no config key), each the
  union of axis-aligned squares of side 2h + 1 px centred on SMPL-24 joints *(surrogate
  ruling 2026-10-20, batch Q1 (a))*:

  | Region | Joints (index) | h (px) | Square at 3.5 m | Covers (estimates; the SMPL mesh is not available in this container) |
  |---|---|---|---|---|
  | `hands` | `left_hand` 22, `right_hand` 23 | 10 | 21 px ≈ 23 cm | hand joint at the knuckles; wrist 8.6 cm (≈ 8 px) behind it, fingertips ≈ 10 cm beyond |
  | `feet` | `left_ankle` 7, `right_ankle` 8, `left_foot` 10, `right_foot` 11 | 8 | 17 px ≈ 19 cm, two per foot | ankle to ball of the foot 13.4 cm; heel a few cm behind the ankle; toes ≈ 5 cm beyond the foot joint |
  | `head` | `head` 15 | 14 | 29 px ≈ 32 cm | head joint low in the head, 8.3 cm above the neck joint; crown ≈ 13–17 cm above it; chin and nose ≈ 10 cm in front |

  The sizes are fixed a priori from body dimensions at the ring's 91.4 px/m and from crops
  of the fixture, never from val or test numbers (constitution V). They fit the current
  camera geometry only. Generalization, not built now (constitution VI): a metric
  half-size projected at each joint's depth, `h = f · s / z`, with `f` and `z` from the
  sample's `K` and `joints_3d_cam` (both in the sample dict); not the projected torso
  length, which varies 12.5–43 px with pose on the fixture. A dataset version that changes
  the camera distance or the body shape (#15) revisits this table.
- **FR-002** (region mask): `strike_a_pose.eval.metrics` MUST provide a pure batched
  function that builds a region's mask from 2D keypoints `[B, J, 2]` (u, v px), visibility
  `[B, J]`, the image size `(H, W)`, the region's joints and its half-size h: pixel (r, c)
  belongs to it iff some joint of the region is visible and |r − ⌊v⌋| ≤ h and
  |c − ⌊u⌋| ≤ h; only in-image pixels exist (clipping). Output: bool `[B, H, W]` of the
  keypoints' array type (NumPy in → NumPy out), row-major like the image rows.
- **FR-003** (masked MSE): `strike_a_pose.eval.metrics` MUST provide a pure batched
  function: per sample, the mean of `(pred − target)²` over the mask's pixels; NaN when the
  mask is empty; `pred`, `target` and `mask` of equal shape `[B, ...]` (else
  `ValueError`); `[B]` output of `pred`'s array type and float dtype, as `image_mse`.
- **FR-004** (scoring seam): `score_images` MUST accept, as one optional group, the true
  2D keypoints, visibility and image size of the scored rows. Given them, its result gains
  `regions`: for each FR-001 region, `image_mse` = float64 mean of the per-sample masked
  MSE over the samples whose region is non-empty, and `samples` = their count (`image_mse`
  NaN when 0). `pred_fn` stays called once per batch, and that prediction serves the
  headline and the regions; masks are built per batch, never stored per split. The
  headline values MUST be bit-identical with and without labels, and without labels the
  result equals today's.
- **FR-005** (003 baselines): `score_split` MUST pass every scored split's keypoints,
  visibility and image size, so that every image baseline entry (mean image, each PCA k)
  carries `regions`. Pose metrics and the PCA fits are unchanged.
- **FR-006** (004 trainer): the loading of scored splits (val when it is in `eval_splits`,
  every other eval split) MUST keep the true 2D keypoints and visibility next to the
  masks, and every image model scored in the run (`ae`, `mean_image`, `pca_k<K>`) MUST
  carry `regions`. Training, the per-epoch val loss, the weight selection (008),
  `predict_rows` and the figure are unchanged; the regions are scored in the evaluation
  block after `load_selected_weights`, from the predictions that give the headline.
- **FR-007** (`metrics.json`): keys MUST only be added, in the 003 and 004 files alike:
  - in every image-model entry of a split with samples: `"regions": {"feet": {"image_mse":
    …, "samples": …}, "hands": {…}, "head": {…}}`;
  - top level: `"regions": {"<region>": {"joints": [SMPL-24 joint names], "half_size_px":
    h}}`, the FR-001 table (the file says what it measured);
  - `units.regions`: one sentence giving the unit and the aggregation.

  `metrics_version` stays 1 and `train_version` stays 2: no existing key changes meaning
  (003 contract: breaking changes bump; precedent: the additive keys of 007 and of ruling
  #3). Floats 6 significant digits, NaN → null, sorted keys; bytes deterministic for the
  same inputs, code and machine / thread count.
- **FR-008** (config, CLI, figures): no new config key in 003 or 004 (the regions are
  constants; the code commit in `run_info.json` pins them and `metrics.json` records
  them); resolved `config.yaml`, `train.config.LEGACY_VALUES`, CLI flags, CLI output lines
  and reconstruction figures unchanged. Run directories started before this slice resume
  unchanged and report regions when they finish.
- **FR-009** (tests, written first; pytest):
  (a) mask, hand-computed: an interior joint at fractional (u, v) gives exactly the
  (2h + 1)² square at (⌊v⌋, ⌊u⌋); integer coordinates (u = 10.0 → column 10);
  (b) clipping at each border and corner (pixel count = product of the clipped extents),
  including a visible joint at u just below W;
  (c) union: two overlapping squares → the union's pixel count; two joints on one pixel →
  one square;
  (d) visibility: an invisible joint adds nothing, also one just outside the image whose
  square would reach in; all of a region's joints invisible → empty mask;
  (e) masked MSE: error only outside the mask → 0; error only inside → its mean over the
  mask; empty mask → NaN; shape mismatch → `ValueError`; output type and float dtype kept;
  (f) NumPy and torch inputs give identical values for (a)–(e);
  (g) placement on the fixture: every region of every fixture sample holds ≥ 10 % true-body
  pixels (measured minimum 23 %); with u and v swapped, 33 of the 48 sample-region pairs
  fall below 10 %, so the test catches a u / v or row / column mix-up;
  (h) `score_images` with labels: headline values bit-identical to the call without
  labels; `pred_fn` called once per batch; region values equal the mean of the per-sample
  masked MSE over non-empty samples (synthetic images with a known error inside and
  outside a region); `samples` with a sample whose hand joints are invisible (left out of
  `hands` only) and with every sample's region empty (`image_mse` NaN, written null);
  equal results for batch sizes 1, 3 and n;
  (i) 003 fixture run: every image baseline on test has the three regions with `image_mse`
  in [0, 1] and `samples` 8, the top-level `regions` block and `units.regions`; val stays
  `{"samples": 0, "poses": 0}`; `metrics.json` bytes repeat (existing test);
  (j) 004 fixture run: `ae`, `mean_image` and `pca_k4` likewise; the resume tests stay
  byte-equal with regions; `tests/test_train_selection.py` `_rescore` passes the labels,
  so its `ae` equality covers the regions of the selected weights;
  (k) the 003 and 004 fixture runs' `mean_image` and `pca_k4` entries are identical,
  regions included;
  (l) every other existing test stays green, edited only where it compares whole
  image-model entries or lists `metrics.json` keys.
- **FR-010** (documentation, same slice): see *Documentation scope*.
- **FR-011** (constraints): no new dependency, `pyproject.toml` / `uv.lock` untouched;
  Python 3.11-compatible code; `.github/workflows/ci.yml` unchanged (its fixture smoke
  steps run the new code); the 006 probe untouched; `docs/results/` untouched; nothing
  under `data/`, `runs/` or `assets/` committed.

### Documentation scope (FR-010)

- `specs/003-baselines-and-metrics/spec.md`: FR-001 and FR-002 get dated notes "amended
  2026-10-20 by slice 009" (region mask and masked MSE functions; per-region MSE for every
  image model); the *Key Entities* `metrics.json` line gains one clause; original text
  kept.
- `specs/003-baselines-and-metrics/contracts/eval-run.md`: metrics API rows for the two
  functions and the region constants; the `metrics.json` example and text gain the
  top-level `regions`, per-entry `regions` and `units.regions`, with the note that
  `metrics_version` stays 1 (additive, slice 009).
- `specs/004-autoencoder-mvp/contracts/train-run.md`: `metrics.json` example and text:
  `regions` for every image model; `train_version` stays 2.
- `specs/004-autoencoder-mvp/spec.md`: FR-006 dated note (it names the scoring functions;
  008 amended it the same way). *(lead default)*
- `README.md`: status line and slice table (009 row); Usage: the metric-functions
  sentence (region mask, masked MSE) and the baseline and train paragraphs (both report
  the MSE in hands, feet and head regions for every image model; what a region is; models
  are compared within a region); Results: one sentence that the published runs predate
  009 and that per-region numbers come with the next full run.
- `specs/roadmap.md`: open question #16 records the operator's choice (i), 2026-10-20,
  implemented by 009 (options ii–v still open); the "Out of MVP" candidates line; 009 row
  in the post-MVP table; status line; changelog entry dated 2026-10-20, appended at the
  end.
- `docs/HANDOFF.md`: State: 009 implemented (approved at surrogate scope, operator
  sign-off pending). Next work unchanged in order and content, except that item 3 records
  009 as done and item 4 (re-baseline) notes that its `metrics.json` brings the first
  per-region numbers (re-running the 003 baselines, ≈ 3 min, adds them for every PCA k;
  optional). *(lead default)*
- Not edited: `docs/results/`; 001–008 `plan.md`, `research.md`, `tasks.md` (historical
  records); `.github/workflows/ci.yml`; the 006 spec and contract.

### Key Entities

- **Region**: name, SMPL-24 joints, half-size h in px (FR-001).
- **Region mask**: per sample, the pixels of a region (FR-002).
- **Region record** (`metrics.json`, per split, image model and region): `image_mse`,
  `samples` (FR-007).
- **Region definitions** (`metrics.json` top-level `regions`): the FR-001 table.

## Success Criteria *(mandatory)*

- **SC-001**: ruff, ruff format, pytest and the CI smoke commands are green locally, and CI
  is green on Python 3.11 and 3.13 after the orchestrator's push; pytest stays under ~45 s
  locally (31.7 s at `5382075`); per-leg CI wall time stays < 5 min.
- **SC-002**: the fixture baseline and train smoke runs at `5382075` and after this slice:
  `metrics.json` objects equal once the added keys are removed; figures, resolved configs
  and `history.csv` (except `epoch_s`) identical; CLI output lines identical.
- **SC-003**: every new function has hand-computed tests run on NumPy and torch inputs, and
  both array types give identical values.
- **SC-004**: in the 003 and 004 fixture runs every image-model entry on test has the three
  regions with `image_mse` in [0, 1] and `samples` 8, and the `mean_image` / `pca_k4`
  entries of the two runs are identical, regions included.
- **SC-005**: scoring the regions of one model on one split of 1,600 images of 256 × 192
  takes ≤ 5 s on the 4-vCPU dev machine (prototype 2.2 s; measured once with a synthetic
  benchmark in the tasks' verification record), i.e. at most a minute added to a full 003
  run (≈ 3 min today) and at most 30 s to a full train run.
- **SC-006**: `docs/results/` is byte-unchanged and no document reports per-region numbers
  for the published 003 / 004 runs.
- "Done" does not depend on a full-dataset run.

## Assumptions

- The first real numbers come with the re-baseline train run (after SMPL access and F1):
  regions for `ae`, `mean_image` and `pca_k64` on val and test. Until then the README and
  the roadmap say they come with the next full run.
- Region numbers are comparable only between runs on datasets with the current camera
  geometry (ring at 3.5 m, focal 320 px, 256 × 192) and the same FR-001 table.
- Regions are image areas around projected joints, not body-part segmentations (001 has
  no part labels): they also hold background and neighbouring body (forearm, shin, neck,
  shoulders); the same pixels are scored for every model.
- The next slice keeps every predictor behind `predict_rows` → `score_images`, so its
  variants get regions without further change.

## Notes for the plan (specify session, 2026-10-20)

Facts found while specifying that shape the plan and the tasks (not requirements; checked
at `5382075`):

1. **First step, before any source edit:** run the fixture baseline and train smoke
   commands (README, "Development checks") into a run root outside the repository and keep
   the run directories and their console output: they are the SC-002 "before" reference
   (`metrics.json`, `recon_test.png`, `config.yaml`, `history.csv`, output lines).
2. **Trainer evaluation path (FR-006).** `train/trainer.py` `load_split(ds, batch_size,
   masks=False)` keeps only `image` (uint8 `[n, 1, H, W]`), `pose_index` and, with
   `masks=True`, `mask` (bool `[n, H*W]`): it holds **no** keypoints or visibility today.
   Val is loaded before training with `masks="val" in cfg.eval_splits` and held for the
   whole run; every other eval split is loaded in `_train_and_evaluate` with
   `masks=True`. Keep `keypoints_2d` (`[n, 24, 2]`, float32 as the dataset returns it) and
   `visibility` (bool `[n, 24]`) whenever the masks are kept (≈ 0.35 MB for 1,600
   samples), and pass them with `a["image_size"]` to the `score_images` call of the
   evaluation loop (`r = score_images(fn, x, a["mask"], cfg.batch_size,
   cfg.iou_threshold)`), which runs after `load_selected_weights`.
3. **003 path (FR-005).** `eval/baselines.py` `load_arrays` already holds `kp2d` (float64
   `[n, 24, 2]`), `visible` (`[n, 24]`) and `image_size` for every eval split; only the
   `score_images` call in `score_split` changes.
4. **`score_images` (FR-004)** receives flattened rows `[n, H*W]`, so it needs the image
   size to build the masks; a `[B, H, W]` mask reshaped to `[B, H*W]` is row-major like
   the rows (images `[n, 1, H, W]` reshaped with `reshape(n, -1)`). The headline lists
   (`mse`, `ps`, `iou`) and their float64 means stay exactly as they are.
5. **`metrics.json` assembly (FR-007).** 003: the `metrics` dict in `run_baselines`, units
   from `UNITS` in `eval/baselines.py`; 004: the `metrics` dict in `_train_and_evaluate`,
   whose units are `IMAGE_UNITS`, a selection of `eval.baselines.UNITS` (add `regions` to
   that selection too). Joint names for the top-level `regions` block come from
   `datagen.skeleton.SMPL24_JOINT_NAMES`.
6. **Existing tests to edit (FR-009 (l)).** `tests/test_train_selection.py` `_rescore`
   loads val with `load_split(ds, cfg.batch_size, masks=True)` and calls `score_images`
   without labels; its result is compared with whole `ae` entries (`m["splits"]["val"]
   ["image"]["ae"] == …`, two places), so it must pass the keypoints, visibility and image
   size. Its run-versus-run comparisons of `splits` stay valid (both runs carry regions).
   `tests/test_eval_report.py::test_score_images_matches_metrics` calls without labels and
   stays valid as it is (FR-004). `tests/test_eval_cli.py` and `tests/test_train.py` check
   model-name sets and value ranges, which stay valid; `tests/test_probe.py` is unaffected.
7. **Fixture facts for the tests.** All 24 joints are visible in all 16 fixture samples,
   so every fixture region has `samples` 8 on test and the empty-region paths can only be
   tested with synthetic inputs. Body fill of the FR-001 regions on the fixture: minimum
   23 % (hands), 32 % (feet), 53 % (head); with u and v swapped, 33 of the 48
   sample-region pairs fall below 10 % (FR-009 (g)). The 003 and 004 fixture runs give
   identical `mean_image` and `pca_k4` entries today (FR-009 (k)).
8. **Cost (SC-005).** A plain torch prototype (per joint, a row-range and a column-range
   mask combined into `[B, H, W]` and OR-ed into the region; masked sum of the squared
   error; B = 256, 7 joints, 3 regions) takes ≈ 2.2 s per model per 1,600-image split on
   4 vCPU: ≈ 27 s per full 003 run (6 image models × 2 splits) and ≈ 13 s per full train
   run (3 × 2). The masks depend only on the labels; rebuilding them for each model is
   affordable at this cost.

## Clarifications

### Session 2026-10-20 (autonomous run, Fable surrogate scope)

Defaults chosen by the lead within the surrogate's scope are marked *(lead default)*; the
questions sent to the surrogate in one batch carry *(surrogate ruling 2026-10-20)*. The
surrogate ruled (a) on all three questions and accepted every lead default, naming in
particular: no version bump, no config key and no `LEGACY_VALUES` entry; invisible
joints dropped, with `samples` counts; one region per body part; keypoints, visibility
and image size carried through the trainer's evaluation path and `score_images`; the
top-level `regions` block and `units.regions`; the cross-path test (the 003 and 004
fixture runs give identical `mean_image` and `pca_k4` entries, regions included); the
placement test (≥ 10 % body pixels); depth scaling recorded as the future
generalization; the occlusion limitation recorded. Every answer is provisional and
overturnable by the operator.

1. Q: Slice number and slug? → A: `009-region-metrics` (next free number; names the
   change). *(lead default)*
2. Q: Which joints and square sizes form each region? → A: FR-001: hands = hand joints,
   h 10; feet = ankle and foot joints, h 8; head = head joint, h 14; squares of side
   2h + 1 (batch Q1, option (a)). *(surrogate ruling 2026-10-20)*
3. Q: Square or disc? → A: square (the operator's "head box"; exact pixel counts that a
   test can state by hand). *(lead default)*
4. Q: Which pixel holds a keypoint? → A: row ⌊v⌋, column ⌊u⌋: the renderer samples pixel
   centres, so pixel (r, c) covers [c, c + 1) × [r, r + 1), and `visibility` (`0 ≤ u < W`,
   `0 ≤ v < H`) keeps that pixel in the image. *(lead default)*
5. Q: Joints that are not visible? → A: dropped; a region with no visible joint is empty
   and the sample is left out of that region's mean, as `samples` records. The label's
   visibility decides, as in 003's `keypoint_error_2d`. Considered: use every joint in
   front of the camera and clip its square, which would score the in-frame fingers of a
   hand joint just outside the image; rejected for consistency with the labels and because
   an occlusion-aware visibility (later) must drop hidden joints. *(lead default; brief)*
6. Q: Which pixels of a region are scored? → A: all of them, body and background,
   matching the headline MSE, which counts background (batch Q2, option (a)).
   *(surrogate ruling 2026-10-20)*
7. Q: Which metrics per region? → A: MSE only; an IoU per region can be added later as
   new keys (batch Q3, option (a)). *(surrogate ruling 2026-10-20)*
8. Q: Aggregation over a split? → A: per-sample masked MSE, then the mean over the samples
   with a non-empty region (each image counts once, as for the headline MSE), accumulated
   in float64. Considered: a pixel-weighted mean over all region pixels of the split
   (weights large, unclipped regions more). *(lead default)*
9. Q: Left and right separately? → A: no: one region per body part, both sides together
   (the questions are extremity versus whole image and model versus model). *(lead
   default)*
10. Q: Where does the code live? → A: `eval/metrics.py` (region constants, mask function,
    masked MSE: pure, batched, NumPy or torch, next to the other 003 metrics);
    `score_images` takes the labels as one optional group. Masks are built per batch:
    stored per split they would take 3 × 1,600 × 49,152 B ≈ 236 MB. Function and constant
    names are fixed in the plan. *(lead default)*
11. Q: `metrics.json` layout and versions? → A: FR-007: nested `regions` per image-model
    entry (the count sits next to the value it qualifies, and there is room for more
    metrics), a top-level definitions block, `units.regions`; no version bump (additive).
    Considered: flat keys (`hands_mse`) with counts once per split. *(lead default)*
12. Q: Config keys? → A: none (FR-008): constants, recorded in `metrics.json`; a key would
    need validation and a `LEGACY_VALUES` entry for a value no run should tune. *(lead
    default; brief)*
13. Q: Figure and CLI output? → A: unchanged. Considered: drawing the squares on the
    figure's `true` column (a visual placement check), but it changes the 003 / 004 figure
    bytes; FR-009 (g) checks placement instead. *(lead default)*
14. Q: The published numbers? → A: unchanged and not re-scored (their datasets and
    checkpoints no longer exist); README and roadmap say per-region numbers come with the
    next full run. *(lead default)*
15. Q: Out of scope? → A: depth-scaled squares (FR-001), per-side regions, body-part
    segmentation, region overlays, per-region numbers for the train split, training,
    config, CI and probe changes, re-running published runs. *(lead default)*
16. Q: Dates? → A: the session clock, 2026-10-20; the roadmap changelog entry is appended
    at the end. *(lead default)* *(Plan, 2026-10-21: phase 2 runs on 2026-10-21 by the
    session clock, so the dated notes of the Documentation scope ("amended 2026-10-21 by
    slice 009") and the roadmap changelog entry carry 2026-10-21.)*

### Clarification batch for the surrogate (sent and ruled 2026-10-20)

- **Q1 — Region definitions (FR-001).** (a) Per-region squares sized from body dimensions
  at the ring: hands = `left_hand` / `right_hand` (22, 23), h = 10 (21 px); feet =
  `left_ankle` / `right_ankle` / `left_foot` / `right_foot` (7, 8, 10, 11), h = 8 (17 px);
  head = `head` (15), h = 14 (29 px). (b) One size for every region, around the end joints
  only: h = 12 (25 px) around 22 / 23, 10 / 11 and 15: one constant, but in side views the
  heel sits at or past the square's edge (the ankle is 13.4 cm ≈ 12 px behind the foot
  joint), and upright heads lose part of the crown. (c) As (a) with the wrists (20, 21)
  added to hands: (a) already contains the wrist (≈ 8 px from the hand joint), so this
  adds ≈ 10 cm of forearm per hand, which reconstructs well and dilutes the hand signal.
  **Lead proposal: (a)**: it covers each part in every view at the ring's scale with the
  fewest neighbouring pixels; three constants with a stated rationale.
  **Ruling (surrogate, 2026-10-20): (a)** (hands = joints 22 / 23, h 10; feet = 7 / 8 /
  10 / 11, h 8; head = 15, h 14; squares of side 2h + 1).
- **Q2 — Which pixels of a region are scored?** (a) All pixels in the region, body and
  background: the 003 MSE restricted to the region; it sees both detail lost on the body
  (fingers) and blur or ghost limbs spilling onto the background; values depend on each
  region's background share, so models are compared within a region. (b) Only true-body
  pixels in the region (region ∩ mask): no background share, but blind to halos and to
  limbs drawn where the body is not. (c) Both, as two keys per region. Fixture (test split,
  hands / feet / head): blur σ = 1 px — all 0.0080 / 0.0072 / 0.0049, body-only 0.0087 /
  0.0102 / 0.0042; mean image — all 0.122 / 0.121 / 0.193, body-only 0.187 / 0.272 /
  0.253. **Lead proposal: (a)**: the operator's "MSE" over a box, consistent with the
  headline MSE (which counts background), and one number per region.
  **Ruling (surrogate, 2026-10-20): (a)**: all pixels inside the region, body and
  background, matching the headline MSE.
- **Q3 — Metrics per region.** (a) MSE only. (b) MSE and silhouette IoU inside the region
  (`|(pred > 0.075) ∧ mask ∧ R| / |((pred > 0.075) ∨ mask) ∧ R|`): whether fingers and toes
  survive thresholding; one more function, key and test set. Fixture: under blur the region
  IoU stays close to the whole-image IoU except for the feet (σ = 1 px: hands 0.88 / feet
  0.78 / head 0.93 vs whole 0.88; σ = 2 px: 0.80 / 0.64 / 0.88 vs 0.78), while the region
  MSE singles out the extremities (11–14× the whole-image MSE for hands and feet, 8–9× for
  the head). (c) MSE, IoU and PSNR. **Lead proposal: (a)**: what the operator chose; PSNR
  is a per-sample transform of the MSE; an IoU can be added later as additive keys if the
  next slice's silhouette-oriented variants need it.
  **Ruling (surrogate, 2026-10-20): (a)**: MSE only; an IoU per region can be added
  later as new keys.
