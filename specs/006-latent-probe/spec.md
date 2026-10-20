# Feature Specification: Latent Probe

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `006-latent-probe`)
**Created**: 2026-10-20
**Status**: Approved (surrogate ruling, operator sign-off pending) — surrogate
rulings 2026-10-20 #1 (exactly one post-MVP slice) and #2 (images as model input),
recorded in [`../roadmap.md`](../roadmap.md); provisional and overturnable.
**Input**: Roadmap post-MVP slice 006 + lead brief 2026-10-20 (Fable surrogate
scope): "freeze the 004 encoder; compute latents once from
`runs/ae-smpl_proc_v1-20261020T030038Z/checkpoint.pt` (final weights) for
train/val/test of `smpl_proc_v1` with `splits/smpl_proc_v1.json`, cached under
git-ignored `runs/`. Targets in order: (1) root-relative 3D joints → MPJPE mm via
the 003 `mpjpe`, next to the 003 mean-pose (253 mm) and pose-PCA (k=8 94 mm, k=16
34 mm) baselines — PCA sees the true joints, the probe sees only images; (2) camera
yaw as 8-way classification (accuracy + confusion, e.g. front/back mix-ups; say
why not regression); (3) SMPL axis-angle pose [72] optional. Models: closed-form
ridge / linear probe first (deterministic; λ chosen on val), then a small MLP probe
(fit on train, tuned on val, test scored once at the end). No new dependencies.
CLI + config + fixture config, run dir via `strike_a_pose.runlog`, `metrics.json`
with baselines side by side, a small PIL figure, summary in
`docs/results/006-probe-smpl_proc_v1/`, README + roadmap, fixture pytest, CI smoke
only if per-leg CI stays < 5 min. No edits to datagen / data / models / train
beyond additive helpers. Each full-dataset run ≤ 30 min CPU."

Settled context: the 004 `ConvAutoencoder` (latent 64) was trained on train images
only, by reconstruction (MSE) — it never saw joints, pose parameters or camera
labels. Its final-epoch weights are the reported MVP result (surrogate ruling #3).
Data: `smpl_proc_v1`, 2,000 procedural poses × 8 ring cameras (yaw 0°, 45°, …,
315°; fixed distance, height and focal length; root orientation zeroed, mean body
shape), frozen pose-level split 1600 / 200 / 200 poses (12,800 / 1,600 / 1,600
images). 003 world-frame pose baselines (val / test): mean pose 253.4 / 250.3 mm,
PCA k=8 94.0 / 93.6 mm, k=16 34.2 / 34.6 mm.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Probe joints from the frozen latent (Priority: P1)

A researcher runs `strike-a-pose-eval probe --config configs/probe.yaml`; the
encoder from a 004 checkpoint maps every image to its latent once (cached), linear
and MLP probes are fit on train latents to predict root-relative 3D joints, tuned on
val, and scored with MPJPE (mm) next to the mean-pose and pose-PCA baselines.

**Why this priority**: answers "what pose information does the AE latent hold?",
the old linear-model probe of the predecessor; constitution V (baselines side by
side).

**Independent Test**: fixture run (fixture AE checkpoint trained in the test) writes
a run dir whose `metrics.json` has `ridge`, `mlp`, `mean_pose`, `pca_k<K>` entries
for each joint target and evaluated split.

**Acceptance Scenarios**:

1. **Given** a 004 checkpoint, a dataset and its split, **When** the probe CLI runs,
   **Then** a new run directory holds the resolved config, `run_info.json`
   (003 fields + checkpoint path / SHA-256 / epoch / commit of its run), and
   `metrics.json` with every probe and baseline per target and split.
2. **Given** the same inputs twice, **When** the probe runs, **Then** `metrics.json`
   bytes are identical (deterministic ridge, seeded MLP, CPU).
3. **Given** a second run on the same checkpoint + dataset + split, **When** it
   starts, **Then** it reuses the cached latents instead of re-encoding (cache keyed
   by checkpoint, manifest and split hashes) and the results are unchanged.
4. **Given** a missing checkpoint, an invalid config, or a split that does not fit
   the dataset, **When** the CLI runs, **Then** it exits non-zero with a clear
   message and no run directory.

### User Story 2 - Probe camera yaw (Priority: P1)

The same run predicts which of the 8 ring cameras took each image (8-way
classification) with a linear and an MLP probe and reports accuracy, the confusion
matrix and the error breakdown by angular distance (45° neighbours, 90°, 135°,
180° front/back flips) next to a majority-class baseline.

**Independent Test**: fixture run has `yaw` entries with `accuracy`, an 8×8
`confusion` whose total equals the split size, and `errors_by_offset_deg`.

### User Story 3 - Full-dataset probe recorded (Priority: P2)

One probe run on `smpl_proc_v1` from the 004 final-epoch checkpoint; numbers,
figure, wall time and an honest interpretation recorded in docs citing run dir,
config and commit.

### Edge Cases

- Empty val split (fixture) → no selection possible: the first grid value is used
  and recorded as `selected_by: "default (empty val)"`; the split is reported with
  `samples: 0`.
- Checkpoint image size ≠ dataset image size → config error before the run dir.
- Cached latents from another checkpoint / dataset / split → different cache key, so
  never reused; a corrupt or mismatching cache file is recomputed.
- Grid ties on val → first value in config order wins.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Module `strike_a_pose.eval.probe` MUST encode every image of the
  train / val / test splits with the frozen encoder of a 004 checkpoint (eval mode,
  no gradient) into `[n, latent_dim]` float32 latents, together with the labels the
  probes need, and cache them under a git-ignored directory keyed by the SHA-256 of
  the checkpoint, the dataset manifest and the split file.
- **FR-002**: Probe targets: `joints_world` (root-relative 3D joints, world frame =
  the 003 baseline frame, `[24×3]` metres), `joints_cam` (root-relative, camera
  frame, i.e. as seen in the image), `yaw` (camera index 0–7 of the ring), and
  `smpl_pose` (SMPL axis-angle `[72]`, optional / last).
- **FR-003**: Linear probe: closed-form ridge regression on train-standardized
  latents with intercept, float64 (`(ZᵀZ + λ·n·I) W = ZᵀY`), λ from a config grid
  chosen on val; for `yaw`, ridge on one-hot targets with arg-max prediction.
- **FR-004**: MLP probe: `latent → hidden layers (ReLU) → target`, Adam, seeded,
  fixed epochs; MSE on train-standardized targets for regression, cross-entropy for
  `yaw`; weight decay chosen on val from a config grid.
- **FR-005**: Fitting uses train only; selection uses val only; test is scored once,
  after selection, with the selected models; test numbers never feed back.
- **FR-006**: Baselines fit on train in the same run: mean target (mean pose /
  majority class) for every target; pose PCA with `pose_pca_ks` components on the
  true joints for both joint frames (oracle reference: PCA sees the true joints, the
  probes see only image latents).
- **FR-007**: Metrics: joints — 003 `mpjpe` (root-relative, ×1000 → mm) and 003
  `keypoint_error_2d` (px; prediction + true root projected with the true camera);
  yaw — accuracy, 8×8 confusion (rows true, columns predicted, ring order) and
  error counts by angular offset {45, 90, 135, 180}°; smpl_pose — mean geodesic
  rotation error (degrees) over the posed joints 1–21.
- **FR-008**: CLI `strike-a-pose-eval probe --config PATH [--seed INT]
  [--run-root DIR] [--checkpoint PATH]`; overrides are written into the resolved
  config; exit codes as 003 (0 OK, 1 dataset/split invalid, 2 config error incl.
  missing / incompatible checkpoint, 4 run dir exists).
- **FR-009**: Run dir (`strike_a_pose.runlog`): `config.yaml`, `run_info.json`,
  `metrics.json` (sorted keys, 6 significant digits), `skeleton_<split>.png`
  (true vs predicted skeletons on the input images) and `yaw_confusion_<split>.png`;
  no checkpoint or latents in the run dir.
- **FR-010**: Optional random-encoder control (`random_encoder_control: true`):
  the same probes on latents of an untrained, seeded `ConvAutoencoder` encoder of
  the same architecture, to separate what training put into the latent from what any
  conv feature map of the image gives.
- **FR-011**: Fixture config `configs/probe_fixture.yaml`; pytest end-to-end on the
  committed fixture; CI smoke step only if per-leg CI stays < 5 min.
- **FR-012**: No edits to `datagen`, `data`, `models` or `train` except additive
  helpers (and surrogate ruling #3's additive `best.pt` in `train`); the AE
  checkpoint and dataset are read-only inputs.

### Key Entities

- **Probe config** (YAML), **latent cache**, **run directory**:
  `contracts/probe-run.md`.

## Success Criteria *(mandatory)*

- **SC-001**: Fixture probe CLI (incl. a fixture AE train) < 60 s in pytest; CI leg
  < 5 min.
- **SC-002**: Full-dataset probe ≤ 30 min CPU on the 4-vCPU machine (Fable cap).
- **SC-003**: `metrics.json` reports, for val and test, ridge + MLP probes next to
  mean / PCA baselines (joints), majority class (yaw) and mean pose (smpl_pose); the
  docs state that PCA sees the true joints.
- **SC-004**: `uv sync --locked`, ruff, pytest green locally and on CI.

## Assumptions

- 003 metrics / PCA, 002 loader and 004 `load_checkpoint` are reused unchanged.
- The 004 final checkpoint and `smpl_proc_v1` + split are present locally; neither
  is regenerated or committed.

## Clarifications

### Session 2026-10-20 (autonomous run, Fable surrogate scope)

Defaults chosen by the lead within the surrogate's scope; each is provisional.

1. Q: Which frame for the joint target? → A: both. `joints_world` is the headline
   because it is the frame of the 003 baselines (root orientation is zeroed in the
   data, so the world-frame skeleton is the same for all 8 views of a pose: the
   probe must undo the view). `joints_cam` is what the image actually shows; its
   mean-pose / PCA baselines are refit in the camera frame in the same run.
2. Q: Why classification for yaw, not regression? → A: the ring has exactly 8 fixed
   yaws (0°, 45°, …, 315°), so the label is discrete; a scalar angle wraps at
   0°/360° (a 350° vs 10° error looks like 340°), and (sin, cos) regression still
   has to be snapped to 8 values. Classification is exact for this label set and
   its confusion matrix directly exposes front/back (180°) and left/right mirror
   mix-ups.
3. Q: Ridge λ grid / scale? → A: latents standardized with train mean / std (std
   floor 1e-8); penalty `λ·n` with `λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}`; selection
   on val (joints: MPJPE; yaw: accuracy; smpl_pose: geodesic error); ties → first.
4. Q: MLP shape / training? → A: hidden `[256, 256]`, ReLU, Adam lr 1e-3, batch 256,
   100 epochs, cosine lr decay, seeded shuffling; weight decay grid
   `{0, 1e-4, 1e-3}` chosen on val (same criteria as ridge). No early stopping
   (val only picks the grid value).
5. Q: Baselines? → A: mean pose + pose PCA `[8, 16]` per joint frame (world-frame
   numbers must equal 003's: consistency check), majority class for yaw, mean pose
   vector for smpl_pose. PCA is an oracle reference (it compresses the true joints),
   not an image-based competitor.
6. Q: smpl_pose target (optional)? → A: included (cheap once latents exist):
   ridge / MLP on the 72-vector, scored by geodesic angle between predicted and true
   joint rotations over joints 1–21 (root and hands are zero in the data). Linear
   regression in axis-angle space is crude near ±π; reported as is.
7. Q: Random-encoder control? → A: on by default for the full run (FR-010): seeded
   untrained encoder, same probes. Lead addition: without it a good probe number
   could come from image features alone rather than from what AE training learned.
8. Q: Which checkpoint weights? → A: the final-epoch weights in `checkpoint.pt` of
   run `ae-smpl_proc_v1-20261020T030038Z` (surrogate ruling #3; that run has no
   `best.pt`).
9. Q: Latent cache? → A: `runs/probe-cache/<sha12(ckpt)>-<sha12(manifest)>-
   <sha12(split)>.pt` (one torch file: labels + latents of every encoder per split
   + the three full hashes, seed and encoder names; any mismatch → recompute).
   Default `cache_dir: runs/probe-cache`; `null` disables caching (fixture tests).
   *(Amended by the analyze pass: one file for both encoders, not one per encoder.)*
10. Q: Figures? → A: PIL only. `skeleton_val.png`: 6 evenly spaced val images with
    the true skeleton and the selected ridge / MLP `joints_world` predictions
    projected with the true root + camera (one column per probe); and
    `yaw_confusion_val.png`: 8×8 count grid for the MLP yaw probe (and ridge).
11. Q: Test split? → A: scored once in the full run, after selection; a pilot (if
    any) scores val only (`eval_splits: [val]`).
12. Q: Where does the code live? → A: `strike_a_pose.eval.probe` (+
    `eval/probe_config.py`), subcommand of the existing `strike-a-pose-eval` CLI;
    no new console script, no new dependency.
