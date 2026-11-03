# Feature Specification: Baselines and Metrics

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `003-baselines-and-metrics`)
**Created**: 2026-10-20
**Status**: Approved (autonomous run, operator delegation 2026-10-20); Operator sign-off 2026-10-21 ("approve all")
**Input**: Roadmap slice 003 + lead brief 2026-10-20: "package `strike_a_pose.eval`:
metrics — image reconstruction (per-pixel MSE + PSNR and/or silhouette IoU on
thresholded output), MPJPE (root-relative, metres → mm), 2D keypoint error (px);
pure batched torch/numpy functions, unit-tested with hand-computed cases.
Baselines — image: mean image and PCA with k components (fit on train only);
pose: mean pose (root-relative 3D joints) and PCA on joints. Config-driven eval
CLI writing a run directory (resolved config, git commit, environment, seed,
metrics JSON, reconstruction figure). Never tune on test; report val and test.
Run the baselines on the full dataset and record the numbers. Tests on the
committed fixture only."

Settled context (operator): MVP = image autoencoder that *works* (slice 004);
baselines are reported for context, **not** a bar to beat (roadmap Q3). Images
are 1×256×192 grey-on-black in [0, 1]; labels include 2D/3D keypoints (smpl24)
and pose parameters (002 sample contract). Split unit = pose; frozen split files
under `splits/`. Prerequisite (001 T025): the full default dataset
`smpl_proc_v1` is generated and its split is frozen as `splits/smpl_proc_v1.json`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Score a reconstruction with standard metrics (Priority: P1)

A researcher (or the 004 eval code) passes batches of predicted and true images,
3D joints or 2D keypoints to library functions and gets per-sample errors in
stated units, from NumPy arrays or torch tensors alike.

**Why this priority**: 004 reports its autoencoder with exactly these functions;
the baselines below are only comparable if both use the same code.

**Independent Test**: Hand-computed cases (tiny arrays whose answers are worked
out by hand) for every metric, run on NumPy and torch inputs.

**Acceptance Scenarios**:

1. **Given** a prediction equal to the target, **When** any metric runs,
   **Then** errors are 0, IoU is 1 and PSNR is at its documented cap.
2. **Given** a batch `[B, ...]`, **When** a metric runs, **Then** it returns `[B]`
   per-sample values of the input's array type.
3. **Given** 3D joints offset by a constant translation, **When** MPJPE runs,
   **Then** the error is unchanged (root-relative).
4. **Given** joints marked not visible, **When** the 2D keypoint error runs,
   **Then** they are excluded from the per-sample mean.

### User Story 2 - Produce baseline numbers from one logged run (Priority: P1)

The researcher runs one CLI command with a versioned config; it fits trivial
baselines on the train split only, evaluates them on val and test, and writes a
self-describing run directory whose numbers can be quoted in docs.

**Why this priority**: Constitution V (each model reported against a trivial
baseline on a frozen held-out split) and III (a number is a result only if it
can be regenerated from the run's artifacts).

**Independent Test**: Run the CLI on the committed fixture with a fixture config
into a temp dir; check the run directory contents; run twice and compare
`metrics.json` bytes.

**Acceptance Scenarios**:

1. **Given** a dataset, its split file and a config, **When** `strike-a-pose-eval
   baseline --config C` runs, **Then** a new run directory holds the resolved
   config, run info (git commit + dirty flag, argv, seed, environment, dataset
   manifest hash, split file hash), `metrics.json` with val and test results
   for every baseline, and a reconstruction figure PNG.
2. **Given** the same config, data and code, **When** the command runs twice,
   **Then** both `metrics.json` files are byte-identical.
3. **Given** a split file that does not match the dataset, or an invalid config,
   **When** the command runs, **Then** it exits non-zero with a clear message and
   no metrics.
4. **Given** a split with no samples (e.g. the fixture's empty val), **When** the
   command runs, **Then** that split is reported with `samples: 0` and no
   metric values, without failing.

### User Story 3 - Full-dataset baseline numbers recorded (Priority: P2)

The baseline run on `smpl_proc_v1` is done once and its key numbers (val and
test) are recorded in this slice's verification record and the roadmap
changelog, citing the run directory, config and git commit.

**Independent Test**: The cited numbers equal the run's `metrics.json`.

### Edge Cases

- PCA `k` larger than the training data's numerical rank (tiny fixture: 8 train
  images; 1 train pose) → components with ~zero variance are dropped; the
  effective `k` is recorded next to the requested `k`.
- MSE of 0 → PSNR capped (MSE floored at 1e-10 → 100 dB), never `inf`.
- Empty predicted and true silhouettes → IoU 1.
- A sample with no visible joints → its 2D error is NaN and it is excluded from
  the split mean (count recorded).
- Config with unknown keys, empty `k` lists, non-positive `k`, or unknown split
  names → config error (exit 2). The fit split is always `train` and is not
  configurable.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Package `strike_a_pose.eval` MUST provide pure, batched metric
  functions accepting NumPy arrays or torch tensors (`[B, ...]` → `[B]`, same
  array type): `image_mse`, `psnr`, `silhouette_iou`, `mpjpe`,
  `keypoint_error_2d`.
  *(Amended 2026-10-21 by slice [009](../009-region-metrics/spec.md): also
  `region_mask` (a region's pixels, bool `[B, H, W]`, from 2D keypoints and visibility)
  and `masked_mse` (per-sample MSE over a mask, NaN when it is empty), with the region
  constants `REGIONS`.)*
- **FR-002**: Image metrics work on images in `[0, 1]` (002 normalization):
  per-image mean squared error over pixels; PSNR = `10·log10(1 / max(MSE, 1e-10))`
  dB; silhouette IoU between `pred > threshold` and the true body mask.
  *(Amended 2026-10-21 by slice [009](../009-region-metrics/spec.md): every image model
  is also scored by its MSE inside three regions built from the true 2D keypoints —
  `hands`, `feet`, `head` (squares of side 21 / 17 / 29 px on joints 22 / 23,
  7 / 8 / 10 / 11 and 15, clipped to the image) — averaged over the images whose region
  is non-empty; additive `metrics.json` keys, headline metrics unchanged.)*
- **FR-003**: MPJPE MUST subtract the root joint (pelvis, index 0) from both
  prediction and target, then average the Euclidean joint error; inputs in
  metres, reports in **millimetres** (`× 1000`, key suffix `_mm`).
- **FR-004**: 2D keypoint error MUST be the mean Euclidean pixel distance over
  joints, optionally restricted to visible joints (key suffix `_px`).
- **FR-005**: Baselines MUST be fit on the train split only (not configurable)
  and evaluated on a configured list of splits (default val, test):
  - image: per-pixel **mean image**; **PCA** reconstruction for each `k` in a
    configured list (outputs clamped to `[0, 1]`) *(amended 2026-10-21 by slice 010,
    FR-014: the list may be empty — no image PCA, `fit.image_pca` null — for datasets
    whose image PCA does not fit in memory)*;
  - pose: **mean pose** of root-relative 3D joints; **PCA** reconstruction of
    root-relative 3D joints for each `k` in a configured list;
  - pose baselines are also scored in 2D by projecting the predicted joints
    with the sample's camera and true root position.
- **FR-006**: An eval CLI `strike-a-pose-eval baseline --config PATH` MUST be
  driven by a versioned YAML config (dataset dir, split file, seed, run root,
  `k` lists, threshold, figure options); optional CLI overrides (`--seed`,
  `--run-root`) are written into the resolved config.
- **FR-007**: Each run MUST write a new run directory containing
  `config.yaml` (resolved), `run_info.json` (git commit + dirty flag, argv, cwd,
  seed, Python/NumPy/torch/Pillow versions, platform, CPU count, start/finish
  UTC, dataset dir + manifest SHA-256, split file path + SHA-256), `metrics.json`
  (deterministic: no timestamps), and a reconstruction figure PNG (true images vs
  mean and PCA reconstructions on a few samples of a configured split).
- **FR-008**: Python, NumPy and torch seeds MUST be set from the config seed and
  logged (constitution III), even if the baselines are deterministic.
- **FR-009**: Tests MUST run on the committed fixture and synthetic arrays only
  (no SMPL, no full dataset) and keep CI per-leg wall time < 5 min.
- **FR-010**: The full default dataset (001 T025) MUST be generated, validated,
  timed, and its frozen split committed at `splits/smpl_proc_v1.json`.

### Key Entities

- **Eval config** (YAML): see `contracts/eval-run.md`.
- **Run directory**: see `contracts/eval-run.md`.
- **metrics.json**: nested `splits → {image, pose} → baseline → metric`; image
  entries also hold `regions → region → {image_mse, samples}` and the file a top-level
  `regions` definitions block (amended 2026-10-21 by slice
  [009](../009-region-metrics/spec.md); `metrics_version` stays 1).

## Success Criteria *(mandatory)*

- **SC-001**: Every metric has at least one hand-computed test case, run on both
  NumPy and torch inputs.
- **SC-002**: Fixture baseline run (CLI, temp run root) completes in < 30 s on
  CPU and twice gives byte-identical `metrics.json`.
- **SC-003**: Full-dataset baseline run completes on a 4-core CPU in < 30 min
  and its val/test numbers are recorded in docs with the run directory cited.
- **SC-004**: `uv sync --locked`, ruff, pytest green locally and on CI for both
  Python versions; per-leg CI wall time < 5 min.

## Assumptions

- Datasets follow 001 layout v1 and the 002 sample contract; the camera ring is
  fixed, so held-out evaluation is over novel poses, not novel viewpoints.
- Pose labels have zero global orientation (001), so world-frame root-relative
  joints are identical across the 8 views of a pose.

## Clarifications

### Session 2026-10-20 (autonomous run, defaults chosen by lead)

1. Q: Package and CLI names? → A: `strike_a_pose.eval` (`metrics`, `pca`,
   `baselines`, `config`, `cli` modules); console script `strike-a-pose-eval`
   with subcommand `baseline` (room for 004's `evaluate` without a new binary).
   Run-directory helpers (seeding, environment summary, git commit) live in
   `strike_a_pose.runlog` so 004's train CLI reuses them without importing eval.
2. Q: Which image metrics? → A: all three cheap ones: per-pixel MSE (primary,
   matches an AE's L2 loss), PSNR (dB, from per-image MSE, averaged over images)
   and silhouette IoU. IoU threshold default **0.075** = half the renderer's
   ambient floor (body pixels are ≥ 0.15 = `AMBIENT`, background exactly 0), so
   a true image thresholded gives exactly its mask.
3. Q: Which frame for MPJPE? → A: root-relative (pelvis subtracted), Euclidean,
   reported in mm. Pose baselines operate on **world-frame** root-relative
   joints: with zero global orientation these are view-independent, and because
   each camera's rotation is fixed, a world-frame prediction rotated into a
   camera has the same MPJPE as in the world frame (rotation invariance). So the
   world-frame numbers equal camera-frame root-relative MPJPE of a per-camera
   baseline. No Procrustes-aligned variant (VI).
4. Q: 2D error for pose baselines? → A: project predicted joints (root-relative +
   **true** root position) with the sample's true camera (K, R, t) and compare to
   the labelled 2D keypoints over visible joints. This isolates articulation
   error from global placement; stated in the metrics docs.
5. Q: PCA `k` defaults? → A: image `k ∈ {8, 16, 32, 64, 128}` — headline
   **k = 64** = the predecessor AE latent size that 004 starts from; the sweep is
   one fit (top-128 components) and shows the error-vs-k context. Pose
   `k ∈ {4, 8, 16, 32}` (root-relative joints have 69 free dims, so `k = 64`
   would be near-lossless and uninformative); no headline pose `k`. Chosen a
   priori, not from test numbers.
6. Q: PCA algorithm? → A: exact PCA by eigendecomposition of the smaller of
   the Gram (`n×n`) or covariance (`d×d`) matrix in float64 (data in float32),
   torch on CPU; no randomized SVD (no approximation knob, deterministic).
   Components with eigenvalue ≤ 1e-6 × the largest are dropped (numerical
   rank); `k_effective = min(k, rank)` is recorded. Full dataset: ~12.8k × 49,152
   train images (≈ 2.5 GB float32) fit in memory on the 15 GB dev machine; a
   config cap `image.max_fit_samples` is **not** added (VI) — revisit if needed.
7. Q: PCA output range? → A: image reconstructions are clamped to `[0, 1]`
   before scoring (images live in `[0, 1]`; an AE with a sigmoid output does
   the same). Pose reconstructions are not clamped.
8. Q: Which split for the figure? → A: `val` by default (looking at test images
   is avoided as a habit; test is only scored). Figure: N = 6 evenly spaced
   samples × columns (true, mean, PCA for each `k`), PIL only (no matplotlib, VI).
9. Q: Where do runs go? → A: `run_root` from config (default `runs/`, now
   git-ignored); run dir name `<run_name>-<UTC yyyymmddThhmmssZ>`; refuses to
   reuse an existing dir. Only numbers (and a committed copy of the full-dataset
   `metrics.json` + `run_info.json` under `docs/results/`) enter git.
10. Q: Aggregation? → A: per split: mean over samples of per-sample values (all
    images have equal size, so mean per-image MSE = per-pixel MSE). Pose metrics
    are averaged over samples (each pose counted 8×, equal to the per-pose mean).
    Values rounded to 6 significant digits in `metrics.json` for byte stability.
11. Q: Exit codes? → A: 0 OK; 1 dataset/split invalid; 2 config error; 4 run
    dir exists (mirrors 001/002).
12. Q: Full dataset prerequisite (001 T025) — who runs it? → A: this slice (the
    SMPL model is present locally, git-ignored). Split frozen with default
    fractions 0.8/0.1/0.1, seed 0. The split file records the manifest SHA-256,
    so it is valid only for the dataset produced by that exact config + seed +
    code (documented in README).
