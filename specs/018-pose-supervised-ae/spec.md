# Feature Specification: Pose-Supervised Autoencoder

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `018-pose-supervised-ae`)
**Created**: 2026-10-24
**Status**: Approved at surrogate scope; operator decision 2026-10-24 (direction change: pose
supervision on the encoder) as approval basis. Clarification batch 1 answered by surrogate
ruling 2026-10-24 (below); the operator may overrule.
**Input**: The plain conv AE on `smpl_proc_v3` reaches val MSE 0.00158 and world-frame MLP probe
MPJPE 131 mm (three-seed mean 132.8 mm, sd 1.7 mm; random encoder 212 mm; mean pose 258 mm).
Mask-weighted MSE gains −8.4 mm at +14 % MSE; more poses has diminishing returns (≈ 117 mm
fitted at 16k poses). Operator target (grill point 8): **≤ 80 mm** world MPJPE. Add an
auxiliary head on the latent, trained jointly with reconstruction from labels the dataset
already stores, and evaluate it without weakening the 015 discipline.

Settled context (measured at `9075172`, see [research.md](research.md) R1–R3):

- The trainer reads every label file already (each dataset item carries `joints_3d_world`
  `[24, 3]`, `joints_3d_cam`, `pose` 72-d axis-angle, `camera.yaw_deg`, `camera_index`), but
  `load_split` keeps only images (+ masks / 2D keypoints for scored splits). No new data.
- v3 root-relative world joints (4,000 sampled images): per-coordinate sd median 135 mm, max
  431 mm; **12 of 72 coordinates have sd 0** (root, and joints rigid to the pelvis because the
  world frame fixes global orientation); mean-pose MPJPE 253 mm (published 258 mm on val).
- The probe (`eval/probe.py`) loads `ckpt["model"]` into `ConvAutoencoder` strictly; extra
  checkpoint keys are ignored, so a head stored under its own keys keeps every probe,
  sampling and round-trip consumer working unchanged.

## Clarifications

### Session 2026-10-24 (batch 1, surrogate rulings; operator may overrule)

- Q1 Targets: (a) root-relative `joints_world` (72 = 24 × 3); (b) SMPL pose as 6D rotations;
  (c) both; (d) any of these plus the yaw class. **Ruling (a)**. It is the metric the operator
  target is stated in and the one the probe ranks by; the joints are a deterministic function of
  the pose (FK with betas 0), so (b) adds no information but adds a rotation representation and a
  second normalization; world-frame joints are camera-invariant, so predicting them from an
  image already forces the encoder to undo the yaw. Yaw is 8 classes the probe already gets
  near-perfect from the plain AE (README "Results"), so a CE term would spend gradient on a
  solved problem. Rejected targets are recorded in research.md R4 as follow-ups.
- Q2 Loss form and normalization: **MSE on per-coordinate standardized targets** (train-split
  mean and population sd per coordinate, fitted once at run start, stored in the checkpoint);
  coordinates with sd ≤ `POSE_STD_FLOOR_M = 1e-6` m are excluded from the loss (the 12 constant
  ones), the loss is the mean over the remaining coordinates and the batch. Same scaling as the
  probe's `Standardizer`, so the λ grid reads in "fraction of variance". Rejected: raw-metre MSE
  (dominated by the long chains, scale-dependent λ); per-joint L2 / MPJPE loss (non-smooth at
  0, no evidence it helps at this error level).
- Q3 λ grid: total loss = `recon + λ · pose`. Calibration (R2): recon ≈ 0.0016 at convergence;
  the standardized pose MSE at the plain-AE probe level is ≈ (131/253)² ≈ 0.27. **λ ∈ {0.003,
  0.03}**: pose term ≈ 0.5× and ≈ 5× the reconstruction term at that level, a decade apart,
  bracketing "balanced" from both sides. A third plain-λ run (0.3, ≈ 50×) would turn the AE
  into a regressor and is left out to fit the budget.
- Q4 Mask weighting: **yes, one combination run** at λ 0.03 + `mask_weight` 4 (the R1-v3
  setting). R1-v3's −8.4 mm (5 sd) is the only real latent gain measured so far; testing whether
  it adds to pose supervision costs one run. Its MSE is reported against R0-v3 like R1-v3's.
- Q5 Budget: 3 runs × ≈ 5–6 h (R0-v3 wall time; the head adds < 1 % FLOPs) + 3 probes
  (minutes each) ≈ **17 h** on 4 vCPU, under the 20 h ceiling. Memory as R0-v3 / R1-v3 (≤ 6 GB
  peak RSS) + 15 MB of labels; 12 GB gate unchanged.
- Q6 Weight selection and early stop: unchanged — on **val pixel MSE** (`eval_loss`, which
  never sees the head). Pixel quality stays the gate (operator); the head is trained, never
  selected, on labels.
- Q7 Honest evaluation: (i) the 006 probe is still fit on frozen latents of the **train** split
  of the scored weights (`best.pt`), hyper-parameters on val, test scored once, random-encoder
  control kept — the pose head is invisible to it; (ii) the head's own val / test MPJPE is
  recorded separately in the training run's `metrics.json` (`splits.<s>.pose_head`) and never
  substituted for the probe; (iii) `scripts/promotion.py --split both` with R0-v3 / R0b-v3 as
  before; the 015 rule (`MSE_DROP_REL` 5 %, `LATENT_GAP_FACTOR` 2) is **unchanged**; (iv) a
  separate pose-track report ranks variants by probe MPJPE against the three-seed mean and sd.
- Q8 Pose-track "real gain" bar: Δ vs the three-seed mean < **−2 sd** (sd computed from the
  three seed probes per split, val 1.72 mm, test 1.5 mm). Same 2× factor as the 015 latent bar.
- Q9 VAE: `pose_weight > 0` with `variational: true` is refused (`ConfigError`): no VAE run is
  planned and the head's input (μ vs a sampled z) would be one more design choice.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train a pose-supervised AE from a config (Priority: P1)

The researcher sets `pose_weight: 0.03` in a train config and runs `strike-a-pose-train`; the
run trains the AE and a small MLP head jointly, logs the pose term per epoch, stores the head in
`checkpoint.pt` / `best.pt`, and reports the head's world MPJPE on val and test.

**Independent Test**: fixture runs (CI) — `pose_weight: 0` is bitwise the current trainer;
`> 0` repeats, resumes equal to uninterrupted, and records the head scores.

**Acceptance Scenarios**:

1. **Given** any shipped config without `pose_weight`, **When** it trains, **Then** losses,
   weights and metrics equal today's bit for bit (no head built, no RNG draw).
2. **Given** a fixture config with `pose_weight: 0.03`, **When** it trains twice, **Then**
   `history.csv` and `metrics.json` (minus timings) are identical and
   `splits.val.pose_head.mpjpe_mm` is a finite number.
3. **Given** the `best.pt` of a pose run, **When** `strike-a-pose-eval probe` runs on it,
   **Then** it loads and scores as for a plain run.

### User Story 2 - Pose-track report (Priority: P1)

The lead runs `scripts/pose_track.py` with the three R0-v3 seed runs and the 018 variants and
gets, per split, variants ranked by probe MPJPE with Δ vs the seed mean, Δ / sd, a "real" flag,
the head's MPJPE and whether probe / head reach ≤ 80 mm.

### User Story 3 - Campaign queue (Priority: P2)

`docs/campaign-scripts/pose018_queue.sh` (copied to `data/`) trains and probes the three
variants idempotently under the watchdog, with the 015 lock, memory gate, exit-143 handling.

### Edge Cases

- `pose_weight < 0` → `ConfigError`; `pose_weight > 0` with `variational: true` → `ConfigError`.
- A coordinate with zero train variance → excluded from the loss (`active` false), predicted as
  the train mean on unstandardizing (its error is exactly 0 for a world-frame constant).
- Resume of a pre-018 run directory (no `pose_weight` in its config) → `LEGACY_VALUES` 0.0;
  checkpoints without head keys resume as plain runs.
- Old `history.csv` rows (no `train_pose`) → `_complete_row` fills `train_pose: None`.
- `pose_track.py` with fewer than 2 seeds → exit 2 ("need >= 2 --seed").

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** Config key `pose_weight: float = 0.0` (≥ 0; `LEGACY_VALUES` 0.0; refused with
  `variational`). `Objective` gains `pose_weight`.
- **FR-002** `models/pose_head.py`: `PoseHead` MLP latent_dim → 256 → 256 → 72 (ReLU),
  `PoseTargetStats` (train mean / sd / active mask), `root_relative`, `pose_loss`
  (data-model.md). Built only when `pose_weight > 0`, after the AE, from the global torch RNG.
- **FR-003** Training: `load_split(..., joints=True)` keeps `joints_world` float32 `[n, 24, 3]`;
  the head reads the encoder output (μ, before dropout); loss `recon + pose_weight · pose_loss`;
  one AdamW over AE + head parameters; history column `train_pose` (empty when off).
- **FR-004** Checkpoints (`checkpoint.pt`, `best.pt`) gain `pose_head`, `pose_head_hparams`,
  `pose_stats` (None / absent when off); resume and best-val selection restore the head with the
  AE; `ckpt["model"]` and `model_hparams` are unchanged so `load_checkpoint` users are unaffected.
- **FR-005** Evaluation: for each eval split with the head on, `splits.<s>.pose_head =
  {"target": "joints_world", "mpjpe_mm": float}` (root-relative world frame, joints 0–23, same
  `mpjpe` as the probe); top-level `metrics.pose_head` = hparams, parameter count, active
  coordinates, `pose_weight`; units entry for `pose_head_mpjpe_mm`.
- **FR-006** `eval/pose_track.py` + `scripts/pose_track.py` (pose track, US2): ranks variants by
  probe `splits.<s>.targets.joints_world.mlp.mpjpe_mm`; seed mean / sd (ddof 1) from ≥ 2 seed
  probes per split; `REAL_GAIN_SD = 2.0`, `POSE_TARGET_MM = 80.0`; `--split {val,test,both}`.
- **FR-007** Configs `configs/smpl_proc_v3/ae_pose_l0.003.yaml`, `ae_pose_l0.03.yaml`,
  `ae_pose_l0.03_mask.yaml` (copies of `ae.yaml` / `ae_mask.yaml` + `pose_weight`); queue
  `docs/campaign-scripts/pose018_queue.sh`.
- **FR-008** Docs: HANDOFF campaign procedure names the 018 queue and both reports; results go
  to `docs/results/pose-campaign-018/README.md` (written by the lead after the campaign).

### Key Entities

`PoseHead`, `PoseTargetStats`, `Objective.pose_weight`, pose-track row (data-model.md).

## Success Criteria *(mandatory)*

- **SC-001** Default configs bitwise unchanged (US1 scenario 1); full suite green.
- **SC-002** The campaign finishes within ≈ 17 h ± 4 h, peak RSS < 12 GB, with three probes.
- **SC-003** Reports: `promotion.py --split both` (015 rule, unchanged) and `pose_track.py
  --split both` for the three variants, plus the head's val / test MPJPE and whether any of
  probe or head reaches ≤ 80 mm. No result threshold is pre-registered for "success" of the
  direction beyond the 2-sd real-gain bar; the operator target is reported, not tuned toward.
