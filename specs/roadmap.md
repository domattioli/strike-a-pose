# strike-a-pose Roadmap (spec of specs)

Status: **Slice list + order APPROVED (operator, 2026-10-17).** MVP done (2026-10-20).
Q2 confirmed by surrogate ruling 2026-10-20 (provisional, operator may overturn).
Post-MVP: exactly one slice, 006 `latent-probe` (surrogate ruling 2026-10-20); stop after it.

Each slice → `specs/<NNN-slug>/{spec,plan,tasks}.md` via a speckit pipeline
run. Each slice's `spec.md` needs operator sign-off before plan/implement
(constitution, Development Workflow). Plan changes are recorded in the
Changelog below.

## Goal (MVP)

Reproducible successor to `Conv_AE-Human_Pose`: from a scripted, versioned
synthetic dataset of rendered human poses, train an image autoencoder end-to-end that *works*: trains
without error, loss decreases, held-out reconstructions are recognizable, and
baseline numbers are reported alongside for context (not as a bar to beat).

## Slices

| # | Slug | Purpose | Depends on | Done when |
|---|------|---------|------------|-----------|
| 001 | `dataset-generation` | Scripted CLI that renders body-model poses from defined cameras and writes images + labels (pose params, keypoints, camera, mask) with a manifest | — | Same config + seed → byte-identical manifest; small fixture set committed for tests |
| 002 | `data-pipeline` | Package `strike_a_pose.data`: dataset loader, transforms, normalization, frozen train/val/test split | 001 (manifest schema) | Loader unit tests pass on fixture; split file versioned |
| 003 | `baselines-and-metrics` | Metrics (e.g., MPJPE, image recon error) + mean-pose / PCA baselines, eval CLI | 002 | Baseline numbers produced by a logged run |
| 004 | `autoencoder-mvp` | Config-driven train/eval CLI for the AE, CPU smoke test in CI | 002, 003 | Runs end-to-end from config; loss decreases; held-out recon figure + metrics logged |
| 005 | `repo-foundation` | `pyproject.toml`, ruff, pytest, CI workflow (lint + tests + smoke) | — | CI green on an empty package; can land before or with 001 |

Out of MVP (later candidates): generation/interpolation, real-image datasets, GPU
rails (Kaggle), per-region (hands/face/feet) image MSE (surrogate ruling 2026-10-20 #4),
training improvements / VAE, dataset realism (open questions below).

Post-MVP slices (surrogate ruling 2026-10-20 #1: exactly one, then stop):

| # | Slug | Purpose | Depends on | Done when |
|---|------|---------|------------|-----------|
| 006 | `latent-probe` | Frozen 004 encoder → latents; linear (ridge) and small MLP probes for root-relative 3D joints (MPJPE mm), camera yaw (8-way), SMPL pose; eval CLI `strike-a-pose-eval probe` | 002, 003, 004 (checkpoint) | Probe numbers next to the 003 pose baselines from a logged run; CI green |

## Open questions (batched; answer all at once)

Slice-level (answered 2026-10-17):
1. ~~Approve slice list and order?~~ Approved. 005 stays separate (default).
2. ~~Model input: proposed **images in** (as predecessor), keypoints/pose params
   generated as labels by 001 for metrics + later probes.~~ Confirmed (surrogate
   ruling 2026-10-20 #2, provisional; implicit in the 004 sign-off).
3. ~~Success margin?~~ None — MVP bar is "working", not "better".

Slice 001 specifics (carried into its clarify step):
4. Source: synthetic renders (body model: SMPL / SMPL-X / other) or a public dataset?
5. Renderer: port `body-model-visualizer` or new headless one (pyrender / Blender / PyTorch3D)?
6. Pose sampling: joint-limit random, AMASS mocap, or curated set?
7. Cameras: fixed named views vs randomized extrinsics?
8. Per-sample outputs: image, mask, 2D/3D keypoints, pose params, camera?
9. Size and resolution (render directly at 256×192?).
10. Compute + data storage location.
11. SMPL/AMASS non-commercial licenses acceptable?

Deferred to the operator (surrogate rulings 2026-10-20 #6; not acted on):

12. PyTorch CPU wheel index: keep PyPI torch for now (the trusted-network allowance
    did not take effect in the 2026-10-20 session; 002 research R1). Switch?
13. Early stopping / fewer epochs as the training default? (004 val minimum at
    epoch 34 of 60; ruling #3 only adds an informational `best.pt`.)
14. Split-hash contract change. Proposal: hash the manifest *without*
    `code.commit` / `code.dirty`, store the commit informationally, and warn
    instead of refusing on a commit mismatch (today a regenerated-but-identical
    dataset at another commit is rejected by the frozen split).
15. Dataset realism: arms-down poses, random betas (body shape), camera variation
    (height, distance, pitch, non-ring yaw)?
16. Training improvements / VAE (blur, lost extremities, mild overfitting in 004)?

## Changelog

- 2026-10-17: initial draft roadmap (dataset-first ordering).
- 2026-10-17: slice list approved; MVP bar set to "working"; image input proposed.
- 2026-10-17: slice 001 spec approved (8 clarifications at defaults). Plan,
  tasks, analyze done; implementation landed on `claude/charming-wozniak-5ouldf`
  (package `strike_a_pose.datagen`, CLI `strike-a-pose-data`, layout contract
  v1, 32 tests green without SMPL/AMASS). Deviation: in-package NumPy
  rasterizer + NumPy SMPL LBS instead of pyrender/smplx (determinism, no GL/torch
  deps; FR-003 still met; see 001 research R1/R2). Minimal scaffolding
  (pyproject, uv.lock, ruff, pytest) landed with 001; CI workflow stays in 005.
  Open operator tasks: obtain SMPL/AMASS, generate + commit fixture (T022),
  first full run timing (T025).
- 2026-10-20: slice 001 amended (operator decision): AMASS skipped for MVP.
  New default `pose_source: procedural` — per-joint uniform rotations within a
  the predecessor's range table + k≥5 sparsification (`predecessor-v1`, research R9), recorded in the manifest;
  AMASS kept as optional config source. Fixture (T022) regenerated from real
  SMPL with the procedural source and committed. Default dataset renamed
  `smpl_proc_v1`.
- 2026-10-20: operator answers on 001 open questions: defaults (keep arms-out limitation, framing, run_info.json, predecessor hand-free table) except
  hand joints zeroed (`predecessor-v1-nohands`; predecessor AE did hands poorly). Fixture regenerated with seed 3 (seed 0 near-T-poses).
- 2026-10-20: slice 005 `repo-foundation` (autonomous run, operator delegation
  2026-10-20; clarifications at lead defaults in its spec). Added
  `.github/workflows/ci.yml` (push + pull_request; ubuntu, Python 3.11/3.13;
  `uv sync --locked`; ruff check, ruff format --check, pytest, `validate` on the
  committed fixture), `README.md`, pyproject `readme`/`authors`/`urls`.
  "Done when" read as CI green on the current package (no longer empty).
  Smoke-train step left as a marked placeholder in `ci.yml`; slice 004 owns it.
- 2026-10-20: slice 002 `data-pipeline` (autonomous run, operator delegation
  2026-10-20; clarifications at lead defaults in its spec). Package
  `strike_a_pose.data`: `PoseImageDataset` (manifest + file checks on load,
  opt-in full 001 validation, per-item label checks; dict samples — 1-channel
  float32 image in [0,1], bool mask, keypoints/joints/pose/camera in source
  units, ids), normalization helpers (image ↔ tensor, root-relative 3D,
  image-normalized 2D), frozen **pose-level** split (`pose-permutation-v1`,
  records manifest SHA-256; all camera views of a pose in one split to avoid
  leakage) via `strike-a-pose-data split`; fixture split committed at
  `splits/smpl_fixture.json`. Contract: `specs/002-data-pipeline/contracts/sample-contract.md`.
  PyTorch added from PyPI (locked 2.14.1): the CPU wheel index is unreachable
  from the agent session (egress policy), so Linux installs pull CUDA runtime
  wheels; switching to the CPU index is an open operator question (002 research R1).
  Note for 004: the fixture split has train 1 pose / val 0 / test 1 (8 samples each).
- 2026-10-20: 001 T025 done (slice 003, operator delegation): full `smpl_proc_v1`
  generated at `406b68f` in 290 s on 4 vCPU (16,000 samples, 351 MB; SC-004 met);
  frozen split `splits/smpl_proc_v1.json` (1600/200/200 poses). The split pins the
  manifest SHA-256, which includes `code.commit` → valid only for that config + seed
  + commit (README).
- 2026-10-20: slice 003 `baselines-and-metrics` (autonomous run, operator delegation
  2026-10-20; clarifications at lead defaults in its spec). Package
  `strike_a_pose.eval`: metrics (`image_mse`, `psnr`, `silhouette_iou` @ 0.075,
  root-relative `mpjpe` → mm, `keypoint_error_2d` px; NumPy or torch), exact PCA
  (Gram/covariance eigh, rank clamp), mean/PCA baselines fit on train only, CLI
  `strike-a-pose-eval baseline --config` writing run dirs under git-ignored `runs/`
  (resolved config, run_info with commit/seed/env/hashes, `metrics.json`, recon
  figure). Shared run helpers in `strike_a_pose.runlog` (for 004). CI gains a fixture
  baseline smoke step (additive edit to 005's workflow). Contract:
  `specs/003-baselines-and-metrics/contracts/eval-run.md`.
  Full-dataset run `runs/baselines-smpl_proc_v1-20261020T022918Z` (commit `aded3a4`,
  clean; 182 s; copy in `docs/results/003-baselines-smpl_proc_v1/`): val / test —
  mean image MSE 0.01403 / 0.01385 (IoU 0.416 / 0.420); PCA k=64 MSE 0.00459 /
  0.00452 (PSNR 23.7 / 23.8 dB, IoU 0.644 / 0.646); mean pose MPJPE 253.4 / 250.3 mm
  (2D 18.4 / 18.2 px); pose PCA k=8 94.0 / 93.6 mm. Context for 004, not a bar.
- 2026-10-20: slice 004 `autoencoder-mvp` (autonomous run, operator delegation
  2026-10-20; clarifications at lead defaults in its spec, plus the operator's
  restart-resilience requirement, Session 2026-10-20b). **MVP done.** Packages
  `strike_a_pose.models` (`ConvAutoencoder`, latent 64, 3.1 M parameters) and
  `strike_a_pose.train` (strict YAML config, seeded Adam/MSE trainer with cosine lr,
  full-state end-of-epoch checkpoint, CLI `strike-a-pose-train --config` /
  `--resume RUN_DIR`). Contract: `specs/004-autoencoder-mvp/contracts/train-run.md`.
  Cross-slice touches: 003 figure/scoring helpers moved to
  `strike_a_pose.eval.report` (behaviour-preserving, 003 byte test guards it); 005's
  `ci.yml` smoke-train placeholder replaced by a fixture train step.
  Full run `runs/ae-smpl_proc_v1-20261020T030038Z` (commit `f9619c6`, clean; resumed
  across two container restarts, code unchanged; copy in
  `docs/results/004-ae-smpl_proc_v1/`): 60 epochs, 84 min compute on 4 vCPU (2 h 45 min
  calendar incl. ≈ 80 min with the process dead). val / test — AE MSE 0.00221 /
  0.00219 (PSNR 27.1 dB, IoU 0.879 / 0.881) vs PCA k=64 0.00459 / 0.00452 (IoU 0.644 /
  0.646) and mean image 0.01403 / 0.01385 (baselines identical to 003). Train MSE
  0.0352 → 0.00106; val 0.0100 → min 0.00217 (epoch 34) → 0.00221: mild overfitting,
  final weights kept as specified. Held-out reconstructions recognizable but blurred;
  hands/fingers, faces and feet are lost, crossing limbs merge (candidate topics for
  post-MVP slices, not acted on).
  Lesson (process): the agent container is reclaimed when the session idles, which
  kills background jobs (two kills during this run). Long jobs must be resumable
  (end-of-epoch checkpoint + `--resume`), launched detached with the resume command
  written down in tasks.md, and the session kept active (or heartbeated) while they
  run; plan wall-clock budgets with restarts in mind.
  PyTorch CPU wheel index re-tested at the end of the slice: still 403 → unchanged
  (open operator question, 002 research R1).
- 2026-10-20: **Surrogate rulings 2026-10-20** (operator AFK; rulings delegated to
  the "Fable" surrogate). Each ruling is **provisional and overturnable** by the
  operator; none changes the constitution.
  1. Continue past the MVP with exactly **one** slice, 006 `latent-probe`; stop
     after it lands green (do not start 007). *(provisional)*
  2. Q2 "images as model input": **confirmed** (implicit in the 004 sign-off);
     Q2 struck above, status line updated. *(provisional)*
  3. Keep the 004 **final-epoch** weights as the reported MVP result (no
     re-scoring). Future runs additively save `best.pt` (best val loss) next to
     `checkpoint.pt` and log the best-val epoch; best-val selection / early
     stopping is **not** the default; `configs/ae.yaml` unchanged. *(provisional)*
  4. Headline image metrics unchanged (MSE + silhouette IoU @ 0.075); per-region
     (hands / face / feet) MSE → deferred candidate. *(provisional)*
  5. Pin the CI runner `ubuntu-latest` → `ubuntu-24.04` in `.github/workflows/ci.yml`.
     *(provisional)*
  6. Deferred to the operator (open questions 12–16 above): torch CPU wheel index;
     early-stopping / fewer-epochs default; split-hash contract change; dataset
     realism; training improvements / VAE.
- 2026-10-20: slice 006 `latent-probe` (autonomous run, Fable surrogate scope;
  spec approved by surrogate ruling, operator sign-off pending). Module
  `strike_a_pose.eval.probe` + subcommand `strike-a-pose-eval probe` (configs
  `probe.yaml`, `probe_fixture.yaml`; contract `specs/006-latent-probe/contracts/probe-run.md`):
  frozen 004 encoder → hash-keyed latent cache → closed-form ridge + small MLP probes
  (λ / weight decay chosen on val, test once) for root-relative 3D joints (world +
  camera frame), camera yaw (8-way classification) and SMPL pose, next to mean /
  majority / pose-PCA (oracle) baselines and an untrained-encoder control; PIL
  skeleton + confusion figures. CI gains a fixture probe smoke step (legs ≈ 1 min).
  Ruling #3 implemented: the trainer additively writes `best.pt` and logs
  `best_val_epoch` (resume-safe, tested); the final weights stay the evaluated ones.
  Full run `runs/probe-smpl_proc_v1-20261020T060632Z` (commit `ae32db9`, clean,
  7.5 min; copy in `docs/results/006-probe-smpl_proc_v1/`), val / test: world-frame
  MPJPE MLP 172.0 / 172.8 mm, ridge 214.5 / 212.5 vs mean pose 253.4 / 250.3 and
  untrained-encoder MLP 214.6 / 214.5 (oracle pose PCA k=8 94.0 / 93.6); camera
  accuracy MLP 0.965 / 0.959, ridge 0.826 / 0.819, untrained MLP 0.685 / 0.670;
  remaining camera errors mostly front ↔ back flips. Reading: the latent holds the
  view almost fully and the body configuration only coarsely (≈ a 4-component pose
  PCA). Per surrogate ruling #1 work stops here; open questions 12–16 remain.
  New observation for the operator: the MLP weight-decay grid picked its edge value
  (1e-3) for 3 of 4 targets; not widened after test was seen.
