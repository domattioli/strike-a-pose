# strike-a-pose Roadmap (spec of specs)

Status: **Slice list + order APPROVED (operator, 2026-10-17).** MVP done (2026-10-20).
Q2 confirmed by surrogate ruling 2026-10-20. The operator's "approve all" (labelled 2026-10-21) covers slices 002–006
(incl. their autonomous clarifications) and the surrogate rulings of 2026-10-20 only; slices 007–012 are approved at
surrogate scope and await operator sign-off (list in `docs/HANDOFF.md`).
Post-MVP: exactly one slice, 006 `latent-probe` (surrogate ruling 2026-10-20); stop after it.
Then 007 `split-content-hash`, which implements the operator-approved open question #14
(2026-10-20; approved at surrogate scope, operator sign-off pending), 008
`best-val-selection`, which implements the operator's choice B for open question #13
(2026-10-20; approved at surrogate scope, operator sign-off pending), 009
`region-metrics`, which implements the operator's choice (i) for open question #16
(chosen 2026-10-20, implemented 2026-10-21; approved at surrogate scope, operator sign-off
pending), 010 `dataset-realism`, which implements the operator's decision on open
question #15 (chosen 2026-10-20, implemented 2026-10-21; approved at surrogate scope,
operator sign-off pending; datasets and runs wait for SMPL), and 011
`training-variants`, which implements the operator's options (ii)–(iv) for open question #16
(chosen 2026-10-20, implemented 2026-10-21; approved at surrogate scope, operator sign-off
pending; runs wait for SMPL), and 012 `run-guards`, which implements the operator's 2026-10-21
decision to run the ≈ 20 h campaign with checks for early termination and iteration
(implemented 2026-10-21; approved at surrogate scope, operator sign-off pending; the campaign
waits for SMPL), and 014 `realism-ablations`, which implements the operator's 2026-10-21
decisions amending 010 (independent realism datasets v2a–v2d, `armsdown-v1`, an automatic
limb–torso intersection count; implemented 2026-10-21; datasets wait for SMPL).

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
rails (Kaggle), per-region (hands/face/feet) image MSE (surrogate ruling 2026-10-20 #4;
implemented in 009 under the operator's choice (i) for #16, 2026-10-20: hands / feet /
head box, SMPL-24 has no face joint), training improvements / VAE (options (ii)–(iv)
implemented in 011 under the operator's #16 choice: regularization, mask-weighted MSE, a VAE;
runs once SMPL is available), dataset realism
(implemented as opt-in generator options in 010 under the operator's #15 decision; the
datasets `smpl_proc_v2`…`v5` and their runs follow its run procedure once SMPL is
available; open questions below).

Post-MVP slices (surrogate ruling 2026-10-20 #1: exactly one, then stop; 007 implements
the operator-approved open question #14, 008 the operator's choice B for #13, 009 the
operator's choice (i) for #16, 010 the operator's decision on #15, 011 the operator's
options (ii)–(iv) for #16):

| # | Slug | Purpose | Depends on | Done when |
|---|------|---------|------------|-----------|
| 006 | `latent-probe` | Frozen 004 encoder → latents; linear (ridge) and small MLP probes for root-relative 3D joints (MPJPE mm), camera yaw (8-way), SMPL pose; eval CLI `strike-a-pose-eval probe` | 002, 003, 004 (checkpoint) | Probe numbers next to the 003 pose baselines from a logged run; CI green |
| 007 | `split-content-hash` | Split files pin the manifest content (manifest without its generating `code`) instead of the raw bytes: `split_version` 2, `SplitCommitWarning` on a code mismatch, v1 read-only (open question #14) | 002 | Fixture split rewritten as v2 with the same pose lists; no numerical change; CI green; follow-up F1 rewrites `splits/smpl_proc_v1.json` once SMPL is available |
| 008 | `best-val-selection` | Training runs score the weights of the epoch with the lowest val loss (`best.pt`) by default: `select_weights: best_val \| final`, final weights when val is empty or never finite, `selection` block in `metrics.json` (`train_version` 2), resume-safe; 60 epochs and cosine lr unchanged (open question #13, option B) | 004 | Selection, fallback and resume tested on the fixture (scripted val curve); fixture smoke numbers unchanged; CI green; the published 004 / 006 numbers stay final weights until the re-baseline run |
| 009 | `region-metrics` | Every image model (003 mean image / PCA k; 004 `ae`, `mean_image`, `pca_k<K>`) also reports its MSE in `hands` / `feet` / `head` regions (squares on the true 2D keypoints of joints 22 / 23, 7 / 8 / 10 / 11, 15; `region_mask`, `masked_mse`, labels passed to `score_images`); additive `metrics.json` keys, `metrics_version` 1 / `train_version` 2 unchanged; no training or config change (open question #16, option (i)) | 003, 004 | Fixture runs report the regions (003 and 004 identical for the baselines), every existing number unchanged; CI green; first real numbers with the re-baseline run |
| 010 | `dataset-realism` | Opt-in generator options with omit-when-default manifest serialization: `ranges_version: armsdown-v1` (named `predecessor-v2-armsdown` until 014; arms lowered ∓65° after sparsification, shoulder z `[-20, 20)`, collar z `[-10, 10)`), `body_shape` (per-pose betas N(0, 1) clipped at ±2, own random stream), `camera_jitter` (camera moved in height / distance, aimed at the ring's look-at point, tilted down; projected-vertex in-frame check, recorded ring-camera fallback); configs for the chain `smpl_proc_v2` → `v5` (replaced by 014's independent v2a–v2d); 003 `image_pca_ks: []` (open question #15) | 001, 003 | Every option tested on the synthetic body model; today's datasets byte-identical (fixture pins, before/after synthetic comparison); CI green; datasets and runs after SMPL access and F1 (run procedure) |
| 011 | `training-variants` | Config-selectable training variants of the 004 trainer / model, defaults bitwise today's run: `weight_decay` (AdamW), `shift_px` (per-sample integer shifts), `dropout` (latent), `mask_weight` (α, `w = 1 + α·[x > 0]`), `variational` + `beta` (VAE: `logvar` head, `beta·KL/(H·W)`, `encode` = μ); `history.csv` + `train_recon`, `train_kl`; `metrics.json` `model.variational`, `training.objective`; configs `ae_{mask,vae,reg}.yaml` (open question #16, options (ii)–(iv)) | 004, 006, 008, 009 | Defaults bit-equal to the pre-011 loop and in a same-machine before/after fixture run; every knob, resume and legacy resume tested; a VAE checkpoint probed; CI green; runs R1–R3 after SMPL and the re-baseline (experiment procedure) |
| 012 | `run-guards` | Opt-in trainer guards, defaults bitwise today's run: `early_stop {patience, min_epochs, min_delta}` (stop on a val plateau; schedule and selection unchanged; `training.stopped_early`), `abort_on_divergence` + `divergence_ratio` (`aborted.json`, exit 5, not resumable); a run-dir lock; batched mean image (the 4,000-pose OOM); `env.cpu_model`, `dataset.checksums_sha256`; `scripts/run_status.py`; campaign configs and the campaign procedure (operator decision 2026-10-21) | 004, 008, 011 | Guards, lock and status script tested on the fixture (scripted losses); fixture 003 / 004 outputs byte-identical; CI green; the campaign after SMPL access |
| 013 | `capsule-body` | Interim license-free body (operator decision 2026-10-21; SMPL unavailable): `datagen/capsule.py`, 28 rigid capsules on SMPL's rest skeleton recovered at build time from the committed fixture, SMPL-format `.npz` with a pinned SHA-256, `strike-a-pose-data make-capsule-body`; test-enforced twin configs `configs/capsule_proc_{v1,v2a,v2b,v2c,v2d}/`; results only under `docs/results/013-capsule/` (infrastructure tests) | 001, 012, 014 | Body tested (skeleton, rigid kinematics, closed capsules, digest pin, render sanity against the fixture's SMPL masks, end-to-end generation); CI green; `capsule_proc_v1` generated, split frozen, pose baselines reproduce the published numbers |
| 014 | `realism-ablations` | Amends 010 (operator decisions 2026-10-21): independent single-change datasets `smpl_proc_v2a` (arms-down `armsdown-v1`) / `v2b` (shape) / `v2c` (jitter) / `v2d` (4,000 poses, no PCA), each vs v1; the table renamed (no alias); `datagen/intersection.py` + `scripts/intersection_check.py` (arm vertices inside per-segment torso elliptic cylinders; v2a split gate 5 %) | 010, 012 | Configs tested against v1; intersection count tested on the synthetic body; CI green; datasets after SMPL (campaign procedure) |
| 015 | `pose-scale-and-sampling` | Operator decisions 2026-10-22: `smpl_proc_v3` (8,000 poses) configs + v3 R0b / R2 / R1; baselines keep train images uint8 without image PCA; `peak_rss_mb` in run_info; promotion rule pixel-or-latent win (`eval/promotion.py`, supersedes 012's); `strike-a-pose-eval sample` (VAE prior: valid-silhouette fraction, NN-MSE, reference bars); 015 queue | 012, 014 | Tests for each piece; CI green; campaign done 2026-10-24 (docs/results/smpl-campaign-015) |
| 016 | `probe-round-trip` | Operator decision 2026-10-24 (grill point 5): `strike-a-pose-eval round-trip` on a 015 sample run + the 006 probe run of its VAE: re-fit the selected `smpl_pose` / `yaw` MLPs from the probe cache, encode each image (μ), re-render the predicted pose (mean shape, predicted ring camera) with the NumPy SMPL rasterizer, silhouette IoU; plausible = IoU ≥ τ = p5 of real-val IoU; shuffled-pose control; augmented `samples.json` + figures in a new run dir | 006, 015 | Tests (IoU, τ, re-render identity, refit, refusals, determinism); CI green; the three existing sample runs round-tripped, README table |

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
13. ~~Early stopping / fewer epochs as the training default? (004 val minimum at
    epoch 34 of 60; ruling #3 only adds an informational `best.pt`.)~~ Operator choice
    B, 2026-10-20 (report `best.pt`, the lowest val loss, as the evaluated model; 60
    epochs and the cosine schedule unchanged), implemented in 008 `best-val-selection`.
    Operator decision 2026-10-21: the ≈ 20 h campaign may run "so long as we have checks to
    ensure early termination and iteration if the run looks to be spinning its wheels": early
    stopping as an opt-in guard (patience 10 from epoch 20 in the campaign configs), not as the
    reported-model rule (B stands); implemented in 012 `run-guards`.
14. ~~Split-hash contract change. Proposal: hash the manifest *without*
    `code.commit` / `code.dirty`, store the commit informationally, and warn
    instead of refusing on a commit mismatch (today a regenerated-but-identical
    dataset at another commit is rejected by the frozen split).~~ Approved by the
    operator ("approve all"); implemented in slice 007 `split-content-hash`
    (2026-10-20). Follow-up F1 (needs SMPL): rewrite `splits/smpl_proc_v1.json` as v2.
15. Dataset realism: arms-down poses, random betas (body shape), camera variation
    (height, distance, pitch, non-ring yaw)?
    Operator decision, 2026-10-20: all four changes, each as its own dataset version and
    experiment, in this order: (a) arms-down poses, (b) random body shape, (c) camera ring
    jitter (small height / distance / pitch around the 8 named cameras, in-frame check, no
    free cameras), (d) more poses (config only); defaults reproduce today's dataset byte for
    byte. Implemented in 010 `dataset-realism` (2026-10-21) as opt-in options and the config
    chain `smpl_proc_v2` (arms-down) → `v3` (+ shape) → `v4` (+ jitter) → `v5` (+ 4,000
    poses); the 001 answer "keep arms-out limitation" holds for the default table only.
    Datasets, splits and runs: 010 run procedure (needs SMPL; after F1).
    Operator decisions 2026-10-21 (012 campaign procedure): independent single-change datasets
    v2a (arms-down, table renamed `armsdown-v1` with an automatic intersection count), v2b
    (shape), v2c (jitter), v2d (4,000 poses), each compared with v1, instead of the cumulative
    chain; configs and 010 documents amended by 014 `realism-ablations` (2026-10-21):
    `configs/smpl_proc_v2{a,b,c,d}/`, `scripts/intersection_check.py` (v2a's split fails above
    5 % of poses with an arm vertex inside the torso), the confound of v2a documented (less
    pose variety, more overlap). The winners' dataset is pending (default v2a).
16. Training improvements / VAE (blur, lost extremities, mild overfitting in 004)?
    Operator choice (i), 2026-10-20: per-region MSE metrics first (hands / feet / head
    box; metrics only, before the re-baseline run), implemented in 009 `region-metrics`
    (2026-10-21). Options (ii)–(iv), 2026-10-20: regularization (weight decay, small pixel
    shifts, bottleneck dropout), loss shaping (mask-weighted MSE first; a silhouette term
    deferred) and a VAE (small beta), each config-selectable with defaults reproducing
    today's run bit for bit and the plain AE as control; implemented in 011
    `training-variants` (2026-10-21); runs: its experiment procedure (needs SMPL). Option (v)
    (capacity sweep) still open. Operator decisions 2026-10-21 (slice 012): the runs follow the
    012 campaign procedure with checks; a seed-1 control R0b as the noise bar; the first VAE
    run at β = 0.01 (collapse retry 0.001, healthy follow-up 0.1). Promotion metrics per
    variant: surrogate ruling 2026-10-21, operator confirmation requested.

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
  spec approved by surrogate ruling, operator sign-off 2026-10-21 ("approve all")). Module
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
- 2026-10-21: operator "approve all": signs off slices 002–006 (incl. every
  autonomous-run clarification recorded in their specs) and all surrogate rulings
  2026-10-20 (#1–#6). Open-question dispositions: #12 approved — switch torch to the
  CPU wheel index once `download.pytorch.org` is reachable from a session; #14
  approved — implement the split-hash proposal (hash manifest without
  `code.commit`/`code.dirty`, commit informational, warn on commit mismatch,
  refuse on content mismatch); #13, #15, #16 still need a concrete choice (they are
  options, not single proposals) — carried to the next session.
- 2026-10-20: slice 007 `split-content-hash` (third session that day; autonomous run,
  Fable surrogate scope; spec approved at surrogate scope, operator sign-off pending).
  Dated by the session clock: the "approve all" entry above is labelled 2026-10-21 by
  the previous session but was written on 2026-10-20 UTC, like every commit of this
  session. Implements open question #14: split files are `split_version` 2 and pin
  `manifest_content_sha256` (SHA-256 of the 001 serializer's output for the manifest
  without its `code`) instead of the raw manifest bytes, keep the generating code as
  `manifest_code` (informational), warn (`SplitCommitWarning`) when a dataset with the
  same content was generated by other code, and still refuse other content;
  `split_version` 1 files stay readable with their strict raw-bytes check and a
  migration hint. 003 / 004 / 006 `run_info.json` gain `dataset.manifest_content_sha256`;
  the 004 resume check and the 006 latent cache keep the raw manifest hash (surrogate
  ruling: the manifest does not hash the images). Fixture split rewritten as v2 (same
  pose lists; pre-007 bytes kept as a test asset). Contracts updated in place: 002
  sample contract (split file v2), 003 / 004 / 006 run contracts. Implementation
  `187fa3b`; pytest 249 passed (28 s); fixture baseline / train / probe `metrics.json`
  before vs after differ only in `dataset.split_file_sha256`. Follow-up F1 (blocked:
  the SMPL model is not obtainable in this session; access to `domattioli/smpl-assets`
  was refused): regenerate `smpl_proc_v1`, check it reproduces the `406b68f` content,
  rewrite `splits/smpl_proc_v1.json` as v2 (commands in 007 `tasks.md` and
  `docs/HANDOFF.md`). #12 re-tested from the agent session 2026-10-20:
  `download.pytorch.org` still refused (HTTP 403 at the proxy CONNECT, 3 attempts);
  torch stays on PyPI; the switch waits for a session whose egress policy allows that
  host (002 research R1).
- 2026-10-20: slice 008 `best-val-selection` (autonomous run, Fable surrogate scope; spec
  approved at surrogate scope, operator sign-off pending; approval basis: the operator's
  choice B for open question #13, 2026-10-20). Training runs score the weights of the
  epoch with the lowest finite val loss by default (`select_weights: best_val | final`),
  read back from `best.pt` after an epoch / val-loss / config check, and fall back to the
  final weights when val is empty (the fixture, so CI) or never finite; 60 epochs and the
  cosine schedule unchanged, no early stopping; training itself is unchanged.
  `metrics.json` gains a `selection` block (rule, applied weights, reason, epoch, val loss,
  file) and `train_version` 2. Found while specifying and fixed: `best.pt` was written
  before `checkpoint.pt`, safe on resume only on the same machine, so `checkpoint.pt` is
  now written first and resume rewrites `best.pt` from it after a kill between the two
  writes (and refuses when the selected weights are lost); a NaN val loss at epoch 1
  pinned the best epoch (NaN is now never selected); a new config key made every older run
  dir non-resumable ("checkpoint does not match config.yaml"), so a legacy-value table
  resumes run dirs that predate a key with the value they started under
  (`select_weights` → `final`). Surrogate ruling (batch Q1, option (a)): only the selected
  weights are scored, test once; their gain over the final weights is known on val only,
  the selecting split, as the README and the 004 contract state. The probe code is
  unchanged; `configs/probe.yaml` points at the re-baseline run's `best.pt` (placeholder
  path). Contracts updated in place: 004 `train-run.md` (Train v2), 006 `probe-run.md`
  (`checkpoint` row); dated notes in the 004 and 006 specs. Implementation `5b1befa`;
  pytest 267 passed (31 s; 249 before); fixture smoke before / after: the AE
  `metrics.json` differs only in `train_version` and the `selection` block (every number,
  the history, the figure and the weights identical), 003 byte-identical, 006 only in
  `encoder.checkpoint_sha256` (byte-identical on the pre-008 checkpoint). The published
  004 / 006 numbers (final weights of a pre-008 run) stay as reported until the
  re-baseline run (after SMPL access, F1 and slice 009).
- 2026-10-21: slice 009 `region-metrics` (autonomous run, Fable surrogate scope; spec
  approved at surrogate scope 2026-10-20, operator sign-off pending; approval basis: the
  operator's choice (i) for open question #16, 2026-10-20). Every image model scored by
  the project (003 mean image and each PCA k; 004 `ae`, `mean_image`, `pca_k<K>`) also
  reports its MSE inside three regions built from the true 2D keypoints: `hands` (21-px
  squares on joints 22 / 23), `feet` (17-px squares on 7 / 8 / 10 / 11) and `head` (a
  29-px square on joint 15; SMPL-24 has no face joint), each the union of its visible
  joints' squares clipped to the image, all pixels counted, averaged over the images
  whose region is non-empty (`samples`). Surrogate rulings (batch Q1–Q3, all (a)): these
  joints and sizes (fixed a priori from body dimensions at the ring's 91 px/m), all
  pixels of a region, MSE only. Two pure functions in `strike_a_pose.eval.metrics`
  (`region_mask`, `masked_mse`, constants `REGIONS`); `score_images` takes the
  keypoints, visibility and image size as one optional keyword group and scores the
  regions from the same predictions as the headline (bit-identical with and without);
  the 004 trainer keeps the labels of its scored splits. `metrics.json` gains `regions`
  per image entry, a top-level `regions` definitions block and `units.regions`;
  `metrics_version` 1 and `train_version` 2 unchanged; no config key, figure or output
  line changed; training untouched. Contracts updated in place (003 `eval-run.md`, 004
  `train-run.md`); dated notes in the 003 and 004 specs. Implementation `1750c99`; pytest
  307 passed (34–41 s; 267 before); fixture smoke before / after: the 003 and 004
  `metrics.json` only gain the new keys (25 leaves each, no value changed), figures,
  configs, history and output lines identical, the 006 probe byte-identical on the same
  checkpoint; the 003 and 004 runs give identical `mean_image` / `pca_k4` entries,
  regions included. Cost: 1.0–1.3 s per model per 1,600-image split. The published 003 /
  004 / 006 numbers predate 009; the re-baseline run brings the first per-region
  numbers.
- 2026-10-21: slice 010 `dataset-realism` (autonomous run, Fable surrogate scope; spec
  approved at surrogate scope 2026-10-21, operator sign-off pending; approval basis: the
  operator's decision on open question #15, 2026-10-20). Three opt-in generator config
  blocks, each left out of the manifest `config` while at its default, so today's
  datasets regenerate byte for byte (pins against the committed fixture manifest; a
  before/after generation of synthetic datasets with today's configs is byte-identical):
  `ranges_version: predecessor-v2-armsdown` (shoulder z narrowed to `[-20, 20)`, collar z
  to `[-10, 10)`, arms lowered by ∓65° after sparsification; every other draw identical to
  the default table for the same seed, so `smpl_proc_v2` vs `smpl_proc_v1` is a
  controlled comparison; this reopens the 001 answer "keep arms-out limitation" for the
  new table only, and the collar narrowing departs from the predecessor's table),
  `body_shape` (per-pose betas N(0, std²) clipped at ±clip from their own random stream,
  shared by the pose's views) and `camera_jitter` (per-sample camera moved in height and
  distance, aimed at the ring's look-at point and tilted down, accepted only if every mesh
  vertex projects inside the image with a margin, else the ring camera; labels record
  `camera.jitter`; names, yaw and the manifest camera list stay the ring's). Surrogate
  rulings (batch Q1–Q4): a cumulative chain `smpl_proc_v2` (arms-down) → `v3` (+ shape) →
  `v4` (+ jitter) → `v5` (+ 4,000 poses), each compared with its predecessor; the
  aim-and-tilt jitter; the standard split for 4,000 poses; 003 `image_pca_ks: []` for the
  4,000-pose dataset, whose image PCA needs an estimated 25–30 GB. Configs
  `configs/smpl_proc_v{2,3,4,5}/{generate,baselines,ae,probe}.yaml`. No dataset, split or
  run exists yet: the run procedure (F1 → `smpl_proc_v1` re-baseline → v2 → v3 → v4 → v5,
  ≈ 8.5 h CPU after the re-baseline, with an arms-down visual check before the v2 split) is
  in 010 `tasks.md`.
- 2026-10-21: slice 011 `training-variants` (autonomous run, Fable surrogate scope; spec
  approved at surrogate scope 2026-10-21, operator sign-off pending; approval basis: the
  operator's options (ii)–(iv) for open question #16, 2026-10-20). Six train config keys,
  each defaulting to today's computation and carrying that value in `LEGACY_VALUES`: a
  default run equals the pre-011 code's bit for bit (regression test against an inline copy
  of the old loop; a same-machine fixture train + probe before/after comparison with
  identical history, weights, optimizer state, figure and metrics apart from the additive
  keys). `weight_decay` (AdamW, decoupled, one group), `shift_px` (per-sample integer shifts
  of the train batches, zero fill), `dropout` (on the latent, training only), `mask_weight`
  (α: `sum(w·d²) / sum(w)` with `w = 1 + α·[x > 0]`, the rendered silhouette),
  `variational` + `beta` (VAE: a `logvar` head built last, `z = μ + σ·ε` in training only,
  `loss = recon + β·KL/(H·W)`; `encode` = μ, so val loss, selection, metrics, figure and
  probe never sample). `history.csv` appends `train_recon`, `train_kl`; `metrics.json` adds
  `model.variational` and `training.objective`, the probe `encoder.variational` (versions
  unchanged). Surrogate rulings (batch Q1–Q3): one combined regularization run R3 first; R1–R3
  on `smpl_proc_v1` against the re-baseline R0, the 010 chain stays plain and each winning
  variant runs once on the last 010 dataset next to its plain run; β = 0.1 first. Configs
  `configs/ae_{mask,vae,reg}.yaml`; no run exists yet (experiment procedure in 011
  `spec.md`; ≈ 5 h CPU after R0).
- 2026-10-21: slice 012 `run-guards` (operator present; orchestrator as the operator's
  surrogate; spec approved at surrogate scope 2026-10-21, operator sign-off pending; approval
  basis: the operator's decision 2026-10-21 to run the ≈ 20 h campaign "so long as we have
  checks to ensure early termination and iteration if the run looks to be spinning its
  wheels"). Trainer guards, off by default and in `LEGACY_VALUES`: `early_stop {patience,
  min_epochs, min_delta}` (Keras semantics; decided after each epoch and on resume from the
  checkpointed history; the cosine schedule stays planned for all epochs and the 008 selection
  is unchanged, so a stopped run is a prefix of the full run; `training.stopped_early` only when
  configured) and `abort_on_divergence` + `divergence_ratio` (non-finite train loss, train loss
  above 10× epoch 1's after epoch 3, three non-finite val losses: `aborted.json`, no
  `metrics.json`, exit 5, `--resume` refuses). Red-team additions: a run-dir lock (a live
  owner refuses a second start, exit 2; dead, reused or other-boot pids are stale: the
  published 004 run had two resumes 19 s apart); a batched float64 mean image (the old one
  allocated a float64 copy of the train matrix: ≈ 16.7 GB peak for the 4,000-pose AE run, an
  OOM on 15 GB; now ≈ 3 GB, bit-identical); `env.cpu_model` and `dataset.checksums_sha256` in
  003 / 004 / 006 `run_info.json`; documentation honesty fixes (approval scope, 007–012
  verified on the fixture only, the 004 run's exact sessions). `scripts/run_status.py`
  (read-only status for check-ins). Campaign configs: guards in `configs/ae{,_mask,_vae,_reg}.yaml`,
  new `ae_seed1.yaml`, `ae_vae_beta0.1.yaml`, `ae_vae_beta0.001.yaml`; `ae_vae.yaml` β 0.01.
  Hygiene: the generator's omitted defaults frozen for layout v1 (contract + literal pin); F1
  reports a SMPL-file mismatch instead of stopping, adds a manifest check with the recorded
  SMPL hash substituted and a pixel diagnostic (the 003 baselines rerun against the published
  numbers, ≤ 1e-5). Surrogate rulings (batch Q1–Q3): (a) with per-variant promotion metrics
  (operator confirmation requested); (a) plus the pixel diagnostic; (a) unattended check-ins
  advance the campaign and commit results. Operator decisions 2026-10-21: F1 failure →
  continue with a caveat naming the part that differs (SMPL file, labels, pixels); independent
  realism datasets v2a–v2d (010 amendment by a separate lead); `armsdown-v1` with an automatic
  intersection count; a seed-1 control R0b; the first VAE β 0.01. Campaign procedure (F1 → R0
  → R0b → v2a–v2d → R1 → R2 → R3 → winners, ≈ 18 h before the winners) with decision rules,
  R0 acceptance tolerances and the verbatim check-in text: `docs/HANDOFF.md` and 012 `tasks.md`.
  pytest 446 passed (48.8 s; 374 before); fixture 003 / 004 `metrics.json` and figures
  byte-identical to `e98134e`'s. #12 (torch CPU index) on hold until the campaign's runs are done.
- 2026-10-21: slice 014 `realism-ablations` (operator decisions 2026-10-21, amending 010; no
  clarification batch). The realism datasets are independent single-change datasets, each
  compared with `smpl_proc_v1`: `configs/smpl_proc_v2{a,b,c,d}/` replace the chain
  `configs/smpl_proc_v{2,3,4,5}/`; their AE configs carry the 012 guards and the 010 series test
  no longer exempts them. The arms-down table is renamed `armsdown-v1` (not the predecessor's;
  no alias, never generated); its confound (narrowed shoulder / collar z: less pose variety; more
  limb–torso overlap) is documented in 010 FR-003. New `datagen/intersection.py` (NumPy): arm
  vertices (dominant joints 16–23) inside per-segment elliptic cylinders of the torso (joints 0,
  3, 6, 9) in Kabsch-fitted rest frames, 1 cm margin; `scripts/intersection_check.py` gates
  v2a's split at 5 % of poses (the visual check stays). Synthetic body, v1 vs `armsdown-v1`
  table, 2,000 poses: 0.00 % vs 0.05 % (6-cm cube torso), 0.65 % vs 7.70 % (widened torso);
  not predictive of SMPL. Campaign procedure (HANDOFF, 012 `tasks.md`) updated; W: pending
  operator decision, default v2a. Also `fix(012)`: the run-dir lock treats a `python -m
  strike_a_pose.train.cli` trainer as alive (found in live use).
- 2026-10-21: slice 013 `capsule-body` (operator decision 2026-10-21: the licensed SMPL model
  is still unavailable; an interim, license-free "capsule humanoid built on SMPL's own
  skeleton" so the pipeline and campaign run now, results labelled infrastructure tests).
  Surrogate rulings: Q1 mirror the campaign (capsule v1 and its variants, then the
  independent `capsule_proc_v2a`–`v2d`), Q2 (a) results only under `docs/results/013-capsule/`,
  Q3 (b) rest skeleton derived at build time from the committed fixture labels (operator
  confirmation of the license reading pending). Body: 28 rigid capsules (skinned to the bone's
  proximal joint), 24 marker vertices with a one-hot regressor, zero pose blends, shape
  components size / limb thickness / torso width (3–9 zero), cylinder rings ≤ 3 cm (render
  19 ms vs 370 ms without), uncompressed `.npz` with 1 µm rounding, SHA-256 `bc1f3eb7…` pinned.
  Fixture fidelity: joints within 1.1 µm of the labels, IoU 0.73–0.82 against the real-SMPL
  masks. `capsule_proc_v1` generated, split frozen (`splits/capsule_proc_v1.json`, same pose
  assignment as `smpl_proc_v1`), baselines' pose rows equal the published ones (≤ 1e-4).
- 2026-10-22: 007 follow-up F1 done.`SMPL_NEUTRAL.npz` arrived via `domattioli/smpl-assets` (re-converted by the operator from the original `.pkl`; SHA-256 `f09c8ece…`, 300 shape components of which the generator uses 10; not the recorded `f408141b…` file). `smpl_proc_v1` regenerated at `3ba4259`+: manifest checks 1 and 2 both miss `9000c915…` because label values differ by ≤ 5e-6 (6th-decimal rounding of the re-converted arrays; fixture regeneration: joints ≤ 5e-6, pixels ≤ 1 grey level). Pixel diagnostic: the 003 baselines rerun (`runs/baselines-smpl_proc_v1-20261022T011445Z`, copy in `docs/results/007-f1-baselines-smpl_proc_v1/`) reproduces **every** published 003 number to ≤ 1e-5 relative. Verdict: reproduced in substance, not byte for byte (caveat: the published 003/004/006 runs used the `406b68f` dataset, raw manifest `9000c915…`, which differs from the regenerated one at the 1e-6 level). `splits/smpl_proc_v1.json` rewritten as `split_version` 2 with unchanged pose lists. 
- 2026-10-22: SMPL campaign done (R0, R0b, v2a-v2d, R1-R3; results in `docs/results/smpl-campaign/`). R0 reproduces the published 004 run within the pre-registered tolerances. Strongest lever: more distinct poses (v2d: val MSE -18 %, probe 150 vs 175 mm). Training variants under the pre-registered rule: R2 VAE beta 0.01 winner (no cost), R1 mask-MSE dropped on pixels but its latent probes 12 mm better (rule choice for the operator), R3 no gain. The 014 intersection gate is miscalibrated on real SMPL (flags 79 % of v1 poses); recalibration open.
- 2026-10-24: slice 016 `probe-round-trip` designed (operator decision 2026-10-24, grill point 5: the 015 valid-silhouette criterion is too weak). The probe already predicts the 72-d SMPL pose and the yaw class, but saves no weights: the round trip re-fits the selected MLPs from the probe cache (seeded, checked against the recorded scores), re-renders the predicted pose with mean shape at the predicted ring camera, and scores silhouette IoU; τ = 5th percentile (`method="lower"`) of real-val round-trip IoU; shuffled-pose chance control. Surrogate rulings Q1-Q5 all (a). Tasks B1-B3 for a Sonnet implementer.
