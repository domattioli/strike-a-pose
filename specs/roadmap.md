# strike-a-pose Roadmap (spec of specs)

Status: **Slice list + order APPROVED (operator, 2026-10-17).** Q2 pending confirmation.

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

Out of MVP (later candidates): latent → camera/pose regression (the old
linear-model probe), generation/interpolation, real-image datasets, GPU
rails (Kaggle).

## Open questions (batched; answer all at once)

Slice-level (answered 2026-10-17):
1. ~~Approve slice list and order?~~ Approved. 005 stays separate (default).
2. Model input: proposed **images in** (as predecessor), keypoints/pose params
   generated as labels by 001 for metrics + later probes. Awaiting confirm.
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
