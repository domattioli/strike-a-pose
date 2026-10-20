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
  documented range table (`proc-v1`, research R9), recorded in the manifest;
  AMASS kept as optional config source. Fixture (T022) regenerated from real
  SMPL with the procedural source and committed. Default dataset renamed
  `smpl_proc_v1`.
