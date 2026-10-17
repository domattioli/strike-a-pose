# strike-a-pose Roadmap (spec of specs)

Status: **DRAFT — awaiting operator sign-off on slice list and order.**

Each slice → `specs/<NNN-slug>/{spec,plan,tasks}.md` via a speckit pipeline
run. Each slice's `spec.md` needs operator sign-off before plan/implement
(constitution, Development Workflow). Plan changes are recorded in the
Changelog below.

## Goal (MVP)

Reproducible successor to `Conv_AE-Human_Pose`: from a scripted, versioned
synthetic dataset of rendered human poses, train an autoencoder whose latent
reconstructs the pose better than a trivial baseline on a fixed held-out split.

## Slices

| # | Slug | Purpose | Depends on | Done when |
|---|------|---------|------------|-----------|
| 001 | `dataset-generation` | Scripted CLI that renders body-model poses from defined cameras and writes images + labels (pose params, keypoints, camera, mask) with a manifest | — | Same config + seed → byte-identical manifest; small fixture set committed for tests |
| 002 | `data-pipeline` | Package `strike_a_pose.data`: dataset loader, transforms, normalization, frozen train/val/test split | 001 (manifest schema) | Loader unit tests pass on fixture; split file versioned |
| 003 | `baselines-and-metrics` | Metrics (e.g., MPJPE, image recon error) + mean-pose / PCA baselines, eval CLI | 002 | Baseline numbers produced by a logged run |
| 004 | `autoencoder-mvp` | Config-driven train/eval CLI for the AE, CPU smoke test in CI | 002, 003 | Beats baseline on held-out split by the margin set in its spec |
| 005 | `repo-foundation` | `pyproject.toml`, ruff, pytest, CI workflow (lint + tests + smoke) | — | CI green on an empty package; can land before or with 001 |

Out of MVP (later candidates): latent → camera/pose regression (the old
linear-model probe), generation/interpolation, real-image datasets, GPU
rails (Kaggle).

## Open questions (batched; answer all at once)

Slice-level:
1. Approve slice list and order? Should 005 be folded into 001?
2. Model input for 004: rendered images (like the predecessor), keypoints, or both?
3. MVP success margin over baseline (e.g., ≥ X% lower error)?

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
