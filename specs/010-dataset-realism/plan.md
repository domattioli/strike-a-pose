# Implementation Plan: Dataset Realism Options

**Branch**: `claude/charming-wozniak-5ouldf` (slice `010-dataset-realism`) | **Date**: 2026-10-21 | **Spec**: [spec.md](spec.md)
**Input**: [spec.md](spec.md), approved at surrogate scope (surrogate rulings 2026-10-21: Q1
(a) chain, Q2 (a) aim-and-tilt jitter, Q3 (a) standard split, Q4 (b) `image_pca_ks: []`).

## Summary

Three optional generator config blocks, each off by default and left out of the manifest
while off: `ranges_version: predecessor-v2-armsdown` (arms lowered ∓65° after
sparsification, shoulder z `[-20, 20)`, collar z `[-10, 10)`), `body_shape` (per-pose betas
N(0, std²) clipped, own random stream) and `camera_jitter` (per-sample camera moved in height
and distance, aimed at the ring's look-at point, tilted down, accepted only if every mesh
vertex projects inside the image with a margin, ring-camera fallback recorded). More poses is
`num_poses`. One 003 change: `image_pca_ks: []` skips image PCA (for the 4,000-pose
dataset). Four config directories `configs/smpl_proc_v2…v5/` define the chain; datasets,
splits and runs follow the documented run procedure once SMPL is available.

## Technical Context

**Language/Version**: Python ≥ 3.11 (container 3.13; CI 3.11 + 3.13)
**Primary Dependencies**: numpy, pillow, pyyaml (datagen); torch only in 003 (unchanged deps)
**Storage**: datasets under `data/generated/` (git-ignored); configs under `configs/`
**Testing**: pytest with the synthetic body model (`tests/conftest.py`) and the committed fixture
**Target Platform**: Linux CPU (4 vCPU, 15 GB RAM dev machine)
**Project Type**: library + CLI
**Performance Goals**: generation time unchanged within noise (in-frame test ≪ render)
**Constraints**: today's datasets byte-identical (FR-009); no new dependency; no SMPL here
**Scale/Scope**: ~250 lines of source, ~400 lines of tests, 16 config files, docs

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | All logic in `strike_a_pose.datagen` (+ the 003 config / baselines); no notebook. |
| II CLI + config | PASS | Every option is a versioned config key; per-dataset config directories; CLI unchanged. |
| III Reproducibility | PASS | Seeded, independent streams (pose: `default_rng(seed)` unchanged; shape: `[seed, 1, pose]`; jitter: `[seed, 2, pose, ring]`); manifest records the resolved options; v1 bytes pinned. |
| IV Test-first | PASS | Tests per stage written before the code (red run noted in tasks.md); synthetic models incl. a `shapedirs`-bearing variant. |
| V Honest evaluation | PASS | Each new dataset gets 003 baselines (mean image, pose baselines; PCA where memory allows), AE and probe runs with the v1 configs; comparisons named per the Q1 ruling. |
| VI Simplicity | PASS | Three small config blocks, no new module beyond `datagen/shape.py`; no collision model, no free cameras; FR-014 is ≈ 10 lines. |

No violations; Complexity Tracking empty.

## Design

- `datagen/config.py`: `BodyShapeConfig(std=1.0, clip=2.0)`, `CameraJitterConfig(height_m=
  (-0.3, 0.3), distance_m=(0.0, 0.5), pitch_deg=(0.0, 5.0), margin_px=2, max_attempts=16)`,
  fields `ranges_version`, `body_shape`, `camera_jitter` on `GenerationConfig`;
  `ADDED_FIELD_DEFAULTS` = the three defaults, dropped from `to_dict()` when equal (FR-002);
  strict validation (spec edge cases), numbers normalized to float, ranges to lists in
  `to_dict()`.
- `datagen/procedural.py`: `RANGES_TABLES = {version: (ranges, offsets)}`; v1 entry is the
  unchanged `RANGES_DEG` (`RANGES_VERSION` stays the v1 name); `ARMSDOWN_VERSION`;
  `ranges_table(version)` adds `offset_deg` to every row only for tables with offsets;
  `sample_procedural_poses(n, seed, version=RANGES_VERSION)` adds the offsets after
  sparsification.
- `datagen/shape.py`: `SHAPE_STREAM = 1`; `sample_betas(num_betas, std, clip, seed,
  pose_index)` = `clip(default_rng([seed, 1, pose_index]).normal(0, std, num_betas))`.
- `datagen/smpl.py`: `SMPLModel.num_shape_components` (`shapedirs.shape[-1]`).
- `datagen/cameras.py`: `JITTER_STREAM = 2`; `jittered_camera(ring_cam, ring_cfg, dh, dd,
  dp_deg)` (same `look_at` call as `camera_ring`, so zero deltas are bit-identical);
  `in_frame(cam, verts, margin)`; `choose_camera(ring_cam, ring_index, verts, ring_cfg,
  jitter, seed, pose_index) -> (Camera, record)`.
- `datagen/generate.py`: table version into poses, `inputs` and labels; `num_betas` check
  right after loading the body model (before any output); per-pose betas; per-sample camera
  and `camera.jitter`; nothing else changes.
- `eval/config.py`, `eval/baselines.py`: empty `image_pca_ks` allowed; `Baselines.image_pca`
  optional; `fit.image_pca` null.
- `configs/smpl_proc_v{2,3,4,5}/{generate,baselines,ae,probe}.yaml` (FR-010).

## Stages (each green and committed: ruff, ruff format, full pytest)

1. `plan(010)`: spec with rulings, plan, research, data-model, tasks, analyze.
2. `feat(010): config surface`: omit-when-default machinery and the FR-009 pins (fixture
   config dict, fixture poses, `inputs.pose_source`, `cameras`); `image_pca_ks: []`
   (FR-014). Each option's config block lands with its feature (stages 3–5), so no stage
   accepts an option the generator ignores.
3. `feat(010): arms-down table`: `ranges_version` + table + tests.
4. `feat(010): random body shape`: `body_shape` + betas stream + tests.
5. `feat(010): camera ring jitter`: `camera_jitter` + in-frame check + fallback + tests.
6. `feat(010): dataset configs`: the 16 config files + tests; run procedure in tasks.md.
7. `docs(010)`: 001 / 002 / 003 / 006 / 009 notes, README, roadmap, HANDOFF.

## Risks

- A v1 byte change slipping in: pins in CI plus the before/after synthetic comparison (stage 2
  records the "before" datasets first).
- NumPy draw structure (FR-003 test) is a property of the locked numpy; a numpy change would
  alter every dataset anyway.
- Interpenetration with arms down cannot be checked without the mesh: the run procedure's
  visual check gates the `smpl_proc_v2` split.
