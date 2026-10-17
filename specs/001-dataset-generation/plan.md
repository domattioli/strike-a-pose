# Implementation Plan: Synthetic Pose Dataset Generation

**Branch**: `claude/charming-wozniak-5ouldf` (slice `001-dataset-generation`) | **Date**: 2026-10-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-dataset-generation/spec.md` (approved 2026-10-17)

## Summary

A config-driven CLI (`strike-a-pose-data generate --config … --seed …`) poses
the neutral SMPL body (mean shape) with AMASS-sampled poses, renders each pose
from a fixed named 8-camera ring with an in-package deterministic NumPy
rasterizer, and writes per-sample RGB image, mask and JSON labels plus a
timestamp-free `manifest.json` that is byte-identical for the same
config + seed + inputs + commit. A `validate` command checks any dataset
against the layout contract. SMPL/AMASS are licensed and never downloaded;
missing assets fail fast. All tests use a synthetic SMPL-format model and
synthetic AMASS files. The committed real-SMPL fixture is an operator task.

## Technical Context

**Language/Version**: Python ≥ 3.11 (container: 3.13)
**Primary Dependencies**: numpy (arrays/LBS/rasterizer), pillow (PNG I/O), pyyaml (config). Dev: pytest, ruff. No torch/smplx/pyrender — see research R1/R2.
**Storage**: files under `data/generated/<name>/` (git-ignored); licensed assets under `assets/` (git-ignored)
**Testing**: pytest, CPU only, no licensed assets
**Target Platform**: Linux/macOS laptop CPU
**Project Type**: library + CLI
**Performance Goals**: ≤ 0.2 s/image at 256×192 → default 16,000 images < 1 h (SC-004)
**Constraints**: deterministic labels/manifest; never write partial "complete" datasets; fixture validate < 5 s (SC-005)
**Scale/Scope**: default 2,000 poses × 8 cameras; fixture 2 × 8 = 16 samples

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | All logic in `src/strike_a_pose/`; no notebooks. |
| II CLI + config | PASS | `generate` driven by versioned YAML in `configs/`; no hard-coded paths (asset paths are config fields). |
| III Reproducibility | PASS | Seed is a required flag; Python/NumPy seeded; manifest records resolved config, seed, commit+dirty, input SHA-256s; `run_info.json` records environment. No torch in this slice (no torch seed to set; noted). |
| IV Test-first | PASS | Unit tests for skeleton, LBS, cameras/projection, rasterizer, sampling, manifest/labels, determinism, CLI errors — written with the code. Training smoke test N/A (no training in this slice). |
| V Honest evaluation | N/A | No models/metrics in this slice. |
| VI Simplicity | PASS | 3 runtime deps, each justified (R1, R2); no plugin layers; one renderer, one body model. |
| Tech: datasets never committed | PASS w/ note | Only the ≤16-sample fixture (FR-009, Q8 operator-approved) is committed; `data/`, `assets/` ignored. |
| Tech: pinned deps + lockfile | PASS | `pyproject.toml` + `uv.lock`. |
| Tech: license compat | PASS | numpy (BSD), pillow (MIT-CMU), pyyaml (MIT), pytest (MIT), ruff (MIT). |

Re-check after design: PASS (no changes).

## Project Structure

### Documentation (this feature)

```text
specs/001-dataset-generation/
├── spec.md  plan.md  research.md  data-model.md  tasks.md
└── contracts/dataset-layout.md
```

### Source Code (repository root)

```text
pyproject.toml            # package, deps, console script, ruff + pytest config
uv.lock
configs/
├── default.yaml          # 2000 poses × 8 cams (operator edits asset paths/subsets)
└── fixture.yaml          # 2 poses × 8 cams, dataset_name "smpl_fixture"
docs/data-assets.md       # FR-008: how to obtain SMPL + AMASS, licenses, paths
src/strike_a_pose/
├── __init__.py
└── datagen/
    ├── __init__.py
    ├── skeleton.py       # SMPL24 names/parents (FR-010)
    ├── config.py         # GenerationConfig dataclasses + YAML load/validate
    ├── assets.py         # MissingAssetError, asset checks, sha256
    ├── smpl.py           # SMPL model load (.npz/.pkl) + NumPy LBS
    ├── amass.py          # candidate enumeration + seeded sampling → PoseRecord
    ├── cameras.py        # ring construction, projection, visibility
    ├── render.py         # z-buffer rasterizer → (rgb, mask)
    ├── io.py             # labels/manifest JSON writers, readers, validate
    ├── generate.py       # orchestration (generate_dataset)
    └── cli.py            # argparse entry point `strike-a-pose-data`
tests/
├── conftest.py           # synthetic SMPL-format model + synthetic AMASS tree fixtures
├── test_skeleton.py  test_config.py  test_smpl.py  test_amass.py
├── test_cameras.py   test_render.py  test_io.py    test_generate.py
├── test_cli.py       test_fixture.py
└── fixtures/smpl_fixture/   # OPERATOR: real-SMPL fixture (FR-009), absent until generated
```

**Structure Decision**: single package `strike_a_pose`, data generation under
the `datagen` subpackage so slice 002's `strike_a_pose.data` stays separate.
Minimal repo scaffolding (pyproject, ruff, pytest) lands here; CI workflow
stays with slice 005.

## Key design points (normative detail in research.md / data-model.md / contract)

- Pipeline: load config → check assets (fail fast, exit 3, write nothing) →
  check output dir (exit 4 unless `--overwrite`) → enumerate+sample poses →
  hash inputs → write incomplete manifest → for each pose: LBS once, for each
  camera: project, render, write PNGs + labels → write `checksums.json`,
  `run_info.json` → atomically write complete manifest.
- Body model is passed as an object with `faces` and
  `forward(pose72, betas) → (verts (V,3), joints (24,3))`; tests feed a
  synthetic SMPL-format `.npz` (one cube per joint, one-hot weights, zero
  blendshapes) through the real loader and LBS.
- Interrupted run: manifest remains `complete: false`; re-run requires
  `--overwrite` (restart cleanly). Resume is out of scope.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Deviation from spec-input "pyrender" default | Determinism + zero GL deps | pyrender needs EGL/OSMesa, not bit-exact; FR-003 only requires headless CPU |
