<!-- provenance: author=domattioli model=claude-fable-5-1 effort=high date=2026-10-07 skill=speckit-analyze repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
# Implementation Plan: Kill-Test MVP for Calibrated Multi-View Body-Measurement Uncertainty

**Branch**: `001-kill-test-mvp` | **Date**: 2026-10-07 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-kill-test-mvp/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Build a clean-room Python package, `strike_a_pose` with the `sap` command line, that runs one pre-registered experiment from one configuration file and one seed: generate synthetic silhouettes of randomly posed SMPL-X bodies seen by four cameras, train one multi-view model whose per-view posteriors fuse by product of experts, map latent samples to shape coefficients and then to five measurements in cm, calibrate per-cell intervals with split conformal prediction at 90%, evaluate coverage and median width over view count {1, 2, 4} by placement noise {0, 2, 5} degrees, and print the PASS or KILL verdict of FR-014 from saved outputs. Real-image rows (BodyM, SSP-3D) are reported only. Every stage checkpoints and resumes so the full run fits Kaggle sessions. Every module has a CPU smoke test on a procedural stand-in body, so nothing in the test suite needs a GPU, the network, or a licensed asset. The technical choices and their sources are in [research.md](research.md) (R1 to R14).

## Technical Context

**Language/Version**: Python 3.10 or newer (FR-030); the development container runs 3.13.16 and the Kaggle image pins 3.13 (research R10).  
**Primary Dependencies**: numpy, torch (2.6 or newer, cp313 wheels), opencv-python-headless (4.x line), pyyaml, matplotlib. Optional extras: `[body]` smplx (SMPL-X and SMPL meshes from the asset root), `[real]` sam2 (SSP-3D masks). Dev: pytest, pytest-timeout, ruff. Exact pins in `constraints.txt` (research R11).  
**Storage**: files only. Dataset shards as `.npz`, manifests and results as CSV, records and verdicts as JSON, plots as PNG, under an output directory outside version control.  
**Testing**: pytest, CPU only, no network, no licensed asset; stand-in body model (research R4); every pytest test finishes within 60 seconds (`timeout = 60` applies to every collected test; constitution Principle VI); the tiny end-to-end run (FR-027) is a script step of `scripts/cpu_smoke.sh`, checked by `scripts/check_tiny_run.py`, not a pytest test.  
**Target Platform**: Linux CPU container for development and CI (4 cores, 15 GB RAM); Kaggle GPU notebook (T4 or P100) for the full run.
**Project Type**: single project: importable library plus one command-line entry point.  
**Performance Goals**: full run within 24 GPU-hours in sessions of at most 8.5 hours (SC-010, research R10 and R13); test suite and tiny end-to-end run within 10 minutes each on the reference CPU machine (SC-006).  
**Constraints**: no GPU in development; no network and no licensed asset in tests; no licensed asset or derived artifact in the repository; deterministic generation and splits; checkpoint and resume for every long stage; 10-minute CPU budgets.  
**Scale/Scope**: 25,000 synthetic bodies by 4 views at 128 by 128 pixels (20,000 train, 2,500 calibration, 2,500 test; generation stops unless at least 2,000 calibration and 2,000 test bodies are unflagged, SC-002); 9 cells by 5 measurements; BodyM (2,505 subjects, 8,978 silhouettes) and SSP-3D (311 images) for the real-image rows.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Design response | Status |
|---|---|---|---|
| I. Clean-Room Provenance | No material from the excluded sources; public source named per module; PR statement | Every algorithm cites a public source (research R14); module docstrings name it (task T050); the PR template carries the statement (task T051); the excluded local checkout was never opened | PASS |
| II. Public Data and License Compliance | No licensed asset, derived render, per-subject table, or weight in the repository; assets read from a configured path; third-party licenses recorded | `assets.py` resolves `SAP_ASSET_ROOT` and fails with asset and key names; `.gitignore` plus `tests/test_repo_hygiene.py` (FR-022); committed results are aggregate tables; dependency licenses in research R11; `smplx` and `sam2` are install-time extras, never vendored | PASS |
| III. Coverage-First Evaluation | Coverage and width primary; MAE secondary; nominal level, set sizes, seed in every table; band flags; no unreproducible metric | `results.csv` carries nominal level, `n_cal`, `n_test`, seed, coverage, median width, then MAE; `in_band` flag per cell; every table comes from saved per-sample outputs | PASS |
| IV. Kill-Test Gate Before Scope Grows | Criterion fixed in the spec before training; computed by code; no scope growth; no README | `verdict.py` holds the FR-014 constants and refuses a configuration that differs (exit 2); `configs/full.yaml` is frozen before the first full run, and a later change is a new run recorded next to the first; no extra datasets, models, UI, packaging, or README in this feature | PASS |
| V. Reproducibility | Config plus seed determine a run; deterministic generation; pinned dependencies; run record; code-generated tables | `seeding.py` SeedSequence tree; integer rasterization; `constraints.txt` shared by CPU and Kaggle; `run_record.json`; `sap verify` recomputation (research R9) | PASS |
| VI. CPU-Runnable Tests | Smoke test per module; no GPU, network, or asset in tests; suite within 10 minutes; device fallback | Every module task pairs the module with its test; stand-in body (R4); network blocked in `conftest.py`; `device.py` falls back to CPU; `scripts/cpu_smoke.sh` times the suite and runs the tiny end-to-end run as a script step; every pytest test runs under a 60 s timeout | PASS |
| Assets, Data, Compute (section) | Python package with CLI; asset key; no downloads; fixtures at most 100 kB; outputs outside VCS; results tables under the spec directory | `sap` CLI; `assets.root` or `SAP_ASSET_ROOT`; no download code path; fixtures generated inside tests; `--out` outside the repo; `specs/001-kill-test-mvp/results/` receives the tables and plots together with `run_record.json` and `verdict/verdict.json`, the run record that Principle V requires next to every table | PASS |
| Workflow and Quality Gates (section) | Numbered feature branch; CPU suite passes; clean-room statement; no asset; code-generated tables; docstring summaries | Branch `001-kill-test-mvp`; PR gate tasks in the Polish phase | PASS |

Re-check after Phase 1 design (data model, contracts, quickstart): no principle is violated. The three body-model implementations (SMPL-X, SMPL, stand-in) are required by Principle VI and FR-018 and are one interface, not added complexity. Complexity Tracking stays empty.

## Project Structure

### Documentation (this feature)

```text
specs/001-kill-test-mvp/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   ├── cli.md           # commands, options, exit codes, environment
│   ├── config.md        # configuration schema with tiny and full values
│   └── artifacts.md     # output directory layout and file schemas
├── checklists/          # requirements.md (specify), kill-test-validity.md (checklist)
├── results/             # committed tables and plots of the gate decision (written during implementation and the full run)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
pyproject.toml                 # metadata, dependencies, extras [body] [real] [dev], `sap` entry point, pytest and ruff config
constraints.txt                # exact pins shared by the CPU environment and the Kaggle notebook
configs/
├── tiny.yaml                  # CPU end-to-end smoke configuration (stand-in body, limits poses, 64 px)
└── full.yaml                  # Kaggle full run (SMPL-X, AMASS poses, 128 px, 20k/2.5k/2.5k)
notebooks/
└── kaggle_run.ipynb           # thin wrapper: versions, install from private dataset, env, GPU extrapolation gate on the first session, resume copy, `sap run`
scripts/
├── cpu_smoke.sh               # ruff, pytest, tiny end-to-end run as a script step, opt-in --seed-check; fails when a part exceeds 10 minutes
├── check_tiny_run.py          # FR-027 assertions on a tiny-run output directory; exit 1 names the failed assertion
└── build_kaggle_bundle.sh     # wheel + constraints + configs into one folder for upload as a private dataset
src/strike_a_pose/
├── __init__.py                # version
├── cli.py                     # `sap` subcommands (contracts/cli.md)
├── config.py                  # YAML load, defaults, validation, canonical hash
├── assets.py                  # asset root resolution; MissingAssetError names asset and key (FR-021)
├── seeding.py                 # SeedSequence tree (research R9)
├── device.py                  # device selection and hardware class (FR-028)
├── runrecord.py               # run_record.json (FR-025)
├── checkpoint.py              # stage DONE markers, resume decisions, time budget (FR-029)
├── body/
│   ├── base.py                # BodyModel protocol, joint-name table, part ids
│   ├── standin.py             # procedural capsule mannequin (research R4)
│   ├── smplx_body.py          # SMPL-X through the smplx package (asset)
│   └── smpl_body.py           # SMPL through the smplx package, SSP-3D ground truth only
├── pose/
│   ├── source.py              # PoseSource protocol and registry
│   ├── limits.py              # asset-free joint-limit sampler
│   ├── amass.py               # AMASS npz loader (asset)
│   └── filters.py             # joint-angle limits and capsule self-intersection (FR-001)
├── camera.py                  # rig sampling, intrinsics, extrinsics, placement noise, 6D encoding, projection
├── render.py                  # silhouette by polygon-fill union (research R1)
├── measure.py                 # five measurements by joint-anchored slices (research R6)
├── data/
│   ├── manifest.py            # manifest and shard I/O
│   ├── generate.py            # sharded generation with resume (FR-001 to FR-005, FR-024)
│   ├── splits.py              # contiguous splits by body id
│   └── dataset.py             # torch Dataset: view subsets, placement noise per cell, camera encoding
├── model/
│   ├── encoder.py             # per-view CNN with camera conditioning (FR-006)
│   ├── fusion.py              # product of experts with prior expert (FR-007)
│   ├── decoder.py             # heteroscedastic shape head (FR-008)
│   └── vae.py                 # assembly, loss, latent sampling
├── train.py                   # training loop, checkpoints, deterministic flags, time budget
├── predict.py                 # per-cell predictive medians and spreads from K latent samples
├── calibrate.py               # split conformal per cell and measurement (research R8)
├── evaluate.py                # coverage, median width, MAE, band flags, results.csv
├── verdict.py                 # FR-014 rule; verdict.json and verdict.md; verify recomputation
├── report/
│   ├── tables.py              # results.md from results.csv and the verdict
│   └── plots.py               # coverage and width versus view count (Agg backend)
└── real/
    ├── common.py              # mask preprocessing, nominal cameras, usability rule
    ├── bodym.py               # BodyM loader (FR-017)
    ├── ssp3d.py               # SSP-3D loader and SMPL ground truth (FR-018)
    ├── masks.py               # SAM 2 wrapper, optional import (FR-019)
    └── evaluate_real.py       # real-image tables (FR-020)

tests/
├── conftest.py                # tiny config, temporary output dir, stand-in fixtures, network blocker
├── test_<module>.py           # one smoke test file per module above
├── test_repo_hygiene.py       # FR-022 automated repository check
├── test_check_tiny_run.py     # check_tiny_run.py against a fixture output directory built inside the test
└── test_determinism.py        # SC-005 two-run comparison
```

**Structure Decision**: single project. The package root is `src/strike_a_pose/` with one sub-package per stage group (`body`, `pose`, `data`, `model`, `report`, `real`) and flat modules for cross-cutting services. `tests/` mirrors the module list one to one (every module except `__init__.py` has `tests/test_<module>.py`), so constitution Principle VI (one CPU smoke test per module) is checkable by file name.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations. The table stays empty.
