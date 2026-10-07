---

description: "Task list for feature 001-kill-test-mvp"
---
<!-- provenance: author=domattioli model=claude-fable-5-1 effort=high date=2026-10-07 skill=speckit-tasks repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->

# Tasks: Kill-Test MVP for Calibrated Multi-View Body-Measurement Uncertainty

**Input**: Design documents from `/specs/001-kill-test-mvp/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Constitution Principle VI makes one CPU smoke test per new module mandatory. Every module task therefore names the module file and its test file (two files). Story-level integration tests are separate tasks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Builder tiers**: every task line ends with `(tier: haiku)` for bounded, mechanical work or `(tier: sonnet)` for algorithmic work (rendering math, posterior fusion, conformal calibration, measurement geometry, data generation, training). Each task touches at most two files unless the line says otherwise.

## Format: `[ID] [P?] [Story] Description (tier: X)`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1 to US4)
- Include exact file paths in descriptions

## Path Conventions

- Single project: `src/strike_a_pose/` and `tests/` at the repository root, as plan.md states.
- Configuration files under `configs/`, scripts under `scripts/`, the notebook under `notebooks/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create `pyproject.toml` (package `strike_a_pose`, `sap` entry point, dependencies with lower bounds, extras `[body]`, `[real]`, `[dev]`, pytest and ruff configuration) and `src/strike_a_pose/__init__.py` (`__version__ = "0.1.0"`) (tier: haiku)
- [ ] T002 Create `constraints.txt` (exact pins per research R11) and extend `.gitignore` with `assets/`, `data/`, `out/`, `dist/`, `*.pkl`, `*.npz`, `*.pt`, `*.pth`, `*.ckpt`, `.ipynb_checkpoints/` (tier: haiku)
- [ ] T003 [P] Create `configs/tiny.yaml` and `configs/full.yaml` with every key and value of contracts/config.md (tier: haiku)
- [ ] T004 [P] Create `tests/conftest.py` (fixtures: tiny config path, temporary output dir, `SAP_ASSET_ROOT` unset, socket-level network blocker) and `tests/test_repo_hygiene.py` (FR-022: fail on any tracked asset, render, weight, per-subject table, or file over 1 MB; allow `tests/fixtures/` files up to 100 kB) (tier: haiku)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Implement `src/strike_a_pose/config.py` (YAML load, defaults, unknown-key error, canonical dump, SHA-256 `config_hash`, `--set` overrides) and `tests/test_config.py` (tier: haiku)
- [ ] T006 [P] Implement `src/strike_a_pose/assets.py` (asset root from configuration or `SAP_ASSET_ROOT`, `MissingAssetError` naming file and key, `list_assets()` for `sap info`, no download path) and `tests/test_assets.py` (tier: haiku)
- [ ] T007 [P] Implement `src/strike_a_pose/seeding.py` (`rng_for(seed, *path)` from `numpy.random.SeedSequence`, torch seeding helper, deterministic flags) and `tests/test_seeding.py` (same stream on repeat, different streams per path) (tier: haiku)
- [ ] T008 [P] Implement `src/strike_a_pose/device.py` (select `auto`, `cpu`, `cuda`; hardware class; device name) and `tests/test_device.py` (CPU fallback when CUDA is unavailable) (tier: haiku)
- [ ] T009 [P] Implement `src/strike_a_pose/runrecord.py` (fields of contracts/artifacts.md, library versions, stage timings, atomic JSON write) and `tests/test_runrecord.py` (tier: haiku)
- [ ] T010 [P] Implement `src/strike_a_pose/checkpoint.py` (`DONE.json` write and staleness check, atomic rename helper, `TimeBudget.should_stop(next_unit_seconds)`) and `tests/test_checkpoint.py` (tier: haiku)
- [ ] T011 Implement `src/strike_a_pose/body/base.py` (`BodyModel` protocol: `joint_names`, `vertices(betas, pose_root, pose_body)`, `joints(...)`, `faces`, `part_ids`, `canonical()`; joint-name table; bone list) and `src/strike_a_pose/body/standin.py` (procedural capsule mannequin per research R4) (tier: sonnet)
- [ ] T012 Write `tests/test_body_base.py` and `tests/test_standin.py` (joint names match the table; vertex count stable; shape coefficient 0 raises height monotonically; a pose moves child parts only; determinism) (tier: haiku)
- [ ] T013 Implement `src/strike_a_pose/measure.py` (plane-triangle intersection for a part set, 2D convex hull perimeter, the five definitions of research R6, NaN flag) and `tests/test_measure.py` (a capsule of radius r gives 2 pi r within 1%; height equals the mannequin height; NaN on an empty slice) (tier: sonnet)
- [ ] T014 Implement `src/strike_a_pose/camera.py` (intrinsics, look-at extrinsics, rig sampling with minimum separation, placement-noise rotation with fixed angle and random axis, 6D rotation encoding, projection) and `tests/test_camera.py` (rotation angle equals the noise level; separation enforced; projection of a known point; 0 degrees is identity) (tier: sonnet)
- [ ] T015 Implement `src/strike_a_pose/render.py` (project faces, drop faces behind the camera, `cv2.fillPoly` union with `shift=4`, area fraction, border-touch flag) and `tests/test_render.py` (a unit cube renders the expected square area within 2%; bit-identical on repeat; empty when behind the camera) (tier: sonnet)
- [ ] T016 Implement `src/strike_a_pose/pose/filters.py` (joint-angle limits from the configuration table; capsule radii from canonical vertices; non-adjacent capsule overlap; `accept(pose) -> PoseFilterResult`) and `tests/test_pose_filters.py` (over-limit joint rejected; leg-through-leg pose rejected; canonical pose accepted) (tier: sonnet)
- [ ] T017 Implement `src/strike_a_pose/pose/source.py` (`PoseSource` protocol, registry `amass` and `limits`, `draw(rng) -> (pose_root, pose_body)`) and `src/strike_a_pose/pose/limits.py` (uniform axis-angle within the limits, random axis) (tier: haiku)
- [ ] T018 Implement `src/strike_a_pose/pose/amass.py` (discover `<root>/amass/<subset>/**/*.npz`; accept the `pose_body` plus `root_orient` layout and the `poses` layout; log keys; seeded frame draw; zero hands) and `tests/test_amass.py` (synthetic npz fixtures of both layouts written inside the test) (tier: haiku)
- [ ] T019 [P] Write `tests/test_pose_source.py` and `tests/test_pose_limits.py` (registry resolves both sources; limits draws stay within the table; determinism per seed) (tier: haiku)
- [ ] T020 Implement `src/strike_a_pose/body/smplx_body.py` (lazy `smplx` import, `SMPLX_NEUTRAL.npz` from the asset root, canonical pose, part ids from `lbs_weights` argmax, metres) and `src/strike_a_pose/body/smpl_body.py` (same for SMPL male, female, neutral) (tier: haiku)
- [ ] T021 Write `tests/test_smplx_body.py` (skips when the asset is absent; with a fake `smplx` module asserts joint-name mapping and part-id derivation) (tier: haiku)
- [ ] T022 Implement the `src/strike_a_pose/cli.py` skeleton (argparse with the subcommands of contracts/cli.md, common options, exit codes, `sap info [--strict]`) and `tests/test_cli.py` (`sap info` runs; unknown key exits 2; missing asset exits 3) (tier: haiku)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Run the synthetic kill test and read the verdict (Priority: P1) MVP

**Goal**: `sap run` executes generate, train, predict, calibrate, evaluate, verdict, report from one configuration and one seed and prints the verdict line (FR-001 to FR-016, FR-024 to FR-029).

**Independent Test**: `sap run --config configs/tiny.yaml --out <tmp>` on CPU finishes within 10 minutes with 45 result rows, a verdict line in the fixed format, two plots, and a run record; the full configuration runs on Kaggle (quickstart section 4).

### Tests for User Story 1 (CPU smoke test per new module is mandatory, constitution VI; more only if requested)

Module tests live in the module tasks below (two files each). Integration test: T038.

### Implementation for User Story 1

- [ ] T023 [US1] Implement `src/strike_a_pose/data/manifest.py` (manifest CSV columns of contracts/artifacts.md with `repr` floats, shard npz keys, bit-packing of masks, atomic write) and `tests/test_manifest.py` (round trip; byte-identical CSV on repeat) (tier: haiku)
- [ ] T024 [US1] Implement `src/strike_a_pose/data/generate.py` (per shard: seeded betas, pose draws with filter and rejection count, rig, four renders, canonical-pose measurements, flags; skip existing shards; `summary.json`; `DONE.json`) and `tests/test_generate.py` (two tiny runs produce identical shard bytes; resume skips existing shards; rejection rate reported) (tier: sonnet)
- [ ] T025 [P] [US1] Implement `src/strike_a_pose/data/splits.py` (contiguous ranges, `split_of(body_id)`, disjointness check) and `tests/test_splits.py` (tier: haiku)
- [ ] T026 [US1] Implement `src/strike_a_pose/data/dataset.py` (torch Dataset over shards; view subset `0..k-1`; per-cell noise from `noise_axis` and `noise_deg`; training mode draws k and noise per sample from the seeded generator; camera encoding vector; collate with view masks) and `tests/test_dataset.py` (shapes for k in 1..4; cell noise changes `R_given` only; flagged bodies excluded) (tier: sonnet)
- [ ] T027 [US1] Implement `src/strike_a_pose/model/encoder.py` (strided CNN per `model.channels`, camera embedding, `mu_v`, `logvar_v`) and `src/strike_a_pose/model/fusion.py` (product of experts with the prior expert over a variable number of views with a view mask) (tier: sonnet)
- [ ] T028 [US1] Implement `src/strike_a_pose/model/decoder.py` (MLP to shape-coefficient mean and log-scale) and `src/strike_a_pose/model/vae.py` (joint plus sub-sampled single-view NLL, KL with warm-up, `sample_betas(views, K)`) (tier: sonnet)
- [ ] T029 [US1] Write `tests/test_fusion.py` (one view at 0 degrees equals its own posterior; fused variance at most each view's; adding a view never increases variance; analytic two-expert check) and `tests/test_vae.py` (loss finite; backward works; shapes for k in 1..4 at 64 px) (tier: haiku)
- [ ] T030 [US1] Implement `src/strike_a_pose/train.py` (AdamW, deterministic flags, seeded sampler, checkpoint every `train.checkpoint_every` steps and per epoch with RNG states, resume, `history.csv`, time budget, `model_final.pt`, `DONE.json`) and `tests/test_train.py` (two epochs on tiny data; resume from a checkpoint reaches the same final loss within 1e-6 on CPU) (tier: sonnet)
- [ ] T031 [US1] Implement `src/strike_a_pose/predict.py` (per split and cell: K latent samples, decoded shape coefficients, body-model measurements in canonical pose, median and scaled MAD, `latent_var_mean`, cached npz per cell, `DONE.json`) and `tests/test_predict.py` (one file per cell; spread floored; a non-finite prediction raises with the sample id) (tier: sonnet)
- [ ] T032 [US1] Implement `src/strike_a_pose/calibrate.py` (normalized scores, `ceil((n + 1)(1 - alpha))` quantile, floor, refusal below `calibrate.min_cal`, `quantiles.csv`, `DONE.json`) and `tests/test_calibrate.py` (quantile index on a known score list; refusal message names the minimum; coverage on synthetic Gaussian data within 2 points of 90%) (tier: sonnet)
- [ ] T033 [US1] Implement `src/strike_a_pose/evaluate.py` (intervals from quantiles, clip at 0 with count, coverage, median width, MAE, mean signed error, inclusive band flag, `results.csv` in the fixed column order, `per_sample/` CSVs, `DONE.json`) and `tests/test_evaluate.py` (45 rows; boundary coverage is in band; clipped count) (tier: haiku)
- [ ] T034 [US1] Implement `src/strike_a_pose/verdict.py` (FR-014 from `results.csv`: four ratios at 0 degrees, median, threshold, eight-cell band check; `verdict.json`, `verdict.md`, the fixed verdict line; `recompute(out)` for `sap verify`) and `tests/test_verdict.py` (PASS and KILL fixtures; an out-of-band compared cell gives KILL with the invalid-comparison note; height and the 2 and 5 degree rows never change the verdict) (tier: haiku)
- [ ] T035 [P] [US1] Implement `src/strike_a_pose/report/tables.py` (`results.md` from `results.csv`, `verdict.md`, the run record, and the real-image tables when present) and `tests/test_tables.py` (tier: haiku)
- [ ] T036 [P] [US1] Implement `src/strike_a_pose/report/plots.py` (Agg backend; coverage and width versus view count, one subplot per measurement, one series per noise level, band shaded) and `tests/test_plots.py` (two PNG files written) (tier: haiku)
- [ ] T037 [US1] Wire `generate`, `train`, `predict`, `calibrate`, `evaluate`, `verdict`, `report`, `run` (stage order, `--resume`, `--time-budget`, exit 7) in `src/strike_a_pose/cli.py` and extend `tests/test_cli.py` (`sap run` on the tiny configuration writes every stage directory) (tier: haiku)
- [ ] T038 [US1] Write `tests/test_e2e_tiny.py` (FR-027: `sap run --config configs/tiny.yaml` on CPU; asserts 45 result rows, the verdict line format, the run record fields, and elapsed time under 600 s) (tier: haiku)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Evaluate the calibrated model on real public images (Priority: P2)

**Goal**: `sap real-eval` reports coverage, median width, mean signed error, subjects evaluated, and subjects skipped for BodyM (2 views) and SSP-3D (1 view), reported only (FR-017 to FR-020).

**Independent Test**: with synthetic dataset-shaped fixtures built inside the tests, `real-eval` writes `real/<dataset>/results.csv` and `subjects.csv` with the right columns and skip counts; with the real assets at the configured path, the operator runs quickstart section 5.

### Tests for User Story 2 (CPU smoke test per new module is mandatory, constitution VI; more only if requested)

Module tests live in the module tasks below (two files each).

### Implementation for User Story 2

- [ ] T039 [US2] Implement `src/strike_a_pose/real/common.py` (pad to square, area resize to `camera.image_size`, threshold, usability rule, nominal camera builder from `real.nominal_camera`) and `tests/test_real_common.py` (tier: haiku)
- [ ] T040 [US2] Implement `src/strike_a_pose/real/bodym.py` (discover split folders and CSV columns by header name, join subjects to photos, front and side masks, five tape measurements, log columns) and `tests/test_bodym.py` (synthetic BodyM-like directory built inside the test) (tier: haiku)
- [ ] T041 [US2] Implement `src/strike_a_pose/real/ssp3d.py` (load `labels.npz`, log keys, silhouettes, gendered SMPL ground truth in canonical pose through `smpl_body.py` and `measure.py`) and `tests/test_ssp3d.py` (synthetic labels fixture; the ground-truth path exercised with the stand-in body in place of SMPL) (tier: sonnet)
- [ ] T042 [US2] Implement `src/strike_a_pose/real/masks.py` (`MaskBackend` protocol; `ProvidedMasks`; `Sam2Masks` with lazy import, checkpoint from the asset root, box prompt; status per subject) and `tests/test_masks.py` (stub backend; the `sam2` test is skipped when the package is absent) (tier: haiku)
- [ ] T043 [US2] Implement `src/strike_a_pose/real/evaluate_real.py` (apply the trained model and the matching cell's quantiles; `results.csv` and `subjects.csv` per dataset; skipped counts; mean signed error) and `tests/test_evaluate_real.py` (tier: sonnet)
- [ ] T044 [US2] Add `sap real-eval` to `src/strike_a_pose/cli.py` and the real-image section to `src/strike_a_pose/report/tables.py` (tier: haiku)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Validate a change on a CPU-only machine (Priority: P3)

**Goal**: one script runs lint, the full suite, and the tiny end-to-end run on a 4-core, 15 GB machine with no GPU, no network, and no licensed asset, within the budgets (FR-026, FR-027, SC-006).

**Independent Test**: `bash scripts/cpu_smoke.sh` exits 0 within 10 minutes per part on the reference machine.

### Tests for User Story 3 (CPU smoke test per new module is mandatory, constitution VI; more only if requested)

Covered by the module tests of Phases 2 to 4 plus the tasks below.

### Implementation for User Story 3

- [ ] T045 [US3] Create `scripts/cpu_smoke.sh` (ruff, pytest with `--durations=10`, the tiny end-to-end run; fails when either part exceeds 600 s) and `tests/test_no_network_no_assets.py` (asserts the network blocker and the unset asset root are active during the suite) (tier: haiku)
- [ ] T046 [US3] Add `pytest-timeout` with a 60-second per-test limit to `pyproject.toml` (`[dev]` extra and pytest options) and to `constraints.txt` (tier: haiku)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - Reproduce a reported result (Priority: P3)

**Goal**: two runs with the same configuration and seed give identical manifests and metrics within tolerance; the verdict is recomputable from saved outputs (FR-024, FR-025, SC-003, SC-005).

**Independent Test**: `tests/test_determinism.py` passes; `sap verify --out <out>` exits 0 on an untouched output directory and 6 after a tampered `results.csv`.

### Tests for User Story 4 (CPU smoke test per new module is mandatory, constitution VI; more only if requested)

- [ ] T047 [US4] Write `tests/test_determinism.py` (two tiny generations give identical shard bytes and manifest; two tiny trainings on CPU give metrics within the SC-005 tolerance; `verdict.recompute` equals the stored verdict) (tier: haiku)

### Implementation for User Story 4

- [ ] T048 [US4] Add `sap verify --out` and `sap run --seed-check` to `src/strike_a_pose/cli.py` and the comparison helpers to `src/strike_a_pose/verdict.py` (exit 6 on any difference) (tier: haiku)

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T049 Create `notebooks/kaggle_run.ipynb` (version print, install from `/kaggle/input/sap-src` with constraints, environment variables, resume copy from the previous output, `sap run --config configs/full.yaml --out /kaggle/working/out --resume --time-budget 8.5h`, print the verdict) and `scripts/build_kaggle_bundle.sh` (wheel, constraints, configs into `dist/kaggle-bundle/`) (tier: haiku)
- [ ] T050 [P] Add a one-line summary docstring and the public-source citation (research R1, R6, R7, R8) to every module under `src/strike_a_pose/` (docstring-only edits across all modules; constitution Principle I) (tier: haiku)
- [ ] T051 [P] Create `.github/pull_request_template.md` with the clean-room statement line and the CPU-suite checkbox (constitution Principle I and the workflow gate) (tier: haiku)
- [ ] T052 Run `bash scripts/cpu_smoke.sh`, fix the ruff findings it names (any files), and copy the tiny-run `evaluate/results.csv` and `run_record.json` to `specs/001-kill-test-mvp/results/tiny/` as pre-Kaggle evidence (tier: haiku)
- [ ] T053 Move specs to DomI: run `bash /home/user/DomI/skills/speckit-pipeline/scripts/move_specs_to_domi.sh --apply` (tier: haiku)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - US1 first (MVP); US2 needs T030 (trained model) and T032 (quantiles) from US1; US3 and US4 need US1
- **Polish (Phase 7)**: Depends on all desired user stories being complete
- **T053**: depends on every other task; nothing runs after it, because it moves `.specify/` and `specs/` into DomI `specs/consumers/strike-a-pose/`

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after T030 and T032; integrates with US1 outputs but is independently testable with fixtures
- **User Story 3 (P3)**: Can start after T038; packages the gate that every PR runs
- **User Story 4 (P3)**: Can start after T034 and T037

### Within Each User Story

- Module and its test ship in the same task; the test is written first and fails before the module exists
- Data before model, model before training, training before prediction, prediction before calibration, calibration before evaluation, evaluation before verdict and report

### Parallel Opportunities

- Phase 1: T003 and T004 after T001
- Phase 2: T006 to T010 together after T005; T011 then T012 and T013 to T016 together; T017 and T018 then T019; T020 then T021
- Phase 3: T025, T035, T036 in parallel with their neighbours; T027 and T028 in parallel after T026
- Phase 4: T039, T040, T042 together; T041 after T020; T043 after T039 to T042
- Phase 7: T050 and T051 in parallel with T049

### Operator step outside this list

The full-size run (quickstart section 4) happens on Kaggle after T052 and is not an LLM task. Its committed outputs land in `specs/001-kill-test-mvp/results/` before T053, or in DomI's consumer copy of that directory after T053; the orchestrator chooses the order.

---

## Parallel Example: User Story 1

```bash
# After T024 lands, launch together:
Task: "T025 data/splits.py + tests/test_splits.py"
Task: "T035 report/tables.py + tests/test_tables.py"
Task: "T036 report/plots.py + tests/test_plots.py"

# After T026 lands, launch together:
Task: "T027 model/encoder.py + model/fusion.py"
Task: "T028 model/decoder.py + model/vae.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: run `tests/test_e2e_tiny.py` and `sap run --config configs/tiny.yaml`
5. Hand the Kaggle bundle to the operator for the full run

### Incremental Delivery

1. Setup + Foundational: every service and geometry module with its CPU test
2. User Story 1: the complete synthetic pipeline and verdict (MVP)
3. User Story 2: real-image rows
4. User Stories 3 and 4: the CPU gate script and the determinism checks
5. Polish: notebook, docstrings, PR template, tiny-run evidence, move to DomI

### Parallel Team Strategy

With several builders: one sonnet builder takes the geometry chain (T011, T013 to T016), another the model chain (T026 to T032); haiku builders take the service modules (T005 to T010, T017 to T022) and the reporting modules (T033 to T038) as their inputs land.

---

## Task counts

- Total: 53 tasks. Sonnet: 14 (T011, T013, T014, T015, T016, T024, T026, T027, T028, T030, T031, T032, T041, T043). Haiku: 39.
- Per story: US1 16 (T023 to T038), US2 6 (T039 to T044), US3 2 (T045, T046), US4 2 (T047, T048). Setup 4, Foundational 18, Polish 5 (T049 to T053).

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group, with the clean-room statement in every PR description
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
