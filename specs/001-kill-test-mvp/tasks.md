---

description: "Task list for feature 001-kill-test-mvp"
---
<!-- provenance: author=domattioli model=claude-opus-5-5 effort=max date=2026-10-07 skill=speckit-implement repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->

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

- [X] T001 Create `pyproject.toml` (package `strike_a_pose`, `requires-python = ">=3.10"`, `sap` entry point, dependencies with lower bounds, extras `[body]`, `[real]`, `[dev]` including `pytest-timeout`, pytest option `timeout = 60` applied to every collected test with no marker-based deselection, ruff configuration) and `src/strike_a_pose/__init__.py` (`__version__ = "0.1.0"`) (tier: haiku)
- [X] T002 Create `constraints.txt` (exact pins per research R11, including `pytest-timeout`) and extend `.gitignore` with `assets/`, `data/`, `out/`, `dist/`, `*.pkl`, `*.npz`, `*.pt`, `*.pth`, `*.ckpt`, `.ipynb_checkpoints/` (tier: haiku)
- [X] T003 [P] Create `configs/tiny.yaml` and `configs/full.yaml` with every key and value of contracts/config.md (tier: haiku)
- [X] T004 [P] Create `tests/conftest.py` (fixtures: tiny config path, `small_config` returning the overrides of contracts/config.md section "Test overrides", temporary output dir, `SAP_ASSET_ROOT` unset, socket-level network blocker) and `tests/test_repo_hygiene.py` (FR-022 rules: deny-list `.npz .pkl .pt .pth .ckpt .h5 .obj .ply` anywhere; `.png` and `.jpg` allowed only under `specs/*/results/` and `tests/fixtures/`; any tracked CSV whose header contains `body_id`, `subject_id`, or `pose_body_0` fails; any tracked file over 1 MB fails; `tests/fixtures/` files at most 100 kB) (tier: haiku)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Implement `src/strike_a_pose/config.py` (YAML load, defaults, unknown-key error, canonical dump, SHA-256 `config_hash`, `--set` overrides) and `tests/test_config.py` (tier: haiku)
- [X] T006 [P] Implement `src/strike_a_pose/assets.py` (asset root from configuration or `SAP_ASSET_ROOT`, `MissingAssetError` naming file and key, `list_assets()` for `sap info`, no download path) and `tests/test_assets.py` (tier: haiku)
- [X] T007 [P] Implement `src/strike_a_pose/seeding.py` (`rng_for(seed, *path)` from `numpy.random.SeedSequence`, torch seeding helper, deterministic flags) and `tests/test_seeding.py` (same stream on repeat, different streams per path) (tier: haiku)
- [X] T008 [P] Implement `src/strike_a_pose/device.py` (select `auto`, `cpu`, `cuda`; hardware class; device name) and `tests/test_device.py` (CPU fallback when CUDA is unavailable) (tier: haiku)
- [X] T009 [P] Implement `src/strike_a_pose/runrecord.py` (fields of contracts/artifacts.md, library versions, stage timings, atomic JSON write) and `tests/test_runrecord.py` (tier: haiku)
- [X] T010 [P] Implement `src/strike_a_pose/checkpoint.py` (`DONE.json` write and staleness check, atomic rename helper, `TimeBudget.should_stop(next_unit_seconds)`) and `tests/test_checkpoint.py` (tier: haiku)
- [X] T011 Implement `src/strike_a_pose/body/base.py` (`BodyModel` protocol: `joint_names`, `vertices(betas, pose_root, pose_body)`, `joints(...)`, `faces`, `part_ids`, `canonical()`; joint-name table; bone list) and `src/strike_a_pose/body/standin.py` (procedural capsule mannequin per research R4) (tier: sonnet)
- [X] T012 Write `tests/test_body_base.py` and `tests/test_standin.py` (joint names match the table; vertex count stable; shape coefficient 0 raises height monotonically; a pose moves child parts only; determinism) (tier: haiku)
- [X] T013 Implement `src/strike_a_pose/measure.py` (batched API `measure_batch(vertices[B, V, 3], faces, part_ids, joints[B, J, 3], step_cm) -> [B, 5]` with `step_cm` from `measure.step_cm`, that vectorizes the plane-triangle intersection over B meshes and all slice heights at once, torch and device-aware, chunked for memory; convex-hull perimeter by the Cauchy projection formula over 64 directions, error under 0.1%; the five definitions of research R6; NaN flag) and `tests/test_measure.py` (a batch of capsules of radius r gives 2 pi r within 1%; height equals the mannequin height; NaN on an empty slice; the batched result equals the single-mesh result) (tier: sonnet)
- [X] T014 Implement `src/strike_a_pose/camera.py` (intrinsics, look-at extrinsics, rig sampling with minimum separation, placement-noise rotation with fixed angle and random axis, 6D rotation encoding, projection) and `tests/test_camera.py` (rotation angle equals the noise level; separation enforced; projection of a known point; 0 degrees is identity) (tier: sonnet)
- [X] T015 Implement `src/strike_a_pose/render.py` (project faces, drop faces behind the camera, `cv2.fillPoly` union with `shift=4`, area fraction, border-touch flag) and `tests/test_render.py` (a unit cube renders the expected square area within 2%; bit-identical on repeat; empty when behind the camera) (tier: sonnet)
- [X] T016 Implement `src/strike_a_pose/pose/filters.py` (joint-angle limits from the configuration table; capsule radii from canonical vertices; non-adjacent capsule overlap; `accept(pose) -> PoseFilterResult`) and `tests/test_pose_filters.py` (over-limit joint rejected; leg-through-leg pose rejected; canonical pose accepted) (tier: sonnet)
- [X] T017 Implement `src/strike_a_pose/pose/source.py` (`PoseSource` protocol, registry `amass` and `limits`, `draw(rng) -> (pose_root, pose_body)`) and `src/strike_a_pose/pose/limits.py` (uniform axis-angle within the limits, random axis; `pose_root = 0`, an upright body) (tier: haiku)
- [X] T018 Implement `src/strike_a_pose/pose/amass.py` (discover `<root>/amass/<subset>/**/*.npz`; accept the `pose_body` plus `root_orient` layout and the `poses` layout; discard the root orientation (`root_orient`, or `poses[:, :3]`) and return `pose_root = 0`, research R2; log keys; seeded frame draw; zero hands) and `tests/test_amass.py` (synthetic npz fixtures of both layouts written inside the test; the returned root is zero for both) (tier: haiku)
- [X] T019 [P] Write `tests/test_pose_source.py` and `tests/test_pose_limits.py` (registry resolves both sources; limits draws stay within the table; determinism per seed) (tier: haiku)
- [X] T020 Implement `src/strike_a_pose/body/smplx_body.py` (lazy `smplx` import, `SMPLX_NEUTRAL.npz` from the asset root, canonical pose, part ids from `lbs_weights` argmax, metres) and `src/strike_a_pose/body/smpl_body.py` (same for SMPL male, female, neutral) (tier: haiku)
- [X] T021 Write `tests/test_smplx_body.py` and `tests/test_smpl_body.py` (both skip when the asset is absent; with a fake `smplx` module assert joint-name mapping, part-id derivation from `lbs_weights` argmax, and gender selection for SMPL) (tier: haiku)
- [X] T022 Implement the `src/strike_a_pose/cli.py` skeleton (argparse with the subcommands of contracts/cli.md, common options, exit codes, `sap info [--strict]`) and `tests/test_cli.py` (`sap info` runs; unknown key exits 2; missing asset exits 3) (tier: haiku)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Run the synthetic kill test and read the verdict (Priority: P1) MVP

**Goal**: `sap run` executes generate, train, predict, calibrate, evaluate, verdict, report from one configuration and one seed and prints the verdict line (FR-001 to FR-016, FR-024 to FR-029).

**Independent Test**: `sap run --config configs/tiny.yaml --out <tmp>` on CPU finishes within 10 minutes with 45 result rows, a verdict line in the fixed format, two plots, and a run record; the full configuration runs on Kaggle (quickstart section 4).

### Tests for User Story 1 (CPU smoke test per new module is mandatory, constitution VI; more only if requested)

Module tests live in the module tasks below (two files each). The tiny end-to-end run is a script step of `scripts/cpu_smoke.sh` (T045), checked by `scripts/check_tiny_run.py` (T038); it is not a pytest test, and every pytest test finishes within 60 s. Encoder and decoder tests: T054.

### Implementation for User Story 1

- [X] T023 [US1] Implement `src/strike_a_pose/data/manifest.py` (manifest CSV columns of contracts/artifacts.md with `repr` floats, shard npz keys, bit-packing of masks, atomic write) and `tests/test_manifest.py` (round trip; byte-identical CSV on repeat) (tier: haiku)
- [ ] T024 [US1] Implement `src/strike_a_pose/data/generate.py` (per shard: seeded betas, pose draws with filter and rejection count, rig aimed at the posed mesh's bounding-box center, four renders, canonical-pose measurements with `measure.step_cm`, flags; skip existing shards; `summary.json`; after the last shard, exit 4 without `DONE.json` when unflagged calibration or test bodies are fewer than `data.min_unflagged`, naming both counts; otherwise `DONE.json`) and `tests/test_generate.py` (two generations with the `small_config` overrides produce identical shard bytes; resume skips existing shards; rejection rate reported; a `data.min_unflagged` above the unflagged count exits 4 and names both counts) (tier: sonnet)
- [X] T025 [P] [US1] Implement `src/strike_a_pose/data/splits.py` (contiguous ranges, `split_of(body_id)`, disjointness check) and `tests/test_splits.py` (tier: haiku)
- [ ] T026 [US1] Implement `src/strike_a_pose/data/dataset.py` (torch Dataset over shards; view subset `0..k-1`; per-cell noise rotates about the stored per-(body, view) `noise_axis` by the cell's `noise_deg`; training mode draws k and the noise angle per sample from the seeded generator, about the stored axis; camera encoding vector; collate with view masks) and `tests/test_dataset.py` (shapes for k in 1..4; cell noise changes `R_given` only; flagged bodies excluded) (tier: sonnet)
- [X] T027 [US1] Implement `src/strike_a_pose/model/encoder.py` (strided CNN per `model.channels`, camera embedding, `mu_v`, `logvar_v`) and `src/strike_a_pose/model/fusion.py` (product of experts with the prior expert over a variable number of views with a view mask) (tier: sonnet)
- [ ] T028 [US1] Implement `src/strike_a_pose/model/decoder.py` (MLP to shape-coefficient mean and log-scale) and `src/strike_a_pose/model/vae.py` (joint plus sub-sampled single-view NLL, KL with warm-up, `sample_betas(views, K)`) (tier: sonnet)
- [ ] T029 [US1] Write `tests/test_fusion.py` (with one view the fused posterior equals that view's posterior, the product of the prior expert and the encoder expert, exactly; fused variance at most each view's; adding a view never increases variance; analytic two-expert check) and `tests/test_vae.py` (loss finite; backward works; shapes for k in 1..4 at 64 px) (tier: haiku)
- [ ] T030 [US1] Implement `src/strike_a_pose/train.py` (AdamW, deterministic flags, seeded sampler, checkpoint every `train.checkpoint_every` steps and per epoch with RNG states, resume, `history.csv`, time budget, `model_final.pt`, `DONE.json`; the last 5% of the train range (`n_monitor = ceil(0.05 n_train)` bodies) is a loss-monitoring slice: its `split` stays `train` in the manifest, its loss is logged as `monitor_loss` in `history.csv`, it takes no gradient step, and no selection reads it or the cal or test splits) and `tests/test_train.py` (two epochs on `small_config` data with `train.epochs=2`; resume from a checkpoint reaches the same final loss within 1e-6 on CPU; every body id yielded by the training sampler lies in `[0, n_train - n_monitor)`) (tier: sonnet)
- [ ] T031 [US1] Implement `src/strike_a_pose/predict.py` (per split and noise level: encode each (body, camera 0 to 3) once and reuse the per-view posteriors across the view-count cells, 4 encodings per body and noise level instead of 7 (1 + 2 + 4); fuse over a fixed four-slot layout in camera order 0 to 3 with zero precision for absent views, so a fewer-view fused precision never exceeds a more-view one in floating point; then per cell: K latent samples, decoded shape coefficients, measurements by `predict.measure_mode`: `exact` runs `measure_batch` on (bodies x K) batches of canonical-pose meshes, `linearized` uses a first-order expansion of the measurement map around each body's median sample (11 mesh evaluations per body and cell); median and scaled MAD, `latent_var_mean`, cached npz per cell with its `split` key, `DONE.json`) and `tests/test_predict.py` (one file per split and cell with the `small_config` overrides; with reused posteriors `latent_var_mean` never increases with the view count, exactly; `linearized` equals `exact` at the median sample and lies within 1% of it for shape coefficients one unit away on the stand-in body; spread floored; a non-finite prediction raises with the sample id) (tier: sonnet)
- [ ] T032 [US1] Implement `src/strike_a_pose/calibrate.py` (normalized scores, `ceil((n + 1)(1 - alpha))` quantile, floor, refusal below `calibrate.min_cal`, `quantiles.csv`, `DONE.json`) and `tests/test_calibrate.py` (quantile index on a known score list; refusal message names the minimum; a predict file whose `split` is not `cal` is refused; coverage on synthetic Gaussian data within 2 points of 90%) (tier: sonnet)
- [ ] T033 [US1] Implement `src/strike_a_pose/evaluate.py` (intervals from quantiles, clip at 0 with count, coverage, median width, MAE, mean signed error, inclusive band flag compared as integers `100 * covered >= 87 * n_test` and `100 * covered <= 93 * n_test`, `results.csv` in the fixed column order with `code_version` and `hardware_class` copied from the run record, `per_sample/` CSVs, the SC-004 check per test body and noise level, `latent_var_mean` non-increasing over the view counts of `evaluate.views` in ascending order (`v4 <= v2 <= v1` by default) within a relative tolerance of 1e-5 of the fewer-view value, written to `evaluate/sc004.json`, refusal of a predict file whose `split` is not `test`, `DONE.json`) and `tests/test_evaluate.py` (45 rows; boundary coverage is in band; clipped count; a fixture with one SC-004 violation is counted; a wrong split is refused) (tier: haiku)
- [ ] T034 [US1] Implement `src/strike_a_pose/verdict.py` (holds the FR-014 constants: threshold 0.70; chest, waist, hip, thigh; 0 degrees; views 1 and 4; band 0.87 to 0.93; exits 2 when the configuration differs from them; four ratios at 0 degrees from `results.csv`, `numpy.median` of the four, eight-cell band check; a compared cell outside the band is a legitimate KILL: `cells_in_band = false`, `out_of_band_cells` lists it, the verdict line ends with `; comparison invalid for cells <ids>`, and the command exits 0; a non-finite ratio or a missing compared cell gives KILL with `invalid_comparison = true`; `sc004_violations` is copied from `evaluate/sc004.json`; when `invalid_comparison` is true or `sc004_violations` is above 0, the command writes `verdict.json` and `verdict.md` first, then exits 4; the fixed verdict line; `recompute(out)` for `sap verify`) and `tests/test_verdict.py` (PASS and KILL fixtures; an out-of-band compared cell gives KILL with `cells_in_band=false` and the invalid-cells note, exit 0; a non-finite ratio or a missing compared cell gives KILL with `invalid_comparison`, written to `verdict.json` and `verdict.md` before exit 4; an SC-004 violation gives KILL and writes both files before exit 4; an overridden threshold is refused with exit 2; height and the 2 and 5 degree rows never change the verdict) (tier: sonnet)
- [X] T035 [P] [US1] Implement `src/strike_a_pose/report/tables.py` (`results.md` from `results.csv`, `verdict.md`, and the run record) and `tests/test_tables.py` (tier: haiku)
- [X] T036 [P] [US1] Implement `src/strike_a_pose/report/plots.py` (Agg backend; coverage and width versus view count, one subplot per measurement, one series per noise level, band shaded) and `tests/test_plots.py` (two PNG files written) (tier: haiku)
- [ ] T037 [US1] Wire `generate`, `train`, `predict`, `calibrate`, `evaluate`, `verdict`, `report`, `run` (stage order, `--resume`, `--time-budget`, exit 7; a KILL from `verdict` exits 0 and `run` continues to `report`; an exit 4 from `verdict` stops `run` after `verdict.json` and `verdict.md` exist) in `src/strike_a_pose/cli.py` and extend `tests/test_cli.py` (`sap run` with the `small_config` overrides writes every stage directory, `report/` included, within 60 s) (tier: haiku)
- [ ] T038 [US1] Create `scripts/check_tiny_run.py` (asserts on a tiny-run output directory: 45 data rows in `evaluate/results.csv`; the first line of `verdict/verdict.md` matches the verdict-line regex of contracts/cli.md; the `; comparison invalid for cells` suffix is present exactly when `cells_in_band=false`; `run_record.json` holds `config_hash`, `seed`, `code_version`, `hardware_class`; exits 1 naming the failed assertion) and `tests/test_check_tiny_run.py` (a fixture output directory built inside the test passes; a missing row fails; a malformed verdict line fails; under 60 s); the tiny run itself is a script step of `scripts/cpu_smoke.sh` (T045), not a pytest test (tier: haiku)
- [ ] T054 [US1] Write `tests/test_encoder.py` and `tests/test_decoder.py` (encoder output shapes per `model.channels`; the camera embedding changes `mu_v`; decoder mean and log-scale finite with the right shapes) (tier: haiku)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Evaluate the calibrated model on real public images (Priority: P2)

**Goal**: `sap real-eval` reports coverage, median width, mean signed error, subjects evaluated, and subjects skipped for BodyM (2 views) and SSP-3D (1 view), reported only (FR-017 to FR-020).

**Independent Test**: with synthetic dataset-shaped fixtures built inside the tests, `real-eval` writes `real/<dataset>/results.csv` and `subjects.csv` with the right columns and skip counts; with the real assets at the configured path, the operator runs quickstart section 5.

### Tests for User Story 2 (CPU smoke test per new module is mandatory, constitution VI; more only if requested)

Module tests live in the module tasks below (two files each).

### Implementation for User Story 2

- [ ] T039 [US2] Implement `src/strike_a_pose/real/common.py` (pad to square, area resize to `camera.image_size`, threshold; mask status in this order: no mask gives `skipped_no_mask`, then the multi-person rule (two connected components that each hold at least 10% of the mask area) gives `skipped_multi_person`, then the usability rule gives `skipped_unusable`, otherwise `ok`; nominal camera builder from `real.nominal_camera`) and `tests/test_real_common.py` (a two-person mask is `skipped_multi_person`, not `skipped_unusable`) (tier: haiku)
- [ ] T040 [US2] Implement `src/strike_a_pose/real/bodym.py` (discover split folders and CSV columns by header name, join subjects to photos, front and side masks, five tape measurements, log columns) and `tests/test_bodym.py` (synthetic BodyM-like directory built inside the test) (tier: haiku)
- [ ] T041 [US2] Implement `src/strike_a_pose/real/ssp3d.py` (load `labels.npz`, log keys, silhouettes, gendered SMPL ground truth in canonical pose through `smpl_body.py` and `measure.py`) and `tests/test_ssp3d.py` (synthetic labels fixture; the ground-truth path exercised with the stand-in body in place of SMPL) (tier: sonnet)
- [ ] T042 [US2] Implement `src/strike_a_pose/real/masks.py` (`MaskBackend` protocol; `ProvidedMasks`; `Sam2Masks` with lazy import, checkpoint from the asset root, box prompt; status per subject) and `tests/test_masks.py` (stub backend; the `sam2` test is skipped when the package is absent) (tier: haiku)
- [ ] T043 [US2] Implement `src/strike_a_pose/real/evaluate_real.py` (apply the trained model and the matching cell's quantiles; `results.csv` with `n_cal` of the matching synthetic cell and `code_version` and `hardware_class` from the run record, and `subjects.csv` per dataset; skipped counts; mean signed error) and `tests/test_evaluate_real.py` (tier: sonnet)
- [ ] T044 [US2] Add `sap real-eval` to `src/strike_a_pose/cli.py`; the real-image section of `src/strike_a_pose/report/tables.py` already exists (written in T035), so check it against the `real/<dataset>/results.csv` columns that T043 writes and fix any mismatch there (tier: haiku)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Validate a change on a CPU-only machine (Priority: P3)

**Goal**: one script runs lint, the full suite, and the tiny end-to-end run on a 4-core, 15 GB machine with no GPU, no network, and no licensed asset, within the budgets (FR-026, FR-027, SC-006).

**Independent Test**: `bash scripts/cpu_smoke.sh` exits 0 within 10 minutes per part on the reference machine.

### Tests for User Story 3 (CPU smoke test per new module is mandatory, constitution VI; more only if requested)

Covered by the module tests of Phases 2 to 4 plus the tasks below.

### Implementation for User Story 3

- [ ] T045 [US3] Create `scripts/cpu_smoke.sh` (`set -euo pipefail`; three parts, each under `timeout 600`: (1) ruff; (2) `pytest --durations=10`, where every test runs under the 60 s `timeout`; (3) the tiny end-to-end run as a script step: `tmp=$(mktemp -d)`, `timeout 600 sap run --config configs/tiny.yaml --out "$tmp/out" | tee "$tmp/stdout.txt"`, `python scripts/check_tiny_run.py "$tmp/out"`, `grep -E '^VERDICT: ' "$tmp/stdout.txt"`; fails when any part fails or exceeds 600 s; `--dry-run` prints the parts and exits 0) and `tests/test_no_network_no_assets.py` (asserts the network blocker and the unset asset root are active during the suite) (tier: haiku)
- [ ] T046 [US3] Add the opt-in `--seed-check` part to `scripts/cpu_smoke.sh`: only with the flag, after part 3 it runs `timeout 600 sap run --config configs/tiny.yaml --out "$tmp/out2" --seed-check "$tmp/out"`, which is itself the second run and compares against part 3's output; the default gate stays three parts; and write `tests/test_cpu_smoke_script.py` (`bash -n` syntax check; `--dry-run` lists three parts, and four with `--seed-check`; under 60 s) (tier: haiku)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - Reproduce a reported result (Priority: P3)

**Goal**: two runs with the same configuration and seed give identical manifests and metrics within tolerance; the verdict is recomputable from saved outputs (FR-024, FR-025, SC-003, SC-005).

**Independent Test**: `tests/test_determinism.py` passes; `sap verify --out <out>` exits 0 on an untouched output directory and 6 after a tampered `results.csv`.

### Tests for User Story 4 (CPU smoke test per new module is mandatory, constitution VI; more only if requested)

- [ ] T047 [US4] Write `tests/test_determinism.py` as three tests, each under 60 s, on `configs/tiny.yaml` with the `small_config` overrides of contracts/config.md, one `--set` per key: `data.n_train=64`, `data.n_cal=32`, `data.n_test=32`, `data.shard_size=32`, `data.min_unflagged=16`, `calibrate.min_cal=16`, `train.epochs=1`, `predict.n_samples=4`, `camera.image_size=32`, `camera.focal_px=32` (the focal length moves with the image size, so the field of view stays 53 degrees), `evaluate.views=[1,4]`, `evaluate.noise_deg=[0]` (two generations give identical shard bytes and manifest; two CPU trainings give metrics within the SC-005 tolerance; `verdict.recompute` equals the stored verdict); the full-tiny two-run comparison is the opt-in `scripts/cpu_smoke.sh --seed-check` (T046) (tier: haiku)

### Implementation for User Story 4

- [ ] T048 [US4] Add `sap verify --out` and `sap run --seed-check` to `src/strike_a_pose/cli.py` and the comparison helpers to `src/strike_a_pose/verdict.py` (exit 6 on any difference) (tier: haiku)

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T049 Create `notebooks/kaggle_run.ipynb` (version print, install from `/kaggle/input/sap-src` with constraints, environment variables including `CUBLAS_WORKSPACE_CONFIG=:4096:8`; a gate cell that runs only when no previous output is attached: `sap run --config configs/tiny.yaml --set device=auto --out /kaggle/working/tiny` on the GPU, then the extrapolation gate of quickstart section 4 step 0, stopping the notebook when the gate fails; resume copy from the previous output, `sap run --config configs/full.yaml --out /kaggle/working/out --resume --time-budget 8.5h`, print the verdict; committed with outputs cleared) and `scripts/build_kaggle_bundle.sh` (wheel, constraints, configs into `dist/kaggle-bundle/`) (tier: haiku)
- [ ] T050 [P] Add a one-line summary docstring and the public-source citation (research R1, R6, R7, R8) to every module under `src/strike_a_pose/` (docstring-only edits across all modules; constitution Principle I) (tier: haiku)
- [ ] T051 [P] Create `.github/pull_request_template.md` with the clean-room statement line and the CPU-suite checkbox (constitution Principle I and the workflow gate) (tier: haiku)
- [ ] T052 Run `bash scripts/cpu_smoke.sh`, fix the ruff findings it names (any files), copy the tiny-run `evaluate/results.csv` and `run_record.json` to `specs/001-kill-test-mvp/results/tiny/` as pipeline evidence (not a gate artifact), and record there the CPU upper bound of the extrapolation gate of quickstart section 4 step 0 (tiny `timings.predict` on CPU times the mesh-evaluation ratio times the face ratio); the gate is decided on the GPU by the gate cell of T049, and a CPU bound under 12 h already passes it (tier: haiku)
- [ ] T055 Move specs to DomI: run `bash /home/user/DomI/skills/speckit-pipeline/scripts/move_specs_to_domi.sh --apply` (tier: haiku)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - US1 first (MVP); US2 needs T030 (trained model) and T032 (quantiles) from US1; US3 and US4 need US1
- **Polish (Phase 7)**: Depends on all desired user stories being complete
- **T055**: depends on every other task; nothing runs after it, because it moves `.specify/` and `specs/` into DomI `specs/consumers/strike-a-pose/`

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
- Phase 3: T025, T035, T036 in parallel with their neighbours; T027 and T028 in parallel after T026; T029 and T054 together after T028
- Phase 4: T039, T040, T042 together; T041 after T020; T043 after T039 to T042
- Phase 7: T050 and T051 in parallel with T049

### Operator step outside this list

The full-size run (quickstart section 4) happens on Kaggle after T052 and is not an LLM task. Its committed outputs land in `specs/001-kill-test-mvp/results/` before T055, or in DomI's consumer copy of that directory after T055; the orchestrator chooses the order.

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
4. **STOP and VALIDATE**: run `sap run --config configs/tiny.yaml --out <tmp>/out`, then `python scripts/check_tiny_run.py <tmp>/out`
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

- Total: 54 tasks (T053 is unused after analyze cycle 1: the move task became T055 so it stays last). Sonnet: 15 (T011, T013, T014, T015, T016, T024, T026, T027, T028, T030, T031, T032, T034, T041, T043). Haiku: 39.
- Per story: US1 17 (T023 to T038, T054), US2 6 (T039 to T044), US3 2 (T045, T046), US4 2 (T047, T048). Setup 4, Foundational 18, Polish 5 (T049 to T052, T055).

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group, with the clean-room statement in every PR description
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
