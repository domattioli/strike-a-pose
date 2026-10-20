# Tasks: Baselines and Metrics (slice 003)

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[contracts/eval-run.md](contracts/eval-run.md)
**Tests**: required (constitution IV, spec FR-009/SC-001).

## Phase 0: Full dataset (001 T025)

- [x] T001 Generate `smpl_proc_v1` (default config, seed 0), time it, `validate`; record in 001 tasks.md T025.
- [x] T002 Freeze `splits/smpl_proc_v1.json` (seed 0, default fractions); README note: split tied to manifest hash (config + seed + code commit).

## Phase 1: Setup

- [ ] T003 `.gitignore` `/runs/`; `pyproject.toml` console script `strike-a-pose-eval`; `uv lock --check`.

## Phase 2: Foundational — run log (US2)

- [ ] T004 [P] `tests/test_runlog.py`: `seed_everything` makes `random`/NumPy/torch draws repeat; `environment_summary` keys; `make_run_dir` name format + refuses existing; `write_json_exact` sig-digit rounding, NaN → null, sorted keys, trailing newline.
- [ ] T005 `src/strike_a_pose/runlog.py` per plan.

## Phase 3: US1 — metrics (P1)

- [ ] T006 [P] `tests/test_eval_metrics.py`: hand-computed cases for `image_mse`, `psnr` (incl. cap at MSE 0), `silhouette_iou` (incl. empty union, threshold edge), `mpjpe` (3-4-5 triangle, translation invariance, root choice), `keypoint_error_2d` (visibility mask, NaN when none visible), `project_points` (vs fixture `keypoints_2d` labels, < 1 px); fixture true images thresholded at the default 0.075 give IoU = 1 against their masks (R3); every case on NumPy and torch, output type follows input, shapes `[B]`.
- [ ] T007 `src/strike_a_pose/eval/metrics.py` + `eval/__init__.py`.

## Phase 4: US2 — PCA, config, runner, CLI (P1)

- [ ] T008 [P] `tests/test_eval_pca.py`: synthetic rank-r data (n < d and n > d): k ≥ r reconstructs exactly, k < r error equals the dropped variance, components orthonormal, `k_effective` clamps to rank, `n = 1` → mean only, explained-variance ratio monotone and 1 at full rank; Gram path ≈ covariance path.
- [ ] T009 `src/strike_a_pose/eval/pca.py`.
- [ ] T010 [P] `tests/test_eval_config.py`: defaults; `configs/baselines.yaml` and `configs/baselines_fixture.yaml` load; errors for unknown key, missing required, empty / non-positive / non-int k, bad split names, duplicate splits, figure split not evaluated, threshold out of range, bad run_name.
- [ ] T011 `src/strike_a_pose/eval/config.py`, `configs/baselines.yaml`, `configs/baselines_fixture.yaml`.
- [ ] T012 [P] `tests/test_eval_cli.py` (fixture end-to-end, temp run root): exit 0; run dir files; `config.yaml` = resolved (override `--seed` recorded); `run_info.json` has git, env, seed, manifest + split hashes, timings; `metrics.json` keys per contract (val `samples: 0`; test image + pose baselines; `k_effective` ≤ rank; pose PCA on 1 train pose → rank 0); figure PNG size = header + rows × columns tiles; second run → identical `metrics.json` bytes; split-file mismatch → exit 1 and no run dir; bad config → exit 2; SC-002 timing < 30 s.
- [ ] T013 `src/strike_a_pose/eval/baselines.py`, `src/strike_a_pose/eval/cli.py`.

## Phase 5: Verify + run + record

- [ ] T014 Local: `uv sync --locked`, `uv lock --check`, ruff check, ruff format --check, pytest; fixture CLI smoke.
- [ ] T015 CI: add fixture baseline smoke step to `.github/workflows/ci.yml`; push; CI green on both legs; record per-leg time vs SC-004.
- [ ] T016 [US3] Full run: `strike-a-pose-eval baseline --config configs/baselines.yaml` on a clean tree; record wall time (SC-003), val/test numbers, run dir; copy `metrics.json` + `run_info.json` to `docs/results/003-baselines-smpl_proc_v1/`; figure checked visually.
- [ ] T017 README (status row, usage), roadmap changelog, this file's verification record.

## Dependencies

T003 → all code. T004 → T005. T006 → T007. T008 → T009. T010 → T011.
T005, T007, T009, T011 → T012 → T013. T013 → T014 → T015, T016 → T017.

## Analyze pass (lead, 2026-10-20)

Coverage: FR-001 → T006, T007 · FR-002–FR-004 → T006 · FR-005 → T008, T009,
T012, T013 · FR-006 → T010–T013 · FR-007 → T004, T005, T012 · FR-008 → T004, T012
· FR-009 → T012, T015 · FR-010 → T001, T002. SC-001 → T006 · SC-002 → T012 ·
SC-003 → T016 · SC-004 → T014, T015. Every spec edge case maps to a T006, T008,
T010 or T012 test (rank clamp, PSNR cap, empty union, no visible joints, empty
split, config errors).

Findings and resolutions:
1. Inconsistency: spec edge case said "`test` used for fitting → config error"
   while the plan hard-codes train as the fit split → spec edge case and FR-005
   reworded (fit split not configurable; nothing to reject).
2. Misplaced check: the "true image thresholded → IoU 1" sanity was in the CLI
   test (T012) but is a metric/threshold property → moved to T006 (fixture).
3. Underspecified: `metrics.json` float format. 001 `write_json` rounds to 6
   decimals (3–4 significant digits for small MSEs) → contract + research R4 fix
   6 significant digits, NaN → `null`, own writer in `runlog`.
4. Constitution II: CLI overrides (`--seed`, `--run-root`) could hide run
   inputs → they are written into the resolved `config.yaml` (contract, T012).
5. Constitution III: baselines are deterministic, but seeds are still set and
   logged (FR-008); `metrics.json` byte-identity across runs is tested (T012), and
   `run_info.json` carries git commit + dirty flag → T016 runs on a clean tree.
6. Constitution IV: no training here; the eval CLI gets a CPU smoke step in CI on
   the fixture (T015), separate from 004's smoke-train placeholder.
7. Constitution V: train-only fitting is structural (no config key); figure uses
   val by default; numbers in docs cite the run dir + commit and a committed copy
   of `metrics.json` (T016/T017). Split tied to the manifest hash, which includes
   the code commit → documented (README, 001 T025) and raised as an operator open
   question (whether the split identity should exclude `code.commit`) — not
   changed here (002 contract).
8. Constitution VI: no new dependency (PIL figure, torch linear algebra); flat
   config; no baseline registry.
9. Cross-slice touches (001 tasks.md T025, README, CI step in 005's workflow) are
   additive and recorded in the roadmap changelog.
No constitution violations; Complexity Tracking empty.
