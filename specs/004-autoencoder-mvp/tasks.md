# Tasks: Autoencoder MVP (slice 004)

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[contracts/train-run.md](contracts/train-run.md)
**Tests**: required (constitution IV, spec FR-008).

## Phase 0: Step 0 — 003 figure check

- [x] T001 Investigate the 003 recon figure (true vs mean/PCA extent): tiles vs arrays, bounding boxes, cross-correlation shift over full val → research R0 (no bug).
- [x] T002 Re-test the PyTorch CPU index (research R1: still 403 → unchanged).

## Phase 1: Setup

- [ ] T003 `pyproject.toml` console script `strike-a-pose-train`; `uv lock --check`.

## Phase 2: Foundational — shared report helpers

- [ ] T004 [P] `tests/test_eval_report.py`: grid tiles at documented offsets equal column inputs (non-square H×W, several rows/cols, header labels region); `figure_indices` spacing; `score_images` equals per-batch metric means.
- [ ] T005 `src/strike_a_pose/eval/report.py`; `baselines.py` uses it (003 tests incl. byte-identical fixture `metrics.json` still pass).

## Phase 3: US1 — model, config, training (P1)

- [ ] T006 [P] `tests/test_model.py`: encode → `[B, 64]`, decode/forward → `[B, 1, 256, 192]` in [0, 1]; custom latent/channels; non-divisible size → `ValueError`; parameter count of the default model; checkpoint round trip.
- [ ] T007 `src/strike_a_pose/models/{__init__,autoencoder}.py`.
- [ ] T008 [P] `tests/test_train_config.py`: defaults; `configs/ae.yaml` + `configs/ae_fixture.yaml` load; errors: unknown/missing key, bad types, non-positive ints/lr, empty/duplicate channels, bad schedule, `train` or duplicate in eval_splits, figure split not evaluated, threshold range, bad run_name, `context_pca_k` null allowed.
- [ ] T009 `src/strike_a_pose/train/config.py`, `configs/ae.yaml`, `configs/ae_fixture.yaml`.
- [ ] T010 [P] `tests/test_train.py`: overfit on a fixed batch of 4 fixture images (15 steps, final < ½ initial); fixture CLI end-to-end (temp run root, < 60 s): exit 0, run dir files, resolved config with `--seed` override, run_info fields, `history.csv` rows = epochs with empty val and last train loss < first, `metrics.json` keys (val `samples: 0`; test `ae`/`mean_image`/`pca_k4`), figure size for 4 columns, checkpoint loads; second run → identical `history.csv` losses and `metrics.json` bytes; split mismatch → exit 1, no run dir; bad config → exit 2; non-divisible channels → exit 2.
- [ ] T011 `src/strike_a_pose/train/{__init__,trainer,cli}.py`.

## Phase 4: Verify + CI

- [ ] T012 Local: `uv sync --locked`, `uv lock --check`, ruff check, ruff format --check, pytest; fixture CLI smoke timing.
- [ ] T013 CI: replace the 005 placeholder with the fixture smoke-train step; push; green on both legs; record per-leg time.

## Phase 5: US3 — full run + record

- [ ] T014 Pilot (val only, 1 epoch, scratch config) to check loading, memory, epoch time and that loss falls.
- [ ] T015 Full run `strike-a-pose-train --config configs/ae.yaml` on a clean tree (background); record wall time, loss curve, val/test metrics; check context baselines equal 003; inspect figure (SC-003).
- [ ] T016 Copy `metrics.json`, `run_info.json`, `config.yaml`, `history.csv`, figure (< 500 KB) to `docs/results/004-ae-smpl_proc_v1/`; README (status, usage, results), roadmap changelog (MVP status), this file's verification record.

## Dependencies

T003 → code. T004 → T005. T006 → T007. T008 → T009. T005, T007, T009 → T010 → T011
→ T012 → T013, T014 → T015 → T016.

## Analyze pass (lead, 2026-10-20)

Coverage: FR-001 → T006, T007 · FR-002 → T008–T011 · FR-003 → T010, T011 (train-only
fit; val loss logged only) · FR-004 → T010 · FR-005 → T010 (seed in run_info,
repeatability) · FR-006 → T004, T005, T010, T015 · FR-007 → T004, T010 · FR-008 →
T010, T013 · FR-009 → T016 (only summaries copied; `runs/` ignored). SC-001 → T010,
T013 · SC-002 → T015 · SC-003 → T015 · SC-004 → T012, T013. Edge cases: non-divisible
size (T006, T010), partial last batch (T010 fixture: 8 images / batch 4 exact → also
covered by overfit/`train_epoch` with batch 3 in T010), rank clamp (T010
`pca_k4` k_effective), empty val (T010), existing run dir (exit 4 shares the 003
path; covered by `make_run_dir` test in 003).

Findings and resolutions:
1. Underspecified: whether the per-epoch val loss requires `val` in `eval_splits`
   → contract: computed whenever val has samples (monitor), independent of the
   final scoring list.
2. Coverage gap: partial last batch was only implicit (fixture 8 / batch 4 divides
   evenly) → T010 adds a `train_epoch` call with batch 3 (8 = 3 + 3 + 2).
3. Inconsistency: spec SC-001 said "< 60 s" while constitution IV says "~2 min" →
   both hold (60 s is the tighter slice target); kept.
4. Constitution III: training determinism is machine/thread dependent → stated in
   spec Assumptions + research R4; `run_info.json` logs `torch_threads`; repeatability
   tested on one machine only (T010).
5. Constitution V: test usage — the CLI scores `eval_splits` (default val+test) at
   the end of every run; the pilot (T014) uses a val-only config so test is touched
   once, by T015; final weights only (no selection) → no tuning surface on val/test
   beyond the a-priori hyper-parameters.
6. Constitution V: AE vs baseline comparability — baselines are refit in the same
   run with the 003 code (R6), and T015 checks equality with the 003 numbers.
7. Constitution VI: `lr_schedule` adds one knob; kept (two values, one line,
   standard for fixed-epoch runs). Resume not implemented (optional in brief).
8. Cross-slice touches: 003 `baselines.py` refactor (T005, guarded by 003 tests),
   005's `ci.yml` placeholder replaced (T013) — recorded in the roadmap changelog.
No constitution violations; Complexity Tracking empty.
