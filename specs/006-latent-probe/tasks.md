# Tasks: Latent Probe (slice 006)

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[contracts/probe-run.md](contracts/probe-run.md)
**Tests**: required (constitution IV).

## Phase 0: Surrogate rulings (roadmap / CI / 004 additive)

- [x] T001 Roadmap: "Surrogate rulings 2026-10-20" changelog (all provisional), Q2 struck + status line, 006 row, deferred open questions 12–16.
- [x] T002 CI: pin `ubuntu-latest` → `ubuntu-24.04` (separate commit).
- [x] T003 [P] `tests/test_train.py`: with a val split, `best.pt` holds the epoch of the minimum val loss (loads via `load_checkpoint`); interrupted + resumed run gives the same `best.pt` epoch / weights and `metrics.json` as uninterrupted; empty val → no `best.pt`.
- [x] T004 `train/trainer.py`: additive `best.pt` + `best_val_epoch` / `best_val_loss` in `metrics.json.training` and `run_info.json`; 004 contract note. `configs/ae.yaml` unchanged; selection stays off.

## Phase 1: Probe core (US1, US2)

- [x] T005 [P] `tests/test_probe_config.py`: defaults; `configs/probe*.yaml` load; errors (unknown / missing key, bad types, empty / duplicate / unknown targets, non-positive λ, negative weight decay, `val` missing from eval_splits, figure split not evaluated, bad run_name).
- [x] T006 `src/strike_a_pose/eval/probe_config.py`, `configs/probe.yaml`, `configs/probe_fixture.yaml`.
- [x] T007 [P] `tests/test_probe.py` units: ridge equals the normal-equation / lstsq solution and recovers a noiseless linear map; `rotation_angle_deg` hand cases (identity 0°, 90° about an axis, π flips); `yaw_scores` confusion / offsets on a hand case; MLP fits a small separable problem; empty-val selection fallback.
- [x] T008 `src/strike_a_pose/eval/probe.py`: extraction, cache, ridge, MLP, metrics, baselines, selection, figures, `run_probe`.
- [x] T009 [P] `tests/test_probe.py` end-to-end: fixture AE (train CLI, temp dir) → probe CLI (< 60 s): exit 0, run dir files, resolved config (checkpoint override), run_info checkpoint fields, `metrics.json` keys per target / split (val `samples: 0`, `selected_by` default), figure sizes; second run identical `metrics.json`; cache hit gives identical metrics; exit 2 for missing checkpoint / bad config, exit 1 for split mismatch.
- [x] T010 `eval/cli.py` `probe` subcommand + headline printout.

## Phase 2: Verify + CI

- [x] T011 Local: `uv sync --locked`, ruff check, ruff format --check, pytest; fixture CLI timing. → at ae32db9: ruff clean, pytest 230 passed (24 s); fixture probe CLI 4.4 s (+ fixture AE train ≈ 4 s). Diff review: `train/trainer.py` change is additive only (ruling #3); `datagen`, `data`, `models` untouched.
- [x] T012 CI: fixture probe smoke step (after the AE smoke train) if the leg stays < 5 min; push; green on both legs. → run 39378504271 (ae32db9, runner `ubuntu-24.04`) green: py3.11 leg 59 s (tests 30 s, probe smoke 4 s), py3.13 leg 61 s (tests 31 s, probe smoke 3 s).

## Phase 3: US3 — full run + record

- [x] T013 Full run `strike-a-pose-eval probe --config configs/probe.yaml` on a clean tree (≤ 30 min cap; foreground with `timeout`, nothing needs heartbeats); check world-frame mean pose / PCA equal 003; inspect figures.
  → Run `runs/probe-smpl_proc_v1-20261020T060632Z` at ae32db9 (clean), `timeout 1800`, one process, no restart: wall 450 s (7.5 min; loading + encoding both encoders 80 s, cache `runs/probe-cache/572dd7f2c9bb-9000c915c0b9-4b539833405f.pt` written; probes 28–80 s per encoder × target). Checkpoint: `runs/ae-smpl_proc_v1-20261020T030038Z/checkpoint.pt`, epoch 60 (final weights), its run at f9619c6 clean. No pilot (test was scored once, in this run).
  - Consistency: world-frame `mean_pose`, `pca_k8`, `pca_k16` (MPJPE and 2D px, val and test) equal `docs/results/003-baselines-smpl_proc_v1/metrics.json` exactly.
  - Selection (val): ridge λ 0.01 (joints_world), 1e-4 (joints_cam), 0.1 (yaw, smpl_pose) — the val curve is flat for λ ≤ 0.1; MLP weight decay 1e-3 (largest grid value) for joints_world, yaw, smpl_pose, 1e-4 for joints_cam. Grid edge noted in README; not widened (test already seen).
  - Results (val / test): joints_world MPJPE — MLP 172.0 / 172.8, ridge 214.5 / 212.5, random-encoder MLP 214.6 / 214.5, random ridge 228.4 / 226.7, mean pose 253.4 / 250.3, PCA k=8 94.0 / 93.6, k=16 34.2 / 34.6 (oracle). joints_cam — MLP 161.1 / 159.6, ridge 234.3 / 231.8, random MLP 235.4 / 234.5, mean 313.1 / 308.6, PCA k=8 143.3 / 139.8, k=16 44.1 / 45.2. Yaw accuracy — MLP 0.965 / 0.959, ridge 0.826 / 0.819, random MLP 0.685 / 0.670, random ridge 0.602 / 0.599, majority 0.125. SMPL pose angle — MLP 23.4° / 24.1°, ridge 24.9° / 25.5°, mean 26.9° / 27.5°.
  - Yaw confusions (MLP, val): 56 errors of 1600 — 22 at 180° (`Front_0`→`Back_180` 9, `Back_180`→`Front_0` 12), 17 at 45°, 11 at 90°, 6 at 135°. Ridge errors are mostly 45° neighbours (142 of 279).
  - Figures (lead's visual read): `skeleton_val.png` — torso orientation and outstretched arms predicted well; bent / raised legs and bent arms regress toward the mean pose (rows 2–3: the bent leg is predicted straight); ridge is visibly worse on legs. `yaw_confusion_val.png` — strong diagonal for the MLP, front/back off-diagonal pair the main error.
- [x] T014 Copy `metrics.json`, `run_info.json`, `config.yaml`, figures to `docs/results/006-probe-smpl_proc_v1/`; README (status, usage, results), roadmap changelog, this file's verification record. → copied byte-for-byte (148 KB total); cache and checkpoint not committed. Final CI: see the roadmap changelog entry.

## Dependencies

T001, T002 first. T003 → T004. T005 → T006. T006, T007 → T008 → T009 → T010 → T011
→ T012 → T013 → T014.

## Analyze pass (lead, 2026-10-20)

Coverage: FR-001 → T007–T009 (cache hit test) · FR-002 → T008, T009 · FR-003 → T007,
T008 · FR-004 → T007, T008 · FR-005 → T008 (selection on val, test scored after),
T013 · FR-006 → T008, T009, T013 (003 equality check) · FR-007 → T007, T009 ·
FR-008 → T009, T010 · FR-009 → T009 · FR-010 → T008, T009 · FR-011 → T006, T009,
T012 · FR-012 → T003, T004 (additive only; review of the diff in T011). SC-001 → T009,
T012 · SC-002 → T013 · SC-003 → T013, T014 · SC-004 → T011, T012. Edge cases: empty
val (T007, T009), image-size mismatch (T008 check, exit 2), cache mismatch (T009),
ties (T008, documented).

Findings and resolutions:
1. Inconsistency: spec clarification 9 named one cache file per encoder, the
   contract one file for all encoders → contract kept (one loader pass feeds both
   encoders); spec amended.
2. Underspecified: whether `eval_splits` may omit `val` → no: selection needs val;
   contract requires `val` in `eval_splits` (config error otherwise).
3. Ambiguity: "test scored once" with a cache hit on a second run → the cache holds
   latents and labels only, never scores; each run scores test once after its own
   val selection. The full run is made once on a clean tree.
4. Constitution V wording risk: PCA baselines are not image-based; every table that
   shows them labels them as an oracle reference (spec SC-003, README).
5. Coverage gap check for ruling #3: resume must reproduce `best.pt` → T003 tests it;
   no new checkpoint field is needed because the best epoch follows from the
   checkpointed history.
