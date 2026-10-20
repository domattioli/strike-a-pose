# Tasks: Autoencoder MVP (slice 004)

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[contracts/train-run.md](contracts/train-run.md)
**Tests**: required (constitution IV, spec FR-008).

## Phase 0: Step 0 — 003 figure check

- [x] T001 Investigate the 003 recon figure (true vs mean/PCA extent): tiles vs arrays, bounding boxes, cross-correlation shift over full val → research R0 (no bug).
- [x] T002 Re-test the PyTorch CPU index (research R1: still 403 → unchanged).

## Phase 1: Setup

- [x] T003 `pyproject.toml` console script `strike-a-pose-train`; `uv lock --check`.

## Phase 2: Foundational — shared report helpers

- [x] T004 [P] `tests/test_eval_report.py`: grid tiles at documented offsets equal column inputs (non-square H×W, several rows/cols, header labels region); `figure_indices` spacing; `score_images` equals per-batch metric means.
- [x] T005 `src/strike_a_pose/eval/report.py`; `baselines.py` uses it (003 tests incl. byte-identical fixture `metrics.json` still pass).

## Phase 3: US1 — model, config, training (P1)

- [x] T006 [P] `tests/test_model.py`: encode → `[B, 64]`, decode/forward → `[B, 1, 256, 192]` in [0, 1]; custom latent/channels; non-divisible size → `ValueError`; parameter count of the default model; checkpoint round trip.
- [x] T007 `src/strike_a_pose/models/{__init__,autoencoder}.py`.
- [x] T008 [P] `tests/test_train_config.py`: defaults; `configs/ae.yaml` + `configs/ae_fixture.yaml` load; errors: unknown/missing key, bad types, non-positive ints/lr, empty/duplicate channels, bad schedule, `train` or duplicate in eval_splits, figure split not evaluated, threshold range, bad run_name, `context_pca_k` null allowed.
- [x] T009 `src/strike_a_pose/train/config.py`, `configs/ae.yaml`, `configs/ae_fixture.yaml`.
- [x] T010 [P] `tests/test_train.py`: overfit on a fixed batch of 4 fixture images (15 steps, final < ½ initial); fixture CLI end-to-end (temp run root, < 60 s): exit 0, run dir files, resolved config with `--seed` override, run_info fields, `history.csv` rows = epochs with empty val and last train loss < first, `metrics.json` keys (val `samples: 0`; test `ae`/`mean_image`/`pca_k4`), figure size for 4 columns, checkpoint loads; second run → identical `history.csv` losses and `metrics.json` bytes; split mismatch → exit 1, no run dir; bad config → exit 2; non-divisible channels → exit 2.
- [x] T011 `src/strike_a_pose/train/{__init__,trainer,cli}.py`.

## Phase 4: Verify + CI

- [x] T012 Local: `uv sync --locked`, `uv lock --check`, ruff check, ruff format --check, pytest; fixture CLI smoke timing.
- [x] T013 CI: replace the 005 placeholder with the fixture smoke-train step; push; green on both legs; record per-leg time. → run 39361032154 (de14d25) green: py3.11 leg 41 s (tests 18 s, AE smoke train 4 s), py3.13 leg 48 s (tests 17 s, AE smoke train 3 s).

## Phase 4b: Restart resilience (clarification 2026-10-20b)

- [x] T017 `tests/test_train.py`: interrupted after epoch 4 (+ stray history row past the checkpoint) then `--resume` == uninterrupted run (history, weights, `metrics.json` bytes; bit-exact on CPU); resume errors (finished → 2, no checkpoint → 2, `--seed`/`--config` with `--resume` → usage error 2).
- [x] T018 `trainer.resume_training`, full-state checkpoint (scheduler, RNG states, history, wall times), `run_info.sessions`, CLI `--resume RUN_DIR`; spec FR-010 + contract.

## Phase 5: US3 — full run + record

- [x] T014 Pilot (val only, 1 epoch, scratch config) to check loading, memory, epoch time and that loss falls. → Replaced by watching the first epochs of T015 (no test-split access before the end, so constitution V holds): load 46 s, RSS ≈ 2.3 GB while training, epochs 85 s / 81 s, train loss 0.0352 → 0.0071, val 0.0100 → 0.0055. Plus a real-data restart drill: SIGKILL after epoch 2 (03:04 UTC), `--resume` at 03:04:42 → "resuming … after epoch 2/60", reload 44 s, continued at epoch 3 (session 2 in `run_info.json`; its `git.dirty` = true only because README.md had uncommitted usage text, no code change). Note: a second, concurrent `--resume` started by mistake for debugging was killed by `timeout` during loading (before any write except a stray `sessions` entry; that entry was *not* overwritten after all — see T015) → there is no lock against two resumes of one run dir (analyze finding 9).
  Run state / successor instructions (if this session is killed too):
  - Launched 2026-10-20T03:00:38Z at commit f9619c6 (clean tree), from the repo root:
    `setsid nohup .venv/bin/strike-a-pose-train --config configs/ae.yaml > runs/ae-full.log 2>&1 < /dev/null &`
  - Run dir: `runs/ae-smpl_proc_v1-20261020T030038Z`; log `runs/ae-full.log` (resumed sessions log to `runs/ae-full-resume*.log`).
  - Alive? `ps -eo pid,etime,args | grep "[s]trike-a-pose-train"` (do not `pkill -f` with a pattern that also matches your own shell's command line). Progress: `tail -3 runs/ae-full.log; wc -l runs/ae-smpl_proc_v1-20261020T030038Z/history.csv`.
  - Killed (no process, no `metrics.json` in the run dir) → resume, do not start a new run:
    `setsid nohup .venv/bin/strike-a-pose-train --resume runs/ae-smpl_proc_v1-20261020T030038Z > runs/ae-full-resume.log 2>&1 < /dev/null &`
  - Do not `uv sync`/swap the venv while it runs.
- [x] T015 Full run `strike-a-pose-train --config configs/ae.yaml` on a clean tree (background); record wall time, loss curve, val/test metrics; check context baselines equal 003; inspect figure (SC-003).
  → Run `runs/ae-smpl_proc_v1-20261020T030038Z`, launched at f9619c6 (clean), finished 2026-10-20T05:45:49Z; 60 epochs, 12,000 steps, 3,126,897 parameters, torch 2.14.1+cu130 on CPU, 4 threads.
  - Sessions (`run_info.json`): 5 entries. (1) start, epoch 1, f9619c6 clean; (2) 03:04:42 resume at epoch 3 after the SIGKILL drill, 570f1df dirty (README usage text only, see T014); (3) 03:05:01, the mistaken concurrent resume killed while loading — it trained nothing, but its `sessions` entry *persisted* (correction to T014: the live session 2 never reached its final `run_info.json` write, because it was itself killed by a container restart after epoch 9, and later resumes read the file from disk); (4) 03:16:30 resume at epoch 10, 693b385 clean, killed by a container restart after epoch 10; (5) 04:38:19 resume at epoch 11, 693b385 clean, ran to the end. Code (`src/`) is identical across f9619c6, 570f1df and 693b385 (later commits touched docs/specs only), so the run is one code version.
  - Wall time: process time counted by the run `wall_s` 5057 s (84.3 min) = training `train_wall_s` 4679 s (sum of `epoch_s` 4674 s, 71–106 s/epoch, mean 78 s) + 4 data loads 196 s (46/44/58/48 s) + final eval/figure ≈ 182 s. Calendar 03:00:38 → 05:45:49 = 2 h 45 min, of which ≈ 80 min with no process alive (container reclaimed after epoch 10, resumed 04:38). Inside the ~3 h budget (SC-002); partial epochs lost at kills are not counted in `wall_s`.
  - Loss (per-pixel MSE; train = running mean over the epoch in train mode, val = eval mode after the epoch): train 0.03516 (ep 1) → 0.00106 (ep 60, its minimum); val 0.01003 (ep 1) → min 0.002168 (ep 34) → 0.002208 (ep 60, +1.8 % over the min). Val flattens at ≈ 0.00218–0.00221 from about epoch 30 while train keeps falling (0.00135 at ep 33 → 0.00106): mild overfitting (12,800 train images, 3.1 M parameters, no augmentation). Final weights are evaluated (Q5, no best-val selection), so this costs ≈ 2 % val MSE; not acted on (val is never used to change the run). Figure: `docs/results/004-ae-smpl_proc_v1/loss_curve.png`.
  - Metrics (val / test, 1,600 images each): AE MSE 0.00221 / 0.00219, PSNR 27.15 / 27.13 dB, IoU 0.879 / 0.881; mean image 0.01403 / 0.01385 (IoU 0.416 / 0.420); PCA k=64 0.00459 / 0.00452 (IoU 0.644 / 0.646). AE MSE ≈ 0.48× PCA k=64 (and below PCA k=128's 0.00327 / 0.00321 from 003).
  - Context baselines equal 003 exactly: `mean_image` and `pca_k64` entries (MSE, PSNR, IoU, val and test) are identical to `docs/results/003-baselines-smpl_proc_v1/metrics.json`; dataset manifest and split hashes match.
  - SC-003 figure check (`recon_val.png`, 6 evenly spaced val samples, lead's visual read): all six AE reconstructions are recognizable as the true pose — torso orientation, arm angles (incl. the bent arm in row 3), raised knee / bent leg (rows 1–3) and stance width are right, with no ghosting (unlike PCA k=64's arm/leg streaks). Losses: everything is blurred and shading/surface detail is gone; faces are featureless; hands and fingers are reduced to tapered stumps or vanish; feet are soft blobs; thin parts (wrists, ankles, lower legs) are thinned. Where limbs overlap or cross (row 1 foot near the hand, row 3 crossed lower leg) they merge into one smeared blob. Verdict: meets the MVP "recognizable" bar; fine extremities are the clear weakness (consistent with the predecessor's hand problems).
- [x] T016 Copy `metrics.json`, `run_info.json`, `config.yaml`, `history.csv`, figure (< 500 KB) to `docs/results/004-ae-smpl_proc_v1/`; README (status, usage, results), roadmap changelog (MVP status), this file's verification record.
  → Copied byte-for-byte (figure 171 KB, no downscale) + `loss_curve.png` rendered from the committed `history.csv` by `plot_loss_curve.py` (PIL only; no new dependency). Checkpoint (38 MB) and data not committed. README status (004 implemented, MVP done), results table and train/resume usage; roadmap changelog. PyTorch CPU index re-tested 2026-10-20 (end of slice): still 403 at CONNECT → unchanged. Local checks and CI: see the commit that ticks this task / final report.

## Dependencies

T003 → code. T004 → T005. T006 → T007. T008 → T009. T005, T007, T009 → T010 → T011
→ T012 → T013, T017 → T018 → T014 → T015 → T016.

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
9. (2026-10-20b) Resume has no lock: two concurrent `--resume` of one run dir would
   both write `history.csv`/`checkpoint.pt`. Not guarded (single operator, one
   machine; a pid lock goes stale after a container restart) — documented in the
   successor instructions under T015 instead.
No constitution violations; Complexity Tracking empty.
