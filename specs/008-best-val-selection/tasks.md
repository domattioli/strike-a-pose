# Tasks: Best-Val Weight Selection (slice 008)

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md); normative contracts
[`../004-autoencoder-mvp/contracts/train-run.md`](../004-autoencoder-mvp/contracts/train-run.md)
(Train v2) and [`../006-latent-probe/contracts/probe-run.md`](../006-latent-probe/contracts/probe-run.md)
(`checkpoint` row).
**Tests**: required (constitution IV; spec FR-009), written before the code.

## Phase 0: Baseline

- [x] T001 SC-002 baseline before any code change (at `a5bb162`, clean tree): the three
  fixture smoke commands (README "Development checks": `baseline`, `train`, `probe` on
  the train run's `checkpoint.pt`) with `--run-root <scratchpad>/runs-008-before`; keep
  their `config.yaml` / `metrics.json` / `history.csv`. → done; pytest at `a5bb162`: 249
  passed in 28.6 s.

## Phase 1: Tests first (FR-009)

- [ ] T002 [P] `tests/test_train_config.py`: (a) default `select_weights` = `best_val`;
  `"best"`, `""`, `True`, `None` → `ConfigError`; the shipped AE configs list
  `best_val`; FR-007: a config file without the key → `load_config` gives `best_val`,
  `load_run_config` gives `final`; a present key wins in `load_run_config`.
- [ ] T003 [P] `tests/test_train_selection.py` (new; small model, split with a val pose,
  scripted val loss installed over `trainer.eval_loss`): (b) `best_val` /
  `weight_selection` on synthetic histories (tie → earliest, minimum at the last epoch,
  NaN at epoch 1 and in the middle, empty val, all NaN, `final`); (c) minimum at epoch 2
  of 4 → `selection`, `best.pt` epoch 2, `checkpoint.pt` epoch 4, val `ae` metrics and the
  figure's `ae` column equal `best.pt` rescored and differ from `checkpoint.pt` rescored;
  NaN at epoch 1 → `best.pt` epoch 2; (d) minimum at the last epoch → epoch 4,
  `best.pt`, `splits` and figure bytes equal to a `final` run, histories (without
  `epoch_s`) and final weights equal (US1-2); then, on a copy, `best.pt`, `metrics.json`
  and the figure removed → `--resume` rewrites `best.pt` from the checkpoint and
  reproduces `metrics.json` byte for byte; (e) committed fixture (empty val) → fallback
  record, no `best.pt`, `splits` equal to a `final` run; (f) `final` with val → final
  weights scored, `best.pt` (epoch 2) still written; no finite val loss → fallback
  record, no `best.pt`; (g) kill points (minima at epochs 1, 2, 4 of 6): stop after epoch
  5 + resume; stop after 3, resume to 4, put back the epoch-2 `best.pt` (a kill between
  the two writes) + resume; a finished copy without `metrics.json` and figure + resume →
  `metrics.json` bytes and `best.pt` weights equal to the uninterrupted run's; (h)
  `best.pt` deleted after epoch 5 (best epoch 4) → CLI `--resume` exit 2 naming `best.pt`
  and epoch 4, no `metrics.json`, `run_info.json` sessions unchanged;
  `load_selected_weights` with a selection naming another epoch → `ConfigError`; (i)
  legacy run dir (`select_weights` removed from `config.yaml`, the checkpoint's and
  `best.pt`'s config) → CLI `--resume` exit 0, `selection` = `final` / `config`.
- [ ] T004 [P] `tests/test_train.py`: `train_version` 2; the fixture run's `selection`
  (empty val fallback) and its log line; CLI exit 2 for `select_weights: best` (message
  names the key, no run dir); the ruling #3 test's comment (the scored weights are now
  `best.pt`'s).
- [ ] T005 [P] `tests/test_probe.py`: (j) a small fixture train run on a split with a val
  pose (CLI), then `strike-a-pose-eval probe --checkpoint <run>/<selection.checkpoint>`
  (= `best.pt`) → exit 0, `run_info.checkpoint.epoch` = `selection.epoch`.
- [ ] T006 Run the new and changed tests on the unchanged code: they must fail (record
  the count).

## Phase 2: Implementation

- [ ] T007 `src/strike_a_pose/train/config.py`: `WEIGHT_SELECTIONS`, `select_weights`
  field + validation, `LEGACY_VALUES`, `with_legacy_values`, `load_run_config` (plan
  "Design").
- [ ] T008 `src/strike_a_pose/train/trainer.py`: `TRAIN_VERSION = 2`, file-name
  constants, `_is_new_best` / `best_val` / `weight_selection`, `best.pt` helpers, write
  order (checkpoint first), `reconcile_best` in `resume_training` (after the checks,
  before the session is recorded), `load_selected_weights` + selection log line at the
  evaluation call site, `selection` in `metrics.json`, resume through `load_run_config`
  and `with_legacy_values`.
- [ ] T009 Configs: `configs/ae.yaml`, `configs/ae_fixture.yaml` (`select_weights:
  best_val` + comments), `configs/probe.yaml` (placeholder `best.pt` path with the
  one-line comment; header).
- [ ] T010 New and changed tests green; `ruff check`, `ruff format --check`.

## Phase 3: Contracts and docs

- [x] T011 Contracts in place: 004 `train-run.md` (Train v2: `select_weights` row,
  file roles, write order, resume reconciliation and legacy keys, `selection` block,
  val-only caveat, log line), 006 `probe-run.md` (`checkpoint` row). → in the plan
  commit.
- [ ] T012 004 `spec.md` dated notes "amended 2026-10-20 by slice 008" (US2 intro,
  FR-003, FR-004, FR-006, the empty-val edge case, clarification 5); 006 `spec.md`
  clarification 8 note.
- [ ] T013 README (status line + table; Usage train paragraph and `best.pt` paragraph;
  probe usage; Results: published 004 / 006 numbers are final-epoch weights, runs from
  008 on report best-val weights, gain known on val only; 004 / 006 config links → the
  resolved copies under `docs/results/`), `specs/roadmap.md` (#13 struck: operator
  choice B, 2026-10-20, implemented in 008; 008 row; status line; changelog entry
  appended, 2026-10-20), `docs/HANDOFF.md` (state; next work: the re-baseline run uses
  `select_weights: best_val` and sets `configs/probe.yaml` to its `best.pt`).

## Phase 4: Verify and commit

- [ ] T014 Local checks: `uv run --locked ruff check .`, `ruff format --check .`,
  `pytest` (count + wall time), `strike-a-pose-data validate tests/fixtures/smpl_fixture`,
  the three smoke commands into `<scratchpad>/runs-008-after`, SC-002 diff against T001
  (probe also on the T001 checkpoint); extra: a scratch fixture train on a split with a
  val pose + `strike-a-pose-eval probe --checkpoint <run>/best.pt` exits 0; `git diff
  a5bb162 -- docs/results pyproject.toml uv.lock .github` empty.
- [ ] T015 Commits (`spec`, `plan`, `feat`, `docs`), no push (the orchestrator reviews,
  pushes and watches CI).

## Dependencies

T001 first. T002–T005 (parallel) → T006 → T007 → T008 → T009 → T010 → T012, T013 →
T014 → T015. T011 is done in the plan commit.

## Analyze pass (lead, 2026-10-20)

Coverage: FR-001 → T002, T004, T007, T009 · FR-002 → T003 (b, c, d, f), T008 · FR-003 →
T003 (c, g), T008 · FR-004 → T003 (c, d, h), T008 · FR-005 → T003, T004, T008 ·
FR-006 → T003 (d, g, h), T008 · FR-007 → T002, T003 (i), T007, T008 · FR-008 → T005,
T009 · FR-009 → T002–T006 · FR-010 → T011–T013 · FR-011 → T014 (diff review).
SC-001 → T014 (+ CI after the orchestrator's push) · SC-002 → T001, T014 · SC-003 →
T003 (c, g) · SC-004 → T005, T014 · SC-005 → T013, T014. Edge cases: ties and NaN
(T003 b, c), minimum at the last epoch (T003 d), val not in `eval_splits` (unchanged
code path: the per-epoch val loss does not depend on `eval_splits`; no new test), kill
between the writes (T003 g), hand-deleted `best.pt` (T003 h), config files without the
key (T002), legacy run dir (T003 i).

Findings and resolutions:
1. FR-006 lists a kill *during* evaluation. Evaluation writes only the figure before
   `metrics.json`, and resume re-runs the whole evaluation (overwriting the figure), so
   "after the last epoch, before `metrics.json`" is the same state; T003 (g) simulates it
   with `metrics.json` and the figure removed.
2. The resume refusal must come before anything is written: `reconcile_best` runs after
   the config / dataset / split checks and before the session is appended to
   `run_info.json` (T003 (h) checks the sessions are unchanged).
3. A pre-008 `best.pt` stores a config without `select_weights`; the match check fills
   legacy values on that side too (T003 (i) strips the key from `best.pt` as well).
4. The evaluation-time check cannot fail without a hand edit during the run, so it is
   tested on `load_selected_weights` directly (T003 (h)).
5. SC-002 diff, expected: train `metrics.json` differs in `train_version` (1 → 2) and the
   added `selection` block only; `history.csv` equal except `epoch_s`; `config.yaml`
   gains `select_weights` (besides `run_root`, which names the run root); 003
   `metrics.json` byte-identical; 006 `metrics.json` differs only in
   `encoder.checkpoint_sha256` (a checkpoint stores its resolved config and
   `train_version`), and is byte-identical when the probe runs on the T001 checkpoint.
6. The scripted val double also writes the scripted values into `history.csv` and
   `training.*_val_loss`; tests assert against the script, not against real losses.
7. The 008 tests live in a new module, so slice 009's edits to `tests/test_train.py`
   do not collide with them.
8. Terminology: "scored weights" = the weights evaluated and reported (`ae`); "selected
   epoch" = `selection.epoch`; "the rule" = `select_weights`.
