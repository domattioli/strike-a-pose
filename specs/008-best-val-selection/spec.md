# Feature Specification: Best-Val Weight Selection

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `008-best-val-selection`)
**Created**: 2026-10-20
**Status**: Approved at surrogate scope (surrogate ruling 2026-10-20; operator sign-off pending).
Approval basis: the operator's choice of option B for roadmap open question #13
(2026-10-20), quoted below; the surrogate ruled on the clarification batch and accepted
every lead default.
**Input**: Operator answer to roadmap open question #13 ("Early stopping / fewer epochs
as the training default?"), 2026-10-20: **option B**, chosen from this description:
"report `best.pt` (lowest val loss) as the evaluated model; keep 60 epochs and the cosine
schedule unchanged; small code (config flag, fallback to final weights when val is empty
as in the fixture, `configs/probe.yaml` repointed, 004 FR-006/Q5 amended); gain on the
004 curve 1.8 % val MSE ≈ 0.08 dB, one seed, measured on the selecting split." For future
runs this replaces the part of surrogate ruling 2026-10-20 #3 (approved by the operator
2026-10-21) that kept best-val selection off by default; the rest of ruling #3 stands:
the published 004 numbers (final-epoch weights) remain the reported MVP result until a
new full run exists. Orchestrator brief (Fable surrogate scope): spec and batched
clarifications first; "done" must not depend on a full-dataset run (no SMPL model,
dataset or checkpoint in the container); the next slice (009, per-region image metrics)
also touches `train/trainer.py`, `eval/report.py` and the 003 / 004 contracts.

Settled context (operator decision above; code facts checked in this session at `a5bb162`):

- Training today (`train/trainer.py`, `_train_and_evaluate`): Adam, MSE, cosine per-step
  decay, 60 epochs (`configs/ae.yaml`). After every epoch the val loss (mean per-image
  MSE, model in eval mode; the same quantity as the 003 `image_mse`: the published val
  `ae` MSE 0.00220785 equals `training.final_val_loss`) is logged to `history.csv`
  (8 significant digits) and kept at full precision in the checkpointed history; it never
  changes training. When val has samples, every strict new minimum writes `best.pt` =
  `{"train_version", "epoch", "val_loss", "model_hparams", "model", "config"}`
  atomically, **before** that epoch's `checkpoint.pt`; `metrics.json` `training`
  (`best_val_epoch`, `best_val_loss`) and `run_info.json` (`best_val_epoch`,
  `best_checkpoint`) record it (ruling #3, commit `69da472`). After the last epoch the
  in-memory final weights are scored on `eval_splits` as `splits.<split>.image.ae` (the
  key is `ae`), next to `mean_image` and `pca_k<K>` refit on train, and drawn in
  `recon_<figure_split>.png`.
- `best.pt` loads with `models.load_checkpoint` (which needs only `model_hparams` and
  `model`) and the 006 probe accepts it (`--checkpoint …/best.pt`; the 006 contract row
  says "004 `checkpoint.pt` (or `best.pt`)"). Checked in this session with a scratch probe
  run on a val-bearing fixture split: exit 0, `run_info.checkpoint.epoch` and
  `metrics.encoder.epoch` read from `best.pt`, `train_run_git` from the run dir. No
  loadability change is needed.
- `resume_training` rebuilds the config from the run dir's `config.yaml` and refuses
  (exit 2, "checkpoint does not match config.yaml") when `checkpoint["config"] !=
  cfg.to_dict()`. A new `TrainConfig` key with a default therefore makes every run dir
  started before it non-resumable, with a misleading message (the 007 red-team note).
- `_best_val` and the in-loop update compare `vl < best`: a NaN val loss at epoch 1 would
  pin the "best" epoch to 1 for the rest of the run (`x < nan` is always false).
- Fixture: `splits/smpl_fixture.json` has train 1 / val 0 / test 1 poses (8 images each),
  so the CI smoke train and the fixture tests take the "no val" path (no `best.pt`). The
  val-bearing tests build a (0.5, 0.5, 0.0) split on the fly (train 1 / val 1 / test 0
  poses). On this machine the minimum of that run's val curve is its last epoch in every
  setting tried (6 and 10 epochs with cosine lr 1e-3; 10 epochs with constant lr 3e-3):
  no existing test can tell best-val weights from final weights.
- 004 full run (`docs/results/004-ae-smpl_proc_v1/history.csv`): val minimum 0.0021685 at
  epoch 34, final 0.0022078 (+1.8 %, 0.078 dB); epoch 39 is within 0.12 % of the
  minimum. That run predates `best.pt` and no longer exists; `configs/probe.yaml` still
  points at its `checkpoint.pt`.
- Environment: no SMPL model, dataset or checkpoint in this container; pytest 249 passed
  in 28.6 s at `a5bb162` (4 vCPU, Python 3.13); CI runs Python 3.11 and 3.13.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The AE is reported at its best val epoch (Priority: P1)

The researcher trains with `configs/ae.yaml` (val has samples). Training is unchanged
(same epochs, schedule, history, final weights); afterwards the run scores the weights
of the epoch with the lowest val loss, the ones in `best.pt`, on val and test, draws the
figure with them, and records which weights it scored.

**Why this priority**: this is the operator's decision (#13, option B).

**Independent Test**: fixture run on a val-bearing split whose per-epoch val loss is
scripted to bottom out at epoch 2 of 4: `metrics.json` names epoch 2 and `best.pt`; the
`ae` metrics and the figure equal those recomputed from `best.pt`'s weights and differ
from the final weights'.

**Acceptance Scenarios**:

1. **Given** `select_weights: best_val` and a val split with samples, **When** the run
   finishes, **Then** every eval split's `ae` entry and the figure's `ae` column come from
   the weights of the earliest epoch with the lowest finite val loss, read back from
   `best.pt`, and `metrics.json` `selection` records the rule, the applied weights
   (`best_val`), that epoch, its val loss and the file (`best.pt`).
2. **Given** the same config with `select_weights: final`, **When** both runs train,
   **Then** their `history.csv` (except `epoch_s`) and `checkpoint.pt` weights are
   identical: selection never changes training.
3. **Given** a val curve whose minimum is the last epoch, **When** the run finishes,
   **Then** the selected epoch is the last one, the scored file is `best.pt`, and the
   `ae` metrics equal those of a `select_weights: final` run.
4. **Given** an invalid `select_weights` value, **When** the CLI runs, **Then** it exits 2
   with a message naming the key and creates no run directory.

### User Story 2 - Final weights when there is nothing to select on, or when asked (Priority: P1)

With an empty val split (the committed fixture, hence CI) the run falls back to the final
weights and says so; `select_weights: final` keeps the 004 behaviour.

**Why this priority**: CI and every fixture test use the empty-val fixture; the operator
asked for this fallback.

**Independent Test**: fixture smoke train at `a5bb162` and after this slice: the two
`metrics.json` differ only in `train_version` and the added `selection` block.

**Acceptance Scenarios**:

1. **Given** `select_weights: best_val` and an empty val split, **When** the run
   finishes, **Then** the final weights are scored, no `best.pt` exists, `selection` says
   `weights: final`, `selected_by: "final (empty val)"`, `val_loss: null`,
   `checkpoint: checkpoint.pt`, and a log line says so.
2. **Given** `select_weights: final` and a val split with samples, **When** the run
   finishes, **Then** the final weights are scored, `best.pt` is still written
   (informational) and `selection.selected_by` is `"config"`.
3. **Given** a val split with samples but no finite val loss, **When** the run finishes,
   **Then** the final weights are scored with `selected_by: "final (no finite val loss)"`.

### User Story 3 - Selection survives interruption (Priority: P1)

The full run (~85 min) can be killed by a container restart and resumed; the resumed run
reports the same epoch and the same numbers as an uninterrupted one, wherever it was
killed.

**Why this priority**: the operator's restart-resilience requirement (004 Session
2026-10-20b); the 004 run was resumed twice. With `best.pt` now the reported model, a
resume must never report weights other than those the history selects.

**Independent Test**: scripted val curve; kills at an epoch boundary after the best
epoch, between the `checkpoint.pt` and `best.pt` writes of a new best epoch, and after
the last epoch before `metrics.json`; each resumed run's `metrics.json` equals the
uninterrupted run's byte for byte and its `best.pt` holds the same weights.

**Acceptance Scenarios**:

1. **Given** a run interrupted at any point, **When** `--resume` completes it, **Then**
   `selection`, `best.pt` (epoch, weights) and `metrics.json` equal the uninterrupted
   run's (same machine and thread count, as 004 FR-010).
2. **Given** a `best_val` run whose `best.pt` is missing or holds another epoch than the
   checkpointed history selects, and the checkpoint cannot supply the selected epoch's
   weights, **When** `--resume` runs, **Then** it exits 2 before training and names
   `best.pt` and both epochs (no silent fallback to the final weights).
3. **Given** a run directory started before this slice (no `select_weights` in its
   `config.yaml` or checkpoint config), **When** `--resume` runs, **Then** it continues
   and finishes under the rule it was started with (final weights), and `selection`
   records `select_weights: final`.

### User Story 4 - Downstream use and docs follow the scored weights (Priority: P2)

The probe config targets the scored weights of the next full run, and the docs say which
weights every published number comes from.

**Why this priority**: the operator listed `configs/probe.yaml` in option B; without
the doc note, the published final-weight numbers and future best-val numbers would be
compared as if they came from the same rule.

**Independent Test**: fixture probe run on the `best.pt` of a val-bearing fixture train
run; review of README, roadmap, HANDOFF and the 004 / 006 documents.

**Acceptance Scenarios**:

1. **Given** a val-bearing fixture train run, **When** `strike-a-pose-eval probe
   --checkpoint <run>/best.pt` runs on that split, **Then** it exits 0 and records the
   selected epoch.
2. **Given** the committed `configs/probe.yaml`, **Then** its `checkpoint` targets
   `best.pt` of the future re-baseline run (a placeholder until that run exists), and
   README, roadmap and HANDOFF say that the published 004 / 006 numbers come from the
   final-epoch weights of a pre-008 run and that runs from 008 on report best-val
   weights.

### Edge Cases

- Equal val losses: the earliest epoch wins (the existing `best.pt` rule).
- Near-ties: plain argmin over the full-precision values, no tolerance or smoothing (on
  the 004 curve epochs 34 and 39 are 0.12 % apart; whichever is lower is selected).
- The selection uses the full-precision val losses in the checkpointed history, not the
  8-significant-digit `history.csv`; a recomputation from `history.csv` can differ only
  when two val losses agree to 8 significant digits.
- Minimum at the last epoch: `weights: best_val`, scored file `best.pt`, the same weights
  as `checkpoint.pt`.
- `val` absent from `eval_splits` but non-empty: the per-epoch val loss is still computed
  (unchanged 004 behaviour) and drives the selection; the selected weights are scored on
  the configured splits only.
- NaN val loss: never selected and never written to `best.pt`; no finite value at all →
  fallback to final weights. Infinite values cannot occur (sigmoid output and images in
  [0, 1] bound the MSE by 1).
- Kill between the `checkpoint.pt` and `best.pt` writes of a new best epoch: on resume
  `best.pt` is rewritten from the checkpoint's weights, which are that epoch's (FR-006).
- `best.pt` deleted or replaced by hand: a `best_val` resume refuses when the selected
  epoch is older than the checkpoint (its weights are gone); evaluation refuses a
  `best.pt` that does not match the selection (FR-004). Never a silent fallback.
- Config files without `select_weights` (e.g. a user's own YAML) start new runs with the
  default `best_val`; only run directories, on `--resume`, read an absent key as `final`.
- `--resume` takes the run dir's `select_weights` like every other key (no override).
- A pre-008 run dir killed between its old-order writes (`best.pt` of epoch N+1,
  `checkpoint.pt` of epoch N) resumes under `final`, where `best.pt` is informational;
  the re-run epoch rewrites `best.pt` if it is again a minimum.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (config): `TrainConfig` MUST gain `select_weights` (string, one of
  `best_val`, `final`; default `best_val`), validated like the other keys (unknown keys
  stay an error; a non-string or another value → `ConfigError`, exit 2, before any run
  dir exists) and written into the resolved `config.yaml`. No CLI flag (constitution II;
  the overrides stay `--seed` / `--run-root`). `configs/ae.yaml` and
  `configs/ae_fixture.yaml` MUST list `select_weights: best_val` explicitly, the fixture
  comment noting that its empty val falls back to the final weights.
- **FR-002** (selection rule): one pure function of the full-precision per-epoch history
  and the rule MUST decide the scored weights: `best_val` → the earliest epoch with the
  minimum finite val loss; when val is empty or no val loss is finite → the final epoch;
  `final` → the final epoch. Training is unchanged: epochs, lr schedule, optimizer,
  shuffling and the per-epoch val loss are computed exactly as before, and val never
  stops or alters training (004 FR-003 amended: val now also selects the reported
  weights). `training.best_val_epoch` / `best_val_loss` and `best.pt` follow the same
  finite-minimum rule.
- **FR-003** (`best.pt`): unchanged structure `{"train_version", "epoch", "val_loss",
  "model_hparams", "model", "config"}` (loads with `load_checkpoint`), written atomically
  at every new finite minimum whenever val has samples (under `final` too, informational
  there), never when val is empty. The write order MUST become: that epoch's
  `checkpoint.pt` first, then `best.pt`; a kill between the two writes then leaves the
  new best epoch's weights in `checkpoint.pt`, from which resume rewrites `best.pt`
  (FR-006).
- **FR-004** (evaluation): after the last epoch the run MUST load the selected weights:
  under `best_val`, read back from `best.pt` after checking that its `epoch` and
  `val_loss` equal the selection's and its `config` equals the run's (any mismatch →
  `ConfigError`, exit 2, no `metrics.json`); otherwise the last epoch's weights. The
  existing 004 evaluation then runs unchanged on them: the `ae` entry of every eval split
  and the figure's `ae` column (label unchanged); mean image and PCA unchanged. Only the
  selected weights are scored, and test is scored once, for them (surrogate ruling
  2026-10-20, batch Q1 (a)); the gain of the selected over the final weights is therefore
  known on val only, the selecting split. One log line before evaluation names the
  applied weights, epoch, val loss and file (plus the final epoch's val loss when it
  differs).
- **FR-005** (records): `metrics.json` MUST gain a top-level `selection` block:
  `select_weights` (the configured rule), `weights` (`best_val` | `final`, as applied),
  `selected_by` (`"val loss"` | `"final (empty val)"` | `"final (no finite val loss)"` |
  `"config"`), `epoch`, `val_loss` (val loss of the selected epoch; null without a finite
  value) and `checkpoint` (`best.pt` | `checkpoint.pt`: the file holding the scored
  weights). `training` keeps `final_val_loss`, `best_val_epoch`, `best_val_loss`;
  `run_info.json` keeps `best_val_epoch` / `best_checkpoint`. `train_version` becomes 2
  in `metrics.json` (and in the checkpoints, which share the constant): the meaning of
  `splits.*.image.ae` changes from "always the final weights" (v1) to "the weights
  `selection` names". Bytes stay deterministic for the same inputs, code and machine.
- **FR-006** (resume): the selection MUST be resume-safe. It is recomputed from the
  checkpointed history. On resume, when val has samples, `best.pt` is reconciled before
  training: if the history's best epoch is the checkpoint's epoch and `best.pt` is
  missing or does not match it (epoch, val loss), `best.pt` is rewritten from the
  checkpoint's weights; if a `best_val` run's `best.pt` still does not hold the history's
  best epoch, resume refuses with exit 2 before training (message names `best.pt` and both
  epochs). A run interrupted at any point (an epoch boundary; between the `checkpoint.pt`
  and `best.pt` writes; after the last epoch, before or during evaluation) and resumed
  MUST finish with `metrics.json` bytes and `best.pt` weights equal to the uninterrupted
  run's (same machine and thread count, 004 FR-010).
- **FR-007** (legacy run directories): `--resume` of a run directory whose `config.yaml`
  has no `select_weights` (started before this slice) MUST continue with
  `select_weights: final`, the rule it was started under. The config / checkpoint
  comparison reads the absent key as `final` on both sides, through one table of legacy
  values for config keys added after a run started (later slices add their keys to it).
  New runs from config files without the key get `best_val`.
- **FR-008** (probe, configs): the probe code is unchanged. `configs/probe.yaml` MUST
  point `checkpoint` at `best.pt` of the future full re-baseline run through a placeholder
  path, `runs/ae-smpl_proc_v1-YYYYmmddTHHMMSSZ/best.pt`, which fails with exit 2 ("no such
  file") until replaced (as today: the run it names no longer exists); a one-line comment
  says to set it to the re-baseline run's `best.pt` (or pass `--checkpoint`), and the
  header names the scored weights instead of "final-epoch weights". The CI probe smoke keeps `checkpoint.pt` (fixture val is
  empty, so the final weights are the scored ones). The published 006 run's resolved
  config stays in `docs/results/006-probe-smpl_proc_v1/config.yaml`.
- **FR-009** (tests, written first; pytest):
  (a) config: default `best_val`; `final` accepted; another value or a non-string →
  exit 2, no run dir;
  (b) selection unit cases: earliest of equal minima; minimum at the last epoch; empty
  val → fallback; NaN entries skipped, including a NaN at epoch 1; all NaN → fallback;
  `final`;
  (c) end to end with a scripted per-epoch val loss (a test double, because the real
  fixture curve bottoms out at its last epoch): minimum at epoch k < E → `selection`,
  `best.pt` epoch k, `checkpoint.pt` epoch E; the `ae` metrics of every eval split and
  the figure equal those recomputed from `best.pt`'s weights and differ from the final
  weights';
  (d) scripted monotone curve → selected epoch E, scored file `best.pt`, `splits` equal
  to a `select_weights: final` run of the same config;
  (e) committed fixture (empty val) → fallback record, no `best.pt`, `splits` equal to a
  `final` run;
  (f) `final` with val → final weights scored, `best.pt` still written;
  (g) resume at the FR-006 kill points (epoch boundary after the best epoch; between the
  two writes, simulated by restoring the previous `best.pt`; after the last epoch, before
  `metrics.json`) → `metrics.json` byte-equal and `best.pt` weights equal to the
  uninterrupted run's;
  (h) `best_val` resume with `best.pt` deleted after an earlier best epoch → exit 2, no
  `metrics.json`; a `best.pt` that does not match the selection at evaluation → exit 2;
  (i) legacy: a run dir with `select_weights` removed from `config.yaml` and from the
  checkpoint's config resumes (exit 0) and records `select_weights: final`;
  (j) probe: fixture probe run with `--checkpoint <val-bearing run>/best.pt` → exit 0,
  `checkpoint.epoch` = the selected epoch;
  (k) the existing 004 / 006 tests stay green, edited only where they assert
  `train_version` or describe the scored weights.
- **FR-010** (documentation, same slice): see *Documentation scope*.
- **FR-011** (constraints): no new dependency, `pyproject.toml` / `uv.lock` untouched;
  Python 3.11-compatible code; training numerics unchanged (for the same config the
  history and final weights are bit-identical to `a5bb162` on the same machine); trainer
  changes localized (the selection function and the `best.pt` write / reconcile / load
  helpers; the evaluation code receives the selected model and is otherwise unchanged) so
  009 can extend the evaluation; `.github/workflows/ci.yml` unchanged; `docs/results/`
  untouched; nothing under `data/`, `runs/` or `assets/` committed.

### Documentation scope (FR-010)

- `specs/004-autoencoder-mvp/spec.md`: the US2 intro, FR-003, FR-004 (`best.pt` in the
  file list), FR-006, the empty-val edge case and clarification 5 get dated notes
  "amended 2026-10-20 by slice 008" (original text kept).
- `specs/004-autoencoder-mvp/contracts/train-run.md`: `select_weights` config row;
  `best.pt` row (scored by default, write order, reconciliation on resume);
  `checkpoint.pt` (resume state; the scored file only when `final` applies);
  `metrics.json` `train_version` 2 with the `selection` block and the rule that a v1 file
  was scored on the final weights; the caveat that the gain of the selected over the final
  weights is known on val only (the selecting split); resume text (legacy key table, the
  new exit-2 case); the selection log line.
- `specs/006-latent-probe/contracts/probe-run.md`: `checkpoint` row: the train run's
  scored file is named by its `metrics.json` `selection.checkpoint` (`best.pt` by default
  when val has samples). `specs/006-latent-probe/spec.md` clarification 8: dated note (the
  published 006 run probed final weights; later runs probe the scored weights).
- `configs/ae.yaml`, `configs/ae_fixture.yaml`, `configs/probe.yaml`: FR-001, FR-008.
- `README.md`: status line and slice table (008 row); Usage "Train …" paragraph (scores
  the selected weights: best val loss by default, final weights when val is empty or with
  `select_weights: final`) and the `best.pt` paragraph (no longer informational); the
  probe usage line (`configs/probe.yaml` targets the re-baseline run; pass
  `--checkpoint`); Results: one sentence that the published 004 and 006 numbers come from
  final-epoch weights of a pre-008 run and that runs from 008 on report best-val weights,
  whose gain over the final weights is known on val only (the selecting split), so the two
  are compared with that caveat; the 004 / 006 config links point at the resolved copies
  under `docs/results/` (the committed configs change in this slice).
- `specs/roadmap.md`: #13 struck with the operator's answer (option B, 2026-10-20,
  implemented by 008); 008 row in the post-MVP table; status line; changelog entry dated
  2026-10-20, appended at the end.
- `docs/HANDOFF.md`: state (008 implemented) and next work (#13 off the operator list;
  re-baseline sequence SMPL → F1 → 009 → full AE run with `configs/ae.yaml` → set
  `configs/probe.yaml` to its `best.pt` → probe run → docs).
- Not edited: `docs/results/`; the 003 contract; 001–007 `plan.md`, `research.md`,
  `tasks.md` (historical records); `.github/workflows/ci.yml`.

### Key Entities

- **`select_weights`** (train config): which weights a run scores and reports.
- **Selection record** (`metrics.json` `selection`): the rule, the applied weights, why,
  the epoch, its val loss and the file (FR-005).
- **`best.pt`**: weights of the earliest minimum-val-loss epoch so far; the scored file
  under `best_val`.
- **`checkpoint.pt`**: full resume state of the last finished epoch; the scored file only
  when `final` applies.

## Success Criteria *(mandatory)*

- **SC-001**: ruff, ruff format, pytest and the CI smoke commands are green locally, and
  CI is green on Python 3.11 and 3.13 after the orchestrator's push; pytest stays under
  ~60 s locally (28.6 s at `a5bb162`); per-leg CI wall time stays < 5 min.
- **SC-002**: no numerical change where nothing is selected: the fixture smoke train's
  `metrics.json` at `a5bb162` and after this slice differ only in `train_version` and the
  added `selection` block, and its `history.csv` is identical except `epoch_s`.
- **SC-003**: with a scripted val curve whose minimum is at epoch k < E, the reported
  `ae` metrics and figure are those of epoch k's weights, and every FR-006 kill point
  resumes to a byte-identical `metrics.json`.
- **SC-004**: the 006 probe runs on the `best.pt` of a fixture train run (exit 0).
- **SC-005**: `docs/results/` is byte-unchanged and no document attributes best-val
  weights to the published 004 / 006 numbers.
- "Done" does not depend on a full-dataset run.

## Assumptions

- The full re-baseline on `smpl_proc_v1` happens later (after the SMPL asset, F1 and
  009); it uses `configs/ae.yaml` as committed here, then `configs/probe.yaml` is set to
  its `best.pt`, and its docs report best-val weights.
- The selection metric stays the val loss the operator chose ("lowest val loss").
- Byte-identical resume holds on one machine and thread count (004 FR-010). Across
  machines the selection stays internally consistent (FR-004 check, FR-006
  reconciliation); low-order digits may differ.
- 009 adds per-region image metrics to the evaluation of the selected weights; this slice
  keeps that evaluation a single code path that receives the model.

## Clarifications

### Session 2026-10-20 (autonomous run, Fable surrogate scope)

Defaults chosen by the lead within the surrogate's scope are marked *(lead default)*; the
question sent to the surrogate in one batch carries *(surrogate ruling 2026-10-20)*. The
surrogate accepted every lead default, and asked to keep the NaN lock-in fix (3) and the
write-order fix (9) in the spec and the tests. Every answer is provisional and
overturnable by the operator.

1. Q: Slice number and slug? → A: `008-best-val-selection` (next free number; names the
   change). *(lead default)*
2. Q: Config key, values, default? → A: `select_weights: best_val | final`, default
   `best_val` (the operator made best-val selection the default for future runs), strict
   validation as for every key, listed explicitly in both AE configs; no CLI flag (runs are
   config-defined and the run dir's `config.yaml` records the rule). `final` keeps the 004
   behaviour for comparisons. *(lead default)*
3. Q: Selection rule? → A: the earliest epoch with the minimum finite val loss: plain
   argmin over full-precision values, no tolerance or smoothing (the operator's rule is
   "lowest val loss"); ties → earliest (the existing `best.pt` rule); NaN never selected,
   which also removes the epoch-1 NaN pin of today's `best.pt` update. *(lead default)*
4. Q: Fallback? → A: final weights when val is empty (fixture, CI) or has no finite val
   loss, recorded in `selection.selected_by` and logged; no Python warning (as the 006
   probe's empty-val fallback). *(lead default; operator text)*
5. Q: Which weights are scored, on which splits? → A: batch Q1, option (a): only the
   selected weights, on every eval split; test is scored once, for them. The final
   weights' val loss stays in `training.final_val_loss` and `history.csv` (the val loss is
   the val image MSE). The README and the 004 contract say explicitly that the gain of the
   selected over the final weights is known on val only, the selecting split, so the
   published 004 final-weight numbers and future best-val numbers are compared with that
   caveat. Rejected: (b) final weights on val as well, (c) on val and test as a
   pre-registered diagnostic. *(surrogate ruling 2026-10-20)*
6. Q: Figure? → A: drawn with the selected weights (figure and metrics show one model);
   column label `ae` unchanged; the selection is in `metrics.json`. *(lead default)*
7. Q: Which file holds the reported model; does `checkpoint.pt` stay the resume
   checkpoint only? → A: no new file. `best.pt` (present iff val has samples) holds the
   best-val weights and is the scored file under `best_val`; `checkpoint.pt` stays the full
   resume state of the last finished epoch and is the scored file only when `final`
   applies (by config or fallback); `selection.checkpoint` names the scored file. Rejected:
   a stable `model.pt` copy of the scored weights (a third 12.5 MB file per run and more
   contract surface for a convenience the operator did not ask for). *(lead default)*
8. Q: Records and versioning? → A: a top-level `selection` block in `metrics.json` (the
   name parallels the probe's `selection`; `selected_by` follows its "val …" /
   "default (empty val)" style); `training` and `run_info.json` unchanged. `train_version`
   1 → 2: the meaning of `splits.*.image.ae` changes (v1: always the final weights; v2: the
   weights `selection` names), and the 004 contract bumps `train_version` on breaking
   changes; the checkpoints carry the same constant (nothing reads it). Considered: keep 1
   as an additive change. *(lead default)*
9. Q: Write order and resume safety? → A: FR-003, FR-004, FR-006. Today `best.pt` is
   written before `checkpoint.pt`. On the same machine a kill between the two writes is
   harmless (the interrupted epoch re-runs bit-identically and rewrites `best.pt`); on
   other hardware (a reclaimed container can come back on another CPU) the re-run may
   no longer beat the previous best, leaving `best.pt` with weights the history does not
   select while the earlier best weights are gone. Writing `checkpoint.pt` first makes
   every interrupted state repairable from the checkpoint, and the evaluation-time check
   turns any remaining mismatch into an error instead of silently reporting other
   weights. *(lead default)*
10. Q: Run directories started before 008? → A: FR-007: they resume under the rule they
    started with (`final`) through a legacy-value table for config keys added after a run
    started; without it every such run dir is refused with the misleading "checkpoint does
    not match config.yaml" (the 007 red-team note). A few lines and one test, reusable by
    009; no such run dir exists today. Rejected: refusing them (cheaper, loses in-flight
    runs); resuming them under `best_val` (changes the rule mid-run). *(lead default)*
11. Q: Probe and `configs/probe.yaml`? → A: no probe code change (`best.pt` loads and is
    probed today, verified in this session; the probe records the file path and epoch).
    `configs/probe.yaml` targets `best.pt` of the future re-baseline run through a
    placeholder path: the run dir name carries its start time, and repointing the old run
    to `…20261020T030038Z/best.pt` would name a file that never existed (that run predates
    `best.pt`). The README probe line passes `--checkpoint`. The CI probe smoke stays on
    `checkpoint.pt`. *(lead default)*
12. Q: How are the tests made to tell best-val from final weights? → A: FR-009: the
    per-epoch val loss is replaced by a scripted sequence in the selection tests (the
    real fixture curve has its minimum at the last epoch on this machine, and a scripted
    curve does not depend on the machine); short fixture runs (≤ 6 epochs) keep pytest
    fast. *(lead default)*
13. Q: The published 004 / 006 numbers? → A: unchanged and still the reported MVP result
    (ruling #3; operator: until a new full run exists); `docs/results/` untouched; README
    states they are final-epoch weights. The 004 run cannot be re-scored (it predates
    `best.pt` and no longer exists). *(lead default; orchestrator brief)*
14. Q: Out of scope? → A: early stopping (option C) and an epoch change (option D); a full
    run or re-baseline; CI changes; probe code changes. *(lead default)*
15. Q: Dates? → A: the session clock, 2026-10-20; the roadmap changelog entry is appended
    at the end (insertion order). *(lead default)*

### Clarification batch for the surrogate (sent and ruled 2026-10-20)

- **Q1 — Besides the selected weights, should the run also score the final-epoch weights?**
  (a) No: only the selected weights are scored (val and test, once); the final weights'
  val MSE is already `training.final_val_loss`. (b) Also the final weights on val only, as
  an `ae_final` entry (adds PSNR / IoU on val; biased toward the selected weights, since val
  chose the epoch). (c) Also the final weights on val and test, as a pre-registered
  diagnostic `ae_final` that is never used to choose: the only in-run, unbiased estimate of
  what selection gains on held-out data (the operator's 1.8 % was measured on the selecting
  split); about 30 s more per full run, plus one more entry in the contract, the tests and
  009's per-region metrics. **Lead proposal: (a)**: constitution V (test scored once, for
  the reported model) and the smallest change; a later full run can still be compared with
  the published 004 final-weight numbers.
  **Ruling (surrogate, 2026-10-20): (a)**, with the val-only caveat stated in the README and
  the 004 contract (clarification 5).
