# Feature Specification: Run Guards and Campaign Procedure

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `012-run-guards`)
**Created**: 2026-10-21
**Status**: Approved at surrogate scope 2026-10-21 (operator present; the orchestrator is the
operator's surrogate): rulings Q1 (a) with per-variant promotion metrics, Q2 (a) with a pixel
diagnostic, Q3 (a); the five lead deviations accepted; an independent red-team review's
additions and the operator's decisions of 2026-10-21 folded in (below). Operator sign-off
pending; the variant promotion rule is a surrogate ruling with operator confirmation
requested.
**Input**: Operator decision 2026-10-21 (verbatim intent): "I'm ok with that much CPU time [the
≈ 20 h campaign in `docs/HANDOFF.md`: F1 → re-baseline R0 → realism datasets → variants R1–R3
→ winners] so long as we have checks to ensure early termination and iteration if the run looks
to be spinning its wheels." It supplements the operator's 2026-10-20 choice B on roadmap #13
(report `best.pt`): early termination is wanted as a guard, not as the rule that picks the
reported model (the 008 selection is unchanged). Orchestrator brief: (A) trainer run guards,
config-selectable, off by default, in `LEGACY_VALUES`; (B) the campaign procedure with decision
rules, a heartbeat check-in and a read-only status script; (C) hygiene: freeze the generator's
omitted defaults; make F1 robust to a re-converted SMPL file. Red-team additions (before any
compute): batched mean image (an OOM on the 4,000-pose dataset), CPU model and dataset
checksums hash in `run_info.json`, a run-dir lock, documentation honesty fixes, R0 acceptance
tolerances, results committed after every run. Operator decisions 2026-10-21: F1 failure →
continue with a caveat; realism datasets become independent single-change datasets v2a–v2d,
each compared with v1 (configs and the 010 documents amended by a separate lead later); the
arms-down table becomes `armsdown-v1` with an automatic intersection count; a seed-1 control
R0b; the first VAE run at β = 0.01. "Done" is code + tests + docs + configs; no full-dataset
run (no SMPL model in this container).

Settled context (checked at `e98134e`):

- **Trainer loop** (`train/trainer.py`, `_train_and_evaluate`): per epoch `train_epoch` →
  `eval_loss` (val, eval mode; None when val is empty) → history row appended and flushed →
  `checkpoint.pt` → `best.pt` on a new finite minimum → log line → the test hook
  `stop_after_epoch`. After the loop: 008 `weight_selection` → `load_selected_weights` →
  evaluation → `metrics.json`, `run_info.json`. The lr schedule is a per-step cosine
  `LambdaLR` over `epochs × steps_per_epoch`. Only `best.pt` and the selection read the val
  loss (004 FR-003 as amended by 008).
- **Resume** refuses a finished run (`metrics.json`, exit 2), a missing checkpoint or a config
  mismatch (exit 2); `LEGACY_VALUES` fills keys added after a run started (008 / 011); the
  history is the checkpointed full-precision list; `reconcile_best` repairs `best.pt`. Nothing
  stops a second process on the same run dir: the published 004 run's `run_info.json` records
  two `--resume` sessions 19 s apart at epoch 3, at commit `570f1df` with a dirty tree.
- **CLI exit codes** (`train/cli.py`): 0, 1 dataset, 2 config, 4 run dir exists; 5 is free.
- **Published 004 curve** (`docs/results/004-ae-smpl_proc_v1/history.csv`): epoch-1 val loss
  0.010030; val minimum 0.0021685 at epoch 34; final (epoch 60) 0.0022078; epochs 30–60 stay
  within −1.8 % … +0.1 % of the final value. `early_stop {patience: 10, min_epochs: 20,
  min_delta: 0}` would stop after epoch 44 (≈ 27 % of the epochs saved) and select the same
  epoch-34 weights. Median epoch 77.6 s; epoch-1 train loss 0.0352, the largest later one
  0.0036 (a 10× threshold is 0.352; the MSE of a sigmoid output on [0, 1] images is ≤ 1).
- **Mean image**: `trainer.py` and `eval/baselines.py` compute `x.sum(dim=0, dtype=float64)`
  over the whole float32 image matrix, which allocates a float64 copy (measured: an extra peak
  of the float32 matrix's size or more). For 25,600 train images: float32 5.0 GB + float64
  10.1 GB + uint8 1.3 GB ≈ 16.4 GB in the AE run: an OOM on the 15 GB machine after ≈ 170 min
  of training. Pixel values are `k/255` in float32 (multiples of 2⁻³¹, ≤ 1), so any float64
  summation order is exact for fewer than 2²² images.
- **Metrics writer** (`runlog.dumps_exact`): 6 significant digits, NaN → null, `inf` raises.
- **Provenance**: `environment_summary()` records Python, platform, library versions, CPU and
  thread counts, not the CPU model; `run_info.dataset` holds the raw and content manifest
  hashes, which do not cover the pixels (`checksums.json` does).
- **Generator defaults** (`datagen/config.py`): `ADDED_FIELD_DEFAULTS = {ranges_version:
  "predecessor-v1-nohands", body_shape: None, camera_jitter: None}`, omitted from the manifest
  while at these values (010 FR-002); not stated as frozen.
- **F1** (three identical copies: `docs/HANDOFF.md`, `specs/007-split-content-hash/{spec,
  tasks}.md`): step 1 stops unless `sha256(SMPL_NEUTRAL.npz) == f408141b…`; the provenance
  check re-hashes the regenerated manifest with `code` = `406b68f` (clean / dirty) against the
  raw hash `9000c915…`. The manifest records `inputs.smpl_model.sha256` and every sample's
  `labels_sha256`, not the images. The published 003 run
  (`docs/results/003-baselines-smpl_proc_v1/metrics.json`: 122 numbers, 6 significant digits)
  used today's `configs/baselines.yaml` values and is deterministic.
- **Environment**: 4 vCPU, 15 GB RAM, torch 2.14.1 (PyPI), Python 3.13 locally, CI 3.11 + 3.13;
  no `scripts/` directory yet; pytest ≈ 36 s.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A stalled run stops early (Priority: P1)

The operator trains a campaign config; once val loss has not improved for `patience` epochs
(after `min_epochs`), the run stops at the end of that epoch and is scored as usual.

**Why this priority**: the operator's condition for the campaign's CPU time.

**Independent Test**: fixture runs with a scripted val curve (008 pattern).

**Acceptance Scenarios**:

1. **Given** `early_stop {patience: P, min_epochs: M, min_delta: δ}` and a val curve whose last
   improvement is at epoch b, **When** it trains, **Then** it stops after the first epoch
   `e ≥ max(b + P, M)` with a finite val loss; `history.csv` and `checkpoint.pt` end at e; the
   log says why; `metrics.json` `training.stopped_early = {epoch: e, reason}`.
2. **Given** the same run, **Then** its e history rows (train / val loss, lr) equal those of a
   run without `early_stop`, the lr schedule still spans the configured `epochs`, and when the
   best epoch is ≤ e the `selection`, the scored `splits` and `best.pt` equal the full run's.
3. **Given** a run killed at any point and resumed, **Then** it stops at the same epoch with a
   byte-identical `metrics.json` (the decision is recomputed from the checkpointed history).
4. **Given** an empty val split (the fixture), **Then** the guard never fires and a log line
   says so.

### User Story 2 - A diverging run aborts (Priority: P1)

With `abort_on_divergence: true`, a run whose losses blow up ends with exit code 5 and an
`aborted.json` instead of burning its remaining epochs.

**Why this priority**: "spinning its wheels" includes a numerically broken run.

**Independent Test**: scripted train / val losses; exit 5, `aborted.json`, no `metrics.json`;
`--resume` refuses.

**Acceptance Scenarios**:

1. **Given** a non-finite epoch train loss, or from epoch 4 on a train loss above
   `divergence_ratio ×` epoch 1's, or three consecutive non-finite val losses, **When** that
   epoch ends, **Then** the process writes `aborted.json` (epoch, reason, last losses), no
   `metrics.json`, and exits 5.
2. **Given** an aborted run dir, **When** `--resume` runs, **Then** it exits 5 naming the reason
   and changes nothing.
3. **Given** a run killed after a diverging epoch's checkpoint but before `aborted.json`,
   **When** resumed, **Then** it aborts the same way without training another epoch.

### User Story 3 - One writer per run dir (Priority: P1)

A second `strike-a-pose-train` on a run dir whose process is alive is refused; a lock left by a
killed process does not block the resume.

**Why this priority**: the check-in resumes runs unattended; the published 004 run was resumed
twice 19 s apart.

**Independent Test**: a live helper process named in `run_dir/lock` → `--resume` exits 2; the
same lock after the helper exits → the resume proceeds; the lock is gone after a normal exit.

### User Story 4 - Defaults and older run dirs are unchanged (Priority: P1)

Every existing config (fixture and CI included) trains and reports as before; run dirs started
before 012 resume.

**Why this priority**: brief ("defaults OFF … bitwise unchanged"); R0 is the campaign's control.

**Independent Test**: same-machine before / after fixture smoke (`e98134e` vs slice head)
(SC-002); the legacy resume test.

**Acceptance Scenarios**:

1. **Given** a config without the 012 keys, **Then** no new file in the finished run dir, no new
   `metrics.json` key and no new log line (the resolved config lists the keys at their
   defaults; `run_info.json` gains the FR-011 provenance fields).
2. **Given** a run dir whose `config.yaml` / `checkpoint.pt` / `best.pt` lack the 012 keys,
   **When** resumed, **Then** it continues with the guards off and finishes like an
   uninterrupted run.

### User Story 5 - The campaign runs with checks and iteration (Priority: P1)

A session or the orchestrator's heartbeat follows one written procedure: ordered steps with
commands and time estimates, a status script, decision rules applied after every run and at
every check-in, results committed as they arrive.

**Why this priority**: the second half of the operator's condition.

**Independent Test**: `scripts/run_status.py` on fixture run dirs; review of the procedure
(`docs/HANDOFF.md` and this slice's `tasks.md`, identical text); the committed configs load.

**Acceptance Scenarios**:

1. **Given** a run dir, **When** `uv run --locked python scripts/run_status.py RUN_DIR` runs,
   **Then** it prints the FR-013 fields and a state, and writes nothing.
2. **Given** a finished, stopped, aborted or dead run, **When** a check-in applies the rules,
   **Then** it resumes, records, retries once, drops, or stops and reports exactly as the
   procedure says, and the next step starts only when its rule passes.

### User Story 6 - Large runs fit in memory and say where they ran (Priority: P2)

The 4,000-pose dataset's baselines and AE run fit on the 15 GB machine; every run records the
CPU model and the hash of the dataset's pixel checksums.

**Why this priority**: red-team findings; v2d would fail after ≈ 170 min, and "same machine
type" and "same pixels" are not checkable today.

**Independent Test**: the batched mean equals the one-shot float64 mean bit for bit; fixture
003 / 004 metrics byte-identical before / after; `run_info.json` fields present.

### User Story 7 - Hygiene (Priority: P3)

The generator's omitted defaults are frozen for layout v1; F1 tells which part of a
regenerated dataset differs (SMPL file, labels, pixels) and continues with a caveat.

**Independent Test**: the pin test; the second manifest check reproduces the fixture's raw
manifest hash after its SMPL hash is altered and substituted back; the comparison script on the
published 003 metrics.

### Edge Cases

- `min_delta > 0`: the early-stop anchor (last epoch that improved by more than `min_delta`) and
  `best.pt` (strict finite minimum, 008) can differ; the selection is always 008's.
- NaN val loss: never an improvement; such epochs count toward `patience`, but the stop is
  decided only after an epoch with a finite val loss; three in a row abort when the divergence
  guard is on.
- `min_epochs ≥ epochs` or a `patience` longer than the run: the guard never fires (valid).
- Stop and divergence at the same epoch: the abort wins (checked first).
- `stop_after_epoch` (test hook) at the stop or abort epoch: the simulated kill comes first; the
  resume then stops or aborts from the history without training.
- `select_weights: final` with an early stop: the final weights are the stop epoch's.
- `val` non-empty but not in `eval_splits`: the val loss still drives both guards (as 008).
- Infinite train loss without the divergence guard: unchanged (the `metrics.json` write raises,
  as today); with the guard the run aborts first.
- `aborted.json` writes non-finite losses as the strings `"nan"`, `"inf"`, `"-inf"`.
- Lock: a lock naming a dead pid, a pid of another boot, or a live process that is not
  `strike-a-pose-train` (pid reuse after a restart) is stale and replaced; `kill -9` or a
  container reclaim leaves the lock, the next resume replaces it. Two launches within
  milliseconds can both pass (no OS lock; documented).
- Status script on a run dir without rows (data still loading): "no epoch yet"; the dead
  verdict then counts from the last session's start.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (config): `TrainConfig` MUST gain, validated like the existing keys (exit 2 before
  any run dir exists):

  | Key | Type, range | Default | Legacy value |
  |---|---|---|---|
  | `early_stop` | null, or a mapping `{patience: int ≥ 1 (required), min_epochs: int ≥ 0 (default 0), min_delta: finite float ≥ 0 (default 0.0)}`, no other keys | `null` | `null` |
  | `abort_on_divergence` | YAML bool | `false` | `false` |
  | `divergence_ratio` | finite float > 1; a value other than 10.0 needs `abort_on_divergence: true` | `10.0` | `10.0` |

  `early_stop` is a frozen dataclass inside `TrainConfig` (`to_dict()` writes all three
  fields); no CLI flag; the legacy values go into `LEGACY_VALUES`.
- **FR-002** (early-stop rule): one pure function of the full-precision history rows and the
  `early_stop` block MUST decide the stop. Walking the epochs, an epoch with a finite val loss
  `v` *improves* iff there is no anchor yet or `v < anchor − min_delta`; the anchor (epoch,
  loss) then moves to it. After epoch e with a finite val loss the run stops iff
  `e ≥ min_epochs` and `e − anchor_epoch ≥ patience`. NaN epochs count as epochs without
  improvement; an empty val (all None) never stops. Evaluated after the epoch's history row,
  `checkpoint.pt`, `best.pt`, log line and the test hook, and on resume from the checkpointed
  history before training (a run killed after its stop epoch goes straight to evaluation). The
  loop ends; the 008 selection and the evaluation run unchanged. Optimizer, scheduler (total
  steps of the configured `epochs`), RNG draws and val loss are untouched: a stopped run is a
  prefix of the full run.
- **FR-003** (early-stop records): log `early stop after epoch e/E: <reason>`; with
  `early_stop` set and an empty val, `early stop inactive: val is empty` before epoch 1.
  `metrics.json` `training.stopped_early` MUST be present iff `early_stop` is configured: `null`
  or `{"epoch": e, "reason": "no val loss below <anchor> (epoch <b>) by more than <δ> for
  <e − b> epochs (patience <P>, min_epochs <M>)"}`. `training.epochs` / `steps` stay the
  configured values; `final_*` are epoch e's; `train_version` stays 2 (additive).
- **FR-004** (divergence rule): with `abort_on_divergence: true`, one pure function of the
  history rows and `divergence_ratio` MUST return the first matching reason at the last epoch
  e, in this order: (a) its train loss is not finite; (b) `e > 3` and its train loss
  `> divergence_ratio ×` epoch 1's; (c) the val losses of epochs `e − 2 … e` are all non-finite
  (None never counts). Evaluated after the epoch's records and before the early-stop check,
  and on resume from the checkpointed history before training.
- **FR-005** (abort): on a match the run MUST write `aborted.json` atomically:
  `{"epoch", "reason", "divergence_ratio", "first_train_loss", "last_losses": [{"epoch",
  "train_loss", "val_loss"} for up to the last 3 epochs]}`, log `aborted after epoch e:
  <reason>`, write no `metrics.json` or figure, and raise `RunAborted`; the CLI prints
  `aborted: <reason> (<run dir>/aborted.json)` and exits 5. `run_info.json`, `history.csv`,
  `checkpoint.pt` and `best.pt` stay as written (post-mortem).
- **FR-006** (no resume after an abort): `resume_training` MUST refuse a run dir holding
  `aborted.json` before writing anything (`RunAborted`, exit 5, message: epoch, reason, "start a
  new run").
- **FR-007** (run-dir lock): `run_training` (after creating the run dir) and `resume_training`
  (after the finished / aborted refusals, before any write) MUST hold `run_dir/lock` =
  `{"pid", "boot_id", "argv", "started_utc"}` (written atomically, re-read to confirm) and
  remove it on every exit that runs Python cleanup (finish, abort, error, the test hook). A lock
  whose process is alive refuses the start: `ConfigError`, exit 2, naming the pid and the file.
  Alive = same boot (`/proc/sys/kernel/random/boot_id`, when readable) and a running process
  with that pid whose command line contains `strike-a-pose-train` (without `/proc`: signal 0).
  Otherwise the lock is stale and replaced.
- **FR-008** (legacy run dirs): a run dir without the 012 keys in `config.yaml`, `checkpoint.pt`
  and `best.pt` resumes with the guards off and finishes byte-identical to an uninterrupted run.
- **FR-009** (defaults bitwise): with `early_stop: null` and `abort_on_divergence: false` the
  trainer MUST compute, write and log exactly what `e98134e` does, apart from the resolved
  config's three keys (`config.yaml`, the checkpoints' `config`; so a probe of such a checkpoint
  differs in `encoder.checkpoint_sha256`), the FR-011 `run_info.json` fields and the transient
  lock file.
- **FR-010** (mean image): one helper MUST compute the per-pixel mean of image rows (uint8,
  scaled `/255` per batch, or float32) by accumulating float64 batch sums, never a float64 copy
  of the whole matrix; the trainer and the 003 baselines use it; bit-identical to the one-shot
  float64 sum (pinned by tests). The trainer builds its float32 train matrix only for the PCA
  context column (`context_pca_k` not null). Peak estimates for the 4,000-pose dataset are
  restated in the campaign procedure and 010 `tasks.md`.
- **FR-011** (provenance): `environment_summary()` MUST add `cpu_model` (first `model name` of
  `/proc/cpuinfo`, else `platform.processor()`, else null); the 003, 004 and 006 `run_info.json`
  `dataset` blocks MUST add `checksums_sha256` (SHA-256 of the dataset's `checksums.json`, null
  if absent). Additive; the three contracts say so.
- **FR-012** (configs): `configs/ae.yaml`, `configs/ae_mask.yaml`, `configs/ae_vae.yaml`,
  `configs/ae_reg.yaml` MUST set `early_stop: {patience: 10, min_epochs: 20, min_delta: 0.0}`,
  `abort_on_divergence: true`, `divergence_ratio: 10.0` (commented); `configs/ae_vae.yaml`
  changes to `beta: 0.01` (operator decision). New, each a copy of its parent with its run name:
  `configs/ae_seed1.yaml` (R0b: `seed: 1`), `configs/ae_vae_beta0.1.yaml` (β = 0.1, more KL
  pressure), `configs/ae_vae_beta0.001.yaml` (β = 0.001, the collapse retry). Not edited:
  `configs/smpl_proc_v*` (the 010 amendment's lead adds the v2a–v2d configs with these guards),
  `configs/ae_fixture.yaml`, probe and baseline configs. The 010 series test exempts the three
  guard keys until that amendment.
- **FR-013** (status script): `scripts/run_status.py RUN_DIR [--json] [--now ISO8601]
  [--min-stale-min 10]` (argparse; stdlib + the package's pure guard and lock functions;
  read-only) MUST print: state (`running`, `stalled`, `looks dead`, `not running`,
  `finished`, `stopped early`, `aborted`; "no epoch yet" annotates a run without rows), epochs done / planned, last and median epoch
  time, minutes since the last activity (newest mtime of `history.csv`, `checkpoint.pt`,
  `best.pt`, `run_info.json`, `lock`, the figure), best val epoch and loss, current val vs best,
  the guards (early stop: epochs since the anchor / patience and the earliest stop epoch;
  divergence: off / ok / the reason it would abort), `train_kl` at epoch 10 and at the last
  epoch when present, the lock (pid, alive / stale / none) and, for a finished run, the val
  `image_mse` and the mean of the val region MSEs. **Looks dead** iff no `metrics.json` /
  `aborted.json`, no live lock owner, and the minutes since the last activity exceed
  `max(2 × median epoch time, --min-stale-min)` (a resumed session rewrites `run_info.json`
  at its start, so the clock restarts there); `not running` = no live owner, not yet stale.
  **Stalled** = live owner but no activity for 3× that threshold. Exit 0, or 2 for a path that
  is not a train run dir. Values from `history.csv` (8 significant digits).
- **FR-014** (campaign procedure): one "Campaign procedure" section in `docs/HANDOFF.md`
  replacing the next-work campaign items (F1, re-baseline, 010 runs, 011 runs), and the same
  text in this slice's `tasks.md`: the steps F1 → R0 → R0b → v2a → v2b → v2c → v2d → R1 → R2
  (→ R2r or R2b) → R3 → W (winners, dataset pending), with configs, detached commands
  (`setsid nohup … &`, `--resume`), per-step time estimates and the results copy; the decision
  rules (Clarifications); the check-in prompt verbatim (`send_later`, ≈ 50 min); a campaign log
  table (HANDOFF only).
- **FR-015** (frozen defaults): the layout contract MUST state that the values of
  `datagen.config.ADDED_FIELD_DEFAULTS` are frozen for layout v1 (a manifest without the key
  means that value; changing one requires a new `schema_version`); a test MUST pin each entry
  literally with a comment saying why. No manifest change.
- **FR-016** (F1): every F1 copy (007 `spec.md` with a dated note, 007 `tasks.md`, HANDOFF) MUST
  become, identically: step 1 prints whether the SMPL file matches `f408141b…` and continues;
  generation; manifest diagnostics: check 1 as before, check 2 with
  `inputs.smpl_model.sha256` set to the recorded
  `f408141b4cd5b6ca6466e4e218aeb88a11c60f598114951b1d6d1f8078c1da16` (both with `code` =
  `406b68f` clean / dirty, looking for `9000c915…`); the split rewrite and the pose-list check,
  worded as proving the seed and algorithm, not the dataset's identity; the pixel diagnostic:
  the 003 baselines rerun (`configs/baselines.yaml`) compared number by number with
  `docs/results/003-baselines-smpl_proc_v1/metrics.json` (relative difference ≤ 1e-5, exact for
  zeros; strings and keys added since are skipped). Reproduced = check 1 or 2 (2: with a one-line
  note) and the pixel diagnostic. Otherwise (operator decision 2026-10-21): keep the rewritten
  split, run the campaign, and record in README "Results", the roadmap changelog and HANDOFF that
  the published 003 / 004 / 006 numbers come from a dataset that no longer reproduces, naming
  the part that differs (SMPL file, labels, pixels), and report it prominently in the next
  operator message. The 010 `tasks.md` run procedure's F1 line points to it.
- **FR-017** (tests, written first; pytest, fixture, CPU; scripted losses as in
  `tests/test_train_selection.py`): (a) config: defaults, partial `early_stop` filled, every
  rejection, `LEGACY_VALUES`, YAML round trip; (b) rule units: fires exactly at `b + P`, not
  before `min_epochs`, `min_delta` anchor semantics, NaN epochs, empty val, state fields;
  divergence (a)–(c), order, the epoch-4 boundary, None never counts; (c) end to end: the stop
  epoch, record and log line; history prefix and lr equal the full run's; selection, `splits`
  and `best.pt` equal the full run's when the best epoch precedes the stop; resume after a kill
  before the stop epoch and after its checkpoint → byte-identical `metrics.json`; empty val
  never stops and logs it; (d) abort paths (a)–(c) through the CLI (exit 5, `aborted.json`, no
  `metrics.json`), resume refusal (exit 5, nothing changed), a kill before `aborted.json` →
  the resume aborts without training; (e) lock: live owner refused (exit 2), stale replaced,
  removed after finish / abort / error, foreign process with a reused pid treated as stale;
  (f) legacy resume; (g) default run unchanged (no `stopped_early`, no `aborted.json`, no lock
  left); (h) configs: guards in the four campaign configs, not in the fixture config; each new
  config differs from its parent only in its knob and run name; (i) status script on fixture
  run dirs (finished, stopped early, aborted, killed mid-run with `--now` later, live lock, VAE
  `train_kl`, not a run dir → exit 2), and nothing written; (j) mean image: batched = one-shot
  bit for bit (uint8 and float rows, odd batch sizes); (k) provenance fields in 003 / 004 / 006
  fixture `run_info.json`; (l) FR-015 pins; (m) the F1 comparison methods on the fixture (a
  scripted check in `tasks.md` T019, as 007 did: the F1 snippets are documentation).
- **FR-018** (documentation, same slice): see *Documentation scope*.
- **FR-019** (constraints): no new dependency (`pyproject.toml`, `uv.lock` untouched); Python
  3.11-compatible; `uv run --locked` only; `docs/results/`, `splits/`, `.github/`,
  `configs/smpl_proc_v*` and the 010 spec untouched; CI smoke unchanged.

### Documentation scope (FR-018)

- Contracts: 004 `train-run.md` (config rows, guard rules, `training.stopped_early`,
  `aborted.json`, exit 5, the resume refusal, the lock, `run_info` fields), 003 `eval-run.md`
  and 006 `probe-run.md` (`run_info` fields), 001 `dataset-layout.md` (FR-015).
- Dated notes "amended 2026-10-21 by slice 012": 004 `spec.md` (FR-003 "val never changes the
  run"), 008 `spec.md` (clarification 14: early stopping out of scope → a guard, selection
  unchanged), 011 `spec.md` (first β 0.01 and the VAE retries, operator decision; experiment
  procedure superseded by the 012 campaign procedure), 007 `spec.md` (F1 block).
- 010 `tasks.md` Follow-up R: one dated note (F1 diagnostics and policy; corrected v2d peak
  memory; the operator's independent datasets v2a–v2d, amendment pending).
- `README.md`: status (012 row; 007–012 verified on the fixture and the synthetic body only, no
  full-dataset run yet); "is expected to regenerate byte for byte (verified by F1)"; the exact
  session history of the published 004 run; Usage paragraph (guards, exit 5, lock, status
  script, campaign procedure), the variant commands (β 0.01).
- `specs/roadmap.md`: status line (the 2026-10-21 "approve all" covers 002–006 and the 2026-10-20
  surrogate rulings only); the operator's 2026-10-21 decisions under #13, #15 and #16; 012 row;
  changelog entry 2026-10-21 appended at the end.
- `docs/HANDOFF.md`: state (012; approval scope corrected; a "Pending operator sign-off" list
  007–012, one line each); next work (campaign; #12 on hold until the campaign's runs are done);
  the "Campaign procedure" and "Campaign log" sections; the F1 block.
- Not edited: `docs/results/`, 001–011 `plan.md` / `research.md` / `data-model.md`, the 010
  spec, `.github/`.

## Success Criteria *(mandatory)*

- **SC-001**: ruff check, ruff format --check, full pytest and the CI smoke commands green
  locally; CI green on 3.11 and 3.13 after the orchestrator's push; pytest grows by ≤ 25 s.
- **SC-002**: same-machine fixture smoke at `e98134e` vs the slice head: 003 and 004
  `metrics.json`, figures and weights byte-identical, `history.csv` identical except `epoch_s`,
  `config.yaml` only gains the three keys, `run_info.json` only gains `env.cpu_model` and
  `dataset.checksums_sha256` (besides times), probe `metrics.json` identical except
  `encoder.checkpoint_sha256`.
- **SC-003**: every FR-017 case passes; the status script answers in < 5 s.
- **SC-004**: the procedure text in HANDOFF and `tasks.md` is identical (script check), its
  shell blocks pass `bash -n`, the F1 copies are identical.
- "Done" does not depend on a full-dataset run.

## Assumptions

- The campaign runs on one machine type (4 vCPU, 15 GB) in the locked environment; estimates
  come from the 004 / 010 / 011 measurements.
- The heartbeat is a turn in the orchestrator's session (`send_later`) on the same container; a
  fresh container has no `data/` or `runs/` (the procedure restarts a step from scratch).
- Byte-identical resume holds on one machine and thread count (004 FR-010).
- Out of scope: early stopping as the reported-model rule, guards for the baselines / probe (not
  iterative), an OS-level lock, the v2a–v2d configs and the 010 amendment (separate lead), GPU.

## Clarifications

### Session 2026-10-21 (operator present; orchestrator = operator's surrogate)

Lead defaults are marked *(lead default)*; the batch Q1–Q3 was ruled by the orchestrator;
the red-team additions and the operator's decisions arrived with the rulings.

1. Q: Slice number and slug? → A: `012-run-guards`. *(lead default)*
2. Q: `min_delta` semantics? → A: Keras / Lightning: against the anchor (the last epoch that
   improved by more than `min_delta`), so slow steady gains still reset the counter; with
   `min_delta: 0` (the campaign) the anchor is 008's best epoch. *(lead default)*
3. Q: NaN val epochs and patience? → A: count as epochs without improvement; the stop is
   decided only after an epoch with a finite val loss. *(lead default)*
4. Q: `training.stopped_early` always? → A: only with `early_stop` configured (null when it did
   not fire), so a default run's `metrics.json` stays byte-identical; `training.epochs` /
   `steps` stay the planned values. *(lead default)*
5. Q: `divergence_ratio` validation? → A: finite > 1; a non-default value without the guard is an
   error; 3 grace epochs and 3 consecutive non-finite val losses are constants. *(lead default)*
6. Q: Exit code when `--resume` meets an aborted run? → A: 5, like the abort. *(lead default)*
7. Q: Files after an abort? → A: `aborted.json` plus the diverging epoch's history row,
   checkpoint and `best.pt` (post-mortem); no `metrics.json`, no figure. *(lead default)*
8. Q: Dead-run detection? → A: the brief's "no new row for > 2× median epoch" alone would call a
   live run dead while it loads data or evaluates; the status script also requires no live lock
   owner (FR-007) and a 10-min floor. Stricter, never looser. *(lead default, accepted)*
9. Q: Retry configs? → A: committed, so a check-in only launches commands. *(lead default,
   accepted; adapted to the operator's β decision: `ae_vae_beta0.1.yaml`,
   `ae_vae_beta0.001.yaml`)*
10. Q: Variants carried to the winners step (Q1)? → A: (a), winners only, with the promotion
    metric fixed per variant in advance: **R1** (mask-weighted MSE) wins if its mean val region
    MSE (hands, feet, head) beats R0's while its whole-image val MSE is within +5 % of R0's;
    **R2** (VAE) wins if its whole-image val MSE is within +5 % of R0's, its last-epoch
    `train_kl` ≥ 1 nat/image and its probe's val world-frame MLP MPJPE is not worse than R0's;
    **R3** (regularization) wins if its whole-image val MSE beats R0's. Dropped: more than 5 %
    worse on whole-image val MSE (R1, R3), or collapsed / aborted after the retry (R2). Between:
    reported, not carried (one seed cannot resolve 0–5 %). Margins are read against the R0 vs
    R0b gap: a "beat" must exceed the gap on that metric, and every comparison states the gap.
    The VAE runs considered are R2 and its retry or β = 0.1 run; the largest β that meets the
    rule is carried. *(surrogate ruling 2026-10-21, operator confirmation requested; the
    largest-β choice is a lead default)*
11. Q: The F1 SMPL-file step (Q2)? → A: (a): a mismatch is printed, generation continues; the
    second manifest check (recorded SMPL hash substituted) identifies a re-converted file with
    the same arrays; plus a pixel diagnostic: the 003 baselines rerun compared with the
    published numbers (≤ 1e-5). Failure policy: continue with a caveat naming the part that
    differs (SMPL file, labels, pixels), recorded in README "Results", the roadmap changelog and
    HANDOFF and reported prominently. The pose-list check proves the seed and algorithm only.
    *(surrogate ruling 2026-10-21; failure policy: operator decision 2026-10-21, replacing the
    surrogate's hard stop)*
12. Q: What may a check-in do unattended (Q3)? → A: (a): resume a dead run, stop a collapsed VAE,
    start the next step when its rule passes, commit each finished step's artifacts (explicit
    paths: split file, `docs/results/campaign/<step>/<kind>/` copies, the campaign-log line),
    the orchestrator pushing; README / roadmap prose waits for an attended wrap-up; on a stop
    outcome nothing new starts. *(surrogate ruling 2026-10-21)*
13. Q: Realism datasets? → A: independent single-change datasets, each compared with v1 (R0):
    v2a arms-down table (renamed `armsdown-v1`, with an automatic limb–torso intersection count
    and threshold), v2b body shape, v2c camera jitter, v2d 4,000 poses (`context_pca_k: null`,
    the batched mean of FR-010); no cumulative chain; configs and the 010 documents come from a
    separate lead after this slice. *(operator decision 2026-10-21)*
14. Q: Control noise? → A: R0b = R0 with seed 1 (≈ 85 min + probe) right after R0; its gap to R0
    is the noise bar of every later comparison. *(operator decision 2026-10-21)*
15. Q: VAE β? → A: the first run R2 at β = 0.01 (`configs/ae_vae.yaml`); collapsed (epoch-10
    `train_kl` < 1 nat/image) or aborted → R2r at β = 0.001 once; healthy (last `train_kl` ≥ 1
    and whole-image val MSE within +5 % of R0's) → R2b at β = 0.1 once (more KL pressure is
    affordable). *(operator decision 2026-10-21)*
16. Q: Red-team additions? → A: FR-007 (lock), FR-010 (batched mean), FR-011 (CPU model,
    checksums hash); documentation honesty fixes, R0 acceptance tolerances (epoch-1 val loss
    0.010030 ± 2 %, `final_val_loss` 0.0022078 ± 3 %, else investigate before continuing; the
    published curve stays within that band from epoch 30 on, so an early stop does not leave
    it), results committed after every run, #12 on hold during the campaign.
    *(orchestrator, 2026-10-21)*
17. Q: Dataset rule, timing rule, failures? → A: a v2x AE run whose best val loss
    (`selection.val_loss`) exceeds 2 × R0's stops the dataset series (check that dataset:
    intersection count, jitter fallback rate, visual check; report); the variants still run.
    v2d starts only if the v2c AE run's `train_wall_s` ≤ 1.5 × 85 min. R0 aborted or outside its
    tolerances → the campaign stops; R0b aborted → recorded, comparisons say "no noise bar";
    R1 / R3 aborted → dropped. A resume that adds no history row by the next check-in, or a
    non-train step killed twice (rerun from scratch once) → stop that step and report.
    *(lead default)*
18. Q: Winners' dataset? → A: pending operator decision (the cumulative v5 no longer exists); the
    campaign stops before step W and asks. *(lead default)*
