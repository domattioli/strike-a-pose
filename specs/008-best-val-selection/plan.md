# Implementation Plan: Best-Val Weight Selection

**Branch**: `claude/charming-wozniak-5ouldf` (slice `008-best-val-selection`) | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)
**Input**: `specs/008-best-val-selection/spec.md` (approved at surrogate scope, surrogate ruling 2026-10-20; operator sign-off pending; approval basis: the operator's choice B for roadmap question #13)

## Summary

Training stays exactly as it is (60 epochs, cosine lr, val loss logged every epoch).
What changes is which weights a run scores: by default the weights of the earliest epoch
with the lowest finite val loss, read back from `best.pt`; the final weights when val is
empty or has no finite loss, or when the config says `select_weights: final`. A
`selection` block in `metrics.json` (`train_version` 2) names the scored weights and
file. `checkpoint.pt` is now written before `best.pt`, and `--resume` repairs or refuses
a `best.pt` that does not hold the checkpointed history's best epoch, so the selection
survives a kill at any point. Run directories started before this slice resume under
the rule they started with (`final`), through a table of legacy values for config keys.
The probe code is unchanged; `configs/probe.yaml` points at the future re-baseline run's
`best.pt`.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.11 + 3.13); 3.11-compatible code only.
**Primary Dependencies**: none new (torch, PyYAML, stdlib `math`); `pyproject.toml` /
`uv.lock` untouched.
**Storage**: run directories under git-ignored `runs/`; `best.pt` keeps its structure;
no new file.
**Testing**: pytest. Selection rule as a pure function on synthetic histories; end-to-end
fixture runs on a split with a val pose, where a test double replaces the per-epoch val
loss with a scripted curve (research R4), using a small model (channels `[4]*6`, latent
8; ≈ 0.4 s per 4–6 epoch run vs ≈ 2 s at full width); the 006 probe on a fixture
`best.pt`; config and legacy-resume cases.
**Target Platform**: Linux CPU (4 vCPU container; CI `ubuntu-24.04`).
**Project Type**: single library + CLIs.
**Performance Goals**: pytest stays < ~60 s locally (28.6 s at `a5bb162`); per-leg CI
< 5 min. Per full run, reading back `best.pt` (12.5 MB) costs < 1 s.
**Constraints**: training numerics unchanged (history and final weights bit-identical for
the same config on the same machine); `docs/results/` untouched; CI workflow unchanged;
nothing under `data/`, `runs/`, `assets/` committed.
**Scale/Scope**: two source files (`train/config.py`, `train/trainer.py`), three configs,
tests, contracts and docs.

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | Selection, `best.pt` repair and loading live in `strike_a_pose.train`; the CLI is unchanged. |
| II CLI + config | PASS | The rule is a config key (`select_weights`), written into the resolved `config.yaml`; no CLI flag, no hidden state. |
| III Reproducibility | PASS | Same config → same history, selection and `metrics.json` bytes; `selection` records rule, epoch, val loss and file; interrupted + resumed = uninterrupted at every kill point (FR-006, tested). |
| IV Test-first | PASS | FR-009 tests (tasks T002–T006) are written and run red before T008–T010. |
| V Honest evaluation | PASS | The selection uses val only; test is scored once, for the selected weights (batch Q1 (a)); the gain over the final weights is stated as known on val only (README, 004 contract); the published final-weight numbers stay as they are and are labelled as such. |
| VI Simplicity | PASS | One pure selection function, three small `best.pt` helpers, one legacy-value table; no new file type, dependency or CLI option. |

## Design

### `strike_a_pose/train/config.py`

- `WEIGHT_SELECTIONS = ("best_val", "final")`.
- `TrainConfig.select_weights: str = "best_val"` (placed after `eval_splits`); added to
  `_STRS` (non-empty string) and to `_check` (one of `WEIGHT_SELECTIONS`, else
  `ConfigError("select_weights: one of ['best_val', 'final'], got …")`).
- `LEGACY_VALUES: dict[str, Any] = {"select_weights": "final"}`: for each config key added
  after runs existed, the value a run directory that lacks the key was started under.
- `with_legacy_values(data)`: `{**LEGACY_VALUES, **data}` for a mapping (present keys
  win), anything else unchanged.
- `load_run_config(path)`: like `load_config`, but through `with_legacy_values`; used only
  by `--resume`. New runs keep `load_config` (absent key → default `best_val`).

### `strike_a_pose/train/trainer.py`

- `TRAIN_VERSION = 2`; file names `CHECKPOINT = "checkpoint.pt"`, `BEST = "best.pt"`.
- `best_val(rows) -> {"epoch", "val_loss"} | None`: earliest epoch with the lowest finite
  val loss (replaces `_best_val`); the in-loop update uses the same predicate
  `_is_new_best(v, best)` (`v` not None, finite, `v < best`), so NaN never becomes or pins
  the best (research R1).
- `weight_selection(rows, rule) -> dict`: the `metrics.json` `selection` block —
  `select_weights`, `weights`, `selected_by`, `epoch`, `val_loss`, `checkpoint` (spec
  FR-005): `final` → last epoch, `"config"`; `best_val` with a best → that epoch,
  `"val loss"`, `best.pt`; otherwise last epoch, `val_loss` null, `"final (empty val)"`
  (every `val_loss` is None) or `"final (no finite val loss)"`.
- `_best_state(epoch, val_loss, model_state, hparams, config)`: the unchanged `best.pt`
  payload. `_read_best(path)`: its contents, or None if missing / unreadable.
  `_best_matches(b, best, cfg)`: same epoch, same val loss (exact float) and
  `with_legacy_values(b["config"]) == cfg.to_dict()`.
- Epoch loop: decide `new_best`, write `checkpoint.pt`, then (if `new_best`) `best.pt`
  (FR-003; the order is the fix, research R2).
- `reconcile_best(run_dir, cfg, ckpt)` (resume, after the config / dataset / split
  checks and before the new session is recorded): no-op without a best epoch or when
  `best.pt` matches; if the best epoch is the checkpoint's epoch, rewrite `best.pt` from
  `ckpt["model"]`; otherwise a `best_val` run raises `ConfigError` ("cannot resume …:
  best.pt (missing | epoch e) does not hold the best val epoch b …"); a `final` run
  leaves the informational file alone.
- `load_selected_weights(model, run_dir, selection, cfg)` (evaluation call site, right
  after the epoch loop): under `best_val`, read `best.pt`, require `_best_matches` (else
  `ConfigError`), `model.load_state_dict(b["model"])` (the model keeps its channels-last
  layout, so the outputs equal those of the same weights at the end of training); under
  `final` the final weights are already in the model. The evaluation code after it is
  unchanged: it scores `model` (the `ae` entries) and draws the figure from it.
- One log line before evaluation (`_selection_line`): "scoring best_val weights: epoch
  34, val 0.00217 (val loss; best.pt); final epoch 60 val 0.00221".
- `metrics.json`: `"selection": selection` next to `training`; `training` keeps
  `best_val_epoch` / `best_val_loss` (now from `best_val`). `run_info.json` unchanged
  (`best_val_epoch`, `best_checkpoint`, `checkpoint` use the constants).
- `resume_training`: `cfg = load_run_config(…)`; the checkpoint check compares
  `with_legacy_values(ckpt["config"])` with `cfg.to_dict()`; then `reconcile_best`.

### Configs

- `configs/ae.yaml`, `configs/ae_fixture.yaml`: `select_weights: best_val` with a comment
  (fixture: "val is empty → final weights"); header comments say which weights are scored.
- `configs/probe.yaml`: `checkpoint: runs/ae-smpl_proc_v1-YYYYmmddTHHMMSSZ/best.pt   # set
  to the re-baseline run's best.pt`; header names the scored weights.

### Tests (FR-009; written first)

- `tests/test_train_config.py`: default `best_val`; invalid values (`"best"`, `""`,
  `True`, `None`); shipped AE configs list `best_val`; `load_config` vs `load_run_config`
  on a file without the key.
- `tests/test_train_selection.py` (new; the 008 tests in one place): `weight_selection` /
  `best_val` unit cases; end-to-end with the scripted val double (`ScriptedVal`,
  installed with `monkeypatch` over `trainer.eval_loss`): best-val weights scored and drawn
  (minimum at epoch 2 of 4), minimum at the last epoch = `final` run (and training
  identical), empty val fallback, no finite val loss, NaN at epoch 1, `final` with val,
  three resume kill points, refusal when the best weights are lost, the evaluation-time
  check, a legacy run dir.
- `tests/test_train.py`: `train_version` 2, the fixture run's `selection` block and log
  line, CLI exit 2 for a bad `select_weights`; the ruling #3 test's comment.
- `tests/test_probe.py`: probe on the `best.pt` of a val-bearing fixture train run.

### Where slice 009 hooks in

- Per-region image metrics: the evaluation block after `load_selected_weights` in
  `_train_and_evaluate` scores whatever `model` holds through `eval/report.py`
  (`score_images`), so 009 changes the scoring helpers and the `img[name] = r` entries,
  not the selection.
- New `metrics.json` fields go into the `metrics` dict assembly next to `selection`
  (`train_version` stays 2 for additive fields).
- A new train config key must add its pre-009 value to `config.LEGACY_VALUES`, so run
  directories started before 009 keep resuming (FR-007).

## Project Structure

### Documentation (this feature)

```text
specs/008-best-val-selection/
├── spec.md
├── plan.md            # this file
├── research.md
├── data-model.md
└── tasks.md
```

Normative contract text stays in the slices that own it (updated in place):
`specs/004-autoencoder-mvp/contracts/train-run.md` (Train v2: config key, file roles,
`selection`, resume) and `specs/006-latent-probe/contracts/probe-run.md` (`checkpoint`
row).

### Source Code (repository root)

```text
src/strike_a_pose/train/config.py       # select_weights, LEGACY_VALUES, load_run_config
src/strike_a_pose/train/trainer.py      # selection, write order, reconcile, load, records
configs/ae.yaml, configs/ae_fixture.yaml, configs/probe.yaml
tests/test_train_selection.py           # new
tests/test_train_config.py, tests/test_train.py, tests/test_probe.py
README.md, specs/roadmap.md, docs/HANDOFF.md,
specs/004-autoencoder-mvp/spec.md, specs/006-latent-probe/spec.md
```

**Structure Decision**: existing single package; no new module in `src/`.

## Complexity Tracking

None.
