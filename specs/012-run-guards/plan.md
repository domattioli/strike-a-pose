# Implementation Plan: Run Guards and Campaign Procedure

**Branch**: `claude/charming-wozniak-5ouldf` | **Date**: 2026-10-21 | **Spec**: [spec.md](spec.md)

## Summary

Three opt-in trainer guards (early stop on a val plateau, abort on divergence, a run-dir lock
that is always on), a batched mean image that removes the 4,000-pose OOM, two provenance
fields, a read-only status script, the campaign configs, and one campaign procedure with
decision rules and a re-armable check-in. Every guard is a pure function of the checkpointed
history, so a resumed run decides exactly as an uninterrupted one. Defaults keep every existing
run's training, history, weights, figure and `metrics.json` byte-identical.

## Technical Context

Python ≥ 3.11 (CI 3.11 + 3.13), torch 2.14.1 (PyPI, locked), NumPy, PyYAML; pytest; ruff
(line 100). No new dependency. Linux container, 4 vCPU, 15 GB. Entry points unchanged
(`strike-a-pose-train`, `strike-a-pose-eval`); the status script is a plain file under
`scripts/` (no `pyproject.toml` change), run with `uv run --locked python scripts/run_status.py`.

## Constitution Check

- I Library-first: guard logic, lock and mean helper live in the package
  (`train/guards.py`, `eval/baselines.py`, `runlog.py`); the script only formats their output.
- II CLI / config: guards are config keys recorded in `config.yaml`; no new flag; the script
  is read-only tooling around run dirs.
- III Reproducibility: decisions are functions of the checkpointed history (resume-exact);
  `run_info.json` gains the CPU model and the pixel checksums hash.
- IV Test-first: every rule has unit tests and fixture end-to-end tests written before the
  code (red runs recorded in `tasks.md`).
- V Honest evaluation: the selection stays 008's; the stop never picks the reported weights;
  the procedure fixes promotion metrics before the runs and reads them against a seed-noise
  bar (R0b).
- VI Simplicity: three config keys, one module of pure functions, one helper, one script; no
  plugin hooks, no OS lock, no scheduler.

No violations; Complexity Tracking empty.

## Design

### D1 Config (`train/config.py`)

`@dataclass(frozen=True) class EarlyStop: patience: int; min_epochs: int = 0; min_delta: float =
0.0`. `TrainConfig` gains `early_stop: EarlyStop | None = None`, `abort_on_divergence: bool =
False`, `divergence_ratio: float = 10.0`. `config_from_dict` builds `EarlyStop` from a mapping
(unknown or missing sub-keys, wrong types → `ConfigError` naming `early_stop.<key>`), keeps
`None`, and accepts an `EarlyStop` instance (internal round trips). `asdict` turns the block
into a plain dict, so `to_dict()`, `config.yaml`, the checkpoints' `config` and the resume
comparison need no special case. `_check`: `divergence_ratio > 1`, and `!= 10.0` only with the
guard. `LEGACY_VALUES` += `early_stop: None`, `abort_on_divergence: False`,
`divergence_ratio: 10.0`.

### D2 Guards (`train/guards.py`, new; no torch import)

- `GRACE_EPOCHS = 3`, `NONFINITE_VAL_EPOCHS = 3`, `ABORTED = "aborted.json"`, `LOCK = "lock"`.
- `class RunAborted(Exception)`.
- `early_stop_state(rows, es) -> dict | None`: walks the rows (FR-002); returns `None` for
  `es is None`, else `{"anchor_epoch", "anchor_loss", "since", "earliest_stop", "fired"}` with
  `fired = None | {"epoch", "reason"}` (the first firing epoch; the walk stops there).
- `divergence_reason(rows, ratio) -> str | None`: FR-004 at the last row.
- `abort_record(rows, reason, ratio) -> dict`: FR-005 (non-finite floats as strings).
- Lock: `boot_id()`, `read_lock(path)`, `lock_alive(info)`, `acquire_lock(run_dir, argv)`
  (raises `ConfigError` for a live owner; writes `{pid, boot_id, argv, started_utc}` through a
  temp file + `os.replace`, re-reads to confirm), `release_lock(run_dir)` (removes the file only
  if it names this pid).

### D3 Trainer (`train/trainer.py`)

- `run_training`: `acquire_lock` right after `make_run_dir`; everything after it inside
  `try / finally: release_lock`.
- `resume_training`: refusals in order — finished (`metrics.json`, exit 2), aborted
  (`aborted.json`, `RunAborted`, exit 5), then `acquire_lock` (live owner → exit 2), then the
  existing checks and work inside `try / finally`.
- `_train_and_evaluate`: after restoring a checkpoint, `_guard(cfg, run_dir, rows, say)` (abort
  check, may raise) and `stop = _stopped(rows, cfg)`; an early-stop log line when the guard is
  set and val is empty. The epoch loop skips when `stop` is set; per epoch after the existing
  records and log line and the `stop_after_epoch` hook: `_guard(...)`, then `stop = …`; on a stop
  log the reason and `break`. `metrics.json` `training` gains `stopped_early` only with
  `early_stop` configured. Nothing else in the loop changes (FR-009).
- `_guard` writes `aborted.json` (temp file + `os.replace`, `write_json_exact` formatting),
  logs, raises `RunAborted`.
- Evaluation: `mean_image = image_mean(train["image"].reshape(n, -1))` (D5); the float32 train
  matrix is built only when `context_pca_k` is set, and freed after the PCA fit.

### D4 CLI (`train/cli.py`)

`EXIT_ABORTED = 5`; `except RunAborted as e: print(f"aborted: {e}", file=sys.stderr); return 5`.

### D5 Batched mean (`eval/baselines.py`)

`image_mean(rows, batch_size=1024) -> Tensor[P] float32`: for each batch, `to float32` (uint8
rows scaled `/255`, the trainer's `to_float` values), `.to(float64).sum(0)` into a float64
accumulator; `/ n`, `.to(float32)`. Exact for fewer than 2²² rows of `k/255` values (research
R4), hence bit-identical to the one-shot `sum(dtype=float64)`. `fit_baselines` and the trainer
call it. Peak memory of the 4,000-pose runs: research R4.

### D6 Provenance (`runlog.py` + three run writers)

`cpu_model()` (first `model name` in `/proc/cpuinfo`, else `platform.processor() or None`) as
`environment_summary()["cpu_model"]`; `checksums_sha256(dataset_dir)` (SHA-256 of
`checksums.json`, None if absent) in the `dataset` block of `run_info.json` written by
`run_baselines` (003), `run_training` (004) and the probe (006). Resume keeps the stored value.

### D7 Status script (`scripts/run_status.py`)

`argparse`; `status(run_dir, now, min_stale_min) -> dict` (pure but for reads) and `render(d)
-> str`; `main(argv) -> int`. Reads `config.yaml` (via `train.config.load_run_config`, so legacy
dirs work), `history.csv`, `run_info.json`, `lock`, `metrics.json`, `aborted.json` and file
mtimes; calls `early_stop_state`, `divergence_reason`, `lock_alive`. Tests import it with
`importlib.util.spec_from_file_location` and pass `--now` (deterministic staleness) and
`--json`.

### D8 Configs

The four campaign configs gain the guard block (commented); `ae_vae.yaml` → `beta: 0.01`; new
`ae_seed1.yaml`, `ae_vae_beta0.1.yaml`, `ae_vae_beta0.001.yaml`. The 011 config test learns the
new β; the 010 series test exempts the three guard keys (the 010 amendment's lead adds them to
the v2a–v2d configs).

### D9 Docs

Campaign procedure (identical in `tasks.md` and HANDOFF, script-checked), F1 block (identical
in the 007 spec, 007 tasks and HANDOFF), contracts, dated notes, README, roadmap (FR-018).

## Where things live

| Change | Files |
|---|---|
| config keys | `src/strike_a_pose/train/config.py` |
| guard rules, lock | `src/strike_a_pose/train/guards.py` (new) |
| loop, resume, records, mean use | `src/strike_a_pose/train/trainer.py` |
| exit 5 | `src/strike_a_pose/train/cli.py` |
| batched mean, 003 `run_info` | `src/strike_a_pose/eval/baselines.py` |
| 006 `run_info` | `src/strike_a_pose/eval/probe.py` |
| CPU model, checksums hash | `src/strike_a_pose/runlog.py` |
| status script | `scripts/run_status.py` (new) |
| configs | `configs/ae.yaml`, `ae_mask.yaml`, `ae_vae.yaml`, `ae_reg.yaml`, `ae_seed1.yaml`, `ae_vae_beta0.1.yaml`, `ae_vae_beta0.001.yaml` |
| tests | `tests/test_train_guards.py`, `tests/test_run_status.py`, `tests/test_run_records.py` (new); `tests/test_realism.py`, `tests/test_train_variants.py` (edited) |

## Stages and commits

1. `plan(012)`: plan, research, data model, tasks (with the campaign procedure).
2. `feat(012): run guards`: D1–D4 with `tests/test_train_guards.py` (red first).
3. `feat(012): campaign status script + configs`: D5–D8 with `tests/test_run_status.py`,
   `tests/test_run_records.py`, the FR-015 pins (red first); the SC-002 comparison.
4. `docs(012)`: D9, roadmap, README, HANDOFF, verification record.

Before each commit: ruff check, ruff format --check, full pytest (count, wall time); for
trainer changes the fixture train + probe smoke commands.

## Risks

- A guard that touches RNG or tensors would break FR-009: the guards read the history only;
  the default-config comparison (SC-002) and the defaults test catch it.
- Lock false positives after a restart (pid reuse): boot id + command-line check (research R3).
- Status script drift from the trainer's rule: it imports the same functions.
- `history.csv` precision (8 significant digits) in the script: near-ties can differ from the
  trainer's full-precision decision; the script says so, the trainer's record is authoritative.
