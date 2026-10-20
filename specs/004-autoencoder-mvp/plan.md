# Implementation Plan: Autoencoder MVP

**Branch**: `claude/charming-wozniak-5ouldf` (slice `004-autoencoder-mvp`) | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)
**Input**: `specs/004-autoencoder-mvp/spec.md` (approved, autonomous run 2026-10-20)

## Summary

Step 0: investigate the suspected 003 figure bug (research R0: not a bug; add a
grid-placement regression test). Add `strike_a_pose.models.ConvAutoencoder` and
`strike_a_pose.train` (config, trainer, CLI `strike-a-pose-train`). Training loads
the split images into memory once, trains with Adam/MSE (seeded shuffling, cosine
lr), logs per-epoch losses and checkpoints, then scores the final model plus the
mean-image and PCA k=64 baselines (fit on train) on val and test with the 003
metrics and writes the 003-style figure with an AE column. CI gets a fixture smoke
train. One full-dataset run (60 epochs, ≈ 1.5 h) recorded in docs.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.11 + 3.13)
**Primary Dependencies**: existing only — torch (`nn`, `optim`), numpy, pillow,
pyyaml. No new dependency.
**Storage**: reads 001 datasets via the 002 loader; run dirs under git-ignored
`runs/`; small summaries under `docs/results/004-ae-smpl_proc_v1/`.
**Testing**: pytest — model shapes/range/errors, config validation, fixed-batch
overfit, fixture CLI end-to-end (artifacts, loss decrease, repeatability), grid
placement (R0).
**Performance Goals**: fixture CLI smoke < 60 s (SC-001); full run ≤ ~3 h on 4 vCPU
(SC-002; planned ≈ 1.5 h); CI leg < 5 min.
**Constraints**: train-only fitting; final-epoch weights; test scored once in the
real run; 003 numbers/bytes unchanged by the refactor.

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | Model, trainer, scoring, figure in `src/strike_a_pose/`; CLI thin; no notebooks. |
| II CLI + config | PASS | `strike-a-pose-train --config configs/ae.yaml`; every knob in YAML; overrides written into resolved `config.yaml`. |
| III Reproducibility | PASS | Seeds (Python/NumPy/torch) set + logged; seeded shuffle generator; run dir: resolved config, commit + dirty flag, env (incl. threads), dataset + split hashes, history, metrics; repeatability tested on fixture; full run on a clean tree. |
| IV Test-first | PASS | Model/config/overfit/CLI/grid tests written with the code; CPU smoke train in CI on the fixture (< 60 s, constitution's ~2 min). |
| V Honest evaluation | PASS | Frozen pose-level split; train-only fit for AE and baselines; final weights (no val/test selection); test scored once; AE reported next to mean image + PCA k=64 in the same `metrics.json`; docs cite run dir + commit. |
| VI Simplicity | PASS | One plain AE class, flat config, in-memory uint8 data, resume only as far as restart resilience needs (2026-10-20b), no VAE, no new deps; 003 helpers shared rather than duplicated. |
| Tech: pinned deps | PASS | No dependency change; new console script (`uv lock --check`). |

## Design

### `strike_a_pose/eval/report.py` (moved from `baselines.py`, behaviour unchanged)

- `figure_indices(n, rows)`; `grid_figure(columns: list[(label, imgs [n, H·W])],
  image_size) -> PIL.Image` (header labels + rows; constants `FIG_PAD`,
  `FIG_HEADER`, `FIG_BG`); `score_images(pred_fn, images, masks, batch_size,
  threshold) -> {"image_mse", "psnr_db", "silhouette_iou"}` (batched per-image
  metrics, float64 means). `baselines.py` uses them (tests import the constants from `report`).

### `strike_a_pose/models/autoencoder.py`

`ConvAutoencoder(nn.Module)` per contract; `parameter_count()`;
`load_checkpoint(path)`.

### `strike_a_pose/train/config.py`

`TrainConfig` frozen dataclass + `config_from_dict` / `load_config` / `to_dict` /
`with_overrides`, strict checks (`ConfigError` from `strike_a_pose.eval.config`
reused). Image-size divisibility checked in the trainer after reading the manifest
(exit 2, before the run dir).

### `strike_a_pose/train/trainer.py`

- `load_split(ds, batch_size, masks) -> {"image": uint8 [n,1,H,W], "mask": bool
  [n, H·W] | None, "pose_index": [n]}` via `DataLoader` (no shuffle, 0 workers).
- `train_epoch(model, opt, sched, images, batch_size, gen) -> float` (sample-weighted
  mean MSE; channels_last; `/255` per batch).
- `eval_loss(model, images, batch_size) -> float | None`.
- `overfit_steps(model, batch, steps, lr) -> list[float]` (used by the overfit test).
- `run_training(cfg, argv) -> Path`: construct datasets (DatasetError → exit 1) →
  check divisibility → run dir → `config.yaml`, `run_info.json` (start) → seed →
  load train (+ val) → model, Adam, cosine `LambdaLR` per step → epochs: train,
  val loss, append `history.csv`, save `checkpoint.pt` (tmp + replace), print →
  context fit (mean image, PCA `context_pca_k` on float train) → score each eval
  split (`ae`, `mean_image`, `pca_k{K}`) → figure → `metrics.json` →
  `run_info.json` (finish).

### `strike_a_pose/train/cli.py`

argparse (`--config`, `--seed`, `--run-root`); exit-code mapping per contract;
prints run dir, first/final losses and headline metrics.

### Packaging, configs, CI

- `pyproject.toml`: `strike-a-pose-train = "strike_a_pose.train.cli:main"`.
- `configs/ae.yaml` (full dataset defaults), `configs/ae_fixture.yaml` (fixture:
  batch 4, 10 epochs, `context_pca_k 4`, figure `test`, 4 rows).
- `ci.yml`: replace the 005 placeholder with `strike-a-pose-train --config
  configs/ae_fixture.yaml --run-root "$RUNNER_TEMP/runs"`.

## Project Structure

```text
src/strike_a_pose/eval/report.py          # new (moved helpers)
src/strike_a_pose/eval/baselines.py       # uses report helpers
src/strike_a_pose/models/__init__.py      # new
src/strike_a_pose/models/autoencoder.py   # new
src/strike_a_pose/train/__init__.py       # new
src/strike_a_pose/train/config.py         # new
src/strike_a_pose/train/trainer.py        # new
src/strike_a_pose/train/cli.py            # new
configs/ae.yaml, configs/ae_fixture.yaml  # new
tests/test_eval_report.py                 # new (R0 regression)
tests/test_model.py                       # new
tests/test_train_config.py                # new
tests/test_train.py                       # new (overfit + fixture CLI)
docs/results/004-ae-smpl_proc_v1/         # new (metrics, run_info, config, history, figure)
pyproject.toml, .github/workflows/ci.yml, README.md, specs/roadmap.md
```

## Complexity Tracking

None. (Moving 003 figure/scoring helpers into `eval/report.py` is a
behaviour-preserving refactor; the 003 fixture byte test guards it.)
