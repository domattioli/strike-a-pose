# Implementation Plan: Latent Probe

**Branch**: `claude/charming-wozniak-5ouldf` (slice `006-latent-probe`) | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)
**Input**: `specs/006-latent-probe/spec.md` (approved by surrogate ruling 2026-10-20, operator sign-off pending)

## Summary

Add `strike_a_pose.eval.probe`: load a 004 checkpoint, encode every split once
(trained encoder + optional seeded random encoder) into a hash-keyed latent cache,
fit closed-form ridge and small MLP probes on train latents for root-relative 3D
joints (world + camera frame), camera yaw (8-way) and SMPL pose, select their one
hyper-parameter on val, score val and test once, and write a 003-style run dir with
the mean-pose / pose-PCA / majority baselines side by side plus two PIL figures.
New subcommand `strike-a-pose-eval probe`. Separately (surrogate ruling #3), the 004
trainer additively saves `best.pt` at each new val-loss minimum.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.11 + 3.13)
**Primary Dependencies**: existing only — torch (`linalg.solve`, `nn`, `optim`),
numpy, pillow, pyyaml. No scikit-learn (spec).
**Storage**: reads the 001 dataset (002 loader) and a 004 checkpoint; latent cache
and run dirs under git-ignored `runs/`; summaries in `docs/results/006-probe-smpl_proc_v1/`.
**Testing**: pytest — ridge closed form vs lstsq, geodesic angle and yaw scoring on
hand cases, config validation, fixture CLI end-to-end (fixture AE trained in the
test), cache hit equality, repeatability, exit codes; `best.pt` + resume test.
**Performance Goals**: full run ≤ 30 min CPU (measured encoder 750 img/s →
≈ 25 s / encoder for 16,000 images; loading ≈ 60 s; 24 MLP fits ≈ seconds each);
fixture CLI < 60 s; CI leg < 5 min.
**Constraints**: train-only fit; val-only selection; test once; AE checkpoint and
dataset read-only; final-epoch weights.

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | All logic in `src/strike_a_pose/eval/probe.py`; CLI thin. |
| II CLI + config | PASS | `strike-a-pose-eval probe --config configs/probe.yaml`; every knob in YAML; overrides written into `config.yaml`. |
| III Reproducibility | PASS | Seeds set + logged; ridge closed form in float64; MLP seeded; run_info records commit, env, dataset / split / checkpoint hashes; cache keyed by hashes; repeatability tested. |
| IV Test-first | PASS | Ridge, angle, yaw metrics, config and CLI tests written with the code; fixture smoke in pytest (+ CI step if time allows). |
| V Honest evaluation | PASS | Frozen split; probes and baselines fit on train only; val selection; test scored once; baselines (mean, PCA oracle, majority) in the same `metrics.json`; random-encoder control; docs cite the run. |
| VI Simplicity | PASS | One module + config, closed-form ridge, one small MLP class, no new deps, no new console script. |

## Design

- `eval/probe_config.py`: `ProbeConfig` frozen dataclass, strict `config_from_dict`,
  `load_config`, `with_overrides` (pattern of `eval/config.py`, reusing `ConfigError`).
- `eval/probe.py`:
  - `extract(ds, encoders, batch_size)` → labels + latents per encoder (one pass over
    the loader; encoders in eval mode, channels_last, `no_grad`).
  - `load_or_extract(...)` + cache I/O (atomic write).
  - `Standardizer` (train mean / std, floor 1e-8); `ridge_fit(z, y, lam)` /
    `ridge_predict` (float64 `linalg.solve`).
  - `MLPProbe(nn.Module)`; `fit_mlp(z, y, kind, ...)` (MSE on standardized targets
    or cross-entropy; Adam + cosine per step; seeded).
  - Metrics: `rotation_angle_deg(pred_aa, true_aa)` (Rodrigues + geodesic angle),
    `yaw_scores(pred, true, n_classes, step_deg)`; joints via 003 `mpjpe` and
    `keypoint_error_2d` / `project_points`.
  - `run_probe(cfg, argv)`: check inputs (datasets, checkpoint, image size) → run
    dir → latents → per target: baselines, ridge grid, MLP grid, select on val,
    score eval splits → figures → `metrics.json` → `run_info.json`.
- `eval/cli.py`: `probe` subparser and headline printout.
- `train/trainer.py` (ruling #3, additive): after each epoch with a val loss below
  every earlier one, write `best.pt` (`{"model", "model_hparams", "epoch",
  "val_loss", "train_version", "config"}`, atomic) before `checkpoint.pt`;
  `metrics.json.training` and `run_info.json` gain `best_val_epoch` /
  `best_val_loss` (null without val). On resume, the best so far is recomputed from
  the checkpointed history, so resumed == uninterrupted. `best.pt` loads with
  `load_checkpoint`.
- Configs: `configs/probe.yaml` (full), `configs/probe_fixture.yaml` (fixture; tiny
  MLP, no cache; checkpoint passed with `--checkpoint`).
- CI: step after the AE smoke train: `strike-a-pose-eval probe --config
  configs/probe_fixture.yaml --checkpoint <fixture run>/checkpoint.pt`, if the leg
  stays < 5 min.

## Project Structure

```text
src/strike_a_pose/eval/probe.py, probe_config.py   # new
src/strike_a_pose/eval/cli.py                      # + probe subcommand
src/strike_a_pose/train/trainer.py                 # + best.pt (additive)
configs/probe.yaml, configs/probe_fixture.yaml     # new
tests/test_probe.py, tests/test_probe_config.py    # new
tests/test_train.py                                # + best.pt / resume test
docs/results/006-probe-smpl_proc_v1/               # new (metrics, run_info, config, figures)
.github/workflows/ci.yml, README.md, specs/roadmap.md, specs/004-autoencoder-mvp/contracts/train-run.md
```

## Complexity Tracking

None. The random-encoder control (FR-010) is one extra encoder in the same pass and
the same probe code; it is the cheapest way to make the probe numbers interpretable.
