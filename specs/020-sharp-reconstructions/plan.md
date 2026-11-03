# Implementation Plan: Sharper Reconstructions

**Branch**: `claude/charming-wozniak-5ouldf` | **Date**: 2026-10-28 | **Spec**: [spec.md](spec.md)

## Summary

Add an optional squared image-gradient term to the reconstruction loss (`gradient_weight` γ,
off by default, bitwise today's trainer), a boundary-F1 sharpness metric in every image score,
and a rescore + fidelity-report tool that compares variants with their references on MSE, IoU,
boundary F1, region MSE and probe MPJPE under a pre-registered verdict. Campaign on
`smpl_proc_v3`: F1 plain + γ 0.5, F2 pose λ 0.03 + γ 0.5, plus rescoring five existing
references; ≈ 14.5 h, after slice 019's campaign, same lock.

## Technical Context

Python ≥ 3.11, PyTorch CPU, pytest, ruff (line 100). No new dependency. 4 vCPU, 12 GB memory
gate. Reuses `smpl_proc_v3` data / split and the R0-v3, R0b-v3, R0c-v3, P2, P2-seed1 runs and
their probes.

## Constitution Check

- I Library-first: loss and metric in `eval/metrics.py`, rescore / report in
  `eval/fidelity.py`; `scripts/fidelity.py` thin. PASS.
- II CLI / config: one config key; two committed configs; queue script. PASS.
- III Reproducibility: the term draws no RNG; default runs bitwise unchanged (test);
  resume equals uninterrupted (test); rescore is deterministic (eval mode). PASS.
- IV Test-first: every batch red first. PASS.
- V Honest evaluation: selection stays val pixel MSE; the sharpness metric is not the trained
  quantity; verdict pre-registered with sd from measured seeds; probe on frozen train latents
  reported beside every fidelity number; test once. PASS.
- VI Simplicity: one loss term, one γ, two runs; no decoder change, no SSIM / VGG. PASS.

## Files

- B1: `src/strike_a_pose/eval/metrics.py`, `eval/__init__.py` (export), `eval/report.py`,
  `eval/baselines.py` (UNITS), `src/strike_a_pose/train/trainer.py` (`IMAGE_UNITS` key list
  only), `tests/test_boundary_f1.py` (new), `tests/test_eval_report.py` (extend).
- B2: `src/strike_a_pose/train/config.py`, `train/objective.py`, `train/trainer.py`,
  `tests/test_train_grad.py` (new), `tests/test_train_config.py` (shipped-config parametrize),
  `tests/test_train_pose.py` (only the `HISTORY_FIELDS` last-field assertion).
- B3: `src/strike_a_pose/eval/fidelity.py` (new), `scripts/fidelity.py` (new),
  `tests/test_fidelity.py` (new).
- B4 (lead, committed with this design; implementer only validates): `configs/smpl_proc_v3/
  ae_grad.yaml`, `ae_pose_l0.03_grad.yaml`, `docs/campaign-scripts/fidelity020_queue.sh`;
  implementer edits `docs/HANDOFF.md`.

## Conflict section (parallel slice 019: latent 128 + pose head)

Slice 019 (`specs/019-latent-capacity/`, committed `964fd0d`) states no `src/` change: configs
(`latent_dim` 128), `tests/test_train_config.py`, `tests/test_latent_capacity.py`, its queue
`latent019_queue.sh` (same lock), roadmap. Shared files both slices may edit; whoever lands second rebases and merges by hand, keeping
both keys / columns (never resolve by taking one side):

| File | 020 change | Likely 019 change |
|---|---|---|
| `src/strike_a_pose/train/config.py` | `gradient_weight` field, legacy value, validation | latent / head config keys |
| `src/strike_a_pose/train/objective.py` | `Objective.gradient_weight`, `from_config` arg | possible pose-head fields |
| `src/strike_a_pose/train/trainer.py` | `train_grad` column, loss term, `IMAGE_UNITS` | head / latent wiring, maybe history columns |
| `tests/test_train_config.py` | parametrize two 020 configs | parametrize 019 configs |
| `tests/test_train_pose.py` | `HISTORY_FIELDS[-1]` assertion | possibly the same file |
| `configs/smpl_proc_v3/` | two new files only | new files only |
| `docs/HANDOFF.md`, `specs/roadmap.md` | own paragraph / bullet | own paragraph / bullet |
| `data/smpl_queue.lock` | shared lock (by design) | shared lock |

Rules: `Objective` gains fields only at the end, by keyword in `from_config`
(`cls(..., pose_weight=cfg.pose_weight, gradient_weight=cfg.gradient_weight)`) so a 019
positional addition cannot silently shift 020's; `HISTORY_FIELDS` additions are appended. 020
never edits `models/autoencoder.py`, `models/pose_head.py` or `specs/019-*`. If 019 lands a
latent-128 change to `configs/smpl_proc_v3/ae.yaml` defaults, 020's configs pin
`latent_dim: 64` explicitly (they do) so F1 / F2 stay comparable to their references.

## Campaign

After the 019 queue prints its own terminal line: copy the queue to
`data/fidelity020_queue.sh`, run under `LOG=data/fidelity020_queue.log
QUEUE=data/fidelity020_queue.sh data/watchdog.sh`. If 019's queue still holds
`data/smpl_queue.lock`, the 020 queue logs "another queue holds the lock" and exits 0: re-run
it later (the watchdog restarts it). After its terminal line the lead runs
`scripts/fidelity.py report --split both`, `promotion.py --split both`, `pose_track.py --split
both`, and writes `docs/results/020-sharp-reconstructions/README.md` with the R0-v3 / F1 and
P2 / F2 `recon_val.png` side by side (same six samples).

## Complexity Tracking

None.
