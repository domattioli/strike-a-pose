# Implementation Plan: Pose-Supervised Autoencoder

**Branch**: `claude/charming-wozniak-5ouldf` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

## Summary

Add an optional MLP pose head (latent → 256 → 256 → 72 root-relative world joints) trained
jointly with the AE: loss = reconstruction + `pose_weight` · standardized-joint MSE. Off by
default (bitwise the current trainer). The head lives in its own checkpoint keys so probe /
sample / round-trip are untouched; its val / test MPJPE is recorded next to (never instead of)
the frozen-latent probe. A pose-track report ranks variants by probe MPJPE against the
three-seed noise; the 015 promotion rule is unchanged. Campaign: λ 0.003, λ 0.03, λ 0.03 +
mask α 4 on `smpl_proc_v3`, ≈ 17 h.

## Technical Context

Python ≥ 3.11, PyTorch CPU, pytest, ruff (line 100). No new dependency. 4 vCPU, 12 GB memory
gate. Reuses `smpl_proc_v3` data / split, R0-v3 / R0b-v3 / R0c-v3 runs and probes as references.

## Constitution Check

- I Library-first: head, stats and loss in `models/pose_head.py`; ranking in
  `eval/pose_track.py`; scripts thin. PASS.
- II CLI / config: one config key; three committed configs; queue script. PASS.
- III Reproducibility: head built only when on, from the seeded global RNG; target stats saved
  in checkpoints; resume equals uninterrupted (test). PASS.
- IV Test-first: every batch red first. PASS.
- V Honest evaluation: probe fit on frozen train latents, head reported separately, val for
  selection (pixel MSE only), test once; promotion rule unchanged; real-gain bar from measured
  seed noise; the 80 mm target is reported, not tuned toward. PASS.
- VI Simplicity: one target, one loss, two λ, one combination; no VAE combination. PASS.

## Files

- B1: `src/strike_a_pose/models/pose_head.py` (new), `src/strike_a_pose/models/__init__.py`,
  `tests/test_pose_head.py` (new).
- B2: `src/strike_a_pose/train/config.py`, `train/objective.py`, `train/trainer.py`,
  `tests/test_train_pose.py` (new), `tests/test_train_config.py` (shipped-config parametrize).
- B3: `src/strike_a_pose/eval/pose_track.py` (new), `scripts/pose_track.py` (new),
  `tests/test_pose_track.py` (new).
- B4 (lead, committed with this design; implementer only validates): `configs/smpl_proc_v3/
  ae_pose_l0.003.yaml`, `ae_pose_l0.03.yaml`, `ae_pose_l0.03_mask.yaml`,
  `docs/campaign-scripts/pose018_queue.sh`; implementer edits `docs/HANDOFF.md`.

## Campaign

Copy the queue to `data/pose018_queue.sh`, run under
`LOG=data/pose018_queue.log QUEUE=data/pose018_queue.sh data/watchdog.sh` (the watchdog's
loading-phase leniency keys on the trainer's `run dir` / `resuming` lines in `$LOG`, which the
queue inherits). After `QUEUE_DONE` the lead runs `promotion.py --split both` and
`pose_track.py --split both` and writes `docs/results/pose-campaign-018/README.md`.
