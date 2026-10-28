# Implementation Plan: Latent Capacity (128-d)

**Branch**: `claude/charming-wozniak-5ouldf` | **Date**: 2026-10-28 | **Spec**: [spec.md](spec.md)

## Summary

Run the existing AE at `latent_dim` 128 on `smpl_proc_v3`: L1 plain (capacity alone), L2 + pose
λ 0.03 (does the +36 % pixel cost shrink?), L3 + λ 0.01 only with `RUN_L3=1`. The wiring is
already generic (research R1–R2); the slice adds configs, fixture tests proving every consumer
handles 128, and a queue. Compare against 64-d with `promotion.py` and `pose_track.py --split
both`, 018 P1 / P2 as extra rows.

## Technical Context

Python ≥ 3.11, PyTorch CPU, pytest, ruff. No dependency. 4 vCPU, 12 GB gate, ≤ 18 h.

## Constitution Check

- I Library-first: no new logic. PASS. II Config-defined: three configs + queue. PASS.
- III Reproducibility: seed 0, as the 64-d references. PASS. IV Test-first: B1 tests first. PASS.
- V Honest evaluation: probe on frozen train latents, val selection on pixel MSE, test once,
  015 rule unchanged, capacity effect judged against measured 64-d seed sd. PASS.
- VI Simplicity: one latent size, two (three) runs. PASS.

## Files

- Lead (committed with this design): `configs/smpl_proc_v3/ae_l128.yaml`,
  `ae_l128_pose_l0.03.yaml`, `ae_l128_pose_l0.01.yaml`, `docs/campaign-scripts/latent019_queue.sh`.
- B1 (implementer): `tests/test_latent_capacity.py` (new), `tests/test_train_config.py`
  (parametrize extension only).
- B2 (implementer): `docs/HANDOFF.md` (019 campaign section).

## Conflict section (parallel slice 020, reconstruction fidelity)

- 019 touches **no** file under `src/`. Slice 020 owns `train/objective.py`, decoder in
  `models/autoencoder.py`, and likely `train/config.py`; 019 must not edit them (FR-005).
- Shared files: `tests/test_train_config.py` (both may extend the `test_shipped_configs_load`
  parametrize list: append-only lines, rebase resolves by keeping both), `docs/HANDOFF.md`
  (separate sections), `specs/roadmap.md` (separate entries), `data/smpl_queue.lock` (one queue
  at a time on the machine: the queues serialize by the shared lock, not run in parallel).
- If 020 changes a default in `TrainConfig` (e.g. a new loss key), 019 configs omit that key and
  so get the default; 020 must keep defaults equal to the current behaviour (its own LEGACY
  rule) or 019's runs stop being comparable to 64-d. Recorded as a dependency check in T030.

## Campaign

Copy to `data/latent019_queue.sh`; run under `LOG=data/latent019_queue.log
QUEUE=data/latent019_queue.sh data/watchdog.sh`. Order L1, mem gate, L2, (L3). After
`QUEUE_DONE` the lead runs:

```
uv run --locked python scripts/promotion.py --split both --r0 <R0-v3 run> --r0-probe <probe> --r0b <R0b-v3 run> \
  --r0b-probe <probe> \
  --variant L1 <L1 run> <L1 probe> --variant L2 ... [--variant L3 ...] \
  --variant P1 <018 P1 run> <probe> --variant P2 <018 P2 run> <probe>
uv run --locked python scripts/pose_track.py --split both --seed <R0-v3> <probe> \
  --seed <R0b-v3> <probe> --seed <R0c-v3> <probe> --variant L1 .. --variant L2 .. \
  --variant P1 .. --variant P2 ..
```

(exact flag names as in `scripts/promotion.py --help` / `docs/results/018-pose-supervised/`),
and writes `docs/results/019-latent-capacity/README.md` with the SC-003 readings.
Duration: L1 + L2 ≈ 12.5–14.5 h; with L3 ≈ 19–22 h.
