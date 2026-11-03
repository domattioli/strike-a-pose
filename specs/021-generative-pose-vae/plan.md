# Implementation Plan: Generative Pose VAE

**Branch**: `021-generative-pose-vae` | **Date**: 2026-11-03 | **Spec**: [spec.md](spec.md)

## Summary

Allow the pose head (and, untested in the campaign, the gradient term) on a VAE: one loss
`recon [+ γ·GDL] [+ β·KL/(H·W)] [+ λ·pose(head(μ))]`, pre-021 combinations bitwise unchanged.
Add a fixed-threshold goal report (`eval/generative.py`, `scripts/generative.py`). Campaign on
`smpl_proc_v3`: V1 128-d VAE β 0.1 + head λ 0.03, V2 128-d VAE β 0.1, prior samples and probe
round trips of both, retroactive round trip of R2b-v3; ≈ 15 h.

## Technical Context

Python ≥ 3.11, PyTorch CPU, pytest, ruff (line 100). No new dependency. 4 vCPU, 12 GB memory
gate. Reuses `smpl_proc_v3`, L1 (sample reference), L2 + seed1 (pixel bar), R0 seeds
(informational pixel column), R2b-v3 sample + probe.

## Constitution Check

- I Library-first: verdict logic in `eval/generative.py`; `scripts/generative.py` thin. PASS.
- II CLI / config: no new key; six committed configs + one retroactive; queue script. PASS.
- III Reproducibility: no new RNG draw; pre-021 loop bitwise (frozen-copy test); VAE + pose
  repeats and resumes equal (tests). PASS.
- IV Test-first: both batches red first. PASS.
- V Honest evaluation: thresholds fixed in the spec before any run; selection stays val pixel
  MSE; head scored on μ as trained; test scored once; V2 control isolates the head's effect on
  sampling; the goal-line pixel reading is printed beside the operator's bar, never hidden. PASS.
- VI Simplicity: two deleted refusals, one reordered block, one ~150-line report. PASS.

## Files

- B1: `src/strike_a_pose/train/config.py`, `src/strike_a_pose/train/trainer.py`,
  `tests/test_train_vae_pose.py` (new), `tests/test_train_pose.py` (drop one parametrize case),
  `tests/test_train_grad.py` (drop one parametrize case), `tests/test_train_config.py`
  (parametrize the two new train configs).
- B2: `src/strike_a_pose/eval/generative.py` (new), `scripts/generative.py` (new),
  `tests/test_generative.py` (new).
- Lead (committed with this design): the seven configs under `configs/smpl_proc_v3/`
  (FR-005..FR-007), `docs/campaign-scripts/vae021_queue.sh`, HANDOFF "## 021" section, roadmap
  bullet. Implementer only validates (T030).

## Conflict section

`docs/HANDOFF.md`, `README.md`, `specs/roadmap.md` are edited concurrently in another worktree.
021 adds only a new `## 021` section at the end of HANDOFF and appends one roadmap bullet; on
rebase keep both sides. 021 never edits README.md. `train/trainer.py` / `config.py`: if another
branch changed `train_epoch`, re-apply the data-model.md block on top and re-run T010.

## Campaign

After B1–B2 merged green and no other queue holds `data/smpl_queue.lock`:

    cp docs/campaign-scripts/vae021_queue.sh data/vae021_queue.sh
    LOG=data/vae021_queue.log QUEUE=data/vae021_queue.sh data/watchdog.sh

Order V1 → probe → 12 GB gate → V2 → probe → samples (V1, V2) → round trips (V1, V2, R2b-v3).
After its done line the lead runs:

    uv run --locked python scripts/generative.py report \
      --v1 <V1> <V1 sample> <V1 round trip> --v2 <V2> <V2 sample> <V2 round trip> \
      --l2 runs/ae-l128-pose-l0.03-smpl_proc_v3-20261028T101221Z \
      --l2 runs/ae-l128-pose-l0.03-seed1-smpl_proc_v3-20261029T055258Z \
      --r0 runs/ae-smpl_proc_v3-20261023T004930Z --r0 runs/ae-seed1-smpl_proc_v3-20261023T065844Z \
      --r0 runs/ae-seed2-smpl_proc_v3-20261024T041446Z \
      --extra R2b-v3 runs/ae-vae-beta0.1-smpl_proc_v3-20261024T065708Z \
        runs/sample-vae-beta0.1-smpl_proc_v3-20261024T143458Z <R2b-v3 round trip> \
      --split both --json docs/results/021-generative-pose-vae/verdict.json

plus `pose_track.py --split both` (V1 next to L2) and writes
`docs/results/021-generative-pose-vae/README.md` with `samples.png` of V1 / V2.

## Complexity Tracking

None.
