# Feature Specification: Latent Capacity (128-d)

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `019-latent-capacity`)
**Created**: 2026-10-28
**Status**: Approved at surrogate scope; operator decision 2026-10-28 ("test a larger latent so
pose and appearance stop competing") is the approval basis. Clarification batch 1 answered by
surrogate ruling 2026-10-28 (below); the operator may overrule.
**Input**: 018 showed the 64-d latent is shared between pose and appearance: plain AE on
`smpl_proc_v3` val MSE 0.00158, probe 132.8 mm (3 seeds, sd 1.7 mm, MSE sd 0.46 %); pose head
λ 0.003: 91 mm at +6 % MSE; λ 0.03: 68.6 mm at +36 % MSE. Question: with `latent_dim` 128, does
(i) capacity alone change pixels / probe, and (ii) the λ 0.03 pixel cost shrink?

Settled context (measured at `dd61515`, see [research.md](research.md) R1–R3):

- `latent_dim` is a single config int wired generically: `ConvAutoencoder` builds
  `Linear(flat, latent_dim)` / `Linear(latent_dim, flat)` with flat = 256 × 4 × 3 = 3072, so
  128 is far below the bottleneck (no structural limit below 3072). Params 3,126,897 → 3,520,177
  (+393,280, +12.6 %, all in the two Linear layers); forward shape unchanged.
- `PoseHead(latent_dim)` already takes the size: 100,936 → 117,320 params.
- Probe (`ckpt` hparams → `ConvAutoencoder`, latents `[n, m.latent_dim]`, ridge / MLP input
  `d`), sample (`vae.latent_dim`), round trip (`vae.latent_dim`), pose-track and promotion
  scripts carry no 64 constant. No code change is expected; this slice proves it with tests.

## Clarifications

### Session 2026-10-28 (batch 1, surrogate rulings; operator may overrule)

- Q1 Latent size: 96, 128, 256? **Ruling 128.** Doubling is the smallest step whose effect
  should exceed seed noise; 256 doubles again the Linear params (+1.2 M) for an untested
  premise. 256 is a follow-up if 128 shows the competition easing (research R4).
- Q2 Variants: **L1 plain 128-d** (λ 0, capacity alone; pixels vs R0-v3 three-seed mean) and
  **L2 128-d + λ 0.03** (pixel cost vs P2's +36 %; probe vs P2's 68.6 mm). **L3 128-d + λ 0.01**
  is queued third but conditional (Q5). One seed each (seed 0, as R0-v3 / P2).
- Q3 Everything else fixed: channels `[16, 32, 64, 128, 256, 256]`, 60 epochs, cosine, batch 64,
  lr 1e-3, early stop as 018, no mask weighting (P3 showed it does not stack), not VAE.
- Q4 Comparison: (i) `scripts/promotion.py --split both` with R0-v3 / R0b-v3 references,
  015 rule unchanged; (ii) `scripts/pose_track.py --split both` with seeds R0-v3, R0b-v3, R0c-v3
  and variants L1, L2 (L3) **plus** the 018 P1 / P2 rows so 64-d and 128-d sit in one table.
  The 64-d seeds stay the reference: there are no 128-d seeds; L1's own MSE delta is read
  against the 64-d MSE sd (0.46 %): a pixel effect of capacity is "real" only at > 2 sd
  (|Δ| > 0.92 %). Pixel-cost comparison for L2: `ΔMSE(L2 vs L1)` against `ΔMSE(P2 vs R0-v3
  mean)` = +36 %; "cost shrank" iff L2-vs-L1 ≤ +18 % (half), reported, not a gate.
- Q5 Budget: ≤ 18 h CPU. L1 + L2 ≈ 2 × (5.5–6.5 h + 45 min probe) ≈ 12.5–14.5 h. L3 adds
  ≈ 6.25–7.25 h → 18.75–21.75 h, over budget. **Ruling: L3 runs only with `RUN_L3=1`**
  in the queue environment (default off); operator question 1.
- Q6 Weight selection: unchanged, best val pixel MSE (`eval_loss`).
- Q7 Memory: +393 k params and 2× latent tensors are negligible against the ≈ 6.0 GB peak of
  018; the 12 GB gate after the first run stays.
- Q8 Code scope: no change to `train/config.py`, losses or the decoder (slice 020 owns
  reconstruction fidelity). This slice ships configs, a queue script and tests only.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train and evaluate a 128-d AE from a config (P1)

The researcher runs `strike-a-pose-train --config configs/smpl_proc_v3/ae_l128.yaml`; the run
trains, scores, and the probe, sampling and round-trip consumers accept its checkpoint.

**Independent Test**: fixture runs at `latent_dim: 128` train, probe and (for a VAE fixture)
sample / round-trip without error; recorded `latent_dim` is 128.

### User Story 2 - Compare 128-d against 64-d (P1)

After the queue, the lead runs promotion and pose-track with both widths in one table.

**Independent Test**: `pose_track.py` already accepts arbitrary `--variant` rows (018 tests).

### Edge Cases

- A 128-d checkpoint passed as `reference_checkpoint` with a 64-d VAE to sample: refused
  (existing `ConfigError`, test pins the message).
- Probe cache from a 64-d run is never reused for 128-d (cache is per probe run dir; test).

## Requirements *(mandatory)*

- **FR-001** Ship configs `configs/smpl_proc_v3/ae_l128.yaml` (λ 0), `ae_l128_pose_l0.03.yaml`,
  `ae_l128_pose_l0.01.yaml`: copies of `ae.yaml` / `ae_pose_l0.03.yaml` with `latent_dim: 128`
  and distinct `run_name`s `ae-l128-smpl_proc_v3`, `ae-l128-pose-l0.03-smpl_proc_v3`,
  `ae-l128-pose-l0.01-smpl_proc_v3`.
- **FR-002** Tests prove the 128-d path end to end on the fixture: train (λ 0 and λ > 0), probe,
  sample, round trip; and that shipped configs load with `latent_dim == 128`.
- **FR-003** Queue `docs/campaign-scripts/latent019_queue.sh` (pattern of `pose018_queue.sh`,
  exit-143 fix kept, terminal tokens only from `log`), L3 gated by `RUN_L3=1`.
- **FR-004** Results in `docs/results/019-latent-capacity/README.md` (lead, after the queue):
  promotion and pose-track tables, the Q4 pixel readings.
- **FR-005** No source change under `src/` unless a test in FR-002 fails; then the minimal fix,
  restricted to `models/`, `eval/` (never `train/config.py`, `train/objective.py`, decoder).

## Success Criteria

- **SC-001** Full suite green; FR-002 tests run in < 60 s total on CPU.
- **SC-002** Campaign: L1, L2 trained and probed in ≤ 15 h; peak RSS ≤ 12 GB.
- **SC-003** The README states, per split: L1 ΔMSE vs 64-d seed mean with the 2 sd reading;
  L2 ΔMSE vs L1 next to P2's +36 %; L2 probe / head vs P2.
