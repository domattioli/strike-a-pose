# Research: Generative Pose VAE (slice 021)

## R1 Why a VAE with the head (and not L2 + a fitted prior)

L2 meets pose (67 / 65 mm) but is a plain AE: `ae_prior` samples (N(0,I) through a plain
decoder) are 0 % valid on v3. A post-hoc density on L2's latents (GMM / flow) would be a second
model and an extra slice of choices; the operator scoped one model whose own prior generates.
The VAE β 0.1 at 64-d already gives 59.0 % valid prior samples (R2b-v3), so the question is
whether a 128-d β 0.1 VAE keeps that with a pose head pulling μ toward joint-linear structure.

## R2 Head on μ, not z

018 / 019 train the head on μ (deterministic). Pose evaluation (`pose_head_scores`) encodes
with `model.encode` = μ for a VAE, so training the head on μ trains exactly what is scored.
Head-on-z (noisy input) would make the head robust to posterior noise and may tie pose
information to the region the prior covers — a candidate if V1 fails on generation, not now.

## R3 Loss composition and bitwise defaults

Pre-021 `train_epoch` composes `recon (+γ·GDL) (+λ·pose)` for the AE and `recon + β·KL/(H·W)`
for the VAE. The unified form adds `+β·KL/(H·W)` after the optional GDL and before the optional
pose term. For the AE paths (no KL) and for the VAE path (no GDL, no pose) every float op is
identical in operands and order; PoseHead construction stays gated on `pose_weight > 0`. Hence
the frozen-loop test (tasks T010) compares weights with `torch.equal`.

## R4 Units of the bars

- 0.56: n 1024, p 0.59 → sd √(0.59·0.41/1024) = 0.01537; 0.59 − 2·0.01537 = 0.5593 → 0.56.
- 13 points: R2-v3 (β 0.01) prior plausible 0.457031 vs control 0.326172 = 13.09 points; the
  bar asks V1 to be at least as far above chance as the best measured VAE round trip.
- Pixel: L2 two-seed mean ×1.05 → val 0.00175955, test 0.00173606. Goal-line reading (R0
  three-seed mean ×1.05) → val 0.00165827, test 0.00163649 (informational).

## R5 Budget

L2 trained 6.4 h (10:12 → 16:34 UTC 2026-10-28); a VAE adds one `Linear(flat, 128)` and the KL
(negligible). V1 ≈ 6.5 h, V2 ≈ 6.5 h (early stop may shorten), two probes ≈ 0.5 h each, two
sample runs ≈ 0.3 h, three round trips ≈ 0.3 h each → ≈ 15 h of the 18 h window.

## R6 Memory

Peak RSS of 128-d runs ≈ 6 GB; β 0.1 on v1 once peaked 11.6 GB without explanation, so the
12 GB gate after V1 stops the queue before V2 if V1 exceeded it.

## R7 Rejected

β sweep (budget), head on z (named follow-up), GDL run (022), fitted latent prior for L2 (second
model), larger latent (cap 128).
