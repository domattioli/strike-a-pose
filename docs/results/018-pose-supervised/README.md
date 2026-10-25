# Slice 018 campaign (2026-10-25): pose-supervised autoencoder on smpl_proc_v3

Seeds for the pose track: R0-v3, R0b-v3, R0c-v3 (plain AE, three seeds; results in `docs/results/smpl-campaign-015/`). Target: ≤ 80 mm world MPJPE (grill point 8). Promotion: 015 rule unchanged (pixel gate); pose track reported separately (spec 018 FR-006).

| Run | Result |
|---|---|
| **P1** λ 0.003 (`ae-pose-l0.003-smpl_proc_v3-20261025T001408Z`) | early-stopped at epoch 58 (best val epoch 48); val / test MSE 0.00167 / 0.00165 (+5.7 % / +5.9 % vs seeds), IoU 0.898; **pose head 90.9 / 88.0 mm**; **frozen-latent probe 91.2 / 88.8 mm** (seeds 132.8 / 129.1; −41.7 mm = −24.8 sd, real); random-encoder control 211.8 mm. 015 verdict **DROPPED** on val and test (MSE just over the +5 % latent-win bound). O2 second seed: no |
| **P2** λ 0.03 (`ae-pose-l0.03-smpl_proc_v3-20261025T061249Z`) | early-stopped at epoch 53 (best val epoch 43); val / test MSE 0.00215 / 0.00212 (+36 % vs seeds), IoU 0.870; **pose head 68.6 / 66.6 mm**; **frozen-latent probe 68.6 / 66.6 mm** (−64.2 mm = −38 sd); both **≤ 80 mm** on val and test; peak RSS 6.0 GB. 015 verdict **DROPPED** (MSE). O2 second seed: no |
| **P3** λ 0.03 + mask α 4 (`ae-pose-l0.03-mask-smpl_proc_v3-20261025T123004Z`) | all 60 epochs (best val epoch 56); val / test MSE 0.00224 / 0.00221 (+42 %), IoU 0.831; pose head 71.9 / 69.9 mm; frozen-latent probe 73.0 / 70.9 mm; peak RSS 6.0 GB. 015 verdict **DROPPED**. Mask weighting on top of pose supervision helps neither pose (+4 mm vs P2) nor pixels |

## Pose track, P1–P3 (`scripts/pose_track.py --split both`)

seeds: n=3 mean 132.8 mm sd 1.7 mm (val)

| Rank | Variant | MSE vs seeds | Probe MPJPE val (mm) | Δ vs seeds (mm) | Δ / sd | Real (< −2 sd) | Head MPJPE val (mm) | Probe ≤ 80 | Head ≤ 80 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | P2 | +36.3 % | 68.6 | -64.2 | -38.1 | yes | 68.6 | yes | yes |
| 2 | P3 | +41.9 % | 73.0 | -59.8 | -35.5 | yes | 71.9 | yes | yes |
| 3 | P1 | +5.7 % | 91.2 | -41.7 | -24.8 | yes | 90.9 | no | no |

second seed for P2 (operator ruling O2, val): yes (within 3 sd of P3)

seeds: n=3 mean 129.1 mm sd 1.5 mm (test)

| Rank | Variant | MSE vs seeds | Probe MPJPE test (mm) | Δ vs seeds (mm) | Δ / sd | Real (< −2 sd) | Head MPJPE test (mm) | Probe ≤ 80 | Head ≤ 80 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | P2 | +35.8 % | 66.6 | -62.6 | -40.6 | yes | 66.6 | yes | yes |
| 2 | P3 | +41.7 % | 70.9 | -58.3 | -37.9 | yes | 69.9 | yes | yes |
| 3 | P1 | +5.9 % | 88.8 | -40.4 | -26.2 | yes | 88.0 | no | no |

second seed for P2 (operator ruling O2, test): yes (within 3 sd of P3)

All three variants: 015 promotion verdict DROPPED on val and test (pixel gate; P1 at +5.7 % misses the +5 % latent-win bound narrowly).

## Reading (pending the O2 second seed of P2)

- **Grill point 8 answered (ruling O1): yes.** With pose supervision at λ 0.03 the pose system reaches 68.6 / 66.6 mm (val / test), under the 80 mm target, and so does the frozen-latent probe (the encoder itself now carries the pose; ridge stays ~180 mm, so the information is non-linear). P2's yaw accuracy is 0.999; SMPL joint-angle error 20.0°.
- **Trade-off is steep and monotone:** λ 0.003 → −42 mm probe for +6 % MSE; λ 0.03 → −64 mm for +36 % MSE (IoU 0.905 → 0.870). Pose and pixels compete for the 64 latent numbers.
- **Mask weighting does not stack** with pose supervision (P3 worse than P2 on both).
- **O2:** P2 is within 3 sd of P3 (gap 4.4 mm vs 3 × 1.7), so a second seed of P2 (`ae-pose-l0.03-seed1-smpl_proc_v3`, ≈ 6 h) is queued to confirm the ranking. The ≤ 80 mm result does not depend on it (P2 and P3 are both under).
