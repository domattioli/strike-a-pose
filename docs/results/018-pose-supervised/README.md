# Slice 018 campaign (2026-10-25): pose-supervised autoencoder on smpl_proc_v3

Seeds for the pose track: R0-v3, R0b-v3, R0c-v3 (plain AE, three seeds; results in `docs/results/smpl-campaign-015/`). Target: ≤ 80 mm world MPJPE (grill point 8). Promotion: 015 rule unchanged (pixel gate); pose track reported separately (spec 018 FR-006).

| Run | Result |
|---|---|
| **P1** λ 0.003 (`ae-pose-l0.003-smpl_proc_v3-20261025T001408Z`) | early-stopped at epoch 58 (best val epoch 48); val / test MSE 0.00167 / 0.00165 (+5.7 % / +5.9 % vs seeds), IoU 0.898; **pose head 90.9 / 88.0 mm**; **frozen-latent probe 91.2 / 88.8 mm** (seeds 132.8 / 129.1; −41.7 mm = −24.8 sd, real); random-encoder control 211.8 mm. 015 verdict **DROPPED** on val and test (MSE just over the +5 % latent-win bound). O2 second seed: no |
| **P2** λ 0.03 (`ae-pose-l0.03-smpl_proc_v3-20261025T061249Z`) | early-stopped at epoch 53 (best val epoch 43); val / test MSE 0.00215 / 0.00212 (+36 % vs seeds), IoU 0.870; **pose head 68.6 / 66.6 mm**; **frozen-latent probe 68.6 / 66.6 mm** (−64.2 mm = −38 sd); both **≤ 80 mm** on val and test; peak RSS 6.0 GB. 015 verdict **DROPPED** (MSE). O2 second seed: no |
| **P3** λ 0.03 + mask α 4 (`ae-pose-l0.03-mask-smpl_proc_v3-20261025T123004Z`) | all 60 epochs (best val epoch 56); val / test MSE 0.00224 / 0.00221 (+42 %), IoU 0.831; pose head 71.9 / 69.9 mm; frozen-latent probe 73.0 / 70.9 mm; peak RSS 6.0 GB. 015 verdict **DROPPED**. Mask weighting on top of pose supervision helps neither pose (+4 mm vs P2) nor pixels |
| **P2-seed1** λ 0.03, seed 1, ruling O2 (`ae-pose-l0.03-seed1-smpl_proc_v3-20261025T205801Z`) | early-stopped at epoch 55 (best val epoch 45); val / test MSE 0.00215 / 0.00212, IoU 0.870; pose head 68.5 / 66.3 mm; probe 68.7 / 66.4 mm. Reproduces P2 within 0.3 mm |

## Final pose track (`scripts/pose_track.py --split both`)

seeds: n=3 mean 132.8 mm sd 1.7 mm (val)

| Rank | Variant | MSE vs seeds | Probe MPJPE val (mm) | Δ vs seeds (mm) | Δ / sd | Real (< −2 sd) | Head MPJPE val (mm) | Probe ≤ 80 | Head ≤ 80 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | P2 | +36.3 % | 68.6 | -64.2 | -38.1 | yes | 68.6 | yes | yes |
| 2 | P2-seed1 | +36.2 % | 68.7 | -64.2 | -38.1 | yes | 68.5 | yes | yes |
| 3 | P3 | +41.9 % | 73.0 | -59.8 | -35.5 | yes | 71.9 | yes | yes |
| 4 | P1 | +5.7 % | 91.2 | -41.7 | -24.8 | yes | 90.9 | no | no |

second seed for P2 (operator ruling O2, val): yes (within 3 sd of P2-seed1)

seeds: n=3 mean 129.1 mm sd 1.5 mm (test)

| Rank | Variant | MSE vs seeds | Probe MPJPE test (mm) | Δ vs seeds (mm) | Δ / sd | Real (< −2 sd) | Head MPJPE test (mm) | Probe ≤ 80 | Head ≤ 80 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | P2-seed1 | +36.0 % | 66.4 | -62.7 | -40.8 | yes | 66.3 | yes | yes |
| 2 | P2 | +35.8 % | 66.6 | -62.6 | -40.6 | yes | 66.6 | yes | yes |
| 3 | P3 | +41.7 % | 70.9 | -58.3 | -37.9 | yes | 69.9 | yes | yes |
| 4 | P1 | +5.9 % | 88.8 | -40.4 | -26.2 | yes | 88.0 | no | no |

second seed for P2-seed1 (operator ruling O2, test): yes (within 3 sd of P2)

(The O2 line now compares P2 with its own second seed; the ruling is satisfied, no third run.)

All variants: 015 promotion verdict DROPPED on val and test (pixel gate; P1 at +5.7 % misses the +5 % latent-win bound narrowly).

## Summary (QUEUE_DONE 2026-10-26 02:41 UTC)

- **Grill point 8 answered (ruling O1): yes.** Pose supervision at λ 0.03 gives a pose system at **68.6 / 66.5 mm** (val / test, mean of two seeds; seed spread 0.1–0.3 mm), under the 80 mm target. The frozen-latent probe matches (68.6–68.7 / 66.4–66.6 mm), so the encoder itself carries the pose; a linear probe stays ≈ 180 mm, so it is non-linear. Yaw 0.999, SMPL joint-angle error 20°.
- **The price is pixels.** λ 0.003: −42 mm probe for +6 % MSE (IoU 0.898). λ 0.03: −64 mm for +36 % MSE (IoU 0.870). The 64-number latent is shared between pose and appearance.
- **Mask weighting does not stack** with pose supervision: P3 is 4.4 mm worse than P2 (2.6 seed sd, against P2's two-seed mean) and 4 % worse in MSE.
- **No memory issue:** peak RSS ≈ 6.0 GB for every run.
- **Operator decision open:** the 015 pixel gate drops every pose variant. Recommended: a separate pose-track promotion (probe gain > 2 seed sd, pixel cost reported) so pose-oriented variants can be carried forward.
