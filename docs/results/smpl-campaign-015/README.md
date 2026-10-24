# Slice 015 campaign (2026-10-22/23): 8,000 poses, VAE β 0.1, sampling

| Run | Result |
|---|---|
| VAE β 0.1, v1 (`ae-vae-beta0.1-smpl_proc_v1-20261022T225733Z`) | val / test MSE 0.00220 / 0.00219 (+1.3 % / +1.5 % vs R0); KL ≈ 167 nats/image at the best epoch 44, 169.4 at the last logged epoch (β 0.01: 274); note: history.csv ends at epoch 54 although metrics.json reports 60 epochs, so the final-epoch KL check cannot be reproduced from the artifact; probe world MPJPE 176.0 / 175.6 mm (R0 174.8 / 175.3): reconstruction and pose information kept |
| Sampling, VAE β 0.1 (`sample-vae-beta0.1-*`) | N(0, I) prior samples valid (one body-sized connected silhouette): **33.1 %** of 1,024; NN-MSE to train 0.0101 (blank image 0.0126, VAE reconstructions 0.0083–0.0091, so samples are only ≈ 20 % closer to train than a blank). Bars: plain-AE decoder on prior samples 0 % (all empty); VAE reconstructions of val 88.4 % |
| Sampling, VAE β 0.01 (`sample-vae-smpl_proc_v1-*`) | prior samples valid 5.7 % (mostly too small); reconstructions 85.6 % |
| baselines v3 (8,000 poses) | peak RSS 3.8 GB (uint8 fix) |

Reading: raising β to 0.1 makes the prior 6× more generative (5.7 % → 33 % valid samples) at ≈ 1 % reconstruction cost and no measurable pose-information loss. Two thirds of prior samples still fail, mostly as fragmented silhouettes (582 / 1,024).
| **R0-v3** plain AE, 8,000 poses (6,400 train) | ran all 60 epochs (best val epoch 56; val flat over the last ≈ 15 epochs as the cosine LR annealed to 0); val / test MSE 0.00158 / 0.00156 (v2d 0.00178, v1 0.00218), IoU 0.905 / 0.906; region MSE val hands 0.025, feet 0.022, head 0.013; peak RSS 5.9 GB; resumed once across a container restart (epoch 35) |
| probe on R0-v3 `best.pt` | world-frame MPJPE MLP 131.0 / 127.5 mm vs mean pose 258.2 / 252.9 (−49 %); camera accuracy 0.981 / 0.983 |

Pose-count curve (plain AE): train poses 1,600 / 3,200 / 6,400 → val MSE 0.00218 / 0.00178 / 0.00158 (−18 %, then −11 %), probe 175 / 150 / 131 mm. Diminishing but not saturated.
| **R0b-v3** plain AE, seed 1 (noise bar) | val / test MSE 0.00159 / 0.00156 (+0.6 % / +0.5 % vs R0-v3); probe world MPJPE 134.4 / 130.5 mm (R0-v3 131.0 / 127.5). **v3 noise bars: seed gap 0.6 % MSE, probe gap 3.3 mm**. v2d→v3 gain −11.2 % ≈ 18× the seed gap (> 3×), so SC-002 holds |
| **R2-v3** VAE β 0.01, v3 (`ae-vae-smpl_proc_v3-20261023T132901Z`) | val / test MSE 0.00159 / 0.00157 (+0.7 % vs R0-v3); KL ≈ 262 nats/image; probe world MPJPE 132.3 / 128.4 mm (R0-v3 131.0 / 127.5). Verdict **REPORTED**: +0.7 % MSE is about the size of the seed gap (0.6 % val / 0.5 % test, just outside it); +1.3 mm probe is inside the 3.3 mm gap. No win, at most a small cost |
| **R1-v3** mask-weighted MSE, v3 (`ae-mask-smpl_proc_v3-20261023T195044Z`) | val / test MSE 0.00181 / 0.00179 (+14.5 % vs R0-v3), IoU 0.856; probe world MPJPE 124.5 / 121.2 mm (−6.58 mm vs R0-v3 = 1.97× the probe gap on val, just under the 2× latent bar; 2.05× on test); peak RSS 5.9 GB. Verdict **DROPPED** (MSE > +5 %). Lowest probe error of the campaign, but it fails the latent criterion on val independently of the MSE bound (passes on test); the val/test agreement on DROPPED comes from the MSE bound alone. Borderline until the third seed (grill point 1) |
| Sampling, R2-v3 (`sample-vae-smpl_proc_v3-*`) | VAE β 0.01 on v3: prior samples valid **32.4 %** of 1,024 (β 0.01 on v1: 5.7 %); fails 497 fragmented, 195 wrong area; NN-MSE 0.0103 (blank 0.0126, reconstructions 0.0083). Bars: plain-AE decoder 0 %; VAE reconstructions of val 93.2 % |

## Summary (QUEUE_DONE 2026-10-24 02:46 UTC)

| Variant | Val MSE vs R0-v3 | Probe val (mm) | Verdict |
|---|---|---|---|
| R0-v3 plain AE | 0.00158 | 131.0 | baseline |
| R0b-v3 seed 1 | +0.6 % | 134.4 | noise bar |
| R2-v3 VAE β 0.01 | +0.7 % | 132.3 | REPORTED |
| R1-v3 mask-weighted | +14.5 % | 124.5 | DROPPED |

- More poses is the biggest lever: 1,600 → 6,400 train poses gives val MSE −27 % and probe 175 → 131 mm, not yet saturated.
- The VAE costs at most about one seed gap on v3 (+0.7 % MSE) and, with 4× the poses, its prior becomes as generative at β 0.01 (32 %) as β 0.1 was on v1 (33 %).
- Mask weighting again trades pixels for pose information (−6.6 mm probe, borderline vs a two-seed noise bar; +14.5 % MSE); the promotion rule drops it. Whether that trade is wanted is an operator call.
- No memory stops. Peak RSS for training: plain AE 5.9 GB, VAE on v3 6.1 GB, VAE β 0.1 on v1 11.6 GB (within 0.4 GB of the 12 GB gate; not explained).

## Grill point 3 (2026-10-24): region rescore, val

Question: is R1-v3's +14.5 % whole-image MSE mostly background it was told to down-weight?

| Run | Whole | Hands | Feet | Head | IoU |
|---|---|---|---|---|---|
| R0-v3 | 0.00158 | 0.02542 | 0.02166 | 0.01338 | 0.905 |
| R0b-v3 | 0.00159 (+0.6 %) | 0.02547 (+0.2 %) | 0.02178 (+0.5 %) | 0.01339 (+0.1 %) | 0.904 |
| R2-v3 | 0.00159 (+0.7 %) | 0.02541 (−0.0 %) | 0.02201 (+1.6 %) | 0.01332 (−0.5 %) | 0.904 |
| R1-v3 | 0.00181 (+14.5 %) | 0.02934 (+15.4 %) | 0.02428 (+12.1 %) | 0.01517 (+13.4 %) | 0.856 |

Answer: no. R1-v3 is 12–15 % worse on every body region and its silhouette IoU drops 0.905 → 0.856,
so a large part of the cost is on the body (the region windows include nearby background and the torso is unmeasured; a body/background MSE split was not computed). The DROPPED verdict stands on fair grounds;
its latent encodes pose a bit better while drawing the body worse. R2-v3 matches R0-v3 on every
region (within the R0/R0b spread, except feet +1.6 % and head −0.5 %).

## Grill point 6 (2026-10-24): val vs test verdicts

```
| Variant | Val MSE vs R0 | Probe MPJPE val (mm) | Δ mm | Verdict (015) | Reason |
|---|---|---|---|---|---|
| R2-v3 | +0.7 % | 132.3 | +1.3 | REPORTED | +0.7 % MSE (gap 0.6 %); +1.3 mm probe (gap 3.3) |
| R1-v3 | +14.5 % | 124.5 | -6.6 | DROPPED | +14.5 % MSE (gap 0.6 %); -6.6 mm probe (gap 3.3); MSE > +5 % |

| Variant | Test MSE vs R0 | Probe MPJPE test (mm) | Δ mm | Verdict (015) | Reason |
|---|---|---|---|---|---|
| R2-v3 | +0.7 % | 128.4 | +1.0 | REPORTED | +0.7 % MSE (gap 0.5 %); +1.0 mm probe (gap 3.0) |
| R1-v3 | +14.8 % | 121.2 | -6.2 | DROPPED | +14.8 % MSE (gap 0.5 %); -6.2 mm probe (gap 3.0); MSE > +5 % |

| Variant | Verdict val | Verdict test | Agree |
|---|---|---|---|
| R2-v3 | REPORTED | REPORTED | yes |
| R1-v3 | DROPPED | DROPPED | yes |
```

Verdicts agree on val and test for both variants (R2-v3 REPORTED, R1-v3 DROPPED).

## Grill point 7 (2026-10-24): is the pose-count curve clean?

Datasets: the v1 / v2d / v3 manifests are identical in `inputs` (pose ranges, Euler order), `skeleton`, `cameras`, betas, image size and generator seed (0); only `num_poses` differs (2,000 / 4,000 / 8,000). Generator code commits differ (b7b2f99 / 32b177d / 2f4450c) but the inputs they record are identical.

Confounds that remain:
- **Gradient steps scale with data.** All three trained for 60 epochs at batch 64, so v3 got 4× the optimizer steps of v1 (at a useful LR; R0-v3's val loss was flat by the end of its cosine schedule). Part of the "more poses" gain may be "more training". A v1 run with 240 epochs (same steps as v3) would separate the two; about 4 h of CPU.
- **Each dataset is scored on its own val split** (same distribution, different poses), so the curve compares slightly different exams. This effect is unmeasured; a weak proxy, mean-image val MSE 0.01403 (v1) vs 0.01406 (v3), suggests the val sets are about equally hard.

Extrapolation (three points, so this is an exact fit, not a test): val MSE 0.00218 / 0.00178 / 0.00158 has differences halving per doubling, which implies a floor near 0.00138 and about 0.00148 (−6 %) at 12,800 train poses (16,000 total). Probe 175 / 150 / 131 mm has differences shrinking ×0.76, which implies about 117 mm at 12,800 and a floor near 71 mm; reaching 80 mm by data alone would need on the order of 10⁵–10⁶ poses (the fit is unreliable that far out). Treat both as rough: a fourth point could change the floor a lot.

## Grill point 5 result (2026-10-24): probe round-trip (slice 016), v1 VAEs

τ = 5th percentile of real-test round-trip IoU (0.32 / 0.31). "Control" = a random training pose re-rendered at the same camera (chance level).

| Set | VAE β 0.01, v1: plausible | plausible & valid | control | VAE β 0.1, v1: plausible | plausible & valid | control |
|---|---|---|---|---|---|---|
| prior samples | 22.0 % | 5.4 % | 20.4 % | 87.2 % | 31.0 % | 75.5 % |
| plain-AE prior | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| VAE recon val | 96.8 % | 82.5 % | 72.4 % | 97.2 % | 85.7 % | 75.3 % |
| real test (ceiling) | 95.0 % | 92.1 % | 73.1 % | 95.0 % | 92.1 % | 75.6 % |

Reading: **the round-trip score barely separates real poses from random ones.** A random training pose at the right camera already "round-trips" for 73–76 % of real images, because silhouettes from one viewpoint overlap heavily (torso, legs). The median real-image IoU is only 0.52 (probe error ≈ 125–175 mm), so the threshold is low. The gap over control is real but small (real test 95 % vs 73 %; β 0.1 prior 87 % vs 75 %). What the metric does support: β 0.1 samples look like bodies about as often as reconstructions do once they are one connected blob (plausible & valid 31 % vs valid 33 %), and β 0.01 samples rarely do. It does not support "most samples are plausible people". A sharper check needs a better probe (point 8) or a pose-prior likelihood; the v3 round-trip is deferred until the training queue ends (it was cut off by the time limit while competing with training).
