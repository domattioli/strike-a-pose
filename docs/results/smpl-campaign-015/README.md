# Slice 015 campaign (2026-10-22/23): 8,000 poses, VAE β 0.1, sampling

| Run | Result |
|---|---|
| VAE β 0.1, v1 (`ae-vae-beta0.1-smpl_proc_v1-20261022T225733Z`) | val / test MSE 0.00220 / 0.00219 (+1.3 % / +1.5 % vs R0); KL ≈ 167 nats/image (β 0.01: 274); probe world MPJPE 176.0 / 175.6 mm (R0 174.8 / 175.3): reconstruction and pose information kept |
| Sampling, VAE β 0.1 (`sample-vae-beta0.1-*`) | N(0, I) prior samples valid (one body-sized connected silhouette): **33.1 %** of 1,024; NN-MSE to train 0.0101. Bars: plain-AE decoder on prior samples 0 % (all empty); VAE reconstructions of val 88.4 % |
| Sampling, VAE β 0.01 (`sample-vae-smpl_proc_v1-*`) | prior samples valid 5.7 % (mostly too small); reconstructions 85.6 % |
| baselines v3 (8,000 poses) | peak RSS 3.8 GB (uint8 fix) |

Reading: raising β to 0.1 makes the prior 6× more generative (5.7 % → 33 % valid samples) at ≈ 1 % reconstruction cost and no measurable pose-information loss. Two thirds of prior samples still fail, mostly as fragmented silhouettes (582 / 1,024).
