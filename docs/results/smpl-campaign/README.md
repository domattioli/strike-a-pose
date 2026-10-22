# SMPL campaign (2026-10-22): re-baseline, realism datasets, training variants

Procedure: `docs/HANDOFF.md`, "SMPL campaign" (012 / 014 rules). Dataset `smpl_proc_v1`
regenerated 2026-10-22 from the re-converted SMPL file (003 baselines reproduce the published
numbers to ≤ 1e-5; see "F1 outcome"). All AE runs: 60 planned epochs, cosine lr, early stop
(patience 10 from epoch 20), best-val weights scored on val and test once.

| Run | Run dir | Result |
|---|---|---|
| baselines `smpl_proc_v1` | `docs/results/007-f1-baselines-smpl_proc_v1/` | every published 003 number reproduced (≤ 1e-5) |
| **R0** AE (seed 0) | `ae-smpl_proc_v1-20261022T012307Z` | early stop after epoch 40, best val epoch 30; val / test MSE 0.00218 / 0.00216, PSNR 27.1 dB, IoU 0.870 / 0.871; region MSE val hands 0.032, feet 0.031, head 0.017. **R0 tolerances pass**: epoch-1 val 0.010016 vs published 0.010030 (−0.1 %, limit ±2 %); final val 0.00219 vs 0.00221 (−0.8 %, limit ±3 %). vs the published 004 final weights (test 0.00219, IoU 0.881): best-val weights −1.4 % MSE, IoU −0.01 |
| probe on R0 `best.pt` | `probe-smpl_proc_v1-20261022T031425Z` | world-frame MPJPE MLP 174.8 / 175.3 mm (published 006: 172.0 / 172.8); camera accuracy MLP 0.956 / 0.956 (published 0.965 / 0.959) |
| **R0b** AE (seed 1, noise bar) | `ae-seed1-smpl_proc_v1-20261022T032318Z` | early stop after epoch 47, best val epoch 37; val / test MSE 0.00216 / 0.00215 (−0.6 % / −0.3 % vs R0), IoU 0.873 / 0.875. **Seed noise: ≈ 0.6 % MSE**; resumed once across a container restart (epoch 12) |
| probe on R0b `best.pt` | `probe-smpl_proc_v1-20261022T043620Z` | world-frame MPJPE MLP 172.0 / 173.9 mm (R0 174.8 / 175.3): **probe noise ≈ 2–3 mm**; camera accuracy 0.958 / 0.954 |
| **v2a** arms-down (`armsdown-v1` table), AE | `ae-smpl_proc_v2a-20261022T045128Z` | early stop after epoch 47, best val epoch 37; val / test MSE 0.00185 / 0.00186 (−15 % / −14 % vs R0), IoU 0.890 / 0.890; region MSE val hands 0.026, feet 0.030, head 0.013. Its own baselines: mean image 0.01295 / 0.01282, PCA-64 0.00394 / 0.00390 (v1: 0.01403 / 0.00459), mean pose 219.3 / 215.0 mm (v1 253.4 / 250.3): **an easier dataset** (arms against the body, narrowed ranges), not a better model; the AE / PCA-64 ratio is 0.47 here vs 0.47 on v1 |
| probe on v2a `best.pt` | `probe-smpl_proc_v2a-20261022T055821Z` | world-frame MPJPE MLP 150.2 / 146.9 mm vs mean pose 219.3 / 215.0 (−32 %; v1: −31 %); camera accuracy 0.956 / 0.963. Intersection gate (diagnostic only, miscalibrated): 100 % of poses flagged (v1 table: 79 %) |
