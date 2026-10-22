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
| **v2b** random body shape, AE | `ae-smpl_proc_v2b-20261022T061420Z` | early stop after epoch 41, best val epoch 31; val / test MSE 0.00239 / 0.00235 (+10 % / +9 % vs R0), IoU 0.862 / 0.863; region MSE val hands 0.033, feet 0.034, head 0.018. Own baselines: mean image 0.0143 / 0.0140, PCA-64 0.00492 / 0.00479 (+7 % vs v1): **harder**, and slightly more so for the AE than for PCA (AE/PCA-64 ratio 0.48 vs 0.47) |
| probe on v2b `best.pt` | `probe-smpl_proc_v2b-*` | world-frame MPJPE MLP 185.4 / 186.1 mm vs mean pose 254.8 / 250.9 (−27 %; v1 −31 %): body shape variation costs ≈ 10 mm of pose information (≈ 4× the seed noise); camera accuracy 0.928 / 0.918 (v1 0.956) |
| **v2c** camera ring jitter, AE | `ae-smpl_proc_v2c-20261022T073339Z` | 0 / 16,000 in-frame fallbacks; early stop after epoch 44, best val epoch 34; val / test MSE 0.00216 / 0.00216 (−1 % / 0 % vs R0: within seed noise), IoU 0.855 / 0.857 (R0 0.870); PCA-64 0.00455 / 0.00448 (≈ v1) |
| probe on v2c `best.pt` | `probe-smpl_proc_v2c-*` | world-frame MPJPE MLP 195.9 / 195.7 mm (R0 174.8 / 175.3; mean pose unchanged 253.4 / 250.3): **−20 mm of pose information** although reconstruction is as good; camera-frame 195.7 mm (v1 163.0); camera accuracy 0.901 / 0.881 (v1 0.956): the latent now spends capacity on the per-sample viewpoint, and the 8-way yaw target is harder because jitter blurs the ring |
| **v2d** 4,000 poses (2× train), AE | `ae-smpl_proc_v2d-20261022T090252Z` | early stop after epoch 53, best val epoch 43; val / test MSE 0.00178 / 0.00176 (**−18 % / −18 % vs R0**, 30× the seed noise), IoU 0.895 / 0.896 (R0 0.870); region MSE val hands 0.028, feet 0.026, head 0.014; mean-image baseline 0.01387 / 0.01388 (≈ v1, so the data is not easier). Train-loss / val-loss at the best epoch 0.00117 / 0.00178 (v1: 0.00143 / 0.00218). Caveat: v2d's val / test are a new pose permutation (≈ 20 test poses shared with v1). Image PCA not run (memory; `image_pca_ks: []`) |
| probe on v2d `best.pt` | `probe-smpl_proc_v2d-*` | world-frame MPJPE MLP 150.2 / 148.1 mm vs mean pose 252.1 / 248.3 (**−40 %**; v1 −31 %); camera accuracy 0.972 / 0.972 (v1 0.956). **More distinct poses is the strongest lever found: the AE is data-limited at 1,600 train poses** |
| **R1** mask-weighted MSE (α 4), v1 | `ae-mask-smpl_proc_v1-20261022T115810Z` | early stop after epoch 53, best val epoch 43; val / test MSE 0.00236 / 0.00235 (+8.4 % / +8.9 % vs R0), IoU 0.829 / 0.830 (R0 0.870); region mean MSE +7.7 % / +8.2 % vs R0. **Promotion rule: DROPPED (whole MSE > +5 %, region MSE worse)** |
| probe on R1 `best.pt` | `probe-smpl_proc_v1-20261022T133914Z` | world-frame MPJPE MLP 162.4 / 162.1 mm (R0 174.8 / 175.3, R0b 172.0 / 173.9): **≈ 12 mm better, 4.4× the 2.8 mm probe gap**; camera accuracy 0.973 / 0.973 (R0 0.956). Same pattern as the capsule campaign: weighting body pixels trades pixel fidelity and silhouette edges for a latent that holds more pose and view information. The pre-registered rule scores pixels only; a probe-based criterion would promote R1 (operator decision) |
| **R2** VAE (β 0.01), v1 | `ae-vae-smpl_proc_v1-20261022T134729Z` | early stop after epoch 46, best val epoch 36; val / test MSE 0.00218 / 0.00217 (+0.2 % / +0.6 % vs R0, inside seed noise), IoU 0.870 / 0.871; KL 242 → 274 nats/image (no collapse) |
| probe on R2 `best.pt` (μ) | `probe-smpl_proc_v1-20261022T150051Z` | world-frame MPJPE MLP 174.4 / 173.4 mm (R0 174.8 / 175.3); camera accuracy 0.954 / 0.959. **Promotion rule: WINNER** (MSE within +5 %, KL ≥ 1 nat, probe not worse): a KL-regularized, sampleable latent at no measurable cost. Next VAE step per the procedure: β 0.1 (`configs/ae_vae_beta0.1.yaml`) |
| **R3** regularization (AdamW wd 0.05, shift 4 px, latent dropout 0.1), v1 | `ae-reg-smpl_proc_v1-20261022T150900Z` | ran all 60 epochs (val still improving; best val epoch 58); val / test MSE 0.00221 / 0.00221 (+1.5 % / +2.3 % vs R0), IoU 0.839 / 0.840 (R0 0.870) |
| probe on R3 `best.pt` | `probe-smpl_proc_v1-20261022T164153Z` | world-frame MPJPE MLP 184.8 / 184.9 mm (R0 174.8 / 175.3): ≈ 10 mm worse; camera accuracy 0.934 / 0.926. **Promotion rule: not a winner (MSE does not beat R0), not dropped (within +5 %); probe clearly worse** |

## Summary (2026-10-22, all runs on the regenerated `smpl_proc_v1` family)

| Run | Val MSE vs R0 | Probe MPJPE val (mm) | Reading | 015 verdict |
|---|---|---|---|---|
| R0 / R0b (control, 2 seeds) | 0.00218 / −0.6 % | 174.8 / 172.0 | reproduces the published 004 run; noise ≈ 0.6 % MSE, 2–3 mm probe | control |
| v2a arms-down | −15 % | 150.2 (mean pose 219) | an easier dataset (narrowed ranges), same relative gains | — (dataset) |
| v2b body shape | +10 % | 185.4 | harder; shape variation costs ≈ 10 mm of pose information | — (dataset) |
| v2c camera jitter | −1 % | 195.9 | same pixels, ≈ 20 mm less pose information (latent spends capacity on the view) | — (dataset) |
| **v2d 4,000 poses** | **−18 %** | **150.2** (mean pose 252) | **the strongest lever: the AE is data-limited at 1,600 train poses** | — (dataset) |
| R1 mask-weighted MSE | +8.4 % (dropped) | **162.4** | worse pixels, clearly better pose latent (also on the capsule body) | DROPPED (misses +5 % by 3.4 points; probe −12.4 mm = 4.4× the gap) |
| **R2 VAE β 0.01** | +0.2 % | 174.4 | **winner** (012 rule): sampleable latent at no cost; next step β 0.1 | REPORTED (no pixel or latent win) |
| R3 regularization bundle | +1.5 % | 184.8 | no gain; slower convergence, worse latent | REPORTED |

Process: guards fired as intended (early stop on 9 of 10 AE runs, none on R3 whose val kept
improving; no divergence aborts); one container restart (R0b resumed at epoch 12); one
self-inflicted incident with the first watchdog (fixed, see HANDOFF).

## 015 re-scoring (promotion rule of spec 015 FR-006, operator decision 2026-10-22)

`scripts/promotion.py` on the run dirs (unrounded numbers; seed gap 0.6 %, probe gap 2.8 mm):

| Variant | Val MSE vs R0 | Probe MPJPE val (mm) | Δ mm | Verdict (015) | Reason |
|---|---|---|---|---|---|
| R1 | +8.4 % | 162.4 | -12.4 | DROPPED | +8.4 % MSE (gap 0.6 %); -12.4 mm probe (gap 2.8); MSE > +5 % |
| R2 | +0.2 % | 174.4 | -0.4 | REPORTED | +0.2 % MSE (gap 0.6 %); -0.4 mm probe (gap 2.8) |
| R3 | +1.5 % | 184.8 | +10.0 | REPORTED | +1.5 % MSE (gap 0.6 %); +10.0 mm probe (gap 2.8) |

R1 misses the +5 % bound of the latent win (surrogate ruling 2026-10-22: rule as worded) with
a probe gain of 12.4 mm, 4.4× the gap: the case to revisit. R2 was the winner under the
superseded 012 rule (whose VAE clause needed only "not worse"); under 015 it is reported, not
carried; its β 0.1 follow-up (R2b) is an operator decision. The v2a–v2d datasets get no verdict
(dataset changes, judged against their own baselines). The 015 campaign (`smpl_proc_v3`, R2b,
sampling) is recorded in its own section when it runs.
