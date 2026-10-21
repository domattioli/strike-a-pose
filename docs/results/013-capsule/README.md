# Interim capsule campaign (spec 013): infrastructure tests

> **Infrastructure tests on the interim capsule body (spec 013); not SMPL.** Image-level
> numbers (MSE, PSNR, silhouette IoU, region MSE, AE, probe image-derived results) are not
> comparable with any SMPL result and are never README headline numbers. Pose-level numbers
> at mean shape (003 mean pose / pose PCA, probe oracle references) use the same poses, split
> and joints as `smpl_proc_v1` and are comparable.

Body: `assets/synthetic/capsule_v1.npz` (built by `strike-a-pose-data make-capsule-body`, never
committed), SHA-256 `bc1f3eb7275ce0fabf6fdab3fb80e36dc2bcd29eb96a6590b009dac5d0b4116a`: 28 rigid
capsules on SMPL's rest skeleton recovered from the committed fixture. Datasets
`capsule_proc_v1` (twin of `smpl_proc_v1`) and the independent single-change datasets
`capsule_proc_v2a`–`v2d`; configs `configs/capsule_proc_*/`; procedure: `docs/HANDOFF.md`,
"Interim capsule campaign".

## Runs

One subdirectory per run (`config.yaml`, `metrics.json`, `run_info.json`, figures), added as
runs finish.

| Run | Run dir | Result |
|---|---|---|
| baselines `capsule_proc_v1` | `runs/baselines-capsule_proc_v1-20261021T110723Z` | pose rows equal the published `smpl_proc_v1` ones to ≤ 1e-4 (spec 013 SC-005 holds) |
| R0 AE `capsule_proc_v1` (seed 0) | `runs/ae-capsule_proc_v1-20261021T111109Z` | early stop fired after epoch 42 (patience 10; best val epoch 32, val MSE 0.00235 vs final 0.00237); best-val weights scored: val / test MSE 0.00235 / 0.00239, IoU 0.864 / 0.865 (mean image 0.01509 / 0.01490; PCA-64 0.00509 / 0.00502); region MSE val hands 0.027, feet 0.037, head 0.018; resumed twice across container restarts (epochs 16, 19) |
| probe on R0 `best.pt` | `runs/probe-capsule_proc_v1-20261021T145935Z` | world-frame MPJPE MLP 183.0 / 184.8 mm, ridge 215.0 / 214.3 (mean pose 253.4 / 250.3; untrained-encoder MLP 217.6 / 214.3); camera accuracy MLP 0.921 / 0.911, ridge 0.763 / 0.758 |
| R0b AE `capsule_proc_v1` (seed 1, noise bar) | `runs/ae-seed1-capsule_proc_v1-20261021T150638Z` | early stop after epoch 47 (best val epoch 37); val / test MSE 0.00236 / 0.00239, IoU 0.865 / 0.866; region MSE val hands 0.027, feet 0.037, head 0.018. **Seed noise vs R0: val MSE +0.2 %, test ±0.0 %** (the bar for the variant comparisons) |
| probe on R0b `best.pt` | `runs/probe-capsule_proc_v1-20261021T161021Z` | world-frame MPJPE MLP 182.4 / 183.8 mm (R0: 183.0 / 184.8), ridge 215.2 / 214.1; camera accuracy MLP 0.929 / 0.915 (R0: 0.921 / 0.911): probe noise ≈ ±1 mm, ±1 % |
| R1 mask-weighted MSE (α 4) | `runs/ae-mask-capsule_proc_v1-20261021T161710Z` | early stop after epoch 52 (best val epoch 42); val / test whole MSE 0.00244 / 0.00244 (+3.6 % / +2.3 % vs R0), IoU 0.822 / 0.824 (R0 0.864 / 0.865); region mean MSE val +0.5 % / test −1.0 % vs R0 (seed noise ±0.7 %). **Promotion rule: not a winner (region MSE does not beat R0 beyond noise), not dropped (whole MSE within +5 %): reported, not carried.** |
| probe on R1 `best.pt` | `runs/probe-capsule_proc_v1-20261021T174113Z` | world-frame MPJPE MLP 172.0 / 174.1 mm (R0 183.0 / 184.8, R0b 182.4 / 183.8): ≈ 10 mm better, ≈ 10× the probe noise; camera accuracy MLP 0.953 / 0.954 (R0 0.921 / 0.911). Up-weighting body pixels puts more pose information in the latent while blurring the silhouette edge (IoU −0.04): a candidate for an operator-chosen promotion metric based on the probe |
| R2 VAE (β 0.01) | `runs/ae-vae-capsule_proc_v1-20261021T174746Z` | early stop after epoch 46 (best val epoch 36); val / test MSE 0.00237 / 0.00238 (+0.8 % / −0.3 % vs R0), IoU 0.862 / 0.863; KL 248 → 282 nats/image (no collapse) |
| probe on R2 `best.pt` (μ) | `runs/probe-capsule_proc_v1-20261021T185006Z` | world-frame MPJPE MLP 184.5 / 185.5 mm (R0 183.0 / 184.8); camera accuracy 0.928 / 0.914. **Promotion rule: MSE and KL criteria pass; probe +1.5 / +0.7 mm vs R0, about twice the R0/R0b probe gap, so "not worse" fails narrowly: borderline, not carried.** KL is healthy and reconstruction within tolerance, so per the 012 procedure the next VAE step is β 0.1 (`ae_vae_beta0.1.yaml`) |
