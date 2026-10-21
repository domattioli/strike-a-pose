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
