# Research: 018 Pose-Supervised Autoencoder

Measured at `9075172`, CPU, cheap reads only (no training); script in the session scratchpad.

## R1 Labels available

`PoseImageDataset.__getitem__` returns `joints_3d_world`, `joints_3d_cam` (`[24, 3]` m),
`pose` (72 axis-angle), `betas`, `camera` (`yaw_deg`, `R`, `t`, `K`), `camera_index` (8 ring
yaws). `load_split` already iterates full items (labels are parsed anyway) but keeps images,
`pose_index` and, for scored splits, masks + 2D keypoints. `smpl_proc_v3`: 64,000 images =
8,000 poses × 8 cameras; train 6,400 poses = 51,200 images (images 2.5 GB uint8 in memory;
joints 51,200 × 72 × 4 B = 15 MB).

## R2 Target statistics and λ calibration

4,000 v3 images (every 8th sample): root-relative world joints per-coordinate sd min 0 /
median 135 / max 431 mm; 12 coordinates have sd 0 (world frame fixes the global orientation, so
the root and pelvis-rigid joints are constant); mean-pose MPJPE 253 mm.

Loss scale: R0-v3 reconstruction MSE ≈ 0.0016 at convergence. A predictor at the plain-AE probe
level (131 mm vs 253 mm mean pose) has standardized MSE ≈ (131/253)² ≈ 0.27; at the 80 mm target
≈ 0.10. Pose / recon term ratio at 0.27: λ 0.003 → 0.5×, λ 0.03 → 5×, λ 0.3 → 50×.
Decision: λ ∈ {0.003, 0.03}, plus λ 0.03 with mask weight 4.

## R3 Consumers of checkpoints

`models.autoencoder.load_checkpoint` reads `model_hparams` and `model` only (strict
`load_state_dict`). Probe, sample and round-trip use it; promotion reads `metrics.json`,
`config.yaml`, `history.csv`. Head under separate keys → no consumer change.
`trainer._holds` compares `best.pt`'s config with `with_legacy_values`, so `pose_weight` must be
in `LEGACY_VALUES`.

## R4 Rejected / deferred

- SMPL pose 6D target, yaw CE term, joints_cam target: deferred (Q1); candidates for a 019 if
  the head plateaus above 80 mm.
- λ 0.3: budget; run only if λ 0.03 is better than λ 0.003 by > 2 sd and MSE allows.
- Head on a sampled VAE z: refused (Q9).
- Seed replication of the winner: operator question 2.
