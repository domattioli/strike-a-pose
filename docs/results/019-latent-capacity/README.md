# Slice 019 campaign (2026-10-28): latent capacity 128 on smpl_proc_v3

References: plain 64-d seeds R0-v3, R0b-v3, R0c-v3 (val MSE mean 0.001579, sd 0.46 %; probe 132.8 mm, sd 1.7) and the 018 pose runs (P2: λ 0.03, 68.6 mm, +36 % MSE). Promotion: 015 pixel track and the 2026-10-28 pose track (`scripts/promotion.py --seed ...`).

| Run | Result |
|---|---|
| **L1** plain AE, latent 128 (`ae-l128-smpl_proc_v3-20261028T022732Z`) | all 60 epochs (best val epoch 56); val / test MSE 0.00112 / 0.00111 (**−29 %**), IoU 0.930 / 0.930 (64-d 0.905); region MSE val hands 0.0196 (−23 %), feet 0.0138 (−36 %), head 0.0102 (−24 %); probe 120.5 / 117.3 mm (−12 mm vs seeds = −7 sd); peak RSS 6.1 GB. **WIN-pixel** on val and test; **PROMOTED (pose)**. Pixel change far outside the 2–3 sd band, so no second L1 seed (ruling R4). Figure: `recon_compare_L1.png` (true / 64-d / 128-d / 64-d + pose head) |

## L1 promotion (`scripts/promotion.py --split both`, three plain seeds)

| Variant | Val MSE vs R0 | Probe MPJPE val (mm) | Δ mm | Verdict (015) | Reason | Pose track | Pixel cost |
|---|---|---|---|---|---|---|---|
| L1 | -28.9 % | 120.5 | -10.5 | WIN-pixel | -28.9 % MSE (gap 0.6 %); -10.5 mm probe (gap 3.3) | PROMOTED (pose) | -29.0 % |

| Variant | Test MSE vs R0 | Probe MPJPE test (mm) | Δ mm | Verdict (015) | Reason | Pose track | Pixel cost |
|---|---|---|---|---|---|---|---|
| L1 | -28.7 % | 117.3 | -10.2 | WIN-pixel | -28.7 % MSE (gap 0.5 %); -10.2 mm probe (gap 3.0) | PROMOTED (pose) | -28.9 % |

| Variant | Verdict val | Verdict test | Agree | Pose track val / test |
|---|---|---|---|---|
| L1 | WIN-pixel | WIN-pixel | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |
