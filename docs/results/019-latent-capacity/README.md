# Slice 019 campaign (2026-10-28): latent capacity 128 on smpl_proc_v3

References: plain 64-d seeds R0-v3, R0b-v3, R0c-v3 (val MSE mean 0.001579, sd 0.46 %; probe 132.8 mm, sd 1.7) and the 018 pose runs (P2: λ 0.03, 68.6 mm, +36 % MSE). Promotion: 015 pixel track and the 2026-10-28 pose track (`scripts/promotion.py --seed ...`).

| Run | Result |
|---|---|
| **L1** plain AE, latent 128 (`ae-l128-smpl_proc_v3-20261028T022732Z`) | all 60 epochs (best val epoch 56); val / test MSE 0.00112 / 0.00111 (**−29 %**), IoU 0.930 / 0.930 (64-d 0.905); region MSE val hands 0.0196 (−23 %), feet 0.0138 (−36 %), head 0.0102 (−24 %); probe 120.5 / 117.3 mm (−12 mm vs seeds = −7 sd); peak RSS 6.1 GB. **WIN-pixel** on val and test; **PROMOTED (pose)**. Pixel change far outside the 2–3 sd band, so no second L1 seed (ruling R4). Figure: `recon_compare_L1.png` (true / 64-d / 128-d / 64-d + pose head) |
| **L2** latent 128 + pose λ 0.03 (`ae-l128-pose-l0.03-smpl_proc_v3-20261028T101221Z`) | early-stopped at epoch 55 (best val epoch 45); val / test MSE 0.00167 / 0.00165 (**+5.7 %** vs 64-d seeds; P2 at 64-d: +36 %), IoU 0.896 / 0.898 (P2 0.870); region MSE val hands 0.0263, feet 0.0238, head 0.0139; **pose head 67.0 / 64.7 mm**, probe 67.1 / 64.9 mm (P2 68.6 / 66.6); peak RSS 6.2 GB. 015 pixel verdict DROPPED (+5.7 % > +5 %); **PROMOTED (pose)** on val and test. Figure: `recon_compare_L2.png` (true / P2 / L1 / L2) |

## Promotion and pose track, L1 + L2 with the 018 references (`--split both`)

| Variant | Val MSE vs R0 | Probe MPJPE val (mm) | Δ mm | Verdict (015) | Reason | Pose track | Pixel cost |
|---|---|---|---|---|---|---|---|
| L1 | -28.9 % | 120.5 | -10.5 | WIN-pixel | -28.9 % MSE (gap 0.6 %); -10.5 mm probe (gap 3.3) | PROMOTED (pose) | -29.0 % |
| L2 | +5.8 % | 67.1 | -63.9 | DROPPED | +5.8 % MSE (gap 0.6 %); -63.9 mm probe (gap 3.3); MSE > +5 % | PROMOTED (pose) | +5.7 % |
| P1 | +5.8 % | 91.2 | -39.9 | DROPPED | +5.8 % MSE (gap 0.6 %); -39.9 mm probe (gap 3.3); MSE > +5 % | PROMOTED (pose) | +5.7 % |
| P2 | +36.5 % | 68.6 | -62.4 | DROPPED | +36.5 % MSE (gap 0.6 %); -62.4 mm probe (gap 3.3); MSE > +5 % | PROMOTED (pose) | +36.3 % |

| Variant | Test MSE vs R0 | Probe MPJPE test (mm) | Δ mm | Verdict (015) | Reason | Pose track | Pixel cost |
|---|---|---|---|---|---|---|---|
| L1 | -28.7 % | 117.3 | -10.2 | WIN-pixel | -28.7 % MSE (gap 0.5 %); -10.2 mm probe (gap 3.0) | PROMOTED (pose) | -28.9 % |
| L2 | +5.9 % | 64.9 | -62.5 | DROPPED | +5.9 % MSE (gap 0.5 %); -62.5 mm probe (gap 3.0); MSE > +5 % | PROMOTED (pose) | +5.7 % |
| P1 | +6.1 % | 88.8 | -38.7 | DROPPED | +6.1 % MSE (gap 0.5 %); -38.7 mm probe (gap 3.0); MSE > +5 % | PROMOTED (pose) | +5.9 % |
| P2 | +36.1 % | 66.6 | -60.9 | DROPPED | +36.1 % MSE (gap 0.5 %); -60.9 mm probe (gap 3.0); MSE > +5 % | PROMOTED (pose) | +35.8 % |

| Variant | Verdict val | Verdict test | Agree | Pose track val / test |
|---|---|---|---|---|
| L1 | WIN-pixel | WIN-pixel | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |
| L2 | DROPPED | DROPPED | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |
| P1 | DROPPED | DROPPED | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |
| P2 | DROPPED | DROPPED | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |

seeds: n=3 mean 132.8 mm sd 1.7 mm (val)

| Rank | Variant | MSE vs seeds | Probe MPJPE val (mm) | Δ vs seeds (mm) | Δ / sd | Real (< −2 sd) | Head MPJPE val (mm) | Probe ≤ 80 | Head ≤ 80 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | L2 | +5.7 % | 67.1 | -65.7 | -39.0 | yes | 67.0 | yes | yes |
| 2 | P2 | +36.3 % | 68.6 | -64.2 | -38.1 | yes | 68.6 | yes | yes |
| 3 | P1 | +5.7 % | 91.2 | -41.7 | -24.8 | yes | 90.9 | no | no |
| 4 | L1 | -29.0 % | 120.5 | -12.3 | -7.3 | yes | – | no | – |

second seed for L2 (operator ruling O2, val): yes (within 3 sd of P2)

seeds: n=3 mean 129.1 mm sd 1.5 mm (test)

| Rank | Variant | MSE vs seeds | Probe MPJPE test (mm) | Δ vs seeds (mm) | Δ / sd | Real (< −2 sd) | Head MPJPE test (mm) | Probe ≤ 80 | Head ≤ 80 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | L2 | +5.7 % | 64.9 | -64.2 | -41.7 | yes | 64.7 | yes | yes |
| 2 | P2 | +35.8 % | 66.6 | -62.6 | -40.6 | yes | 66.6 | yes | yes |
| 3 | P1 | +5.9 % | 88.8 | -40.4 | -26.2 | yes | 88.0 | no | no |
| 4 | L1 | -28.9 % | 117.3 | -11.9 | -7.7 | yes | – | no | – |

second seed for L2 (operator ruling O2, test): yes (within 3 sd of P2)

## Summary (QUEUE_DONE 2026-10-28 17:17 UTC)

- **Capacity was the bottleneck for pictures.** Doubling the latent (L1) cuts val MSE by 29 % (≈ 63 seed sd), raises IoU 0.905 → 0.930, and improves hands / feet / head by 23 / 36 / 24 %; it also improves the frozen-latent probe by 12 mm. First variant with a pixel WIN and a pose-track promotion.
- **With room, the pose head is nearly free in pixels.** L2's pose cost is +5.7 % MSE versus +36 % at 64-d (019 Q4 "cost shrank" criterion: ≤ +18 % vs L1 is **not** met — L2 is +49 % vs L1; the cost shrank against the 64-d plain baseline, not against L1). Pose is the best so far: head 67.0 / 64.7 mm, under 80 mm.
- **Goal status (operator 2026-10-28: both in one model):** pose ≤ 80 mm ✓; reconstructions as good as the best plain AE (L1) ✗ — L2 sits at the old 64-d plain level. Levers: slice 020 (gradient loss, running now), a larger latent (256), or a smaller λ at 128-d (L3, λ 0.01, deferred).
- **Ruling O2:** L2 is within 3 sd of P2 on pose, so a second L2 seed is due to confirm the pose ranking (≈ 6.5 h); queued after slice 020's campaign. The pixel conclusion (+5.7 % vs +36 %) does not depend on it.
- No memory issues (peak ≈ 6.2 GB).
