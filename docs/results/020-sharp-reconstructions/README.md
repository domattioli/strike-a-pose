# Slice 020 campaign (2026-10-28/29): sharper reconstructions (gradient loss) on smpl_proc_v3

Operator ruling 2026-10-28: "renderings" = the AE reconstructions. Gradient-loss weight γ 0.5, latent 64. Sharpness metric: boundary F1 at 1 px (`boundary_f1`). Verdict rule fixed before the runs (spec 020): SHARPER if bF1 gain > 2 seed sd, MSE ≤ +5 % and probe ≤ +2 sd vs the reference; SHARPER-COSTLY if the gain is real but MSE or probe fails; else NO-GAIN. References: three plain seeds (F1) and the two P2 seeds (F2). Figure: `recon_compare_020.png` (true | 64 plain | F1 | P2 | F2).

| Run | Result |
|---|---|
| **F1** plain + γ 0.5 (`ae-grad-smpl_proc_v3-20261028T172823Z`) | early-stopped at 53 (best 43); val / test MSE 0.00160 / 0.00158 (+1.2 %), IoU 0.906 / 0.907, bF1 0.9113 / 0.9136 (−0.9 / −0.4 sd); probe 135.6 / 132.4 mm (+2.8 / +3.3 mm); peak 6.0 GB. **NO-GAIN** (val and test); 015 REPORTED; pose track not promoted |
| **F2** pose λ 0.03 + γ 0.5 (`ae-pose-l0.03-grad-smpl_proc_v3-20261028T234943Z`) | val / test MSE 0.00210 / 0.00208 (−2.3 % / −2.0 % vs P2), IoU 0.876 / 0.878 (P2 0.870), bF1 0.8500 / 0.8530 (+0.0105 / +0.0109 = **+10.6 / +13.3 sd**); hands / feet / head slightly better than P2; pose head 68.7 / 67.1 mm, probe 69.2 / 67.3 mm (+0.5 / +0.8 mm, within noise); peak 6.2 GB. **SHARPER** (val and test); 015 DROPPED (pixel gate vs the plain AE); **PROMOTED (pose)** |

## Fidelity report (`scripts/fidelity.py report --split both`)

| Run (val) | Kind | Ref | MSE | MSE vs ref | IoU | bF1 | Δ bF1 | Δ bF1 / sd | hands | feet | head | Probe (mm) | Δ probe (mm) | Δ probe / sd | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ae-smpl_proc_v3-20261023T004930Z | plain | seeds | 0.00158 | — | 0.9052 | 0.9132 | n/a | n/a | 0.02542 | 0.02166 | 0.01338 | 131.0 | n/a | n/a | — |
| ae-seed1-smpl_proc_v3-20261023T065844Z | plain | seeds | 0.00159 | — | 0.9044 | 0.9112 | n/a | n/a | 0.02547 | 0.02178 | 0.01339 | 134.4 | n/a | n/a | — |
| ae-seed2-smpl_proc_v3-20261024T041446Z | plain | seeds | 0.00157 | — | 0.9049 | 0.9122 | n/a | n/a | 0.02542 | 0.02178 | 0.01324 | 133.1 | n/a | n/a | — |
| ae-pose-l0.03-smpl_proc_v3-20261025T061249Z | pose | pose-ref | 0.00215 | — | 0.8702 | 0.8389 | n/a | n/a | 0.03257 | 0.03183 | 0.01726 | 68.6 | n/a | n/a | — |
| ae-pose-l0.03-seed1-smpl_proc_v3-20261025T205801Z | pose | pose-ref | 0.00215 | — | 0.8705 | 0.8399 | n/a | n/a | 0.03265 | 0.03163 | 0.01750 | 68.7 | n/a | n/a | — |
| F1 | plain | seeds | 0.00160 | +1.2 % | 0.9056 | 0.9113 | -0.0009 | -0.91 | 0.02588 | 0.02208 | 0.01344 | 135.6 | +2.8 | +1.66 | NO-GAIN |
| F2 | pose | pose-ref | 0.00210 | -2.3 % | 0.8761 | 0.8500 | +0.0105 | +10.60 | 0.03231 | 0.03119 | 0.01713 | 69.2 | +0.5 | +0.32 | SHARPER |

seeds: n=3 bF1 mean 0.9122 sd 0.0010; probe mean 132.8 sd 1.68

| Run (test) | Kind | Ref | MSE | MSE vs ref | IoU | bF1 | Δ bF1 | Δ bF1 / sd | hands | feet | head | Probe (mm) | Δ probe (mm) | Δ probe / sd | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ae-smpl_proc_v3-20261023T004930Z | plain | seeds | 0.00156 | — | 0.9060 | 0.9147 | n/a | n/a | 0.02520 | 0.02123 | 0.01308 | 127.5 | n/a | n/a | — |
| ae-seed1-smpl_proc_v3-20261023T065844Z | plain | seeds | 0.00156 | — | 0.9053 | 0.9131 | n/a | n/a | 0.02509 | 0.02141 | 0.01305 | 130.5 | n/a | n/a | — |
| ae-seed2-smpl_proc_v3-20261024T041446Z | plain | seeds | 0.00156 | — | 0.9056 | 0.9139 | n/a | n/a | 0.02515 | 0.02151 | 0.01295 | 129.4 | n/a | n/a | — |
| ae-pose-l0.03-smpl_proc_v3-20261025T061249Z | pose | pose-ref | 0.00212 | — | 0.8718 | 0.8420 | n/a | n/a | 0.03222 | 0.03133 | 0.01665 | 66.6 | n/a | n/a | — |
| ae-pose-l0.03-seed1-smpl_proc_v3-20261025T205801Z | pose | pose-ref | 0.00212 | — | 0.8717 | 0.8422 | n/a | n/a | 0.03215 | 0.03125 | 0.01680 | 66.4 | n/a | n/a | — |
| F1 | plain | seeds | 0.00158 | +1.2 % | 0.9066 | 0.9136 | -0.0003 | -0.37 | 0.02561 | 0.02173 | 0.01316 | 132.4 | +3.3 | +2.11 | NO-GAIN |
| F2 | pose | pose-ref | 0.00208 | -2.0 % | 0.8776 | 0.8530 | +0.0109 | +13.31 | 0.03196 | 0.03075 | 0.01656 | 67.3 | +0.8 | +0.55 | SHARPER |

seeds: n=3 bF1 mean 0.9139 sd 0.0008; probe mean 129.1 sd 1.54

## Summary (QUEUE_DONE 2026-10-29 05:51 UTC)

- **The gradient loss sharpens only where outlines were damaged.** On the plain AE it does nothing (NO-GAIN: outlines already near the plain ceiling, bF1 0.912). On the pose model it recovers a real part of the lost outline (bF1 0.839 → 0.850, +11 sd; IoU 0.870 → 0.876; MSE −2 %) at no pose cost.
- It recovers ≈ 15 % of the outline gap between P2 and the plain AE (0.073 bF1). Capacity (slice 019) remains the big lever: L1 at 128-d cut MSE by 29 %.
- **Next lever for the operator goal (pose ≤ 80 mm and L1-level pictures in one model):** L2 (128-d + pose) + γ 0.5, since both effects are cheap and compatible. Decoder upsampling stays the follow-up if that is not enough.
