# 021 — Generative pose VAE (one 128-d latent for pose + pictures + sampling)

Campaign 2026-11-03 (02:30–15:32 UTC, `docs/campaign-scripts/vae021_queue.sh`). Dataset smpl_proc_v3 (8,000 poses, 800 val / 800 test poses, 6,400 images each).

- **V1** = 128-d VAE, β 0.1, pose head λ 0.03 (`ae-l128-vae-beta0.1-pose-l0.03-smpl_proc_v3-20261103T023050Z`, probe `probe-smpl_proc_v3-20261103T090644Z`)
- **V2** = 128-d VAE, β 0.1, no pose head (control; `ae-l128-vae-beta0.1-smpl_proc_v3-20261103T094030Z`, probe `probe-smpl_proc_v3-20261103T143426Z`)
- **R2b-v3** = slice 015's 64-d VAE β 0.1 (reference; new round trip `roundtrip-vae-beta0.1-smpl_proc_v3-20261103T152627Z`)

## Verdict: **PARTIAL** — failing axis: **valid samples**; next step: **pose head reads the sampled z ("head on z")**, then β 0.03

| Check (thresholds fixed before runs) | V1 | Bar | |
|---|---|---|---|
| Pose head MPJPE val / test | 66.8 / 64.8 mm | ≤ 80 mm | PASS |
| Pixel MSE val / test | 0.00171 / 0.00169 (L2 ×1.022 / ×1.023) | ≤ L2 mean × 1.05 | PASS |
| N(0,I) samples valid | 41.6 % (n 1,024) | ≥ 56 % | **FAIL** |
| Round-trip margin (plausible vs shuffled control) | 37.7 % vs 22.6 % = 15.1 pts | ≥ 13 pts | PASS |

- V2 vs V1 prior validity: 46.0 % vs 41.6 % — the head **HURTS** sampling (diff −4.4 pts, 2.0 sd).
- Going to 128-d itself costs validity: R2b-v3 (64-d) 59.0 % vs V2 (128-d) 46.0 %.
- Goal-line pixel axis (≤ +5 % of the 64-d plain seed mean): V1 is +8.4 % → not met either; the slice gate was L2 + 5 % (operator plan), which V1 passes.
- Frozen-latent probe: V1 68.0 / 66.1 mm (L2 67.1 / 64.9) — pose lives in the latent, not only the head. Pose track: PROMOTED. 015 pixel track: DROPPED (+8.6 %).
- V2: best pictures of any VAE (val MSE 0.00117, −26 % vs R0), WIN-pixel; probe 127.4 mm.

Full tables: `generative.md`, `promotion.md`, `pose_track.md`, `verdict.json`. Figures: `recon_compare_V1.png` (true | L2 | V1, 6 val poses), `samples.png` (top V1, bottom V2: prior sample | nearest training image). Run summaries copied in the run folders.

## ELI5

V1 is one model that both draws a body and says where its joints are. It reads poses as well as our best model (≈6.5 cm per joint) and its redrawings are only 2 % blurrier. But when we ask it to invent a brand-new body from random numbers, only 4 in 10 come out as a proper body (we needed 5.6 in 10). Making the latent bigger (64→128) is the main reason; adding the pose head makes it a bit worse.
