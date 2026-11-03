# Research: 019 Latent Capacity

Measured at `dd61515`, CPU, cheap checks only (model construction, no training).

## R1 Bottleneck and wiring

`ConvAutoencoder((256, 192), d, [16, 32, 64, 128, 256, 256])`: bottleneck `(256, 4, 3)`,
flat 3072. Encoder ends `Linear(3072, d)`, decoder starts `Linear(d, 3072)`; VAE `logvar`
`Linear(3072, d)`. Any `1 <= d` is legal; 128 ≪ 3072.

| d | AE params | PoseHead params | forward `[2,1,256,192]` |
|---|---|---|---|
| 64 | 3,126,897 | 100,936 | ok |
| 128 | 3,520,177 | 117,320 | ok |

## R2 Consumers

`grep latent_dim src/`: trainer passes `cfg.latent_dim` to the AE and `PoseHead`; probe
allocates `[n, m.latent_dim]` and rebuilds the random encoder from checkpoint hparams; sample
and round trip draw `randn(n, vae.latent_dim)`; sample refuses a reference checkpoint of a
different `latent_dim`. No literal 64 on these paths (`context_pca_k` is null on v3).

## R3 Cost

Extra FLOPs: two 3072×64 matmuls per image, < 0.1 % of the conv stack. Wall time per v3 run as
R0-v3 / P2 (5.5–6.5 h incl. early stop at epochs 53–58), probe ≈ 45 min. Peak RSS 018 ≈ 6.0 GB.

## R4 Follow-ups (not in this slice)

256-d if L2's pixel cost halves; a 128-d second seed (only if L1's pixel delta is within
2–3 sd, i.e. ambiguous); 128-d + λ 0.1 if pose keeps improving with λ.
