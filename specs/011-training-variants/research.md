# Research: 011 Training Variants

Checks run 2026-10-21 at `19479e2` in the locked environment (torch 2.14.1, CPU, Python 3.13)
with scratch scripts; nothing committed from them.

## R1 — AdamW at `weight_decay = 0` equals Adam

Decision: always build `torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)`.
Evidence: 5 steps of a small `ConvAutoencoder` from the same seed with `Adam` and with
`AdamW(weight_decay=0.0)`: parameters bitwise equal. `AdamW` is `Adam(decoupled_weight_decay=
True)`; a saved `Adam` state dict loads into `AdamW` (its `__setstate__` forces
`decoupled_weight_decay` true; `weight_decay` 0 is kept), so pre-011 checkpoints resume.
`AdamW`'s own default `weight_decay` is 0.01, hence the explicit argument. Alternatives:
coupled L2 through Adam (scaled by 1/√v, weak and uneven; not offered); `Adam` when 0 and
`AdamW` otherwise (a branch for no gain once R1 is pinned by a test).

## R2 — The α = 0 branch

Decision: `recon_loss` returns `F.mse_loss(y, x)` when `mask_weight == 0`. Evidence: the
general formula `Σ w·d² / Σ w` with `w ≡ 1` equals `F.mse_loss` bitwise in value and gradient
at 4 × 1 × 64 × 64 and at the full batch 64 × 1 × 256 × 192 (channels-last); the design memo's
"backward kernels differ" did not reproduce. The branch is kept as a guarantee that does not
depend on reduction order or thread count.

## R3 — The silhouette is `x > 0`

Decision: the mask weight uses `m = (x > 0)` of the (shifted) batch; train masks stay
unloaded (≈ 0.6 GB at 12,800 images). Evidence: the renderer writes body pixels as
`round(255·(0.15 + 0.65·shade))` (`AMBIENT`, `DIFFUSE` in `datagen/render.py`) on a zero
background and the mask at the same pixels; fixture: `(image > 0) == mask` for 16 / 16 samples,
darkest body pixel 39, body fraction 7.46 % (so α = 4 gives the body ≈ 29 % of the weight).

## R4 — KL scale and β

Decision: `loss = recon + β·KL/(H·W)`, KL in nats per image (mean over the batch). Then the
loss is `(SSE + β·KL)/(H·W)` per image: β = 2σ² of a Gaussian decoder (β = 1 ↔ σ² = ½); the
004 AE's best val MSE 0.00217 ↔ a "calibrated" β ≈ 0.004. First value β = 0.1 (Q3 ruling):
real pressure toward the prior for later sampling, reconstruction expected somewhat blurrier.

## R5 — `logvar` head placement and initialization

Decision: `logvar = nn.Linear(flat, latent_dim)` on the input of `encoder[-1]`, created after
`encoder` and `decoder`. Evidence: `encoder[-1](encoder[:-1](x))` equals `encoder(x)` bitwise;
the head adds 3072 · 64 + 64 = 196,672 parameters (3,126,897 → 3,323,569). Built last, it
leaves the RNG stream of every other parameter unchanged: a VAE starts from the plain AE's
weights at the same seed, and the probe's random-encoder control (a plain model) is the
untrained VAE's encoder.

## R6 — Dropout in the trainer, not the model

Decision: `F.dropout(z, p, training=True)` in `train_epoch`, only when `p > 0`. Evidence:
`F.dropout` at p = 0 draws nothing either, but the explicit branch avoids depending on it.
Keeping it out of the model means no hparam, no state key, and no eval path can apply it
(memo alternative: an `nn.Dropout` attribute on the model).

## R7 — Shift implementation

Decision: one `torch.randint(-k, k + 1, (B, 2))` per batch, `F.pad` of the uint8 batch by k
zeros, a per-sample crop (64 slices per batch, negligible next to the convolutions). uint8
values are copied, never interpolated; eval batches are never shifted. Zero fill clips bodies
at the frame edge (004 research R0: val silhouettes span x 0–189), accepted at k = 4.

## R8 — Cost

`configs/ae.yaml` ≈ 85 min on 4 vCPU, probe ≈ 8 min; the variants add a 3072 → 64 linear
layer, a few element-wise ops per batch and 64 slices: estimated a few % per step (measured on
the fixture in stage 5). R1–R3 ≈ 4.7 h sequential.
