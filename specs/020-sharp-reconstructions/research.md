# Research: Sharper Reconstructions (slice 020)

## R1 Reading of "fidelity of the renderings"

The operator note cites IoU 0.905 → 0.870 and MSE +36 %, both reconstruction metrics of the
018 results (`docs/results/018-pose-supervised/README.md`). The dataset images come from
`datagen/render.py`, a deterministic NumPy z-buffer rasterizer: flat-shaded grey body on black,
light at the camera, one sample per pixel centre (no anti-aliasing). They are the ground truth
every metric is scored against and look sharp in every `recon_val.png` ("true" column); no
roadmap, HANDOFF or grill entry complains about them. Alternative reading, not designed: the
renderer has no anti-aliasing and flat shading, so "fidelity" could mean smoother edges /
smooth shading. Evidence for that reading is absent; the operator is asked (report Q1).

## R2 Character of the blur (cheap check, 320 val images, every 20th of the split)

Script: a throw-away scratch script (not committed) loading `runs/<run>/best.pt` with
`load_checkpoint` and the v3 val split with `load_split(..., masks=True)`; 2 min on 4 vCPU.

| Model | MSE | grad ratio | GDL² | bF1 @1 px | bF1 @2 px | GDL¹ (L1) |
|---|---|---|---|---|---|---|
| R0-v3 plain | 0.00130 | 0.720 | 0.00232 | 0.929 | 0.978 | 0.0120 |
| P2 λ 0.03 | 0.00181 | 0.709 | 0.00254 | 0.861 | 0.951 | 0.0127 |
| all-black | — | 0 | 0.00383 | — | — | 0.0133 |

grad ratio = mean ‖∇ŷ‖ / mean ‖∇x‖ (forward differences); GDL² = Σ over the two axes of the
mean squared difference of forward differences (FR-002 definition); bF1 = boundary F1 of
(ŷ > 0.075) vs the true mask (FR-002). Subset MSE is below the full-split values (0.00158 /
0.00215); the ordering is the same.

Readings: (i) both models recover only ≈ 40 % of the true gradient energy and the L1 gradient
error is ≈ that of a black image: the interior shading facets are not reconstructed at all;
(ii) pose supervision costs little extra gradient energy (+9 %) but a lot of contour accuracy
(bF1 0.929 → 0.861): its blur is mostly misplaced / missing extremities, consistent with the
region MSE (hands +28 %, feet +47 %); (iii) at 2 px the F1 is near ceiling, so 1 px is the
metric's tolerance. Figures: soft edges, hands / feet shortened, folded limbs smeared, no
checkerboard.

## R3 Calibration of γ

At R0-v3 convergence GDL² ≈ 1.8 × MSE (0.00232 / 0.00130). With `loss = MSE + γ·GDL²`,
γ = 0.5 makes the gradient term ≈ 0.9 × the MSE term. GDL² on 2× average-pooled images is
0.00168 (R0-v3) vs 0.00608 for black (72 % recovered): the coarse scale is already mostly
learnt, so a multi-scale version adds little; single scale chosen.

## R4 Rejected loss candidates

- SSIM / MS-SSIM: window size, constants, and scale weights are extra hyper-parameters; on
  near-binary images with black background SSIM is dominated by the background's undefined
  structure term (constants then decide); no in-repo implementation. Follow-up if GDL fails.
- BCE on the sigmoid output: for soft targets its minimiser is the same conditional mean as
  MSE; it re-weights gradients near 0 / 1 but does not remove averaging; R2's BCE 0.0525 vs
  0.0546 tracks MSE.
- Perceptual (VGG) loss: needs pretrained weights (download, licence check, dependency),
  3-channel natural-image features on 1-channel silhouettes; violates VI.
- Adversarial loss: training instability, a second network, far over budget.
- L1 GDL (p = 1): its magnitude is dominated by unrecoverable interior facets (≈ black-image
  level), so it would push the decoder to hallucinate texture; p = 2 is in MSE units.

## R5 Decoder alternatives (deferred)

Nearest-neighbour upsample + 3×3 conv instead of `ConvTranspose2d(4, 2, 1)`: fixes checkerboard
artefacts, which the figures do not show. More decoder channels: blur grows when the same
decoder shares a latent with pose (P2), so the bottleneck, not decoder capacity, is the
binding constraint. U-Net skips: excluded by design (they bypass the latent, spec Q4). If F1
shows NO-GAIN, the next slice tries the upsampling change (one run, ≈ 6 h).

## R6 Budget

R0-v3 wall 19 396 s (5.4 h), P2 19 944 s (5.5 h). The gradient term is two shifted differences
on the output (< 1 % of the decoder's FLOPs; ≈ +3 % with autograd, estimate) → ≈ 5.7 h per
run. Probe ≈ 45 min. Rescore: one val + test pass (12 800 images: read + forward) per
checkpoint, ≈ 8 min (estimate from the trainer's final evaluation) × 7. Total ≈ 2 × 5.7 +
2 × 0.75 + 7 × 0.13 ≈ **13.8 h**, plus resume slack → ≈ 14.5 h, under 18 h. A third run
(≈ 6.5 h with probe) would exceed 18 h. Memory: as R0-v3 / P2 (≤ 6.0 GB peak RSS); the term
adds two output-size temporaries per batch (64 × 256 × 192 × 4 B ≈ 12.6 MB each).
