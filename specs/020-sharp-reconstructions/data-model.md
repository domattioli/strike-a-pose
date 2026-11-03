# Data Model: Sharper Reconstructions (slice 020)

## Config / Objective

`TrainConfig.gradient_weight: float = 0.0` (after `pose_weight`), validated: finite, ≥ 0
(message `"gradient_weight: must be >= 0"`); `> 0` with `mask_weight > 0` →
`"gradient_weight: not with mask_weight"`; `> 0` with `variational` →
`"gradient_weight: not with variational: true"`. `LEGACY_VALUES["gradient_weight"] = 0.0`.
Must be added to the numeric-key finiteness list beside `mask_weight`.
`Objective.gradient_weight: float = 0.0` (last field); `from_config` passes it.

## `gradient_loss(y, x) -> Tensor` (eval/metrics.py, torch only)

`y, x` float `[B, 1, H, W]`:
```
dh = (y[..., 1:, :] - y[..., :-1, :]) - (x[..., 1:, :] - x[..., :-1, :])
dw = (y[..., :, 1:] - y[..., :, :-1]) - (x[..., :, 1:] - x[..., :, :-1])
return dh.pow(2).mean() + dw.pow(2).mean()
```
Scalar, differentiable in `y`; 0 iff `y − x` is constant per image.

## `boundary_f1(pred, mask, threshold=IOU_THRESHOLD, tol_px=BOUNDARY_TOL_PX) -> Array`

`pred [B, H*W]` or `[B, H, W]` float, `mask` same shape bool (as `silhouette_iou`); image
shape from the caller: signature takes `image_size: tuple[int, int]` keyword when inputs are
flat (`[B, H*W]`). Steps (float32 torch, `[B, 1, H, W]`):
1. `p = (pred > threshold)`, `m = mask`.
2. contour `c(b) = b − erode(b)`, `erode(b) = −max_pool2d(−b, 3, stride 1, padding 1)`
   (padding with −inf, i.e. pixels outside the image do not erode; a body touching the border
   has no contour there).
3. `k = 2·tol_px + 1`; `dil(c) = max_pool2d(c, k, 1, tol_px)`.
4. `P = Σ(c(p)·dil(c(m))) / Σ c(p)`, `R = Σ(c(m)·dil(c(p))) / Σ c(m)` per image.
5. F1 = 2PR / (P + R); both contours empty → 1.0; exactly one empty → 0.0; P + R = 0 → 0.0.
Returns `[B]` in the input's array type (`_out`), dtype float64 for non-float input as IoU.

## `score_images` result

Adds `"boundary_f1": float` (mean per image) next to `silhouette_iou`, from the same
predictions and threshold. Units: `"boundary_f1": "F1 of the 1-px-tolerance match between the
contours of (prediction > iou_threshold) and of the true body mask, per image mean (spec 020)"`.

## History

`HISTORY_FIELDS = (..., "train_pose", "train_grad")`; `train_grad` = epoch mean (sample
weighted as `train_recon`) of `gradient_loss`, `None` (empty cell) when off. Train loss per
batch: `recon + γ·grad [+ λ·pose]` (`recon` the MSE term; with `logvar` the VAE branch is
unreachable because of the config refusal).

## `fidelity.json` (written by `rescore` into the run dir)

```json
{"fidelity_version": 1, "checkpoint": "runs/<run>/best.pt", "run_dir": "runs/<run>",
 "dataset_dir": "...", "split_file": "...", "iou_threshold": 0.075, "boundary_tol_px": 1,
 "pose_weight": 0.03, "gradient_weight": 0.0,
 "splits": {"val": {"samples": 6400, "image_mse": f, "silhouette_iou": f, "boundary_f1": f,
                    "regions": {"hands": {"image_mse": f, "samples": n}, "feet": ..., "head": ...}},
            "test": {...}}}
```
`pose_weight` / `gradient_weight` read from the run's `config.yaml` (missing → 0.0). Dataset,
split file, iou threshold default to the run's `config.yaml` values.

## Fidelity-report row (per split, per variant)

`name, kind ("plain" | "pose"), ref ("seeds" | "pose-ref"), mse, mse_rel (vs ref mean),
iou, bf1, d_bf1, d_bf1_sd, hands, feet, head, probe_mm, d_probe_mm, d_probe_sd, verdict`.
`kind = "pose"` iff `pose_weight > 0`. sd of bF1 and of probe MPJPE: sample sd (ddof 1) over
the `--seed` runs. Probe MPJPE from the run's `probed.txt` → `<probe dir>/metrics.json`
`splits.<s>.targets.joints_world.mlp.mpjpe_mm`. Verdict: `SHARPER`, `SHARPER-COSTLY`,
`NO-GAIN`, `NO-PROBE` (spec Q8). Reference rows (seeds, pose-ref) are printed too, verdict "—".
