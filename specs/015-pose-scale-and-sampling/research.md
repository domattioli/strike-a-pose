# Research: slice 015

## R1 Baselines memory, bit-identity of the uint8 path

`load_arrays` currently fills `a["image"]` float32 from `b["image"]` (dataset tensors, values
`uint8 / 255.0` in float32). `image_mean` already accepts uint8 rows and converts per batch
with `b.to(float32) / 255.0`: the same float32 values, then the same float64 accumulation
(012 R4), so the mean image is bit-identical. Storing `round(x * 255)` as uint8 is lossless
because each value is exactly `k/255` in float32 and `round(float32(k/255) * 255) == k` for
k in 0..255 (the trainer's `load_split` relies on the same identity; checked in a test).
Only the train split is stored as uint8, and only when `image_pca_ks` is empty (PCA needs a
float matrix). Val / test stay float32 (scoring code unchanged): 1.26 GB each at v3, loaded and
freed one at a time. Peak at v3: train uint8 2.52 GB (+ labels ≈ 0.1 GB) during the fit, then
≈ 1.6 GB per scored split: ≈ 2.7 GB. Before: ≈ 10.2 GB during the fit.

## R2 Valid-sample criterion

Silhouette = `decoded > iou_threshold` (0.075, the 003 / IoU threshold, half the render ambient
floor). A sample is valid iff (a) foreground area A > 0, (b) the largest 8-connected component
holds ≥ 95 % of A (tolerates speckle; a split / two-body sample fails), (c) A within
[p1, p99] of the train masks' areas (true masks, `mask.sum()`), so blank-ish or blob-filled
frames fail. Thresholds are constants in `eval/sample.py` (`LARGEST_FRACTION = 0.95`,
`AREA_PERCENTILES = (1.0, 99.0)`), reported in `samples.json`. Not a pose-plausibility measure
(Q3 (b) rejected: circular); it measures "one body-sized figure", which a blurry prior sample
of a poorly regularized AE typically fails.

## R3 Connected components without new dependencies

SciPy is not a declared dependency (check: `pyproject.toml`); do not import it. Implement
`largest_component_fraction(mask: np.ndarray[bool, (H, W)]) -> tuple[float, int]` with an
iterative stack flood fill over 8-neighbours (no recursion) returning (largest / total,
n_components); (0.0, 0) for an empty mask. 256 × 192 × 1,024 masks in pure Python is slow
(≈ 50 k px foreground worst case each); acceptable cost: foreground is typically ≈ 10-20 %
of the frame, ≈ 1 s per mask worst case. To bound it, run labelling on masks only (bool), and
skip it when (c) already fails.

## R4 Nearest-neighbour MSE

For samples S `[n, P]` float32 and train T `[N, P]` uint8: in chunks of 2,048 train rows,
`T_f = T_chunk.float() / 255`; `d = |s|² + |t|² − 2 S T_fᵀ` (float32 matmul), track the
argmin per sample; then recompute the exact MSE for each argmin in float64. Reported: mean and
median over samples of `mse(s, nn(s))`, and the NN indices (for the figure). Float32 ranking
error is ≪ the gaps between candidates at this scale; the exact recompute makes the reported
number exact for the chosen neighbour.

## R5 Reference bars

Lower bar: the same z through R0's plain-AE decoder (an unregularized latent, not N(0, I)
distributed); upper bar: μ reconstructions of `n_recon` val images (in-distribution decodes),
scored identically. A VAE whose prior samples score near the upper bar has a usable prior.

## R6 Promotion gaps

seed_gap is relative MSE (|R0b − R0| / R0 on whole-image val MSE); probe_gap in mm on the
probe val world-frame MLP MPJPE. v1: seed_gap = |0.00216 − 0.00218| / 0.00218 ≈ 0.6 %
(from the unrounded metrics.json values in the script), probe_gap = 2.8 mm.
