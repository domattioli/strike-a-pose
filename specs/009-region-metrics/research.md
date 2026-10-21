# Research: 009 Per-Region Image Metrics

Measurements in this file: committed fixture and synthetic data, this container (4 vCPU,
Python 3.13, torch 2.14.1), 2026-10-21, with a prototype of the planned functions
(identical arithmetic).

## R1 — Box geometry and sizes

A region is the union of axis-aligned squares of side `2h + 1` px, one per visible region
joint, centred on the pixel that holds the joint. The renderer samples pixel (r, c) at its
centre (c + 0.5, r + 0.5), so pixel (r, c) covers [c, c + 1) × [r, r + 1) and the joint
at (u, v) lies in row ⌊v⌋, column ⌊u⌋; the square is `|r − ⌊v⌋| ≤ h ∧ |c − ⌊u⌋| ≤ h`, an
exact pixel set whose size a test can state by hand ((2h + 1)² inside the image, the
product of the clipped extents at a border). A square rather than a disc: the operator's
"head box", and exact counts (clarification 3).

Sizes (FR-001, surrogate ruling Q1 (a)) come from body dimensions at the ring's 91.4 px/m
(3.5 m, focal 320 px) and from fixture crops, never from val / test numbers:
hands h 10 (21 px ≈ 23 cm: knuckle joint, wrist ≈ 8 px behind, fingertips ≈ 9 px beyond),
feet h 8 around both ankle and foot joints (two 17-px squares per foot: heel to toes),
head h 14 (29 px ≈ 32 cm: the head joint sits low in the head; crown ≈ 12–16 px above it).

Placement check on the 16 fixture samples (all 24 joints visible), true-body pixels as a
share of each region's pixels:

| Region | pixels per sample | body fill mean | body fill min |
|---|---|---|---|
| hands | 762–882 | 50 % | 23 % |
| feet | 678–1028 | 39 % | 32 % |
| head | 841 | 70 % | 53 % |

With u and v swapped (the transposition bug a test must catch), 33 of the 48
sample-region pairs fall below 10 % (6 feet regions become empty: ⌊v⌋ used as a column is
beyond the 192-px width). FR-009 (g) therefore asserts ≥ 10 % for every true pair and at
least 24 failing swapped pairs.

Depth scaling (`h = f · s / z` per joint from `K` and `joints_3d_cam`) stays the recorded
generalization (spec FR-001); the fixed squares fit the current camera geometry only.

## R2 — Visibility semantics

The label's `visibility` decides (001 research R6: in front of the camera and
`0 ≤ u < W`, `0 ≤ v < H`; no self-occlusion), as in 003's `keypoint_error_2d`
(clarification 5). An invisible joint adds no pixel, even when its square would reach into
the image; a region whose joints are all invisible is empty for that sample, which is
then left out of the region's mean and of its `samples` count; other regions of the
sample are unaffected. On the ring every joint is in front of the camera, so "invisible"
means "outside the frame" today.

Arithmetic: ⌊·⌋ is taken in float64. A float32 keypoint converts to float64 exactly, so the
004 trainer (float32 keypoints, as the dataset returns them) and the 003 baselines (the
same values held as float64) build identical masks. Coordinates of invisible joints are
never used (the row band is gated by visibility; NaN or inf would compare False anyway).
Edge: a visible joint stored at u < W in the label JSON could round to u = W in float32
(only for u within 2⁻¹⁷ of W); its square is then clipped to h columns instead of h + 1 —
well defined and identical in both paths.

## R3 — Where the keypoints flow (trainer evaluation path)

- 003: `load_arrays` already holds `kp2d` (float64 `[n, 24, 2]`), `visible` and
  `image_size` for every eval split; only the `score_images` call in `score_split`
  changes.
- 004: `load_split(ds, batch_size, masks)` keeps uint8 images, `pose_index` and, with
  `masks=True`, bool masks. It now keeps `kp2d` (float32) and `visible` exactly when it
  keeps masks (≈ 0.2 KB per sample; 0.35 MB for 1,600 samples). Val is loaded once before
  training with `masks="val" in cfg.eval_splits` and held for the whole run; every other
  eval split is loaded in `_train_and_evaluate`'s evaluation loop with `masks=True`. The
  per-epoch val loss reads only the images.
- The evaluation loop runs after `load_selected_weights` (008): its `score_images` call
  gains `labels=(a["kp2d"], a["visible"], a["image_size"])` for every model in `preds`
  (`ae` = `predict_rows` of the selected weights, `mean_image`, `pca_k<K>`), so the `ae`
  regions come from the same predictions as the `ae` headline. Nothing before it changes,
  so training, history, checkpoints, `best.pt` and the selection are untouched, and a run
  directory started before 009 resumes unchanged (no config key, no checkpoint field) and
  reports regions when it finishes.

## R4 — Aggregation inside the seam

Per batch: `p = pred_fn(x)` once; the headline lists exactly as before; then per region a
`[b, H, W]` mask from the batch's labels, reshaped row-major to `[b, H*W]` (the rows are
`[n, 1, H, W]` images reshaped the same way), and `masked_mse(p, x, mask)` (per sample, in
`p`'s float dtype, NaN where empty). After the loop: per region, the float64 mean of the
non-NaN values and their count. Each image counts once, as for the headline MSE
(clarification 8). Masks are never stored per split (3 × 1,600 × 49,152 B ≈ 236 MB); they
depend only on the labels and are rebuilt for each model (R5).

Bit-identity of the headline with and without labels holds by construction (the same `p`,
no in-place operation). Per-sample values do not depend on the batch composition on the
synthetic test images (each row is reduced on its own); the 003 and 004 fixture runs use
the same batch size (4) and the same float values (004 `to_float` of uint8 equals the
loader's floats), so their `mean_image` / `pca_k4` region values are identical, as their
headline values already are.

## R5 — Cost

Prototype (the planned arithmetic), one model, one synthetic split of 1,600 images of
256 × 192 with random labels, mean-image predictor: regions add 2.4 s at batch 256 (003
config) and 1.3 s at batch 64 (004 config) — within SC-005 (≤ 5 s). Per full run: 003
scores 6 image models on 2 splits (≈ 30 s added to ≈ 3 min), 004 scores 3 on 2 (≈ 8 s added
to ≈ 85 min). Building the masks once per split and reusing them across models would save
most of that but needs the 236 MB of stored masks or a cache keyed by batch; not worth it
(VI). Peak extra memory per batch: one `[b, H, W]` bool mask per joint step plus a
`[b, H*W]` float temporary (≈ 63 MB at b = 256).

## R6 — Versions and readers

`metrics_version` stays 1 and `train_version` stays 2: keys are only added — `regions`
inside image-model entries of splits with samples, a top-level `regions` definitions
block and `units.regions` — and no existing key changes meaning (003 contract: breaking
changes bump; precedents: 007's additive `run_info.json` key, ruling #3's additive
`training.best_val_*`). In-repo readers of these files read named headline keys only
(`eval/cli.py` output lines, the trainer's `_headline`, the tests); the 006 probe reads
only the train run's `run_info.json`. Resolved configs, `LEGACY_VALUES`, checkpoints and
`best.pt` are unchanged, so the 008 resume and selection logic needs nothing.
