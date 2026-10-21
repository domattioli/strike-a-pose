# Research: 003 Baselines and Metrics

## R1 — PCA on 12.8k full-resolution images

- Train split of `smpl_proc_v1`: 1600 poses × 8 views = 12,800 images of
  256 × 192 = 49,152 pixels → `X` is 12,800 × 49,152, ≈ 2.5 GB in float32
  (+ 2.5 GB centred copy). Fits the 15 GB dev machine; CI only sees the fixture.
- Options: (a) full SVD of `X` (≈ 8·10¹² flop, needs `U` 12.8k² too);
  (b) eigendecomposition of the Gram matrix `X Xᵀ` (12.8k², 0.65 GB float32,
  ≈ 8·10¹² flop for the product, eigh ≈ 3·10¹² flop); (c) randomized SVD
  (`torch.svd_lowrank`, ≈ 5·10¹¹ flop, approximate, seed-dependent);
  (d) incremental PCA (sklearn — new dependency).
- **Decision**: (b) for `n ≤ d` and the covariance `Xᵀ X` for `n > d` (pose:
  12.8k × 72). Exact, deterministic, torch only, minutes on 4 cores. Products in
  float32, eigendecomposition in float64. Components with eigenvalue
  ≤ 1e-6 × the largest are dropped (numerical rank); `k_effective` recorded.
  Rejected (c): an approximation knob for a baseline adds a hidden choice;
  (d): new dependency (VI).

## R2 — MPJPE frame and the 2D score of pose baselines

- 001 labels: global orientation zeroed, so world-frame root-relative joints are
  the same for all 8 views of a pose; each camera has a fixed rotation `R_c`.
- For any prediction `P` in world frame, camera-frame root-relative error
  `‖R_c (P − G)‖ = ‖P − G‖` per joint: MPJPE is rotation-invariant. A
  world-frame mean pose rotated per camera equals the per-camera mean of
  camera-frame joints (`R_c` is constant per camera). So the pose baselines are
  fit and scored in the world frame; the numbers equal camera-frame
  root-relative MPJPE (the usual H3.6M "protocol 1" form, no Procrustes).
- 2D: predicted root-relative joints + true root (world) → project with true
  `K, R, t` (`u = fx·x/z + cx`, `v = fy·y/z + cy`, same as 001 `cameras.project`)
  → pixel distance to the label `keypoints_2d` over visible joints.
- *Note 2026-10-21 (slice 010)*: with the `camera_jitter` option `R` varies per sample,
  so "`R_c` constant per camera" no longer holds for such datasets; the world-frame
  scoring (MPJPE is rotation-invariant) and the per-sample 2D projection are unaffected.

## R3 — Silhouette threshold

- Renderer (001 `render.py`): body grey = `round(255·(0.15 + 0.65·shade))`, so
  body pixels ≥ 38/255 ≈ 0.149; background exactly 0. Fixture check: min body
  value 0.153, no non-zero background pixel.
- **Decision**: threshold 0.075 (half the ambient floor): a true image
  thresholded equals its mask exactly (tested on the fixture), and the margin to
  both classes is equal.

## R4 — Run directory helpers

- 001 already has `git_commit()` (commit + dirty flag) and `sha256_file`; reuse.
- New `strike_a_pose.runlog`: `seed_everything`, `environment_summary`,
  `make_run_dir`, YAML/JSON writers — shared with 004's train CLI, so not under
  `eval` (avoids train importing eval).
- `metrics.json`: `json.dumps(sort_keys=True, indent=2)` after rounding every
  float to 6 **significant** digits (001's `write_json` rounds to 6 decimals,
  which would leave only 3–4 digits of a small MSE); NaN → `null`. Rounding
  absorbs last-bit BLAS differences, so repeated runs give identical bytes.

## R5 — Figure without matplotlib

PIL is already a dependency: paste uint8 tiles into a grid with a text header
(`ImageDraw.text`, default bitmap font). Adding matplotlib for one grid is not
justified (VI).
