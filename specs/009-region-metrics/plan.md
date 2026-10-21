# Implementation Plan: Per-Region Image Metrics

**Branch**: `claude/charming-wozniak-5ouldf` (slice `009-region-metrics`) | **Date**: 2026-10-21 | **Spec**: [spec.md](spec.md)
**Input**: `specs/009-region-metrics/spec.md` (approved at surrogate scope, surrogate rulings 2026-10-20 batch Q1–Q3 = (a); operator sign-off pending; approval basis: the operator's choice (i) for roadmap question #16)

## Summary

Every image model the project scores (003 mean image and PCA k; 004 `ae`, `mean_image`,
`pca_k<K>`) additionally reports its MSE inside three image regions built from the true 2D
keypoints: `hands` (joints 22 / 23, squares of side 21 px), `feet` (7 / 8 / 10 / 11, 17 px)
and `head` (15, 29 px). Two pure batched functions join the 003 metrics (`region_mask`,
`masked_mse`); the one scoring seam, `score_images`, takes the true keypoints,
visibility and image size as one optional keyword group and, given them, adds a
`regions` record per model from the same predictions as the headline. The 003 baselines
already hold the labels; the 004 trainer's loader now keeps them for the scored splits.
`metrics.json` only gains keys (per-entry `regions`, top-level `regions` definitions,
`units.regions`); `metrics_version` stays 1 and `train_version` 2. No config key,
CLI flag, figure or training change.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.11 + 3.13); 3.11-compatible code only.
**Primary Dependencies**: none new (torch, NumPy); `pyproject.toml` / `uv.lock` untouched.
**Storage**: run directories under git-ignored `runs/`; no new file; `metrics.json` gains
keys.
**Testing**: pytest. Hand-computed masks and masked MSE on NumPy and torch inputs;
`score_images` on synthetic 64 × 48 images with known errors; the committed fixture for
placement (≥ 10 % body pixels per region) and for the 003 / 004 runs (regions present,
003 = 004 for `mean_image` and `pca_k4`).
**Target Platform**: Linux CPU (4 vCPU container; CI `ubuntu-24.04`).
**Project Type**: single library + CLIs.
**Performance Goals**: region scoring of one model on one 1,600-image split ≤ 5 s on 4 vCPU
(SC-005; prototype 2.4 s at batch 256, 1.3 s at batch 64, research R5); pytest stays under
~45 s locally (267 passed in 39.6 s at `702d2f5` in this session, 31.7 s at `5382075` in
the previous one); per-leg CI < 5 min.
**Constraints**: every existing `metrics.json` value, figure, resolved config, history and
CLI output line unchanged (SC-002); headline values bit-identical with and without labels;
`docs/results/` untouched; CI workflow and the 006 probe unchanged; nothing under `data/`,
`runs/`, `assets/` committed.
**Scale/Scope**: four source files (`eval/metrics.py`, `eval/report.py`,
`eval/baselines.py`, `train/trainer.py`) plus the `eval` exports; one new test module, three
edited ones; two contracts and the docs.

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | Region constants, mask and masked MSE are pure functions in `strike_a_pose.eval.metrics`; the scoring seam and the definitions block in `strike_a_pose.eval.report`; both CLIs only pass labels through. |
| II CLI + config | PASS | No config key (FR-008, clarification 12): the regions are code constants pinned by the run's code commit (`run_info.json`) and written into `metrics.json` (top-level `regions`); resolved configs and CLI flags unchanged. |
| III Reproducibility | PASS | Deterministic masks (integer pixel arithmetic) and float64 aggregation; `metrics.json` bytes repeat for the same inputs, code and machine / thread count (003 / 004 byte tests and the 004 resume tests cover the new keys). |
| IV Test-first | PASS | FR-009 tests (tasks T002–T005) are written and run red (T006) before T007–T010. |
| V Honest evaluation | PASS | Sizes fixed a priori from body dimensions and fixture crops, never from val / test numbers; every image model, baselines included, is scored on the same pixels through the same code; the published 003 / 004 / 006 numbers are not re-scored, and the docs say per-region numbers come with the next full run. |
| VI Simplicity | PASS | Two functions, one constant table, one optional argument group, one definitions helper; squares (no discs, depth scaling, per-side regions or segmentation); no new dependency, file type or option. |

## Design

### `strike_a_pose/eval/metrics.py`

- `REGIONS: dict[str, tuple[tuple[int, ...], int]]`: region name → (SMPL-24 joint indices,
  half-size h in px), in the FR-001 order: `hands` ((22, 23), 10), `feet` ((7, 8, 10,
  11), 8), `head` ((15,), 14). Comment: sized for the 001 ring (3.5 m, focal 320 px,
  256 × 192); depth scaling `h = f · s / z` is the recorded generalization (spec FR-001).
- `region_mask(keypoints, visible, image_size, joints, half_size) -> Array`: keypoints
  `[B, J, 2]` (u, v) px, visible `[B, J]` bool, `image_size` `(H, W)`, joint indices,
  `half_size` h ≥ 0 (else `ValueError`); bad shapes → `ValueError`. For each listed joint:
  `c = ⌊u⌋`, `r = ⌊v⌋` in float64 (exact for float32 and float64 inputs), a row band
  `|rows − r| ≤ h` gated by the joint's visibility (`[B, H]`) and a column band
  `|cols − c| ≤ h` (`[B, W]`), OR-ed as their outer product into a bool `[B, H, W]` that
  starts all False. Only in-image pixels exist, so clipping is implicit; NaN / inf
  coordinates of invisible joints compare False. Returns the keypoints' array type
  (research R1, R2).
- `masked_mse(pred, target, mask) -> Array`: equal shapes `[B, ...]` (else `ValueError`);
  squared error `(pred − target)²` as in `image_mse` (non-float `pred` is promoted to
  float64), flattened; per sample `sum over mask / count`, NaN where the count is 0;
  `[B]` of `pred`'s array type and float dtype.
- Module docstring: every function returns per-sample `[B]` values, except `region_mask`
  (a `[B, H, W]` mask); both of the keypoints' / `pred`'s array type.

### `strike_a_pose/eval/report.py`

- `score_images(pred_fn, images, masks, batch_size, threshold=IOU_THRESHOLD, *,
  labels=None)`; `labels = (keypoints [n, J, 2], visible [n, J], (H, W))` of the scored
  rows. Checks with labels: `n` equal to the rows', `H · W` equal to the row length (else
  `ValueError`). Per batch: one `pred_fn` call; the headline lists exactly as today; with
  labels, for each region in `REGIONS`: `region_mask` of the batch's labels, reshaped
  row-major to `[b, H*W]`, then `masked_mse(p, x, mask)` appended to that region's list.
  Result: the three headline keys as today, plus with labels `regions = {name:
  {"image_mse": float64 mean of the non-NaN per-sample values (NaN if none), "samples":
  their count}}` (research R4).
- `region_definitions() -> dict`: `{name: {"joints": [SMPL24_JOINT_NAMES[j] …],
  "half_size_px": h}}` from `REGIONS` (the top-level `regions` block of both
  `metrics.json` files).

### `strike_a_pose/eval/baselines.py` (003, FR-005)

- `UNITS["regions"]`: one sentence, unit and aggregation (data-model).
- `score_split`: `score_images(…, labels=(a["kp2d"], a["visible"], a["image_size"]))`
  for every image predictor; pose metrics unchanged.
- `run_baselines`: `metrics["regions"] = region_definitions()`.

### `strike_a_pose/train/trainer.py` (004, FR-006)

- `IMAGE_UNITS` selects `regions` too.
- `load_split(ds, batch_size, masks=False)`: with `masks` (the scored splits) also
  `kp2d` float32 `[n, 24, 2]` (the dataset's `keypoints_2d`) and `visible` bool `[n, 24]`
  (`visibility`). Val keeps them exactly when it keeps its masks (`"val" in
  cfg.eval_splits`); every other eval split is loaded with `masks=True` as today.
- Evaluation loop (after `load_selected_weights`): `score_images(fn, x, a["mask"],
  cfg.batch_size, cfg.iou_threshold, labels=(a["kp2d"], a["visible"], a["image_size"]))`
  for every image model; `metrics["regions"] = region_definitions()`. Training, the
  per-epoch val loss, the selection, `predict_rows`, the figure and the log lines are
  untouched.

### `strike_a_pose/eval/__init__.py`

Exports `REGIONS`, `masked_mse`, `region_mask` next to the other metrics.

### Tests (FR-009; written first)

- `tests/test_eval_regions.py` (new): `REGIONS` = the FR-001 table (joint names via
  `SMPL24_JOINT_NAMES`) and `region_definitions()`; (a) interior square at fractional and
  integer coordinates; (b) every border and corner, incl. a visible joint at u just below
  W; (c) union counts; (d) invisible joints, one just outside the image, an all-invisible
  region; (e) masked MSE cases, `ValueError`, dtype; (f) NumPy vs torch identical, float32
  vs float64 keypoints identical; (g) fixture placement (all 48 sample-region pairs ≥ 10 %
  body pixels; with u / v swapped, at least half of them fall below 10 % — 33 measured);
  (h) `score_images` with labels: headline bit-identical to the call without labels (which
  has exactly the three headline keys), `pred_fn` called once per batch, hand-computed
  region values (error 0.5 inside the head square, 0.25 elsewhere, zero targets), region
  values = mean of per-sample `masked_mse` over non-empty samples, `samples` with one
  sample's hands invisible and with every sample's head invisible (NaN, written null), the
  same dict for batch sizes 1, 3 and n; the label-length check.
- `tests/test_eval_cli.py` (i): the fixture run's test entries carry the three regions
  (`image_mse` in [0, 1], `samples` 8), the top-level `regions` block and `units.regions`;
  pose entries have no `regions`; val stays `{"samples": 0, "poses": 0}`.
- `tests/test_train.py` (j, k): `load_split` keeps `kp2d` / `visible` equal to the loader's
  only with `masks=True`; the fixture train run's `ae`, `mean_image`, `pca_k4` carry the
  regions, the block and `units.regions`; (k) the 003 fixture run and a small 004 fixture
  run (1 epoch, small model: the baselines do not depend on the AE) have identical
  `mean_image` and `pca_k4` test entries, regions included, and identical `regions`
  definitions and units.
- `tests/test_train_selection.py`: `_rescore` passes the labels (FR-009 (j)), so its
  whole-entry `ae` equalities cover the regions of the selected weights.

### Where the next slice hooks in

Training variants (mask-weighted MSE, VAE, regularization) keep their predictor behind a
`pred_fn` (`predict_rows` for AE-like models) and score it through `score_images` with the
split's `labels`, so their `metrics.json` entries get `regions` without further change.
A per-region IoU, if needed, is new keys inside each region record (`metrics_version`
stays 1).

## Project Structure

### Documentation (this feature)

```text
specs/009-region-metrics/
├── spec.md
├── plan.md            # this file
├── research.md
├── data-model.md
└── tasks.md
```

Normative contract text stays in the slices that own it (updated in place):
`specs/003-baselines-and-metrics/contracts/eval-run.md` (metrics API rows, `REGIONS`,
scoring seam, `metrics.json` keys) and `specs/004-autoencoder-mvp/contracts/train-run.md`
(`metrics.json` keys, labels kept for scored splits).

### Source Code (repository root)

```text
src/strike_a_pose/eval/metrics.py      # REGIONS, region_mask, masked_mse
src/strike_a_pose/eval/report.py       # score_images(labels=…), region_definitions
src/strike_a_pose/eval/baselines.py    # UNITS["regions"], labels passed, regions block
src/strike_a_pose/eval/__init__.py     # exports
src/strike_a_pose/train/trainer.py     # load_split keeps kp2d/visible, labels passed, block
tests/test_eval_regions.py             # new
tests/test_eval_cli.py, tests/test_train.py, tests/test_train_selection.py
README.md, specs/roadmap.md, docs/HANDOFF.md,
specs/003-baselines-and-metrics/spec.md, specs/004-autoencoder-mvp/spec.md
```

**Structure Decision**: existing single package; no new module in `src/`.

## Complexity Tracking

None.
