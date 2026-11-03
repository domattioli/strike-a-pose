# Tasks: Per-Region Image Metrics (slice 009)

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md); normative contracts
[`../003-baselines-and-metrics/contracts/eval-run.md`](../003-baselines-and-metrics/contracts/eval-run.md)
(metrics API, scoring seam, `metrics.json`) and
[`../004-autoencoder-mvp/contracts/train-run.md`](../004-autoencoder-mvp/contracts/train-run.md)
(`metrics.json`).
**Tests**: required (constitution IV; spec FR-009), written before the code.

## Phase 0: Baseline

- [x] T001 SC-002 baseline before any code change (at `702d2f5`, clean tree): the three
  fixture smoke commands (README "Development checks": `baseline`, `train`, `probe` on
  the train run's `checkpoint.pt`) with `--run-root <scratchpad>/runs-009-before`; keep
  the run directories and the console output (`smoke-009-before.log`). → done; pytest at
  `702d2f5`: 267 passed in 39.6 s (this session; 31.7 s at `5382075` in the previous one).

## Phase 1: Tests first (FR-009)

- [x] T002 [P] `tests/test_eval_regions.py` (new): `REGIONS` = the FR-001 table (joint
  names via `SMPL24_JOINT_NAMES`), `region_definitions()`; (a) interior joint at
  fractional (u, v) → exactly the (2h + 1)² square at (⌊v⌋, ⌊u⌋), integer coordinates
  (u = 10.0 → column 10), h = 0 → one pixel; (b) clipping at the four borders and corners
  (count = product of the clipped extents), a visible joint at u just below W; (c) two
  overlapping squares → the union's count, two joints on one pixel → one square; (d) an
  invisible joint adds nothing, also one just outside the image whose square would reach
  in, all of a region's joints invisible → empty; bad shapes / negative h → `ValueError`;
  (e) masked MSE: error only outside → 0, only inside → its mean, empty → NaN, shape
  mismatch → `ValueError`, float32 / float64 kept; (f) every case on NumPy and torch with
  identical values, plus NumPy vs torch and float32 vs float64 keypoints on random inputs;
  (g) fixture: every region of every sample ≥ 10 % true-body pixels; u / v swapped → 33
  of the 48 pairs below 10 % (an empty region counts as 0 %); (h) `score_images` on
  synthetic 64 × 48 images: headline bit-identical to the call without labels (exactly
  the three keys), `pred_fn` called once per batch, hand-computed values (zero targets,
  error 0.5 inside the head square, 0.25 elsewhere → head 0.25, hands / feet 0.0625),
  region values = mean of per-sample `masked_mse` over non-empty samples, `samples` with
  one sample's hand joints invisible (hands only) and with every head joint invisible
  (NaN, written null), identical results for batch sizes 1, 3 and n; mismatched label
  length or image size → `ValueError`. → 38 tests (backend-parametrized).
- [x] T003 [P] `tests/test_eval_cli.py` (i): fixture run, test split: every image baseline
  has the three regions with `image_mse` in [0, 1] and `samples` 8; the top-level
  `regions` block equals `region_definitions()`; `units.regions` present; pose entries
  have no `regions`; val stays `{"samples": 0, "poses": 0}`; the byte-repeat test is kept.
- [x] T004 [P] `tests/test_train.py`: `load_split` keeps `kp2d` (float32) / `visible`
  equal to the loader's keypoints / visibility only with `masks=True`; (j) the fixture
  train run's `ae`, `mean_image`, `pca_k4` on test carry the regions likewise, with the
  top-level block and `units.regions`; (k) the 003 fixture run and a small 004 fixture
  run (1 epoch, channels `[4]*6`, latent 8; batch 4 as both fixture configs) have
  identical `mean_image` and `pca_k4` test entries, regions included, and identical
  `regions` blocks and `units.regions`.
- [x] T005 [P] `tests/test_train_selection.py`: `_rescore` passes `labels=(a["kp2d"],
  a["visible"], a["image_size"])`, so the whole-entry `ae` equalities cover the regions
  of the selected weights (FR-009 (j)).
- [x] T006 Run the new and changed tests on the unchanged code: they must fail (record the
  count). → on the pre-009 code (`49a672d` + the new tests): `test_eval_regions.py` fails
  at collection (`REGIONS` not importable from `strike_a_pose.eval`);
  `test_eval_cli.py`, `test_train.py`, `test_train_selection.py`: 6 failed, 24 passed —
  `KeyError: 'regions'` 3× (the 003 and 004 fixture runs, the cross-path test) and
  `KeyError: 'kp2d'` 3× (`load_split`, the two `_rescore` users).

## Phase 2: Implementation

- [x] T007 `src/strike_a_pose/eval/metrics.py`: `REGIONS`, `region_mask`, `masked_mse`
  (plan "Design"); `src/strike_a_pose/eval/__init__.py` exports.
- [x] T008 `src/strike_a_pose/eval/report.py`: `score_images(…, *, labels=None)` (checks,
  per-batch masks, per-region aggregation), `region_definitions()`.
- [x] T009 `src/strike_a_pose/eval/baselines.py`: `UNITS["regions"]`, labels in
  `score_split`, `metrics["regions"]` in `run_baselines`.
- [x] T010 `src/strike_a_pose/train/trainer.py`: `IMAGE_UNITS` + `regions`, `load_split`
  keeps `kp2d` / `visible` with the masks, labels in the evaluation loop,
  `metrics["regions"]`.
- [x] T011 New and changed tests green; `ruff check`, `ruff format --check`; SC-005
  benchmark (scratch script: synthetic split of 1,600 images of 256 × 192, mean-image
  predictor, `score_images` with and without labels at batch 256 and 64). → the six
  related modules (`test_eval_regions`, `test_eval_cli`, `test_train`,
  `test_train_selection`, `test_eval_report`, `test_eval_metrics`): 87 passed in 28.0 s;
  ruff clean. SC-005 (real code, best of 3, 4 torch threads): without / with labels
  0.54 / 1.83 s at batch 256 and 0.11 / 1.10 s at batch 64, i.e. the regions add 1.3 s and
  1.0 s per model per 1,600-image split (≤ 5 s); per full run ≈ 15 s for 003 (6 image
  models × 2 splits) and ≈ 6 s for 004 (3 × 2).

## Phase 3: Contracts and docs

- [x] T012 Contracts in place: 003 `eval-run.md` (API rows `region_mask`, `masked_mse`,
  `REGIONS`; "Scoring" section; `metrics.json` example and text), 004 `train-run.md`
  (`metrics.json` example and text; labels kept for the scored splits). → in the plan
  commit.
- [x] T013 003 `spec.md` dated notes "amended 2026-10-21 by slice 009" (FR-001, FR-002,
  *Key Entities* `metrics.json` line); 004 `spec.md` FR-006 note.
- [x] T014 README (status line + table; Usage: metric functions sentence, baseline and
  train paragraphs; Results: one sentence that the published runs predate 009 and
  per-region numbers come with the next full run), `specs/roadmap.md` (#16: operator
  choice (i), 2026-10-20, implemented in 009, options (ii)–(v) open; "Out of MVP" line;
  009 row; status line; changelog entry appended, dated 2026-10-21), `docs/HANDOFF.md`
  (state; next work items 3 and 4), 009 `spec.md` status. → also README: the region
  paragraph after the baseline commands and the train paragraph; HANDOFF header date and
  a pointer in the 2026-10-20 session-pause note.

## Phase 4: Verify and commit

- [x] T015 Local checks: `uv run --locked ruff check .`, `ruff format --check .`, `pytest`
  (count + wall time), `strike-a-pose-data validate tests/fixtures/smpl_fixture`, the three
  smoke commands into `<scratchpad>/runs-009-after`; SC-002 diff against T001
  (`metrics.json` objects equal once the added keys are removed; figures, resolved configs
  and `history.csv` except `epoch_s` identical; console output lines identical up to paths
  and timings); `git diff 702d2f5 -- docs/results pyproject.toml uv.lock .github` empty.
  → at `1750c99` (+ docs): `ruff check .` clean; `ruff format --check .` clean (it also
  checks the Markdown files); `pytest` 307 passed in 34.3 s (40.7 s in an earlier run of
  the same code; 267 passed in 39.6 s at `702d2f5`; +40 tests); `validate
  tests/fixtures/smpl_fixture` OK; the three smoke commands into `runs-009-after` OK; the
  changed files compile under Python 3.11.17 (`/usr/bin/python3.11`; the locked 3.11
  environment runs in CI). SC-002 diff against T001:
  - 003 `metrics.json`: 44 → 69 leaves, 25 added, 0 removed, 0 changed — the top-level
    `regions` block (6), `units.regions` (1) and `regions` in `mean_image`, `pca_k2`,
    `pca_k4` on test (3 × 3 regions × {`image_mse`, `samples`}); equal to the before
    file once they are removed; `recon_test.png` byte-identical; `config.yaml` differs
    only in `run_root` (the run root).
  - 004 `metrics.json`: 48 → 73 leaves, 25 added (block 6, units 1, `ae`, `mean_image`,
    `pca_k4` × 3 × 2), 0 removed, 0 changed; equal once removed; `recon_test.png`
    byte-identical; `history.csv` equal except `epoch_s`; `checkpoint.pt` weights
    bit-identical; `config.yaml` only `run_root`.
  - 006 `metrics.json`: only `encoder.checkpoint_sha256` differs (the checkpoint file
    stores the resolved config, whose `run_root` differs, and wall-clock timings); the
    probe run with the new code on the T001 checkpoint is byte-identical to T001's
    (`metrics.json` and both figures).
  - Console output (32 lines each): identical up to run-dir paths and `(N s)` timings.
  - Cross-path: the after runs' 003 and 004 `mean_image` / `pca_k4` test entries are
    identical, regions included.
  Fixture per-region numbers (test split; val is empty in the fixture split; the AE is
  the 10-epoch smoke model; the mean-image values match the spec's prototype figures
  0.122 / 0.121 / 0.193):

  | Model | whole-image MSE | hands | feet | head |
  |---|---|---|---|---|
  | `ae` (fixture smoke) | 0.13721 | 0.0852 | 0.1138 | 0.0752 |
  | mean image | 0.01903 | 0.1219 | 0.1206 | 0.1930 |
  | PCA k=2 (003 only) | 0.01837 | 0.1200 | 0.1210 | 0.1858 |
  | PCA k=4 | 0.01802 | 0.1210 | 0.1175 | 0.1853 |

  `git diff 702d2f5 -- docs/results pyproject.toml uv.lock .github splits configs`
  empty.
- [x] T016 Commits (`plan`, `feat`, `docs`), no push (the orchestrator reviews, pushes and
  watches CI). → `49a672d` plan, `1750c99` feat, docs = the commit holding this record.

## Dependencies

T001 first. T002–T005 (parallel) → T006 → T007 → T008 → T009, T010 → T011 → T013, T014
→ T015 → T016. T012 is done in the plan commit.

## Analyze pass (lead, 2026-10-21)

Coverage: FR-001 → T002 (`REGIONS` table), T007 · FR-002 → T002 (a–d, f, g), T007 ·
FR-003 → T002 (e, f), T007 · FR-004 → T002 (h), T008 · FR-005 → T003, T004 (k), T009 ·
FR-006 → T004, T005, T010 · FR-007 → T002 (h: null), T003, T004 (j, k), T009, T010,
T012 · FR-008 → T015 (SC-002 diff: configs, figures, output lines; no config file
touched) · FR-009 → T002–T006 · FR-010 → T012–T014 · FR-011 → T015 (diff review).
SC-001 → T015 (+ CI after the orchestrator's push) · SC-002 → T001, T015 · SC-003 →
T002 (f) · SC-004 → T003, T004 (j, k) · SC-005 → T011 · SC-006 → T014, T015. Edge cases:
invisible joints and empty regions (T002 d, h), every sample empty (T002 h), joint just
outside the image (T002 d), clipping (T002 b), overlaps within a region (T002 c),
float32 vs float64 keypoints (T002 f; T004 k end to end), pixel edge u = 10.0 (T002 a),
empty split (unchanged code path: no `image`, hence no regions; T003 val check), val not
in `eval_splits` (unchanged: no masks, now no keypoints either; `load_split` test T004),
occlusion and depth limitations (documented, research R1 / R2; not testable on the
fixture).

Findings and resolutions:
1. The spec's Documentation scope dates the amendment notes and the changelog entry
   2026-10-20 (its session clock); phase 2 runs on 2026-10-21. Resolved by clarification
   16's rule (the session clock): the notes and the changelog entry carry 2026-10-21; the
   spec records this in a dated note under clarification 16 (plan commit).
2. FR-009 (g) quotes "33 of the 48 pairs below 10 %" with u / v swapped; 6 of those are
   feet regions that become empty (⌊v⌋ beyond the 192-px width). The test fixes the
   convention (an empty region counts as 0 % body) and asserts the measured 33 (research
   R1).
3. FR-009 (k) needs a 004 fixture run; the AE does not affect `mean_image` / `pca_k4`
   (fit on the same 8 train images, scored with batch size 4 in both fixture configs),
   so (k) uses a 1-epoch small-model run (≈ 0.3 s) while (j) keeps the committed
   `ae_fixture.yaml` run. pytest stays within SC-001.
4. US3-2 (a run directory started before 009 resumes unchanged): nothing a run writes
   before its evaluation changes (config, checkpoint, `best.pt`, history, `run_info.json`),
   so a pre-009 interrupted run directory equals a 009 one at the same epoch; the existing
   resume tests resume such directories with the new code and compare `metrics.json`
   bytes with uninterrupted runs, regions included. No separate test.
5. US1-2 (`ae` regions from the selected weights): `_rescore` (T005) compares whole `ae`
   entries, regions included, with a rescore of `best.pt` / `checkpoint.pt`; the selection
   code is unchanged.
6. The spec's notes name the dataset keys `keypoints_2d` / `visibility`; `load_split`
   stores them as `kp2d` / `visible`, the names 003's `load_arrays` already uses, so both
   loaders hand `score_images` the same label keys.
7. `score_images` takes the labels keyword-only (`*, labels=None`): positional callers
   keep their meaning, and a call without labels returns exactly today's three keys
   (FR-004); `tests/test_eval_report.py::test_score_images_matches_metrics` stays as is.
8. `region_mask` must not assume the joint's pixel lies inside the image: a float32
   keypoint can round to u = W for a visible joint (research R2), and the swapped-axes
   test (g) feeds centres far outside; band comparisons in float64 handle both without
   index arithmetic.
9. Terminology: "region" = name + joints + h (`REGIONS`); "region mask" = its pixels for
   one sample; "region record" = `{"image_mse", "samples"}` in `metrics.json`; "labels" =
   the (keypoints, visibility, image size) group of `score_images`.

Found during implementation:

10. The per-sample region values, and so the region means, came out bit-identical for
    batch sizes 1, 3 and n on the synthetic 64 × 48 images, and the 003 / 004 fixture
    runs agree exactly (both use batch size 4); no tolerance was needed in either test.
11. The fixture smoke AE (10 epochs on 8 images) has a whole-image MSE seven times the
    mean image's (0.137 vs 0.019) but lower region MSEs (hands 0.085 vs 0.122, head 0.075
    vs 0.193): it paints the background grey (mean 0.38), so its error sits there
    (test split: background MSE 0.144, body 0.051; mean image 0.007 and 0.174), and the
    background fills 93 % of an image but on average 30–61 % of a region (head, hands,
    feet). The ranking can flip between the whole image and a region, which is why region
    values are compared across models within a region (README, contracts); the smoke
    numbers say nothing about the real AE.
12. The 006 probe's `encoder.checkpoint_sha256` differs between any two train runs whose
    checkpoints were written in different run roots or with different timings, so the
    SC-002 probe comparison is made on the same checkpoint (as in 008).
