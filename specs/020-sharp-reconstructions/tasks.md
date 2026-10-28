# Tasks: Sharper Reconstructions

Input: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md). Rules for every implementer: tests first, run them red and note
the red result in your report; then code; each batch ends green with
`uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked pytest -q` (whole suite). Never `uv sync`, never add a dependency, never
`git add -A`, never commit or push (the lead commits). Never commit or copy
`assets/smpl/SMPL_NEUTRAL.npz`. No heavy compute: fixture runs only (`configs/ae_fixture.yaml`,
`tests/fixtures/smpl_fixture`); never start a v3 training or a v3 rescore. Python 3.11 syntax.
Touch only the files your batch lists; never edit `specs/019-*`, `models/autoencoder.py`,
`models/pose_head.py`. House style: docstrings cite "spec 020 FR-xxx". If slice 019 has
landed changes in a shared file (plan.md "Conflict section"), keep both sides.

## Stage 0 — design (lead)

- [x] T001 spec.md (surrogate rulings Q1–Q9), plan.md, research.md, data-model.md, tasks.md;
  configs and queue script (B4 files); roadmap entry.

## Batch B1 — gradient loss and boundary F1 (FR-002, FR-004)

Files: `src/strike_a_pose/eval/metrics.py`, `src/strike_a_pose/eval/__init__.py` (export
`gradient_loss`, `boundary_f1`, `BOUNDARY_TOL_PX`), `src/strike_a_pose/eval/report.py`,
`src/strike_a_pose/eval/baselines.py` (UNITS entry), `src/strike_a_pose/train/trainer.py`
(only: add `"boundary_f1"` to the key tuple of `IMAGE_UNITS`), `tests/test_boundary_f1.py`
(new), `tests/test_eval_report.py` (extend).

- [X] T010 Tests (red first), `tests/test_boundary_f1.py`:
  - `test_gradient_loss_zero_for_constant_offset()`: `x = torch.rand(2, 1, 8, 6,
    generator=torch.Generator().manual_seed(0))`; `gradient_loss(x + 0.3, x) == 0` (abs <
    1e-7); `gradient_loss(x, x) == 0`.
  - `test_gradient_loss_value()`: `x = torch.zeros(1, 1, 3, 3)`; `y = x.clone()`;
    `y[0, 0, 1, 1] = 1.0`; dh has 6 entries (2×3), two of them ±1 → mean 2/6; dw likewise →
    `gradient_loss(y, x) == pytest.approx(4 / 6)`; gradient wrt `y` (requires_grad) is finite
    and non-zero at (1, 1).
  - `test_boundary_f1_identical_and_empty()`: `m = torch.zeros(2, 10, 10, dtype=bool)`;
    `m[0, 3:7, 3:7] = True`; `pred = m.float()`; `boundary_f1(pred, m) == [1.0, 1.0]`
    (image 1: both empty → 1.0); `boundary_f1(torch.zeros(2, 10, 10), m)` → `[0.0, 1.0]`.
  - `test_boundary_f1_shift_tolerance()`: square `m[0, 3:7, 3:7]` (one image, 12×12),
    `pred` the same square shifted right by 1 px → F1 1.0 at `tol_px=1`; shifted by 2 px →
    F1 < 1 at `tol_px=1` and 1.0 at `tol_px=2`.
  - `test_boundary_f1_missing_limb_lowers_recall()`: `m` a 4×4 square plus a 1×5 "arm"
    `m[0, 4, 7:12]` (16×16 image); `pred` = the square only → precision 1, recall < 1, F1 in
    (0.5, 1); assert exact value computed by hand in the test from the definition
    (data-model.md) using an explicit numpy implementation in the test (`scipy` not allowed;
    numpy loops ok).
  - `test_boundary_f1_flat_input_and_threshold()`: same as identical case but `pred` flattened
    `[B, H*W]` with `image_size=(10, 10)`; a pred of 0.07 everywhere inside the square scores
    as empty at the default 0.075 threshold.
  - `test_boundary_f1_numpy_in_numpy_out()`: numpy float32 input → numpy `[B]` output.
- [X] T011 Tests (red first), `tests/test_eval_report.py`: extend the existing
  `score_images` test(s) so the result has `"boundary_f1"`, equal to the mean of
  `boundary_f1(pred, mask, threshold, image_size=...)` on the same rows; an identity `pred_fn`
  gives 1.0. `score_images` needs the image size for flat rows: take it from a new keyword
  `image_size: tuple[int, int] | None = None`; when None and `labels` given use
  `labels[2]`; when both None infer nothing and raise `ValueError("image_size required")`.
  Update every caller (`train/trainer.py`, `eval/baselines.py`, grep `score_images(`) to pass
  `image_size` — this is a call-site edit, allowed in those two files.
- [X] T012 Implement per data-model.md (`BOUNDARY_TOL_PX = 1`; torch max-pool on float32; for
  erosion pad with `F.pad(..., value=1.0)` then `-max_pool2d(-b, 3, 1, 0)` so outside pixels
  count as body and do not create contour; dilation `max_pool2d(c, k, 1, tol_px)` with
  zero-padding semantics — note `max_pool2d` pads with −inf, equivalent here since c ≥ 0).
  Units string exactly as data-model.md. Do not bump `METRICS_VERSION`.

## Batch B2 — training integration (FR-001, FR-003)

Files: `src/strike_a_pose/train/config.py`, `src/strike_a_pose/train/objective.py`,
`src/strike_a_pose/train/trainer.py`, `tests/test_train_grad.py` (new),
`tests/test_train_config.py` (extend `test_shipped_configs_load` parametrize with
`"smpl_proc_v3/ae_grad.yaml"`, `"smpl_proc_v3/ae_pose_l0.03_grad.yaml"`),
`tests/test_train_pose.py` (change only `assert HISTORY_FIELDS[-1] == "train_pose"` to
`assert "train_pose" in HISTORY_FIELDS`). Depends on B1.

- [ ] T020 Tests (red first), `tests/test_train_grad.py` (helpers / cwd fixture style of
  `tests/test_train_pose.py`):
  - `test_gradient_weight_default_and_legacy()`: dataclass default 0.0;
    `LEGACY_VALUES["gradient_weight"] == 0.0`; `Objective().gradient_weight == 0.0`;
    `Objective.from_config(cfg).gradient_weight` equals the cfg's, and its `pose_weight` too.
  - `test_gradient_weight_invalid()` parametrized: `-0.1` → match ">= 0"; `"x"` / `nan` →
    match "finite"; `{"gradient_weight": 0.5, "mask_weight": 4.0}` → match "mask_weight";
    `{"gradient_weight": 0.5, "variational": True, "beta": 0.1}` → match "variational".
  - `test_default_run_is_unchanged_bitwise(tmp_path)`: fixture run with defaults vs
    `gradient_weight=0.0` explicitly: identical history minus `epoch_s`, `torch.equal` on every
    `best.pt["model"]` tensor; `HISTORY_FIELDS[-1] == "train_grad"`; `train_grad` empty in
    every row. Plus `test_train_epoch_without_grad_matches(train_images)`: same seed, two
    models, `train_epoch(objective=Objective())` vs `Objective(gradient_weight=0.0)` → equal
    `train_loss`, `torch.equal` weights, `st["train_grad"] is None`.
  - `test_grad_run_records_term(tmp_path)`: fixture run `gradient_weight=0.5`: `train_grad`
    finite > 0 in every row; `train_loss` within 1e-6 relative of `train_recon + 0.5 *
    train_grad`; `metrics.training.objective.gradient_weight == 0.5`;
    `splits.val.image.ae.boundary_f1` in [0, 1].
  - `test_grad_and_pose_run(tmp_path)`: `gradient_weight=0.5, pose_weight=0.03`:
    `train_loss ≈ train_recon + 0.5*train_grad + 0.03*train_pose` (1e-6 rel) per row.
  - `test_grad_run_repeats(tmp_path)` and `test_grad_resume_equals_uninterrupted(tmp_path)`
    (copy the 018 resume test pattern: stop after epoch k via the existing helper, `--resume`,
    compare history minus timings and `best.pt` tensors).
  - `test_pre_020_history_rows_complete()`: `_complete_row({...no train_grad...})["train_grad"]
    is None`.
- [ ] T021 Implement: config field / legacy / numeric list / validation (data-model.md);
  `Objective.gradient_weight` last field, `from_config` by keyword for all fields;
  `train_epoch`: after `recon = recon_loss(y, x, obj.mask_weight)` (store `y = model.decode(z)`
  once), `if obj.gradient_weight: g = gradient_loss(y, x); recon_total_term = recon +
  obj.gradient_weight * g; grad_total += float(g.detach()) * len(idx)` and use that in place of
  `recon` when building `loss` (pose term added as today); `train_recon` still accumulates the
  MSE `recon`; return `"train_grad"`. No computation of `gradient_loss` when the weight is 0.
  `HISTORY_FIELDS` append `"train_grad"`; `_complete_row` fills it.

## Batch B3 — rescore and fidelity report (FR-005)

Files: `src/strike_a_pose/eval/fidelity.py` (new), `scripts/fidelity.py` (new),
`tests/test_fidelity.py` (new). Depends on B1 (B2 for the fixture grad run in tests).

- [ ] T030 Tests (red first), `tests/test_fidelity.py`:
  - `test_rescore_matches_training_metrics(tmp_path)`: train one fixture run (defaults);
    `rescore(run_dir, splits=("val", "test"))` → dict per data-model.md; its
    `splits.val.image_mse`, `silhouette_iou`, `boundary_f1` and region MSEs equal the run's
    `metrics.json` `splits.val.image.ae` values (rel 1e-6); `write_fidelity` writes
    `fidelity.json` that `json.loads` back equal; `pose_weight` / `gradient_weight` 0.0.
  - `test_rescore_deterministic(tmp_path)`: two rescores identical.
  - `test_report_verdicts()`: pure function `fidelity_rows(seeds, pose_refs, variants,
    split)` on hand-built entries (dataclass `FidelityEntry(name, kind, mse, iou, bf1,
    regions, probe_mm)`): seeds bF1 [0.92, 0.93, 0.94] (sd 0.01), probe [131, 133, 134.4];
    variant A plain bF1 0.96 (Δ +0.03 = +3 sd), MSE +3 %, probe +1 mm → `SHARPER`;
    B plain bF1 0.96, MSE +8 % → `SHARPER-COSTLY`; C plain bF1 0.935 → `NO-GAIN`;
    D pose (pose refs bF1 [0.86, 0.862]) bF1 0.90, probe +10 mm vs pose-ref mean →
    `SHARPER-COSTLY`; E probe None → `NO-PROBE`. Δ and Δ/sd fields checked to 1e-9.
  - `test_report_needs_two_seeds()` → `ValueError` match ">= 2".
  - `test_cli_report_exit_codes(tmp_path)`: `scripts/fidelity.py report` with one `--seed` →
    exit 2; a run dir missing `fidelity.json` → exit 2 and its path in stderr.
- [ ] T031 Implement `eval/fidelity.py`: `rescore(run_dir, splits, batch_size=64,
  checkpoint=None)` (default checkpoint from `metrics.json["selection"]["checkpoint"]`, else
  `best.pt`; dataset / split / threshold from the run's `config.yaml` via `load_config`;
  `load_split(..., masks=True)`; `score_images(predict_rows-based fn, ..., labels=...,
  image_size=...)`), `write_fidelity`, `load_entry(run_dir, split)` (reads `fidelity.json`,
  `config.yaml`, `probed.txt` → probe metrics), `fidelity_rows`, `format_markdown`;
  constants `REAL_GAIN_SD = 2.0`, `MSE_COST_REL = 0.05`, `FIDELITY_VERSION = 1`.
  `scripts/fidelity.py`: `rescore RUN_DIR [--split val test] [--checkpoint P]` (skips with a
  message and exit 0 if `fidelity.json` exists, unless `--force`), and `report --seed RUN
  (≥2) --pose-ref RUN (≥1) --variant NAME=RUN ... [--split {val,test,both}]` printing one
  Markdown table per split (columns of data-model.md) plus a line `seeds: n=… bF1 mean … sd
  …; probe mean … sd …`. Never prints the string `QUEUE_DONE`.

## Batch B4 — configs, queue, docs (lead files; implementer validates + HANDOFF)

- [x] T040 (lead) `configs/smpl_proc_v3/ae_grad.yaml`, `ae_pose_l0.03_grad.yaml`,
  `docs/campaign-scripts/fidelity020_queue.sh`.
- [ ] T041 `sh -n docs/campaign-scripts/fidelity020_queue.sh`; both configs load (covered by
  T020's parametrize); `grep -c QUEUE_DONE docs/campaign-scripts/fidelity020_queue.sh` is 1 (only the final
  `log` line; no other text, comment or message contains that string). Add a short "020
  fidelity campaign" paragraph to `docs/HANDOFF.md` (copy step, watchdog command, runs after
  019's queue, the three report commands of plan.md "Campaign").

## Stage 5 — campaign and results (lead, after B1–B3 merged green and the 019 queue done)

- [ ] T050 Run the queue (≈ 14.5 h). After its done line: `scripts/fidelity.py report --seed
  <R0-v3> --seed <R0b-v3> --seed <R0c-v3> --pose-ref <P2> --pose-ref <P2-seed1> --variant
  F1=<F1 run> --variant F2=<F2 run> --split both`; `promotion.py --split both` (F1, F2);
  `pose_track.py --split both` (F2); side-by-side figures; write
  `docs/results/020-sharp-reconstructions/README.md`, roadmap and HANDOFF lines.

## Analyze (lead, 2026-10-28)

- A1 Coverage: FR-001 → B2 T020/T021; FR-002 → B1 T010/T012; FR-003 → B2; FR-004 → B1 T011;
  FR-005 → B3; FR-006 → B4 T040 (done); FR-007 → T041 / T050. Constants agree across files
  (γ 0.5, λ 0.03, `BOUNDARY_TOL_PX` 1, `REAL_GAIN_SD` 2, `MSE_COST_REL` 0.05, threshold 0.075).
  No gap.
- A2 Bitwise default: the history header gains `train_grad` (empty) like 011 / 018 did;
  losses, weights and every pre-existing metric stay bitwise (T020); `metrics.json` gains the
  additive `boundary_f1` key, which no consumer (`promotion.py`, `pose_track.py`, probe)
  reads. Accepted.
- A3 Honest evaluation: selection / early stop stay on val pixel MSE, so γ cannot pick a
  sharper-but-worse epoch; the sharpness metric (binarised contours) is not the trained
  quantity (intensity gradients); verdict and sd fixed before any run; test scored once per
  run (trainer and rescore read the same weights, rescore only re-reads them).
- A4 Reference fairness: F2's reference is the P2 two-seed mean but its sd is the plain seeds'
  (no third P2 seed exists; P2 / P2-seed1 differ by 0.1–0.3 mm in probe, so plain-seed sd is
  the conservative choice). Recorded in the report line.
- A5 Rescore = training metrics (T030 test on fixture): the five references' rescored MSE /
  IoU / regions must equal their published values to 1e-6 rel; the lead checks this for R0-v3
  before reading any variant (else stop: the rescore path differs).
- A6 Budget 13.8–14.5 h < 18 h; peak RSS ≤ 6.0 GB + < 30 MB; 12 GB gate after F1.
- A7 Globs: `ae-grad-smpl_proc_v3-2026*` and `ae-pose-l0.03-grad-smpl_proc_v3-2026*` do not
  match any reference (`ae-pose-l0.03-smpl_proc_v3-2026*` does not match `…-grad-…`); the
  watchdog glob `runs/ae-*smpl_proc_*` covers both.
- A8 019 interaction: 020's configs pin `latent_dim: 64` and no 019 key; if 019 adds config keys
  with defaults that keep today's behaviour, 020's configs remain valid. If 019's merged code
  changes the default trainer bitwise (it must not, by its own rule), T020's bitwise test
  catches it on rebase.
- A9 Queue log hygiene: the string `QUEUE_DONE` appears once in the script (final `log`) and
  `scripts/fidelity.py` never prints it (T031), so the watchdog cannot see a false done line.
