# Tasks: Pose-Supervised Autoencoder

Input: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md). Rules for every implementer: tests first, run them red and note
the red result in your report; then code; each batch ends green with
`uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked pytest -q` (whole suite). Never `uv sync`, never add a dependency, never
`git add -A`, never commit or push (the lead commits). Never commit or copy
`assets/smpl/SMPL_NEUTRAL.npz`. No heavy compute: fixture runs only (`configs/ae_fixture.yaml`,
`tests/fixtures/smpl_fixture`); never start a v3 training. Python 3.11 syntax. Touch only the
files your batch lists. House style: docstrings cite "spec 018 FR-xxx".

## Stage 0 — design (lead)

- [x] T001 spec.md (surrogate rulings Q1–Q9), plan.md, research.md, data-model.md, tasks.md;
  configs and queue script (B4 files); roadmap entry.

## Batch B1 — pose head module (FR-002)

Files: `src/strike_a_pose/models/pose_head.py` (new), `src/strike_a_pose/models/__init__.py`
(export `PoseHead`, `PoseTargetStats`), `tests/test_pose_head.py` (new).

- [X] T010 Tests (red first), `tests/test_pose_head.py`:
  - `test_root_relative_zeroes_the_root()`: `j = torch.arange(2*24*3, dtype=torch.float32)
    .reshape(2, 24, 3)`; `r = root_relative(j)`; `r.shape == (2, 72)`; `r[:, :3] == 0`;
    `r[0, 3:6].tolist() == [3.0, 3.0, 3.0]`.
  - `test_head_shapes_and_parameter_count()`: `torch.manual_seed(0)`; `h = PoseHead(64)`;
    `h(torch.zeros(5, 64)).shape == (5, 72)`; `h.parameter_count() == 100936`;
    `h.hparams() == {"latent_dim": 64, "hidden": [256, 256], "d_out": 72}`;
    `PoseHead(**h.hparams())` builds.
  - `test_stats_standardize_round_trip_and_inactive()`: `y = torch.randn(100, 72,
    generator=torch.Generator().manual_seed(0))`; `y[:, :3] = 0.0`; `y[:, 3] = 0.25`;
    `s = PoseTargetStats.fit(y)`; `s.active.sum() == 68`; `not s.active[:4].any()`;
    `s.scale[:4] == 1`; `torch.allclose(s.unstandardize(s.standardize(y)), y, atol=1e-6)`;
    `s.standardize(y)[:, 4:].mean(0)` abs < 1e-5 and population sd within 1e-4 of 1;
    `s.unstandardize(torch.full((1, 72), 7.0))[0, 3] == pytest.approx(0.25)`.
  - `test_stats_dict_round_trip()`: `PoseTargetStats.from_dict(s.to_dict())` equal tensors;
    `torch.save` / `torch.load(weights_only=True)` of `to_dict()` works.
  - `test_stats_empty_raises()`: `fit(torch.empty(0, 72))` → `ValueError` match "empty".
  - `test_pose_loss_ignores_inactive()`: `active = torch.ones(72, dtype=bool)`;
    `active[:3] = False`; `pred = torch.zeros(4, 72)`; `t = torch.ones(4, 72)`;
    `t[:, :3] = 100.0`; `pose_loss(pred, t, active) == pytest.approx(1.0)`; gradient flows to
    `pred` (requires_grad) and is 0 on the inactive columns.
- [X] T011 Implement per data-model.md. `fit` computes in float64, returns float32.

## Batch B2 — training integration (FR-001, FR-003–FR-005)

Files: `src/strike_a_pose/train/config.py`, `src/strike_a_pose/train/objective.py`,
`src/strike_a_pose/train/trainer.py`, `tests/test_train_pose.py` (new),
`tests/test_train_config.py` (extend the `test_shipped_configs_load` parametrize with
`"smpl_proc_v3/ae_pose_l0.003.yaml"`, `"smpl_proc_v3/ae_pose_l0.03.yaml"`,
`"smpl_proc_v3/ae_pose_l0.03_mask.yaml"`). Depends on B1.

- [X] T020 Tests (red first), `tests/test_train_pose.py` (reuse the helpers / cwd fixture style
  of `tests/test_train_variants.py`: `CFG = configs/ae_fixture.yaml`, `_cfg(tmp_path, root, **kw)`
  via `load_config(CFG).with_overrides(run_root=..., **kw)`, epochs as that file uses):
  - `test_pose_weight_default_and_legacy()`: `TrainConfig(dataset_dir="d", split_file="s")
    .pose_weight == 0.0`; `LEGACY_VALUES["pose_weight"] == 0.0`;
    `Objective().pose_weight == 0.0`; `Objective.from_config(cfg).pose_weight` equals the cfg's.
  - `test_pose_weight_invalid()` parametrized: `{"pose_weight": -0.1}` → match ">= 0";
    `{"pose_weight": 0.03, "variational": True, "beta": 0.1}` → match "variational".
  - `test_load_split_joints(...)`: fixture train split, `load_split(ds, 4, joints=True)`
    `["joints_world"]` shape `(n, 24, 3)`, float32, equal to `ds[i]["joints_3d_world"]` for
    i = 0 and n − 1; without `joints` the key is absent.
  - `test_default_run_is_unchanged_bitwise(tmp_path)`: one fixture run with defaults and one
    with `pose_weight=0.0` explicitly: identical `history.csv` except `epoch_s`, identical
    `best.pt["model"]` tensors (`torch.equal`); `best.pt` has no `"pose_head"` key; history
    header `== HISTORY_FIELDS` and the last field is `"train_pose"`, empty in every row.
    Plus `test_train_epoch_without_pose_matches_pre_018(train_images)`: same seed, two models,
    `train_epoch(..., objective=Objective())` vs `train_epoch(..., objective=Objective(),
    pose=None)` → equal `train_loss` and `torch.equal` weights; `st["train_pose"] is None`.
  - `test_pose_run_records_head(tmp_path)`: fixture run, `pose_weight=0.03`: `history.csv`
    `train_pose` finite in every row; `train_loss` within 1e-6 relative of
    `train_recon + 0.03 * train_pose` per row (epoch means of a linear sum);
    `metrics.json["pose_head"]` has `parameters == 100936`, `pose_weight == 0.03`,
    `target == "joints_world"`; `0 < splits.val.pose_head.mpjpe_mm < 2000` and test likewise;
    `metrics.training.objective.pose_weight == 0.03`; `best.pt` and `checkpoint.pt` contain
    `pose_head`, `pose_head_hparams`, `pose_stats`.
  - `test_pose_run_repeats(tmp_path)`: two identical pose runs → identical history (minus
    `epoch_s`) and identical `splits.val.pose_head.mpjpe_mm`.
  - `test_pose_resume_equals_uninterrupted(tmp_path)`: as `test_resume_equals_uninterrupted`
    in `tests/test_train.py` (`stop_after_epoch=1`, then `resume_training`), with
    `pose_weight=0.03`: identical history (minus `epoch_s`), identical head state dict in
    `best.pt`, identical `pose_head.mpjpe_mm`.
  - `test_pose_head_changes_the_encoder(tmp_path)`: defaults vs `pose_weight=0.03`, same seed:
    `best.pt["model"]` encoder weights differ (the pose gradient reaches the encoder).
  - `test_probe_loads_pose_checkpoint(tmp_path)`: `load_checkpoint(run/"best.pt")` returns a
    `ConvAutoencoder` whose `encode` output on one fixture image has shape `(1, latent_dim)`.
  - `test_head_mpjpe_is_the_probe_metric()`: unit test of the eval helper
    `pose_head_scores(model, head, stats, images_u8, joints_world, batch_size)` with a head
    whose last layer is zeroed (weights and bias 0): predicted joints = train mean, so the score
    equals `mpjpe(mean.expand, root_relative(joints)) * 1000` mean (computed in the test with
    `strike_a_pose.eval.metrics.mpjpe`), `pytest.approx(rel=1e-6)`.
  - `test_pre_018_run_dir_resumes(tmp_path)`: start a default fixture run with
    `stop_after_epoch=1`, delete `pose_weight` from its `config.yaml` and the `train_pose`
    column from `history.csv` / from the checkpoint's `history` rows, then `resume_training`
    finishes (pattern: `test_run_dir_from_before_011_resumes`).
- [X] T021 `config.py`: field, `LEGACY_VALUES`, number parsing (add to `numbers`), `_check`
  rules (data-model.md). `objective.py`: `pose_weight` field and `from_config`.
- [X] T022 `trainer.py`:
  - `HISTORY_FIELDS` += `"train_pose"`; `_complete_row`: add `train_pose: None` when the key is
    missing (keep the 011 behaviour); `_history_row` writes `""` for None (as `train_kl`).
  - `load_split(..., joints: bool = False)` per data-model.md.
  - `_setup`: when `cfg.pose_weight > 0`, `s["pose_head"] = PoseHead(cfg.latent_dim)` built
    **after** the AE; else `None`. `info["pose_head_parameters"]` when on.
  - `_train_and_evaluate`: train loaded with `joints=head is not None`; val / test with
    `joints=head is not None` too; `stats = PoseTargetStats.fit(root_relative(train
    ["joints_world"]).double())`; `targets = stats.standardize(root_relative(...)).float()`;
    optimizer over `list(model.parameters()) + list(head.parameters())` (plain runs: exactly
    `model.parameters()` as today); resume loads `ckpt["pose_head"]` into the head and
    `PoseTargetStats.from_dict(ckpt["pose_stats"])` (assert equal to the refit stats with
    `torch.equal`; mismatch → `ConfigError("pose stats differ from the checkpoint")`).
  - `train_epoch`: new kwarg `pose`; inside the loop after `mu, logvar = model.encode_dist(x)`:
    `pl = pose_loss(pose.head(mu), pose.targets[idx], pose.active)`; `loss = recon +
    obj.pose_weight * pl` (plain branch only; VAE refused by config); accumulate
    `pose_total`; `head.train()` at start. With `pose=None` the code path and float ops are
    exactly today's.
  - Checkpoints: `_save_checkpoint` dict gains the three keys when on; `_best_state(epoch,
    val_loss, weights, hparams, config, extra: dict | None = None)` merges `extra`;
    `reconcile_best` passes the checkpoint's head keys; `load_selected_weights(model, run_dir,
    selection, cfg, head=None)` also loads `b["pose_head"]` into `head` when given.
  - Eval: `pose_head_scores(model, head, stats, images_u8, joints_world, batch_size) -> dict`
    (eval mode, `no_grad`, batches of `batch_size`; `mu = model.encode(x)`; joints =
    `stats.unstandardize(head(mu)).double().reshape(n, 24, 3)`; `mpjpe(pred, root_relative(
    joints_world).double().reshape(n, 24, 3)) * 1000` mean) → `{"target": "joints_world",
    "mpjpe_mm": float}` as `splits[s]["pose_head"]`; top-level `metrics["pose_head"]` and
    `units` entry per data-model.md; `_headline` appends ` head {mpjpe:.1f} mm` when present.
  - Memory: delete `targets` with `train` (`del train`).

## Batch B3 — pose-track report (FR-006)

Files: `src/strike_a_pose/eval/pose_track.py` (new), `scripts/pose_track.py` (new),
`tests/test_pose_track.py` (new). Independent of B2 (reads JSON only).

- [ ] T030 Tests (red first):
  - `test_seed_stats()`: `seed_stats([131.0, 134.4, 133.1])` → mean
    `pytest.approx(132.8333, abs=1e-4)`, sd `pytest.approx(1.71561, abs=1e-5)`; `[131.0]` →
    `ValueError`; `[1.0, 1.0]` → `ValueError` (sd 0).
  - `test_rank_orders_and_flags()`: seeds `PoseEntry("s0", 131.0, None, 0.001578)`,
    `("s1", 134.4, None, 0.001587)`, `("s2", 133.1, None, 0.001573)`; entries
    `("R1-v3", 124.5, None, 0.00181)`, `("P1", 78.0, 70.0, 0.00170)`,
    `("P2", 130.0, 85.0, 0.00160)` → order `["P1", "R1-v3", "P2"]`; R1-v3 `d_mm`
    `approx(-8.3333, abs=1e-4)`, `z approx(-4.8573, abs=1e-4)`, `real True`, `head_ok None`;
    P2 `d_mm approx(-2.8333)`, `z approx(-1.6515, abs=1e-4)`, `real False`, `head_ok False`;
    P1 `probe_ok True`, `head_ok True`; P1 `d_mse_rel approx(0.00170 / 0.0015793333 - 1,
    rel=1e-6)`.
  - `test_script_tables(tmp_path, capsys)`: write minimal run / probe dirs (`metrics.json` with
    `splits.{val,test}.image.ae.image_mse`, `splits.{val,test}.pose_head.mpjpe_mm` for one
    variant only, probe `splits.{val,test}.targets.joints_world.mlp.mpjpe_mm`); `main([...,
    "--split", "both"])` returns 0; output has two tables with header
    `| Rank | Variant | MSE vs seeds | Probe MPJPE val (mm) | Δ vs seeds (mm) | Δ / sd | Real (< −2 sd) | Head MPJPE val (mm) | Probe ≤ 80 | Head ≤ 80 |`
    (and the `test` header), a line `seeds: n=3 mean 132.8 mm sd 1.7 mm` per split, and `–`
    for a missing head; one `--seed` → exit 2 with "need >= 2 --seed"; missing file → exit 2.
- [ ] T031 Implement per data-model.md. Script: `--seed RUN PROBE` (append, nargs 2),
  `--variant NAME RUN PROBE` (append, nargs 3, required), `--split {val,test,both}` default
  `val`; formats: MSE `+x.x %`, mm `.1f`, Δ `+.1f`, Δ / sd `+.1f`, flags `yes` / `no` / `–`.
  Writes nothing. Docstring with the usage line as `scripts/promotion.py`.

## Batch B4 — configs, queue, docs (FR-007, FR-008)

Files (lead-written, already committed): `configs/smpl_proc_v3/ae_pose_l0.003.yaml`,
`ae_pose_l0.03.yaml`, `ae_pose_l0.03_mask.yaml`, `docs/campaign-scripts/pose018_queue.sh`.
Implementer files: `docs/HANDOFF.md`.

- [ ] T040 Validate: the three configs load (T020's parametrize) with `pose_weight` 0.003 /
  0.03 / 0.03 and `mask_weight` 0 / 0 / 4; `sh -n docs/campaign-scripts/pose018_queue.sh`.
- [ ] T041 `docs/HANDOFF.md` campaign procedure: add the 018 queue line (copy to `data/`, the
  watchdog env `LOG=data/pose018_queue.log QUEUE=data/pose018_queue.sh`), and the two reports:
  `scripts/promotion.py --r0 <R0-v3> --r0-probe <…> --r0b <R0b-v3> --r0b-probe <…> --variant
  … --split both` and `scripts/pose_track.py --seed <R0-v3> <probe> --seed <R0b-v3> <probe>
  --seed <R0c-v3> <probe> --variant … --split both`.

## Stage 5 — campaign and results (lead, after B1–B4 merged green)

- [ ] T050 Run the queue (≈ 17 h). After `QUEUE_DONE`: both reports, head val / test MPJPE,
  ≤ 80 mm verdicts, write `docs/results/pose-campaign-018/README.md`, README "## Results" line.

## Analyze (lead, 2026-10-24)

- A1 Consistency spec ↔ data-model ↔ tasks: FR-001–FR-008 each map to a batch (B2: 001/003/004/005,
  B1: 002, B3: 006, B4: 007/008); constants agree (`POSE_HEAD_HIDDEN` (256, 256), 100,936
  parameters, `POSE_STD_FLOOR_M` 1e-6, `REAL_GAIN_SD` 2, `POSE_TARGET_MM` 80). No gap.
- A2 Bitwise default: `HISTORY_FIELDS` gains a column for every run, so the default history
  header changes (an extra empty column, as 011 did with `train_kl`); tests compare to the
  constant, losses and weights stay bitwise (T020). Accepted.
- A3 Honesty: the probe's train latents come from an encoder that saw the train labels, so its
  train fit is optimistic; val / test are untouched by labels and the probe's hyper-parameters
  are chosen on val as in 006. The head's number is a supervised predictor's, reported in its
  own column and never fed to `promotion.py`.
- A4 Selection on pixel val MSE may pick an epoch where the head is not at its best. Accepted
  (Q6, pixel quality stays the gate); the history's `train_pose` shows the head's trajectory.
- A5 Expected 015 verdicts: a variant whose MSE rises > 5 % is DROPPED regardless of the probe
  (as R1-v3); the pose track is the place its pose value is read. No rule change.
- A6 Budget: 3 × 5–6 h + probes ≈ 17 h (< 20 h); peak RSS as R1-v3 (5.9 GB) + 15 MB < 12 GB.
- A7 Run-name globs: `ae-pose-l0.03-smpl_proc_v3-2026*` matches neither `ae-pose-l0.003-…` nor
  `ae-pose-l0.03-mask-…`; the watchdog glob `runs/ae-*smpl_proc_*` covers all three.
