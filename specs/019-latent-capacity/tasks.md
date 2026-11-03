# Tasks: Latent Capacity (128-d)

Input: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md). Rules for every implementer: tests first, run them and note the
result in your report (expected: green on first run, since the wiring is generic; a red test is
a finding, fix per FR-005 only in `models/` or `eval/` and report it); each batch ends green with
`uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked pytest -q` (whole suite). Never `uv sync`, never add a dependency, never
`git add -A`, never commit or push (the lead commits). Never commit or copy
`assets/smpl/SMPL_NEUTRAL.npz`. No heavy compute: fixture runs only (`configs/ae_fixture.yaml`,
`configs/probe_fixture.yaml`, `configs/sample_fixture.yaml`, `configs/roundtrip_fixture.yaml`,
`tests/fixtures/smpl_fixture`); never start a v3 training. Python 3.11. Touch only the files
your batch lists; in particular never edit `src/strike_a_pose/train/*`,
`models/autoencoder.py` or `specs/020-*` (parallel slice 020). Docstrings cite "spec 019 FR-xxx".

## Stage 0 — design (lead)

- [x] T001 spec.md (rulings Q1–Q8), plan.md, research.md, data-model.md, tasks.md; configs
  `configs/smpl_proc_v3/ae_l128.yaml`, `ae_l128_pose_l0.03.yaml`, `ae_l128_pose_l0.01.yaml`;
  `docs/campaign-scripts/latent019_queue.sh`; roadmap entry.

## Batch B1 — 128-d end-to-end tests (FR-002)

Files: `tests/test_latent_capacity.py` (new), `tests/test_train_config.py` (append to the
`test_shipped_configs_load` parametrize list only).

- [X] T010 `tests/test_train_config.py`: append `"smpl_proc_v3/ae_l128.yaml"`,
  `"smpl_proc_v3/ae_l128_pose_l0.03.yaml"`, `"smpl_proc_v3/ae_l128_pose_l0.01.yaml"` to the
  parametrize list (keep existing entries; if slice 020 has appended lines, keep theirs too).
- [X] T011 `tests/test_latent_capacity.py` (module docstring "spec 019 FR-002"; autouse
  fixture `monkeypatch.chdir(ROOT)` like `tests/test_sample.py`):
  - `test_shipped_l128_configs()` parametrized over the three FR-001 files: `latent_dim == 128`;
    `pose_weight` is 0.0 / 0.03 / 0.01 respectively; `run_name` as in spec FR-001; every other
    field equal to `configs/smpl_proc_v3/ae.yaml` (λ 0) or `ae_pose_l0.03.yaml` (λ > 0)
    loaded via `load_config`, compared with `dataclasses.replace(other, run_name=...,
    latent_dim=128, pose_weight=...)` equality.
  - `test_model_sizes_128()`: `ConvAutoencoder((256, 192), 128, [16, 32, 64, 128, 256, 256])
    .parameter_count() == 3520177`, `.bottleneck == (256, 4, 3)`, output shape of
    `zeros(2, 1, 256, 192)` is `(2, 1, 256, 192)`; `PoseHead(128).parameter_count() == 117320`;
    the 64-d model is 3126897 (guards the research R1 table).
  - `test_fixture_train_and_probe_128(tmp_path)` parametrized `pose_weight` ∈ {0.0, 0.03}:
    train with `load_config("configs/ae_fixture.yaml").with_overrides(run_root=str(tmp_path /
    "r"), latent_dim=128, pose_weight=pw)` through the same entry the existing
    `tests/test_train_pose.py` uses (copy its helper; keep its epoch setting); assert the run
    dir's `metrics.json` exists, the selected checkpoint (`metrics["selection"]["checkpoint"]`)
    loads with `models.autoencoder.load_checkpoint` and has `latent_dim == 128`; for 0.03 the
    checkpoint has the pose head keys with `latent_dim` 128 in their hparams (key names as in
    `train/trainer.py`) and `metrics["splits"]["val"]["pose_head"]` exists. Then
    `run_probe(probe_config.load_config("configs/probe_fixture.yaml").with_overrides(
    checkpoint=<ckpt>, run_root=str(tmp_path / "p"), cache_dir=str(tmp_path / "c")))`;
    the probe `metrics.json` records `latent_dim` 128 (the key written at `eval/probe.py`
    ~line 620) and finite world MPJPE for the MLP and the random-encoder control.
  - `test_sample_and_round_trip_128(tmp_path)`: build `_ckpt`-style random checkpoints at
    `latent_dim=128` (copy the helper from `tests/test_roundtrip.py`): VAE seed 0, AE seed 1;
    run sample, probe on the VAE, then `run_round_trip` exactly like `rt_inputs` / `_cfg` in
    `tests/test_roundtrip.py`; assert the round-trip run dir exists with its `metrics.json`.
  - `test_sample_refuses_mixed_latent(tmp_path)`: VAE 128, reference AE 64 → `ConfigError`
    match "latent_dim"; (mirror the existing refusal test in `tests/test_sample.py`).
  - Keep the whole module < 60 s (SC-001): mark nothing slow; reuse the fixture epochs.
- [X] T012 Run the suite; report timings of the new module.

## Batch B2 — handoff (FR-003, FR-004)

Files: `docs/HANDOFF.md`. Depends on B1.

- [X] T020 Add a "019 latent-capacity campaign" section: copy step
  (`cp docs/campaign-scripts/latent019_queue.sh data/ && chmod +x data/latent019_queue.sh`),
  watchdog line from plan.md, `RUN_L3=1` meaning and its +6–7 h, terminal tokens, and the two
  post-queue commands from plan.md with the 018 P1 / P2 / R0-v3 / R0b-v3 / R0c-v3 run and probe
  dirs filled in from `docs/results/018-pose-supervised/README.md` and
  `docs/results/smpl-campaign-015/README.md`.
- [X] T021 `sh -n docs/campaign-scripts/latent019_queue.sh` passes; `grep -c QUEUE_DONE` on the
  script is exactly 1 (the final `log` line; the header comment uses "QUEUE-DONE").

## Stage 3 — campaign (lead / operator, not the implementer)

- [ ] T030 Before starting: confirm slice 020 has not changed any `TrainConfig` default
  (`git diff dd61515 -- src/strike_a_pose/train/config.py` shows only added keys whose
  defaults reproduce current behaviour); otherwise stop and ask.
- [ ] T031 Run the queue; after `QUEUE_DONE` run promotion / pose-track (plan.md) and write
  `docs/results/019-latent-capacity/README.md` (SC-003), roadmap line.

## Analyze (lead, 2026-10-28)

- A1 Coverage: FR-001 → T001 / T011 `test_shipped_l128_configs`; FR-002 → T011; FR-003 → T001
  / T021; FR-004 → T031; FR-005 → B1 rule. SC-001 → T012; SC-002 / SC-003 → T031. No gaps.
- A2 Collision with 020: 019 edits no `src/` file; shared files listed in plan.md conflict
  section are append-only. T030 guards default drift.
- A3 Queue: `latest ae-l128-smpl_proc_v3` globs `runs/ae-l128-smpl_proc_v3-2026*`, which cannot
  match the `ae-l128-pose-*` runs (prefix differs before `-2026`). The done token occurs once
  (T021). L3 off by default keeps the default queue within budget.
- A4 Comparability: L1 / L2 differ from R0-v3 / P2 only in `latent_dim` and `run_name`
  (T011 pins it), same seed 0, same data and split.
- A5 Risk: single seed; the L1 pixel verdict uses the 64-d seed sd as a proxy for 128-d noise
  (stated in spec Q4).
