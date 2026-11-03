# Tasks: Generative Pose VAE

Input: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md). Rules for every implementer: tests first, run them red and note
the red result in your report; then code; each batch ends green with
`uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked pytest -q` (whole suite). Never `uv sync`, never add a dependency, never
`git add -A`, never commit or push (the lead commits). Never commit or copy
`assets/smpl/SMPL_NEUTRAL.npz`. No heavy compute: fixture runs only (`configs/ae_fixture.yaml`,
`tests/fixtures/smpl_fixture`); never start a v3 training, sample or round trip. Python 3.11
syntax. Touch only the files your batch lists; never edit `README.md`, `docs/HANDOFF.md`,
`specs/roadmap.md` (edited concurrently elsewhere), `models/*`, or any committed config. House
style: docstrings cite "spec 021 FR-xxx".

## Stage 0 — design (lead)

- [x] T001 spec.md (surrogate rulings Q1–Q10), plan.md, research.md, data-model.md, tasks.md;
  seven configs, queue script, HANDOFF `## 021` section, roadmap bullet.

## Batch B1 — VAE + pose loss (FR-001..FR-003)

Files: `src/strike_a_pose/train/config.py`, `src/strike_a_pose/train/trainer.py`,
`tests/test_train_vae_pose.py` (new), `tests/test_train_pose.py`, `tests/test_train_grad.py`,
`tests/test_train_config.py`.

- [X] T010 Tests (red first), `tests/test_train_vae_pose.py` (copy the imports, `ROOT`, `CFG`,
  `FIXTURE`, `SPLIT`, `SMALL`, `_cwd`, `train_images`, `_cfg`, `varied`, `_rows`, `_metrics`
  helpers of `tests/test_train_pose.py` verbatim; `VAE = {"variational": True, "beta": 0.1}`):
  - `_train_epoch_pre_021(model, images, opt, batch_size, gen, sched=None, objective=None,
    pose=None)`: a verbatim copy of today's `trainer.train_epoch` (commit of this design,
    `src/strike_a_pose/train/trainer.py` lines 136–206) with only the name changed, importing
    its helpers from `strike_a_pose.train.trainer` / `.objective`. Comment: "frozen pre-021
    loop (spec 021 FR-003); never edit".
  - `test_train_epoch_matches_pre_021(train_images)` parametrized over
    `(variational, objective_kw, with_pose)`: `(False, {}, False)`, `(True, {"beta": 0.1},
    False)`, `(False, {"pose_weight": 0.03}, True)`, `(False, {"gradient_weight": 0.5}, False)`,
    `(False, {"gradient_weight": 0.5, "pose_weight": 0.03}, True)`, `(False, {"mask_weight":
    4.0}, False)`. For each: two identical setups (`torch.manual_seed(0)`; `ConvAutoencoder(
    image_size, 8, [4]*6, variational)`; for pose a `PoseHead(8)` built right after under the
    same seed and a `pose` bundle exactly as `test_train_pose.py::
    test_train_epoch_without_pose_matches_pre_018` builds it, with targets
    `torch.randn(n, 72, generator=torch.Generator().manual_seed(3))` and `active` all True);
    run 2 epochs of `train_epoch` vs `_train_epoch_pre_021` with `torch.Generator()
    .manual_seed(1)` and `torch.manual_seed(5)` before each; assert the returned dicts are
    equal and `torch.equal` on every model (and head) state-dict tensor.
  - `test_variational_accepts_pose_and_gradient()`: `config_from_dict({"dataset_dir": "d",
    "split_file": "s", "variational": True, "beta": 0.1, "pose_weight": 0.03})` and the same
    with `"gradient_weight": 0.5` load; `{"gradient_weight": 0.5, "mask_weight": 4.0}` still
    raises `ConfigError` match "mask_weight".
  - `test_vae_pose_head_reads_mu(train_images, monkeypatch)`: variational model latent 8,
    `PoseHead(8)` wrapped in a module that appends `x.detach().clone()` to a list then calls
    the head; monkeypatch `trainer.reparameterize` with a function that records
    `mu.detach().clone()` and returns `mu + 100.0`; one `train_epoch` with
    `Objective(beta=0.1, pose_weight=0.03)` and batch size ≥ n (one batch); assert
    `torch.equal(head_inputs[0], mus[0])`.
  - `test_vae_pose_run_records(tmp_path, varied)`: `run_training(_cfg(tmp_path, "a", **VAE,
    pose_weight=0.03))`; every row: `train_kl`, `train_pose` finite > 0, `train_grad` "";
    `float(train_loss) == pytest.approx(float(train_recon) + 0.1 * float(train_kl) / (H * W) +
    0.03 * float(train_pose), rel=1e-6)` with `(H, W)` = the fixture dataset's `image_size`;
    `_metrics(d)["splits"]["val"]["pose_head"]["mpjpe_mm"]` finite;
    `_metrics(d)["training"]["objective"]` has `beta` 0.1 and `pose_weight` 0.03;
    `best.pt` holds `pose_head`; `load_checkpoint(best.pt)` returns a model with
    `variational is True`.
  - `test_vae_pose_checkpoint_probes_and_samples(tmp_path, varied)`: from that run,
    `strike_a_pose.eval.cli.main(["probe", "--config", "configs/probe_fixture.yaml",
    "--checkpoint", str(best), "--run-root", str(tmp_path / "p")]) == 0` and
    `main(["sample", "--config", "configs/sample_fixture.yaml", "--checkpoint", str(best),
    "--reference-checkpoint", <a plain fixture AE run's best.pt or checkpoint.pt with latent 8
    trained in the test>, "--run-root", str(tmp_path / "s")]) == 0`; the sample run's
    `samples.json` has `sets.vae_prior.valid_fraction` in [0, 1]. (If the fixture split has no
    val poses for the sample config's `recon_split: test`, use the committed fixture split as
    the existing sample tests do — copy their setup from `tests/test_sample.py`.)
  - `test_vae_pose_run_repeats(tmp_path, varied)` and
    `test_vae_pose_resume_equals_uninterrupted(tmp_path, varied)`: copy the bodies of
    `test_pose_run_repeats` / `test_pose_resume_equals_uninterrupted` with `**VAE` added.
  - `test_vae_grad_run_composes(tmp_path)`: `**VAE, gradient_weight=0.5`: per row
    `train_loss ≈ train_recon + 0.5·train_grad + 0.1·train_kl/(H·W)` (rel 1e-6).
- [X] T011 Edit `tests/test_train_pose.py`: remove the parametrize case
  `({"pose_weight": 0.03, "variational": True, "beta": 0.1}, "variational")`;
  `tests/test_train_grad.py`: remove `({"gradient_weight": 0.5, "variational": True, "beta":
  0.1}, "variational")`; `tests/test_train_config.py`: add
  `"smpl_proc_v3/ae_l128_vae_beta0.1_pose_l0.03.yaml"`, `"smpl_proc_v3/ae_l128_vae_beta0.1.yaml"`
  to the `test_shipped_configs_load` parametrize.
- [X] T012 Implement per data-model.md: delete the two refusals in `config.py::_check`;
  replace the loss block of `train_epoch` with the data-model.md block (y, recon computed once
  as today); update the `train_epoch` docstring (loss = recon + γ·grad + β·KL/(H·W) + λ·pose,
  each when on; head on μ; spec 021 FR-002). No other change (no history column, no
  `Objective` field, no `METRICS_VERSION` bump).

## Batch B2 — goal report (FR-004)

Files: `src/strike_a_pose/eval/generative.py` (new), `scripts/generative.py` (new),
`tests/test_generative.py` (new). Independent of B1.

- [X] T020 Tests (red first), `tests/test_generative.py`. Helper `_e(name="V1", head=(67.0,
  64.7), mse=(0.00170, 0.00168), valid=0.59, n=1024, plaus=0.60, ctrl=0.46)` → `GenEntry(name,
  {"val": head[0], "test": head[1]}, {"val": mse[0], "test": mse[1]}, valid, n, plaus, ctrl)`;
  `L2 = {"val": 0.001675765, "test": 0.001653395}`; `R0 = {"val": 0.0015793, "test":
  0.00155856}`; `S = ("val", "test")`.
  - `test_margin_pts()`: `GenEntry(..., rt_plausible=0.457031, rt_control=0.326172)
    .margin_pts == pytest.approx(13.0859, abs=1e-9)`; None when either is None.
  - `test_goal_met()`: `goal_verdict(axis_checks(_e(), L2, S)) == ("GOAL-MET", [], "")`
    (margin 14.0, MSE val 0.00170 ≤ 0.00175955, test 0.00168 ≤ 0.00173606).
  - `test_bounds_inclusive()`: `_e(head=(80.0, 80.0), valid=0.56, plaus=0.59, ctrl=0.46)`
    (margin 13.0) and `mse=(0.001675765 * 1.05, 0.001653395 * 1.05)` → GOAL-MET.
  - `test_partial_pose()`: `head=(79.0, 80.1)` → `("PARTIAL", ["pose"], "beta 0.03")`.
  - `test_partial_valid()`: `valid=0.5599` → `("PARTIAL", ["valid"], "head on z")`.
  - `test_partial_pixel_and_margin()`: `mse=(0.00176, 0.00168)`, `plaus=0.55, ctrl=0.46`
    (9.0 pts) → `("PARTIAL", ["pixel", "margin"], "head on z; beta 0.03")` (operator ruling 2026-11-03).
  - `test_partial_pose_and_pixel_single_step()`: head 81 both, mse 0.0018 both →
    `("PARTIAL", ["pose", "pixel"], "beta 0.03")`.
  - `test_incomplete()`: `valid=None` → `("INCOMPLETE", ["valid"], "")`; head val None →
    `"pose"` in the failing list and verdict INCOMPLETE.
  - `test_compare_prior()`: `compare_prior(_e(valid=0.62), _e(valid=0.59))` → label
    `"NO-DIFFERENCE"`, diff 0.03 (abs 1e-12), sd 0.0215941796 (abs 1e-9);
    `(0.65, 0.59)` → `"HELPS"`, sd 0.0214102414; `(0.59, 0.65)` → `"HURTS"`; a None → label
    `"INCOMPLETE"`.
  - `test_goal_line_pixel()`: `goal_line_pixel(_e(), R0, S) is False` (0.00170 > 0.00165827);
    `_e(mse=(0.00165, 0.00163))` → True.
  - `test_mean_mse_and_load_entry(tmp_path)`: write minimal files: `t1/metrics.json`
    `{"splits": {"val": {"image": {"ae": {"image_mse": 0.00166906}}, "pose_head":
    {"mpjpe_mm": 66.9733}}, "test": {"image": {"ae": {"image_mse": 0.0016472}}, "pose_head":
    {"mpjpe_mm": 64.7}}}}`, `t2/metrics.json` with 0.00168247 / 0.00165959 and no
    `pose_head`; `s/samples.json` `{"sets": {"vae_prior": {"valid_fraction": 0.589844, "n":
    1024}}}`; `r/samples.json` `{"sets": {"vae_prior": {"round_trip": {"plausible_fraction":
    0.457031, "control_plausible_fraction": 0.326172}}}}`. Assert `mean_mse([t1, t2], S)` ==
    `{"val": approx 0.001675765, "test": approx 0.001653395}` (rel 1e-12);
    `load_gen_entry("V1", t1, s, r)` fields; `load_gen_entry("x", t2).head_mm == {"val": None,
    "test": None}` and generation fields None; missing `metrics.json` → `FileNotFoundError`
    whose message contains the path; `mean_mse([], S)` → `ValueError` match ">= 1".
  - `test_cli_report(tmp_path, capsys)`: same files; `scripts/generative.py` imported via
    `runpy`/`importlib` as `tests/test_fidelity.py` does for `scripts/fidelity.py`;
    `main(["report", "--v1", t1, s, r, "--v2", t1, s, r, "--l2", t1, "--l2", t2, "--r0", t2,
    "--split", "both", "--json", out])` returns 0; stdout contains `"## val"`, `"## test"`,
    `"V2 vs V1 prior: NO-DIFFERENCE"`, and the last line is exactly `"VERDICT GOAL-MET
    failing=- next=-"` (head 66.97 / 64.7 ≤ 80; t1 MSE ≤ L2 mean·1.05; valid 0.5898 ≥ 0.56;
    margin 13.0859 ≥ 13); `json.loads(out)["verdict"] == "GOAL-MET"`; `"QUEUE_DONE" not in`
    stdout. A missing round-trip dir → exit 2 and its path in stderr.
- [X] T021 Implement `eval/generative.py` and `scripts/generative.py` exactly per
  data-model.md (constants, signatures, messages, output format). `scripts/generative.py`
  is thin: argparse (`report` subcommand, `--v1/--v2/--extra` with `nargs=3` / `nargs=4`,
  `--l2`/`--r0` `action="append"`), builds entries, calls the library, prints.

## Batch B3 — validation of lead files

- [X] T030 `sh -n docs/campaign-scripts/vae021_queue.sh`; `grep -c QUEUE_DONE
  docs/campaign-scripts/vae021_queue.sh` is 1 (only the final `log` line); the six train /
  sample / round-trip configs load with their loaders (`load_config`, the sample config
  loader, the round-trip config loader) — add the four sample / round-trip files plus
  `smpl_proc_v3/roundtrip_vae_beta0.1.yaml` to whichever existing "shipped configs load" test
  covers each loader (grep `sample_vae_beta0.1.yaml` / `roundtrip_vae` under `tests/`); if none
  exists for a loader, report it, do not create one.

## Stage 4 — campaign and results (lead, after B1–B3 merged green)

- [X] T040 Run the queue (≈ 15 h) as plan.md "Campaign". After its done line: the report
  command of plan.md, `pose_track.py --split both` (V1 vs L2), figures; write
  `docs/results/021-generative-pose-vae/README.md` (verdict line, table, V2-vs-V1, R2b-v3
  context), roadmap bullet, HANDOFF `## 021` status line.

## Analyze (lead, 2026-11-03)

- A1 Coverage: FR-001/002/003 → B1 T010–T012; FR-004 → B2; FR-005..FR-008 → T001 (lead) +
  T030; FR-009 → T001 / T040. No gap.
- A2 Constants agree across spec / data-model / tasks / configs: β 0.1, λ 0.03, latent 128,
  80 mm, 1.05, 0.56, 13 points, 2 sd, n 1024, L2 means val 0.001675765 / test 0.001653395.
- A3 Bitwise: T010's frozen-loop test covers plain, VAE, AE+pose, AE+grad, AE+grad+pose,
  mask; the reorder only moves the VAE KL into a shared chain whose first operand is `recon`.
  PoseHead construction is still gated on `pose_weight > 0`, so no RNG shift for defaults.
- A4 Honest evaluation: thresholds and verdict fixed here before any run; the pixel bar
  reading (L2 vs 64-d) is surfaced as Q1 and printed both ways; selection unchanged.
- A5 Globs: `ae-l128-vae-beta0.1-smpl_proc_v3-2026*` does not match V1
  (`…-beta0.1-pose-l0.03-…`); `roundtrip-vae-beta0.1-smpl_proc_v3-2026*` is new (v1's is
  `…_v1`); `sample-l128-…` prefixes are new. Verified run dirs exist (2026-11-03): L1
  `ae-l128-smpl_proc_v3-20261028T022732Z`, L2 + seed1, R2b-v3 train / probe / sample.
- A6 Budget ≈ 15 h < 18 h (research R5); 12 GB gate after V1 (R6).
- A7 Round-trip marker `roundtripped.txt` is written into the sample run dir (a read-only
  campaign artefact otherwise); R2b-v3's sample dir gets one too — accepted (marker only).
