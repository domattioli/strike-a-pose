# Tasks: Pose Scale, Latent-Aware Promotion and VAE Sampling

Input: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md). Rules for every implementer: tests first, run them red and note
the red result in your report; then code; the batch ends green with
`uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked pytest -q` (whole suite). Never `uv sync`, never add a dependency (no SciPy),
never `git add -A`, never commit or push (the lead commits). Python 3.11-compatible syntax.
Touch only the files your batch lists. Follow the house style of neighbouring code (docstrings
cite the spec: "spec 015 FR-xxx").

## Stage 0 — design (lead)

- [x] T001 spec.md (surrogate rulings 2026-10-22), plan.md, research.md, data-model.md,
  tasks.md.

## Batch B1 — v3 configs (spec FR-001, FR-002)

Files: `configs/smpl_proc_v3/{generate,baselines,ae,probe,ae_seed1,ae_vae,ae_mask}.yaml`,
`tests/test_realism.py`.

- [x] T010 Tests (red first) in `tests/test_realism.py`:
  - Change `SERIES` to add `"smpl_proc_v3": "num_poses"` (keep the `# fmt: skip` layout).
    `test_series_directories_are_the_independent_datasets` then needs the v3 directory.
  - `test_series_generation_configs_differ_from_v1_in_one_key`: add
    `assert cfg["smpl_proc_v3"].num_poses == 8000`.
  - `test_series_run_configs_match_v1`: the branch `if ds == "smpl_proc_v2d":` becomes
    `if ds in ("smpl_proc_v2d", "smpl_proc_v3"):`; add the same two PCA-key asserts for v3.
  - New `test_v3_variant_configs_match_v1_variants()`: for `(name, v1_file)` in
    `[("ae_seed1", "ae_seed1.yaml"), ("ae_vae", "ae_vae.yaml"), ("ae_mask", "ae_mask.yaml")]`:
    `cur = train_config.load_config(ROOT/"configs"/"smpl_proc_v3"/f"{name}.yaml").to_dict()`,
    `base = load(configs/smpl_proc_v3/ae.yaml).to_dict()`, `v1 = load(configs/{v1_file})`,
    `v1ae = load(configs/ae.yaml)`; assert
    `{k for k in cur if cur[k] != base[k]} == {k for k in v1 if v1[k] != v1ae[k]}` (same variant
    keys changed as on v1) and for each such key other than `run_name`, `cur[k] == v1[k]`;
    assert `cur["run_name"] == {"ae_seed1": "ae-seed1-smpl_proc_v3", "ae_vae":
    "ae-vae-smpl_proc_v3", "ae_mask": "ae-mask-smpl_proc_v3"}[name]`.
  - `test_series_configs_generate_on_synthetic_model` covers v3 automatically via `SERIES`.
- [x] T011 Create the configs: copy each `configs/smpl_proc_v2d/{generate,baselines,ae,probe}.yaml`
  to `configs/smpl_proc_v3/`, replace `smpl_proc_v2d` → `smpl_proc_v3` everywhere,
  `num_poses: 8000` with comment `# 4x smpl_proc_v1: where the more-poses gain saturates (spec
  015)`, header comments updated (spec 015 FR-001; AE run ≈ 4-5 h on 4 vCPU). Then
  `ae_seed1.yaml` = v3 `ae.yaml` with `seed: 1`, `run_name: ae-seed1-smpl_proc_v3`;
  `ae_vae.yaml` = v3 `ae.yaml` with `variational: true`, `beta: 0.01`,
  `run_name: ae-vae-smpl_proc_v3`; `ae_mask.yaml` = v3 `ae.yaml` with `mask_weight: 4.0`,
  `run_name: ae-mask-smpl_proc_v3`. The probe config's `checkpoint:` stays a placeholder
  `runs/ae-smpl_proc_v3-YYYYmmddTHHMMSSZ/best.pt` (the queue passes `--checkpoint`).
  Acceptance: `uv run --locked pytest -q tests/test_realism.py` green.

## Batch B2 — baselines memory + peak RSS (spec FR-004, FR-005)

Files: `src/strike_a_pose/eval/baselines.py`, `src/strike_a_pose/runlog.py`,
`src/strike_a_pose/train/trainer.py`, `src/strike_a_pose/eval/probe.py`,
`tests/test_eval_report.py` or the existing baselines test file (find with
`grep -ln run_baselines tests/`), `tests/test_runlog.py`, `tests/test_train.py`,
`tests/test_probe.py`.

- [x] T020 Tests (red first):
  - `tests/test_runlog.py::test_peak_rss_mb`: `v = runlog.peak_rss_mb()`; `isinstance(v, float)`,
    `v > 10.0` (a Python process with torch imported), `v == round(v, 1)`.
  - In the baselines test file: `test_load_arrays_uint8_train_is_lossless`: on the fixture
    (`PoseImageDataset("tests/fixtures/smpl_fixture", split="train", split_file=
    "splits/smpl_fixture.json")`), `a32 = load_arrays(ds, 8, masks=False)`,
    `a8 = load_arrays(ds, 8, masks=False, image_dtype=torch.uint8)`; assert
    `a8["image"].dtype == torch.uint8`, `torch.equal(a8["image"].float() / 255.0, a32["image"])`,
    and `torch.equal(image_mean(a8["image"]), image_mean(a32["image"]))`; every non-image key
    equal (`torch.equal`).
  - `test_baselines_metrics_unchanged_without_image_pca`: run `run_baselines` twice on the
    fixture config with `image_pca_ks: []` (via `config_from_dict({**load_config(
    "configs/baselines_fixture.yaml").to_dict(), "image_pca_ks": []})` — if the config
    validator refuses an empty list check how `smpl_proc_v2d/baselines.yaml` loads and mirror
    it), once with the new code and compare against a run with `image_dtype` forced to float32
    via monkeypatching `baselines.TRAIN_IMAGE_DTYPE_NO_PCA` to `torch.float32`: the two
    `metrics.json` files are byte-identical after removing nothing (the metrics contain no
    timestamps; if they do, compare the `splits` and `fit` objects).
  - Use two separate `run_root`s (tmp dirs) for the two baselines runs (A9).
  - `run_info.json` of a baselines fixture run, a train fixture run (existing
    `tests/test_train.py` fixture-run test) and a probe fixture run
    (`test_fixture_probe_run`) contains `peak_rss_mb` (float > 0). Add one assert to each
    existing test; do not add new training runs.
- [x] T021 `runlog.py`: add
  ```python
  def peak_rss_mb() -> float:
      """Peak resident set size of this process in MiB (Linux ``ru_maxrss`` is KiB; spec 015
      FR-005). A resumed run reports the resumed process only."""
      import resource

      return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
  ```
  Add `"peak_rss_mb": peak_rss_mb()` to the final `info |= {...}` in `train/trainer.py`
  (next to `finished_utc`, ≈ line 724), `eval/baselines.py` (≈ line 268) and the final
  run_info write of `eval/probe.py::run_probe` (find `finished_utc` there).
- [x] T022 `eval/baselines.py`: module constant
  `TRAIN_IMAGE_DTYPE_NO_PCA = torch.uint8  # spec 015 FR-004: 2.5 GB, not 10 GB, at 51,200 images`.
  `load_arrays(ds, batch_size, masks=True, image_dtype=torch.float32)`: allocate `a["image"]`
  with `image_dtype`; when uint8, fill with `torch.round(b["image"].reshape(m, -1) * 255.0)
  .to(torch.uint8)`; else unchanged. In `run_baselines`, the train load becomes
  `load_arrays(train_ds, cfg.batch_size, masks=False, image_dtype=torch.float32 if
  cfg.image_pca_ks else TRAIN_IMAGE_DTYPE_NO_PCA)`. `fit_baselines` is unchanged
  (`image_mean` takes uint8; `fit_pca` runs only with PCA ks, i.e. float32). Val / test loads
  unchanged. Update the `load_arrays` docstring.
  Acceptance: the batch's tests and the whole suite green.

## Batch B3 — promotion rule (spec FR-006, FR-007 numbers)

Files: `src/strike_a_pose/eval/promotion.py`, `scripts/promotion.py`, `tests/test_promotion.py`.

- [x] T030 Tests (red first), `tests/test_promotion.py`:
  - `R0 = Measures(0.00218, 174.8)`, `R0B = Measures(0.00216, 172.0)` (gaps: seed 0.917 %
    relative = 0.00002 / 0.00218; probe 2.8 mm).
  - `test_v1_campaign_verdicts`: R1 `Measures(0.00236, 162.4)` → `"DROPPED"`; R2
    `Measures(0.00218, 174.4, train_kl=274.0)`, `variational=True` → `"REPORTED"`; R3
    `Measures(0.00221, 184.8)` → `"REPORTED"`. Check `d_mse_rel` of R1 ≈ 0.0826 (`pytest.approx
    (0.00018 / 0.00218)`), `d_mpjpe_mm == pytest.approx(-12.4)`, `probe_gap_mm ==
    pytest.approx(2.8)`.
  - `test_pixel_win`: `Measures(0.00200, 176.0)` → `"WIN-pixel"` (−8.3 % < −0.92 %; +1.2 mm ≤
    2.8); `Measures(0.00200, 178.0)` → not WIN-pixel (+3.2 mm > 2.8) and not a latent win →
    `"REPORTED"`.
  - `test_latent_win`: `Measures(0.00228, 168.0)` (+4.6 %, −6.8 mm < −5.6) → `"WIN-latent"`;
    `Measures(0.00228, 169.5)` (−5.3 mm) → `"REPORTED"`; `Measures(0.00229, 160.0)` (+5.05 %)
    → `"DROPPED"`.
  - `test_boundaries`: ΔMSE exactly +5 % (`Measures(0.00218 * 1.05, 160.0)`) → `"WIN-latent"`
    (≤ is inclusive); ΔMSE exactly −seed_gap → not a pixel win (strict).
  - `test_pixel_wins_over_latent_when_both`: `Measures(0.00200, 160.0)` → `"WIN-pixel"`.
  - `test_vae_needs_kl`: `Measures(0.00200, 170.0, train_kl=0.5)`, `variational=True` →
    `"REPORTED"` with `"KL" in reason`; `train_kl=None` with `variational=True` → `ValueError`.
  - `test_zero_gaps`: `r0b == r0` → gaps 0, rule still evaluates (strict inequalities).
  - `test_script_table(tmp_path)`: write minimal fake run dirs (`metrics.json` with
    `{"splits": {"val": {"image": {"ae": {"image_mse": x}}}}}`; probe `metrics.json` with
    `{"splits": {"val": {"targets": {"joints_world": {"mlp": {"mpjpe_mm": y}}}}}}`;
    `history.csv` with header `epoch,train_loss,val_loss,train_kl` and two rows for the VAE),
    call `scripts/promotion.py`'s `main([...])` (import via `importlib` from the path, as
    `tests/test_run_status.py` does for `run_status.py`) and assert the printed Markdown table
    has one row per variant with the verdict label.
- [x] T031 `eval/promotion.py` (no torch import):
  `MSE_DROP_REL = 0.05`, `LATENT_GAP_FACTOR = 2.0`, `EPS = 1e-12` (float tolerance on the +5 % bound only; DROPPED uses `d_mse > MSE_DROP_REL + EPS`), `KL_MIN_NATS = 1.0`;
  `@dataclass(frozen=True) class Measures: mse: float; mpjpe_mm: float; train_kl: float | None = None`;
  `@dataclass(frozen=True) class Verdict: label: str; d_mse_rel: float; d_mpjpe_mm: float;
  seed_gap_rel: float; probe_gap_mm: float; reason: str`;
  `def verdict(variant: Measures, r0: Measures, r0b: Measures, variational: bool = False) -> Verdict`:
  `seed_gap = abs(r0b.mse - r0.mse) / r0.mse`; `probe_gap = abs(r0b.mpjpe_mm - r0.mpjpe_mm)`;
  `d_mse = (variant.mse - r0.mse) / r0.mse`; `d_mp = variant.mpjpe_mm - r0.mpjpe_mm`;
  `pixel = d_mse < -seed_gap and d_mp <= probe_gap`;
  `latent = d_mse <= MSE_DROP_REL + EPS and d_mp < -LATENT_GAP_FACTOR * probe_gap`;
  if variational: `train_kl` None → `ValueError`; `train_kl < KL_MIN_NATS` → pixel = latent =
  False and reason mentions "KL < 1 nat (collapse)". Label: WIN-pixel if pixel, elif
  WIN-latent if latent, elif `d_mse > MSE_DROP_REL + EPS` DROPPED, else REPORTED. `reason`: one
  short sentence with the numbers (e.g. "+8.3 % MSE > +5 %; −12.4 mm probe (gap 2.8)").
  Module docstring cites spec 015 FR-006 and states that it supersedes the 012 rule.
- [x] T032 `scripts/promotion.py` (plain script like `scripts/run_status.py`, `main(argv) -> int`):
  args `--r0 RUN --r0-probe PROBE --r0b RUN --r0b-probe PROBE` then one or more
  `--variant NAME RUN PROBE` (`action="append", nargs=3`). Reads the numbers at the paths named
  in 012 `tasks.md` ("Where the numbers are"): whole-image val MSE
  `splits.val.image.ae.image_mse`; probe `splits.val.targets.joints_world.mlp.mpjpe_mm`;
  `train_kl` = last non-empty `train_kl` of the run's `history.csv`, read with
  `csv.DictReader` by column name (A8) when the run's
  `config.yaml` has `variational: true`. Prints a Markdown table: `| Variant | Val MSE vs R0 |
  Probe MPJPE val (mm) | Δ mm | Verdict (015) | Reason |` with ΔMSE as `+8.3 %`. Exit 2 on a
  missing file (message to stderr).
  Acceptance: tests green; `uv run --locked python scripts/promotion.py --help` works.

## Batch B4 — sampling command (spec FR-008)

Files: `src/strike_a_pose/eval/sample_config.py`, `src/strike_a_pose/eval/sample.py`,
`src/strike_a_pose/eval/cli.py`, `configs/sample_fixture.yaml`,
`configs/sample_vae_beta0.1.yaml`, `configs/sample_vae.yaml`,
`configs/smpl_proc_v3/sample_vae.yaml`, `tests/test_sample.py`.

- [x] T040 Tests (red first), `tests/test_sample.py`. Helper `_ckpt(path, variational, seed)`:
  `torch.manual_seed(seed)`; `m = ConvAutoencoder((256, 192), 64, [16, 32, 64, 128, 256, 256],
  variational=variational)` (check the constructor keyword name in
  `models/autoencoder.py`); `torch.save({"model_hparams": m.hparams(), "model":
  m.state_dict()}, path)` — `load_checkpoint` needs only these two keys. Tests:
  - `test_largest_component_fraction`: empty 5×5 → `(0.0, 0)`; one 2×2 block → `(1.0, 1)`;
    two blocks of 4 and 1 px apart → `(0.8, 2)`; two pixels touching diagonally → `(1.0, 1)`
    (8-connectivity).
  - `test_sample_validity`: `classify(mask, lo, hi)` returns `"valid"`, `"empty"`, `"area"`,
    or `"components"`; cases: empty → empty; single block area within bounds → valid; area <
    lo → area; two equal blocks, total in bounds → components (largest 0.5 < 0.95).
  - `test_nearest_neighbour_mse`: train uint8 `[5, 12]` random (seeded), samples = train rows
    2 and 4 as float /255 plus small noise; `nn_mse(samples, train, chunk=2)` returns indices
    `[2, 4]` and MSE equal to the float64 MSE to those rows (`pytest.approx(rel=1e-12)`);
    same result with `chunk=10`.
  - `test_config`: `load_config("configs/sample_fixture.yaml")` loads; unknown key, `n_samples:
    0`, `iou_threshold: 1.5`, `figure_samples: 65` → `ConfigError`; `with_overrides(seed=3)`.
  - `test_fixture_sample_run(tmp_path)`: two checkpoints (VAE seed 0, plain AE seed 1) written
    with `_ckpt`; `main(["sample", "--config", "configs/sample_fixture.yaml", "--run-root",
    str(tmp_path/"runs"), "--checkpoint", vae, "--reference-checkpoint", ae])` → 0; the run dir
    holds exactly `{"config.yaml", "run_info.json", "samples.json", "samples.png"}`;
    `samples.json` has the data-model keys; sets `vae_prior`, `ae_prior`, `vae_recon_val`
    each with `n` as configured (recon capped at the fixture val size); `0 <= valid_fraction
    <= 1`; `fail_empty + fail_area + fail_components + round(valid_fraction * n) == n`;
    `run_info.json` has `peak_rss_mb`. A second run with the same seed gives an identical
    `samples.json` except the `checkpoint` paths (compare `sets` and `criteria`).
  - `test_refusals(tmp_path)`: `--checkpoint` = plain AE → exit 2 (config error) and no run
    dir; `--reference-checkpoint` = a VAE → exit 2; reference with `latent_dim` 32 → exit 2.
- [x] T041 `eval/sample_config.py`: `SampleConfig` dataclass with the data-model fields,
  `to_dict`, `with_overrides(**kw)` (ignore None), `config_from_dict`, `load_config(path)`,
  mirroring `eval/probe_config.py` (reuse its `ConfigError`, `RUN_NAME_RE`, `_is_int` style;
  copy small helpers rather than importing private ones).
- [x] T042 `eval/sample.py` (module docstring: spec 015 FR-008, research R2-R5):
  - constants `LARGEST_FRACTION = 0.95`, `AREA_PERCENTILES = (1.0, 99.0)`, `NN_CHUNK = 2048`,
    `METRICS_VERSION = 1`.
  - `largest_component_fraction(mask: np.ndarray) -> tuple[float, int]`: iterative stack flood
    fill, 8-neighbourhood, over a bool `[H, W]` array (research R3).
  - `classify(mask: np.ndarray, lo: float, hi: float) -> str`: area = mask.sum(); 0 → "empty";
    outside `[lo, hi]` → "area" (no labelling); largest fraction < LARGEST_FRACTION →
    "components"; else "valid".
  - `nn_mse(samples: torch.Tensor, train_u8: torch.Tensor, chunk: int = NN_CHUNK) ->
    tuple[torch.Tensor, torch.Tensor]`: samples float32 `[n, P]` in [0, 1], train uint8 `[N, P]`;
    research R4 (float32 squared distances per chunk, running argmin; exact float64 MSE for the
    argmin rows); returns `(idx int64 [n], mse float64 [n])`.
  - `score_set(images: torch.Tensor, train_u8, lo, hi, threshold) -> tuple[dict, torch.Tensor]`:
    the S record of data-model.md and the NN indices.
  - `decode_batches(model, z, batch_size) -> torch.Tensor [n, P]` under `torch.no_grad()`,
    clamped to [0, 1] (check how `predict_rows` in the trainer clamps; match it).
  - `run_sample(cfg, argv=None, log=None) -> Path`: checks before any write (dataset via
    `PoseImageDataset(cfg.dataset_dir, split=..., split_file=cfg.split_file)`; checkpoint
    exists; `load_checkpoint`; `model.variational` true else `ConfigError("checkpoint: not a
    VAE")`; reference: exists, not variational, same `latent_dim` and `image_size`; image size
    equals the dataset's). Then `seed_everything(cfg.seed)`; train images uint8 via
    `train.trainer.load_split(train_ds, cfg.batch_size)` and train mask areas from one pass of
    a `DataLoader` over `train_ds` (`b["mask"].reshape(m, -1).sum(1)`); `lo, hi =
    np.percentile(areas, AREA_PERCENTILES)`; `g = torch.Generator().manual_seed(cfg.seed)`;
    `z = torch.randn(cfg.n_samples, d, generator=g)`; VAE prior decodes; AE prior decodes (if
    reference); μ reconstructions of the first `min(n_recon, len(val))` val images (encode =
    μ for the VAE, as the probe); score the three sets; `make_run_dir(cfg.run_root,
    cfg.run_name, started)`, `config.yaml`, `run_info.json` (argv, cwd, seed, git, env,
    dataset hashes as probe, both checkpoints' sha256, started/finished, wall_s,
    `peak_rss_mb`), `samples.json` via `write_json_exact`, `samples.png`: `grid_figure` with
    columns `[("vae prior", s[:k]), ("nearest train", nn[:k]), ("ae prior", a[:k]),
    ("nearest train", nn_a[:k])]`, k = `cfg.figure_samples` (drop the AE columns without a
    reference).
- [x] T043 `eval/cli.py`: subcommand `sample` (help "decode N(0, I) samples of a VAE checkpoint
  and score them (spec 015)") with `--config` (required), `--seed`, `--run-root`,
  `--checkpoint`, `--reference-checkpoint`; handler `_sample` like `_probe` (same exit codes);
  prints `wrote {run_dir}` and one line per set: `{name}: valid {f:.3f} nn_mse {m:.5f}`.
- [x] T044 Configs: `configs/sample_fixture.yaml` (fixture dataset/split, placeholder
  checkpoints, `n_samples: 8`, `n_recon: 4`, `batch_size: 4`, `figure_samples: 4`; the default `figure_samples` is 32 = a 4 × 8 grid: 4 rows of 8
  samples, each sample tile followed by its nearest train tile (A9));
  `configs/sample_vae_beta0.1.yaml` (v1, run_name `sample-vae-beta0.1-smpl_proc_v1`,
  placeholders `runs/ae-vae-beta0.1-smpl_proc_v1-YYYYmmddTHHMMSSZ/best.pt` and
  `runs/ae-smpl_proc_v1-YYYYmmddTHHMMSSZ/best.pt`), `configs/sample_vae.yaml` (v1, R2,
  `sample-vae-smpl_proc_v1`), `configs/smpl_proc_v3/sample_vae.yaml` (v3, `sample-vae-
  smpl_proc_v3`, reference R0-v3); n_samples 1024, n_recon 1024, seed 0. Header comments
  cite spec 015 FR-008. If B1's `test_series_run_configs_match_v1` iterates over every file in
  `configs/smpl_proc_v3/`, it does not (it names kinds), so no conflict.
  Acceptance: `uv run --locked pytest -q tests/test_sample.py` green and under 120 s; whole
  suite green.

## Batch B5 — campaign queue and docs (lead + one implementer; after B1-B4 are committed)

Files: `docs/campaign-scripts/smpl015_queue.sh`, `docs/campaign-scripts/watchdog.sh`,
`specs/012-run-guards/spec.md`, `specs/012-run-guards/tasks.md`, `docs/HANDOFF.md`,
`docs/results/smpl-campaign/README.md`, `README.md`, `specs/roadmap.md`.

- [x] T050 `docs/campaign-scripts/smpl015_queue.sh`: copy the structure of `smpl_queue.sh`
  (flock on `data/smpl015_queue.lock`, `latest`, `log`, `probe_run`, `train`, ≤ 5 passes,
  `pending`, `QUEUE_DONE` / `QUEUE_INCOMPLETE`). Add `sample_run CONFIG TRAIN_RUN_PREFIX
  REF_PREFIX OUTNAME`: skip if `runs/OUTNAME-2026*/samples.json` exists; else checkpoint from
  each run's `metrics.json` `selection.checkpoint` as `probe_run` does; run
  `strike-a-pose-eval sample --config CONFIG --checkpoint … --reference-checkpoint …`. Add
  `mem_gate RUN_DIR`: read `run_info.json` `peak_rss_mb`; > 12288 → `log "MEMORY STOP $1"` and
  set a flag that skips the remaining v3 steps. Order (Q2 (a)): (1) R2b `train
  configs/ae_vae_beta0.1.yaml ae-vae-beta0.1-smpl_proc_v1 configs/probe.yaml`; (2) `sample_run
  configs/sample_vae_beta0.1.yaml ae-vae-beta0.1-smpl_proc_v1 ae-smpl_proc_v1
  sample-vae-beta0.1-smpl_proc_v1` and the same for R2 (`configs/sample_vae.yaml`); (3) v3:
  generate, split (seed 0), baselines → `mem_gate`; R0-v3 → `mem_gate`; R0b-v3; R2-v3; R1-v3
  (each with `configs/smpl_proc_v3/probe.yaml`); `sample_run configs/smpl_proc_v3/sample_vae.yaml
  ae-vae-smpl_proc_v3 ae-smpl_proc_v3 sample-vae-smpl_proc_v3`. `pending` lists exactly the five train prefixes `ae-vae-beta0.1-smpl_proc_v1`,
  `ae-smpl_proc_v3`, `ae-seed1-smpl_proc_v3`, `ae-vae-smpl_proc_v3`, `ae-mask-smpl_proc_v3`
  and the three sample outputs `sample-vae-beta0.1-smpl_proc_v1`, `sample-vae-smpl_proc_v1`,
  `sample-vae-smpl_proc_v3` (a run with `aborted.json` counts as done). `bash -n` and `sh -n` clean. Lock: the ONE per-machine lock `data/smpl_queue.lock` (analyze
  finding 3). `watchdog.sh`: `LOCK=${LOCK:-data/smpl_queue.lock}`, `LOG=${LOG:-data/smpl015_queue.log}`,
  `QUEUE=${QUEUE:-data/smpl015_queue.sh}` used everywhere; it exits on `QUEUE_DONE` or
  `MEMORY_STOP` in `$LOG` (no relaunch); stall limit `max(3 × median epoch, 1800 s)`.
  OOM guard (A2): a v3 train run dir without `metrics.json` whose `run_info.json` lacks
  `peak_rss_mb` after its first resume attempt (marker file `resume_tried` written by the
  queue before resuming), or a trainer exit status 137, → log `MEMORY_STOP <dir>` and exit 0
  (terminal). The queue times `load_split` once on v3 (`python -c` timing
  `train.trainer.load_split` of the train split) and logs `load_split v3 <s> s` (A4).
- [x] T051 Docs (lead): 012 `spec.md` + `tasks.md` "Promotion" paragraph: mark superseded
  ("superseded 2026-10-22 by spec 015 FR-006") and insert the 015 rule text verbatim; the
  identical procedure text in `docs/HANDOFF.md`; HANDOFF state section for the 015 campaign;
  `docs/results/smpl-campaign/README.md`: "015 verdict" column (spec FR-007, from
  `scripts/promotion.py` output on the real run dirs) and the R1 note (12.4 mm, 4.4× the gap,
  misses +5 %); README results pointer; roadmap row for 015.
- [ ] T052 Verification record (lead): commits, red runs per batch, suite counts and times.

## Dependencies

B1, B2, B3, B4 independent (B4 imports `runlog.peak_rss_mb`: if B2 is not in the tree, B4
adds exactly the T021 function to `runlog.py`, and the lead reconciles). B5 after all four.
The campaign itself (≈ 22-25 h) runs after B5 and is recorded by the lead.

## Analyze (independent analyst, verdict READY WITH FIXES; surrogate rulings 2026-10-22)

- A1 (findings 1-2) watchdog paths: LOCK / LOG / QUEUE parametrized together; terminal
  `MEMORY_STOP` treated like `QUEUE_DONE`. Folded into T050.
- A2 (finding 3) one per-machine lock `data/smpl_queue.lock` kept; OOM detection (no
  `peak_rss_mb` after the first resume, or exit 137) → `MEMORY_STOP`, no 5-pass OOM loop. T050.
- A3 (finding 4) exact pending prefixes (five train, three sample). T050.
- A4 (finding 5) stall floor max(3 × median epoch, 1800 s); `load_split` time on v3 logged. T050.
- A6 T030 fixtures are rounded README numbers; the real verdicts come from
  `scripts/promotion.py` on the run dirs in T051.
- A7 README "≈ 5× the probe noise" for R1 → "4.4×" (T051).
- A8 T032 reads `history.csv` with `csv.DictReader` by name, last non-empty `train_kl`.
- A9 T020 baselines comparison uses two run_roots; sample figure 4 × 8 = 32 tiles
  (`figure_samples` default 32).
- A10 plan R4 NN cost at v3: 45-75 min; SC-002 statement written in T051.

## Verification record

- Implemented by the lead directly (no subagent tool in this session); every batch red first:
  B1 5 failed (no `configs/smpl_proc_v3/`); B2 collection error (`peak_rss_mb` missing);
  B3, B4 collection errors (modules missing).
- Commits: `a7779cb` spec, `db8a4f8` plan, `7f5e963` analyze fixes, `cc7b1e3` B1, `6a12895` B2,
  `6c923cc` B3, `d78fb1b` B4, then `docs(015)` (B5).
- Suite: 476 → 493 passed (≈ 55-75 s); ruff check / format clean before each commit.
- Deviations: (1) `SampleConfig.recon_split` (default `val`) added: the fixture split has no
  val poses, so the fixture config reconstructs `test`; (2) `samples.png` shows the VAE prior
  samples and their nearest train images only (the plain-AE prior is in `samples.json`, not in
  the figure); (3) the queue's OOM guard uses a `resume_tried` marker: a run still unfinished
  after one resume stops the queue (`MEMORY_STOP`), so two container restarts within one run
  also stop it (restart by deleting the marker).
- T052 open: the campaign numbers and SC-002 statement after the runs.
