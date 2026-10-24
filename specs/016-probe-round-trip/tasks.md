# Tasks: Probe Round-Trip Plausibility of VAE Samples

Input: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md). Rules for every implementer: tests first, run them red and note
the red result in your report; then code; the batch ends green with
`uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked pytest -q` (whole suite). Never `uv sync`, never add a dependency (no SciPy at
runtime, no matplotlib), never `git add -A`, never commit or push (the lead commits). Never
commit or copy `assets/smpl/SMPL_NEUTRAL.npz` (tests use the synthetic body of
`tests/conftest.py::make_synthetic_smpl`). A training queue is running (`data/grill_queue.sh`):
do not kill it and do not run real (v1 / v3) round trips; tests only. Python 3.11 syntax.
Touch only the files your batch lists. House style: docstrings cite "spec 016 FR-xxx".

Operator ruling 2026-10-24: `real_test` (first `n_recon` images of the `test` split) is the τ ceiling set,
replacing `real_val` wherever this file names it; `vae_recon_val` still uses `recon_split`.

## Stage 0 — design (lead)

- [x] T001 spec.md (surrogate rulings 2026-10-24), plan.md, research.md, data-model.md, tasks.md.

## Batch B1 — config + pure functions (spec FR-001, FR-005, FR-006, FR-007)

Files: `src/strike_a_pose/eval/roundtrip_config.py`, `src/strike_a_pose/eval/roundtrip.py`,
`configs/roundtrip_fixture.yaml`, `tests/test_roundtrip.py`.

- [X] T010 Tests (red first) in new `tests/test_roundtrip.py` (autouse fixture
  `monkeypatch.chdir(ROOT)` as in `tests/test_sample.py`):
  - `test_config()`: `load_config("configs/roundtrip_fixture.yaml")` loads; defaults
    `calibration_percentile == 5.0`, `control is True`, `figure_samples == 8`, `seed == 0`;
    `config_from_dict` raises `ConfigError` for: unknown key, missing `sample_run` / `probe_run`,
    `calibration_percentile` 0 / 50.5 / `True` / `"5"`, `figure_samples` 0 / 33, `seed` −1,
    `control` = 1 (must be bool), `run_name` "bad name"; `with_overrides(sample_run="x",
    probe_run=None)` changes only `sample_run`.
  - `test_silhouette_iou()`: two 4 × 4 bool masks with 4 and 6 px overlapping in 2 →
    `2 / 8`; identical non-empty → 1.0; disjoint → 0.0; both empty → 0.0; one empty → 0.0;
    shape mismatch → `ValueError`.
  - `test_calibrate_threshold()`: `calibrate_threshold(np.array([0.9, 0.1, 0.5, 0.7]), 5.0)
    == 0.1`; `np.arange(100) / 100, 5.0 → 0.04`; for 1,000 random IoUs (seeded) and p in
    (1, 5, 10): `(x >= tau).mean() >= 1 - p / 100`; empty array → `ValueError`.
  - `test_camera_from_dict_matches_ring()`: for the fixture manifest
    (`json.load(open("tests/fixtures/smpl_fixture/manifest.json"))`) and
    `camera_ring(CameraRingConfig(**m["config"]["camera_ring"]), 256, 192)`: for every k,
    `camera_from_dict(m["cameras"][k])` has equal `name`, `width`, `height`, `yaw_deg` and
    `np.array_equal` `K`, `R`, `t` (dtype float64). (Check `CameraRingConfig`'s constructor in
    `datagen/config.py`; if it is not a plain dataclass, build it the way `generate.py` does.)
  - `test_check_renderable()`: the fixture manifest passes; a copy with
    `config["camera_jitter"] = {...}` (any non-null) raises `ConfigError` mentioning
    `camera_jitter`; with `config["body_shape"] = {...}` raises mentioning `body_shape`;
    `None` values pass.
  - `test_rerender_identity(synthetic_smpl_npz)`: `body = SMPLModel.load(synthetic_smpl_npz)`;
    two poses (zeros and `0.3 * rng.standard_normal(72)` with rows 0, 22, 23 zero), cameras
    `cams = [camera_from_dict(c) for c in manifest["cameras"]]`, `cam_idx = np.array([0, 3])`;
    `rerender_masks(body, poses, cams, cam_idx, num_betas=10)` returns bool `[2, 256, 192]`
    equal to `render(body.forward(p, np.zeros(10))[0], body.faces, cams[k])[1] > 0` per row;
    and `silhouette_iou(out[i], out[i]) == 1.0`.
  - `test_summarize()`: `ious = np.array([0.0, 0.2, 0.6, 0.8])`, `valid = [F, T, T, F]`,
    `yaw = [0, 0, 3, 7]`, `empty = 1`, `control = np.array([0.1, 0.1, 0.7, 0.1])`, `tau = 0.5`,
    `n_classes = 8` → `n 4`, `iou_mean 0.4`, `iou_quantiles["p50"] == np.percentile(ious, 50)`
    (linear), `plausible_fraction 0.5`, `plausible_and_valid_fraction 0.25`,
    `empty_silhouette 1`, `yaw_counts [2,0,0,1,0,0,0,1]`, `control_iou_median 0.1`,
    `control_plausible_fraction 0.25`; with `control=None` both control keys are `None`; with
    `n = 0` every float field and fraction is `None`, `yaw_counts` eight zeros.
  - `test_iou_histogram_figure()`: `iou_histogram_figure({"a": np.array([0.1, 0.9]),
    "b": np.array([])}, 0.5)` returns a PIL image, mode `"L"`, size
    `(HIST_W, HIST_HEADER + 2 * HIST_ROW_H)`; deterministic (two calls `tobytes()` equal).
- [X] T011 `src/strike_a_pose/eval/roundtrip_config.py` modelled on `sample_config.py`:
  `@dataclass(frozen=True) class RoundTripConfig` with fields and defaults of data-model.md
  (`sample_run: str`, `probe_run: str`, `smpl_model_path: str = "assets/smpl/SMPL_NEUTRAL.npz"`,
  `run_name: str = "roundtrip"`, `run_root: str = "runs"`, `seed: int = 0`,
  `calibration_percentile: float = 5.0`, `control: bool = True`, `figure_samples: int = 8`),
  `to_dict()`, `with_overrides(**kw)`; `config_from_dict(data) -> RoundTripConfig` (rules of
  data-model.md; int percentile accepted and stored as float; bool rejected for numbers);
  `load_config(path) -> RoundTripConfig` (YAML errors → `ConfigError`, as `sample_config`).
- [X] T012 `configs/roundtrip_fixture.yaml`: header comment (spec 016, fixture, tests; run line
  `strike-a-pose-eval round-trip --config configs/roundtrip_fixture.yaml --sample-run D
  --probe-run D --smpl-model P`), `run_name: roundtrip-fixture`, `run_root: runs`, `seed: 0`,
  `sample_run: runs/sample-fixture` and `probe_run: runs/probe-fixture` (placeholders,
  overridden), `smpl_model_path: assets/smpl/SMPL_NEUTRAL.npz`, `calibration_percentile: 5.0`,
  `control: true`, `figure_samples: 4`.
- [X] T013 `src/strike_a_pose/eval/roundtrip.py` (module docstring "Probe round-trip
  plausibility of VAE samples (spec 016)"), constants `ROUND_TRIP_VERSION = 1`,
  `CONTROL_STREAM = 16`, `QUANTILES = (5, 25, 50, 75, 95)`, `HIST_BINS = 20`, `HIST_W = 480`,
  `HIST_HEADER = 20`, `HIST_ROW_H = 60`, `SET_ORDER = ("vae_prior", "ae_prior", "vae_recon_val",
  "real_test")`, and:
  - `silhouette_iou(a: np.ndarray, b: np.ndarray) -> float` (bool masks; `ValueError` on shape
    mismatch; 0.0 when the union is empty).
  - `calibrate_threshold(ious: np.ndarray, percentile: float) -> float` =
    `float(np.percentile(ious, percentile, method="lower"))`; `ValueError` if empty.
  - `camera_from_dict(d: dict) -> Camera` (`datagen.cameras.Camera`; `K`, `R`, `t` as float64
    arrays, `width` / `height` int, `yaw_deg` float).
  - `check_renderable(manifest: dict) -> None`: `ConfigError` if `manifest["config"].get(key)`
    is not `None` for `key in ("camera_jitter", "body_shape")`, message
    `f"round trip needs mean-shape ring-camera data; dataset has {key}"`.
  - `rerender_masks(body, poses: np.ndarray, cams: list[Camera], cam_idx: np.ndarray,
    num_betas: int) -> np.ndarray` → bool `[n, H, W]` (H, W from `cams[0]`), one
    `body.forward(poses[i], np.zeros(num_betas))` + `render(...)` per row; `n = 0` → shape
    `[0, H, W]`.
  - `summarize(ious, valid, yaw, empty, tau, n_classes, control=None) -> dict` → the RT record
    of data-model.md (quantiles with `np.percentile` default linear method, keys `p5`…`p95`;
    `plausible = ious >= tau`; Python floats / ints / lists only).
  - `iou_histogram_figure(dists: dict[str, np.ndarray], tau: float) -> Image.Image`: mode "L",
    background 255, width `HIST_W`, a header strip of `HIST_HEADER` px, then one row of
    `HIST_ROW_H` px per set in dict order: the set name (PIL default font, x = 4), `HIST_BINS`
    bars over [0, 1] (`np.histogram(x, bins=HIST_BINS, range=(0, 1))`, heights scaled to the
    row's max count, grey 96), a black vertical line at τ; empty set → name only.
  Acceptance: `uv run --locked pytest -q tests/test_roundtrip.py` green.

## Batch B2 — probe refit, set rebuild, run, CLI (spec FR-002 – FR-004, FR-008)

Files: `src/strike_a_pose/eval/roundtrip.py`, `src/strike_a_pose/eval/cli.py`,
`tests/test_roundtrip.py`.

- [ ] T020 Tests (red first), appended to `tests/test_roundtrip.py`. Module-scoped fixture
  `rt_inputs(tmp_path_factory)` builds, with cwd = ROOT: a random VAE and a random plain AE
  checkpoint (copy `_ckpt` from `tests/test_sample.py`: seeds 0 and 1, `CH = [16, 32, 64, 128,
  256, 256]`); a sample run `run_sample(sample_config.load_config("configs/sample_fixture.yaml")
  .with_overrides(checkpoint=vae, reference_checkpoint=ae, run_root=str(root / "s")))`; a probe
  run `run_probe(probe_config.load_config("configs/probe_fixture.yaml").with_overrides(
  checkpoint=str(vae), run_root=str(root / "p"), cache_dir=str(root / "cache")))` (if
  `with_overrides` drops `cache_dir` when given, pass it via `config_from_dict({**to_dict(),
  ...})`); a synthetic SMPL file `make_synthetic_smpl(root / "SMPL_SYNTH.npz")` (import from
  `tests.conftest`). Returns a dict of paths. Tests:
  - `test_fixture_round_trip_run(rt_inputs, tmp_path)`: `run_round_trip(cfg)` with
    `cfg = load_config("configs/roundtrip_fixture.yaml").with_overrides(sample_run=…,
    probe_run=…, smpl_model_path=…, run_root=str(tmp_path / "a"))` returns a dir containing
    `config.yaml`, `run_info.json`, `samples.json`, `round_trip_per_image.json`,
    `round_trip.png`, `round_trip_iou.png`; `samples.json["sets"]` keys in `SET_ORDER` order
    (`ae_prior` present); every set has `round_trip` with `n == ` the source `n` (8 / 8 / 4) and
    `real_test["round_trip"]["n"] == 4`; `real_test["round_trip"]["plausible_fraction"] >= 0.95`;
    every source set's original fields are unchanged (dict equality with the source
    `samples.json` minus `round_trip`); `round_trip_criteria.calibration.tau` equals
    `calibrate_threshold(per_image["real_test"]["iou"], 5.0)` after `round_sig`;
    `smpl_model.matches_manifest is False` (synthetic body); `probe.refit.split == "test"`,
    `probe.refit.matches is True`; `run_info.json` has `peak_rss_mb > 0` and `wall_s`;
    runtime < 60 s.
  - `test_round_trip_deterministic(rt_inputs, tmp_path)`: two runs into different run roots →
    `samples.json` and `round_trip_per_image.json` byte-identical.
  - `test_round_trip_refusals(rt_inputs, tmp_path)`: each raises `ConfigError` and creates no
    dir under its `run_root`: (a) probe run of a different checkpoint (copy the probe run dir,
    edit `metrics.json["encoder"]["checkpoint_sha256"] = "0" * 64`); (b) sample run with a
    tampered `samples.json` (`sets.vae_prior.foreground_px_mean += 1.0`, copy of the dir) →
    message contains `"not reproduced"`; (c) `smpl_model_path` missing; (d) probe config
    without `smpl_pose` in `targets` (copy, edit `config.yaml`); (e) sample run dir without
    `samples.json`.
  - `test_round_trip_cli(rt_inputs, tmp_path, capsys)`: `main(["round-trip", "--config",
    "configs/roundtrip_fixture.yaml", "--sample-run", …, "--probe-run", …, "--smpl-model", …,
    "--run-root", str(tmp_path)])` returns 0 and prints `wrote <dir>` then one line per set
    starting with two spaces and the set name and containing `plausible` and `valid`; with a
    missing SMPL path returns `EXIT_CONFIG`.
  - `test_refit_matches_probe(rt_inputs)`: `refit_probe(pcfg, metrics, data)` (data from
    `load_or_extract`) returns predictors whose test-split `score("smpl_pose", …)["angle_deg"]`
    and `score("yaw", …)["accuracy"]` equal the probe run's recorded values within 0.05 / 0.005.
- [ ] T021 In `roundtrip.py`:
  - `refit_probe(pcfg: ProbeConfig, metrics: dict, data: dict) -> tuple[Callable, Callable,
    Standardizer, dict]`: `zs = Standardizer.fit(data["train"]["latents"]["trained"])`;
    `ztr = zs(data["train"]["latents"]["trained"])`; pose:
    `ys = Standardizer.fit(target_values(data["train"], "smpl_pose"))`,
    `m = fit_mlp(ztr, ys(y), False, pcfg.mlp_hidden, pcfg.mlp_epochs, pcfg.mlp_batch_size,
    pcfg.mlp_lr, wd_pose, pcfg.seed)`, `pose_fn = lambda z: ys.inverse(_no_grad(m, z))`;
    yaw: `fit_mlp(ztr, data["train"]["camera_index"], True, …, wd_yaw, pcfg.seed, n_classes)`,
    `yaw_fn = lambda z: _no_grad(m2, z).argmax(dim=1)`; `wd_*` =
    `float(metrics["selection"]["mlp"][target]["value"])`; `n_classes = len(metrics["classes"])`.
    Returns `(pose_fn, yaw_fn, zs, {"smpl_pose": wd_pose, "yaw": wd_yaw})`. (Same calls, in the
    same argument order, as `probe.fit_probes`; import `Standardizer, fit_mlp, _no_grad,
    target_values, score, load_or_extract, _encoders` from `.probe`.)
  - `refit_check(pose_fn, yaw_fn, data, metrics, split) -> dict` per data-model.md
    (`split` = the sample run's `recon_split`; null fields when `data.get(split, {"n": 0})["n"]`
    is 0 or the split is absent from `metrics["splits"]`; recorded values from
    `metrics["splits"][split]["targets"]["smpl_pose"]["mlp"]["angle_deg"]` and `…["yaw"]["mlp"]
    ["accuracy"]`, scored on `zs(data[split]["latents"]["trained"])`).
  - `predict(vae, images: torch.Tensor, image_size, batch_size, zs, pose_fn, yaw_fn) ->
    tuple[np.ndarray, np.ndarray]`: images `[n, H*W]` in [0, 1] → batches reshaped
    `[b, 1, H, W]`, `.contiguous(memory_format=torch.channels_last)`, `vae.eval()`,
    `torch.no_grad()`, `mu = vae.encode(x)`; return `pose_fn(zs(mu))` as float64 `[n, 72]`
    and `yaw_fn(zs(mu))` int64 `[n]`.
  - `rebuild_sets(scfg: SampleConfig, vae, ref, val_ds) -> dict[str, torch.Tensor]`: exactly
    `run_sample`'s steps (`seed_everything(scfg.seed)`; `z = torch.randn(scfg.n_samples,
    vae.latent_dim, generator=torch.Generator().manual_seed(scfg.seed))`; `decode_batches` for
    `vae_prior` and, if `ref`, `ae_prior`; `xv = _val_images(val_ds, min(scfg.n_recon,
    len(val_ds)), scfg.batch_size)`; μ-decode → `vae_recon_val`), plus
    `real_test = xv.reshape(xv.shape[0], -1)`; keys in `SET_ORDER` order. Import
    `decode_batches, _val_images, classify` from `.sample`.
  - `check_reproduced(images: dict, source: dict, image_size) -> dict[str, np.ndarray]`: with
    `lo, hi = source["criteria"]["area_bounds_px"]` and `t = source["criteria"]["iou_threshold"]`,
    per set the per-image `valid` bool array (`classify(fg, lo, hi) == "valid"`); for every set
    present in `source["sets"]`, `ConfigError(f"sample run not reproduced: {name} …")` unless
    `round_sig(float(fg.sum(px).mean())) == source.foreground_px_mean` and
    `round_sig(valid.mean()) == source.valid_fraction`; a source set missing from `images` or
    vice versa (except `real_test`) → `ConfigError`. Returns `{set: valid}` for all sets.
  - `run_round_trip(cfg: RoundTripConfig, argv=None, log=None) -> Path` following plan.md
    "Flow" and FR-002 – FR-008: all refusals before `make_run_dir`; the control uses
    `np.random.default_rng([cfg.seed, CONTROL_STREAM, SET_ORDER.index(name)])
    .integers(0, n_train_poses, size=n)` over train rows with `camera_index == 0`; τ from
    `real_test` before summarizing any set; the `real_test` set record gets the 015 fields computed
    with `classify` (fail counts), `nn_mse_mean` / `nn_mse_median` `None`; figure
    `round_trip.png` = `grid_figure([("vae prior", s), ("re-render", r), ("real val", x),
    ("re-render", rx)], image_size)` on the first `min(figure_samples, n)` of each (re-renders
    as float 0 / 1); writes per data-model.md with `write_yaml` / `write_json_exact`;
    `log` lines: run dir, refit check (`WARNING refit differs` when `matches is False`), `WARNING
    SMPL file differs from the dataset manifest` when not matching, τ, one line per set.
    SMPL load: `SMPLModel.load(require_file(Path(cfg.smpl_model_path), "SMPL model file"))`,
    `MissingAssetError` re-raised as `ConfigError`.
  - `__all__` lists the public functions.
- [ ] T022 `src/strike_a_pose/eval/cli.py`: subparser `round-trip` (help "probe round-trip
  plausibility of a sample run's sets (spec 016)") with `--config` (required), `--sample-run`,
  `--probe-run`, `--smpl-model`, `--run-root`; handler `_round_trip(a, argv)` mirroring
  `_sample` (lazy imports; `ConfigError` → `EXIT_CONFIG`, `DatasetError` → `EXIT_DATA`,
  `FileExistsError` → `EXIT_EXISTS`); after `wrote <dir>` prints per set
  `f"  {name}: plausible {rt['plausible_fraction']:.3f} valid {s['valid_fraction']:.3f}
  iou_p50 {rt['iou_quantiles']['p50']:.3f}"` (use `"n/a"` for `None`), then
  `f"  tau {tau:.4f} (p{percentile:g} of real_test)"`. Wire `a.cmd == "round-trip"` in `main`.
  Acceptance: whole suite green.

## Batch B3 — apply to the existing runs + docs (spec FR-009; after B2 is committed)

Files: `configs/roundtrip_vae.yaml`, `configs/roundtrip_vae_beta0.1.yaml`,
`configs/roundtrip_vae_smpl_proc_v3.yaml`, `docs/campaign-scripts/roundtrip016.sh`,
`docs/results/smpl-campaign-015/README.md`, `specs/roadmap.md` (changelog line only).

- [ ] T030 Configs (copy `roundtrip_fixture.yaml`, defaults otherwise):
  - `roundtrip_vae.yaml`: `run_name: roundtrip-vae-smpl_proc_v1`, `sample_run:
    runs/sample-vae-smpl_proc_v1-20261023T002438Z`, `probe_run:
    runs/probe-smpl_proc_v1-20261022T150051Z` (probe of `ae-vae-smpl_proc_v1-20261022T134729Z`).
  - `roundtrip_vae_beta0.1.yaml`: `run_name: roundtrip-vae-beta0.1-smpl_proc_v1`,
    `sample_run: runs/sample-vae-beta0.1-smpl_proc_v1-20261023T002328Z`, `probe_run:
    runs/probe-smpl_proc_v1-20261023T001615Z`.
  - `roundtrip_vae_smpl_proc_v3.yaml`: `run_name: roundtrip-vae-smpl_proc_v3`, `sample_run:
    runs/sample-vae-smpl_proc_v3-20261024T024345Z`, `probe_run:
    runs/probe-smpl_proc_v3-20261023T190738Z` (reference AE in that sample run = R0-v3
    `runs/ae-smpl_proc_v3-20261023T004930Z`).
  Before writing each, verify the probe run's `config.yaml` `checkpoint` is the sample run's
  `checkpoint` (`grep checkpoint runs/<dir>/config.yaml`); a mismatch → stop and report.
- [ ] T031 `docs/campaign-scripts/roundtrip016.sh` (POSIX sh, `set -u`, modelled on
  `smpl015_queue.sh`'s `log` / `$U` helpers): wait while `pgrep -f grill_queue.sh` finds a
  process (poll every 300 s); then for each of the three configs, skip if
  `runs/<run_name>-2026*/samples.json` exists, else
  `uv run --locked strike-a-pose-eval round-trip --config <cfg> >> data/roundtrip016.log 2>&1
  || log "ROUND-TRIP FAILED <cfg>"`; the β 0.1 v3 sample run (grill queue) gets its own config in a later
  batch once its probe run exists (log `BETA0.1-V3 PENDING`); end with `log ROUNDTRIP_DONE`. Do not start it while the queue runs; the lead or operator
  launches it (`nohup sh docs/campaign-scripts/roundtrip016.sh &`).
- [ ] T032 (after the runs) README section "## 016 round-trip plausibility (2026-10-xx)" in
  `docs/results/smpl-campaign-015/README.md`: one table per run, rows `vae_prior`, `ae_prior`,
  `vae_recon_val`, `real_test`, columns n, valid (015), plausible, plausible & valid, IoU p50,
  control p50, control plausible; a line with τ, the refit check and the SMPL match; the
  reading of SC-002 (`vae_prior` plausible fraction vs control plausible fraction, and vs
  `real_test` 0.95); cite run dirs; copy each run's `samples.json`, `round_trip_iou.png` and
  `round_trip.png` into `docs/results/smpl-campaign-015/<run dir name>/` (no checkpoints, no
  SMPL file). Roadmap changelog line "2026-10-xx: slice 016 runs done …".
