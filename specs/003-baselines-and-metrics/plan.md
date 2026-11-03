# Implementation Plan: Baselines and Metrics

**Branch**: `claude/charming-wozniak-5ouldf` (slice `003-baselines-and-metrics`) | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)
**Input**: `specs/003-baselines-and-metrics/spec.md` (approved, autonomous run 2026-10-20)

## Summary

Generate the full default dataset and freeze its split (001 T025). Add
`strike_a_pose.eval` (metrics, exact PCA, baseline runner, config, CLI
`strike-a-pose-eval baseline`) and `strike_a_pose.runlog` (seeding, environment
summary, run dir). Fit mean / PCA baselines for images and root-relative 3D
joints on train, score val and test, write a self-describing run directory,
record the full-dataset numbers in docs.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.11 + 3.13)
**Primary Dependencies**: existing only — torch (metrics, PCA linear algebra),
numpy, pillow (figure), pyyaml (config). No new dependency.
**Storage**: reads 001 datasets via the 002 loader; writes run dirs under
`runs/` (git-ignored); commits `splits/smpl_proc_v1.json` and a copy of the
full-dataset `metrics.json` + `run_info.json` under `docs/results/`.
**Testing**: pytest; hand-computed metric cases; synthetic low-rank PCA data;
CLI end-to-end on the committed fixture (`configs/baselines_fixture.yaml`).
**Performance Goals**: fixture run < 30 s (SC-002); full run < 30 min on 4 CPU
cores (SC-003); CI leg < 5 min (SC-004).
**Constraints**: train-only fitting; `metrics.json` byte-deterministic; no
change to 001/002 behaviour.

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | Metrics, PCA, runner, run-dir helpers in `src/strike_a_pose/`; CLI is a thin wrapper; no notebooks. |
| II CLI + config | PASS | `strike-a-pose-eval baseline --config configs/baselines.yaml`; all paths and knobs in the YAML; overrides recorded in the resolved config. |
| III Reproducibility | PASS | Seeds set + logged; run dir holds resolved config, git commit + dirty flag, env summary, dataset manifest hash, split hash, metrics; exact PCA (no randomness); `metrics.json` byte-stable (tested). |
| IV Test-first | PASS | Metric tests with hand-computed values (NumPy + torch), PCA tests, runlog tests, CLI test on fixture, written with/before the code. CPU smoke (fixture baseline run) also in CI. |
| V Honest evaluation | PASS | Fit on train only (hard-coded); val and test reported; figure from val; frozen pose-level split; units stated (mm, px, dB); docs cite the run dir + commit. Baselines are context, not a bar (roadmap). |
| VI Simplicity | PASS | Flat config, five metric functions, one PCA routine, PIL figure; no new dependency, no plugin/registry for baselines. |
| Tech: pinned deps | PASS | No dependency change; new console script only (`uv lock --check`). |

## Design

### `strike_a_pose/runlog.py`

- `seed_everything(seed)` — `random.seed`, `np.random.seed`, `torch.manual_seed`.
- `environment_summary() -> dict` — python, platform, numpy, torch, pillow,
  `os.cpu_count()`, `torch.get_num_threads()`.
- `git_info() -> dict` — re-export of 001 `generate.git_commit()`.
- `make_run_dir(run_root, run_name, now=None) -> Path` — `<root>/<name>-<stamp>`,
  `mkdir(parents=True, exist_ok=False)` (FileExistsError → exit 4).
- `utc_now() -> datetime`; `write_yaml(path, obj)`; `write_json_exact(path, obj)`
  (sig-digit rounding, NaN → null, sorted keys).

### `strike_a_pose/eval/metrics.py`

Contract table. Implementation: `_to_torch(x)` (NumPy → `torch.from_numpy`, keep
dtype) and `_back(y, like)`; all math in torch; return type follows input.

### `strike_a_pose/eval/pca.py`

`PCA` dataclass + `fit_pca` per contract. `n ≤ d`: `G = Xc Xcᵀ` (input dtype) →
`eigh(G.double())`, components `Vᵀ = (Xcᵀ U_k) / √λ_k`. `n > d`: `C = Xcᵀ Xc`
in float64 → `eigh`. Variances `λ / (n − 1)`; total variance `trace / (n − 1)`.
Mean computed with float64 accumulation. `n == 1` → rank 0 (mean only).

### `strike_a_pose/eval/config.py`

`BaselineConfig` frozen dataclass (contract table), `load_config(path)` /
`config_from_dict(d)` with strict key/type/range checks → `ConfigError(ValueError)`;
`to_dict()` for the resolved YAML.

### `strike_a_pose/eval/baselines.py`

- `load_arrays(ds, batch_size, images=True) -> dict` — iterate a
  `PoseImageDataset` through a `DataLoader` (no shuffle, `num_workers=0`), fill
  preallocated tensors: `image [n, H·W] f32`, `mask [n, H·W] bool`,
  `joints_world [n, 24, 3]`, `kp2d [n, 24, 2]`, `visible [n, 24]`, `K, R, t`,
  `pose_index [n]`.
- `fit_baselines(train, cfg) -> Baselines` — mean image, image PCA at
  `max(image_pca_ks)`, mean root-relative pose, pose PCA at `max(pose_pca_ks)`.
- `image_predictions(b, x, cfg)` yields `(name, k, pred)` batched; `score_split(b,
  arrays, cfg) -> dict` — image metrics per baseline (batched to bound memory);
  pose: predicted root-relative joints → MPJPE (mm) and, after adding the true
  root and projecting, 2D error (px).
- `recon_figure(b, arrays, cfg) -> PIL.Image` — header labels + rows.
- `run_baselines(cfg, argv) -> Path` — check inputs (construct datasets; split
  file vs manifest) → run dir → `config.yaml`, `run_info.json` (start) → fit →
  score → `metrics.json`, figure → `run_info.json` (finish).

### `strike_a_pose/eval/cli.py`

argparse `baseline` subcommand; error → exit-code mapping per contract; prints
run dir and the headline numbers (mean image, PCA k = 64 if present, mean pose)
per eval split.

### Packaging, configs, CI

- `pyproject.toml`: `strike-a-pose-eval = "strike_a_pose.eval.cli:main"`.
- `configs/baselines.yaml` (full dataset defaults), `configs/baselines_fixture.yaml`
  (fixture: `image_pca_ks [2, 4]`, `pose_pca_ks [2]`, figure on `test`, 4 rows).
- `.gitignore`: `/runs/`.
- `ci.yml`: one smoke step `strike-a-pose-eval baseline --config
  configs/baselines_fixture.yaml --run-root "$RUNNER_TEMP/runs"` (cross-slice
  touch of 005's file, additive; recorded in the roadmap).

### Step 0 (001 T025)

`generate` default config seed 0 → `validate` → `split --seed 0 --out
splits/smpl_proc_v1.json`; wall time + size recorded in 001 tasks.md T025;
README notes that the split is tied to the manifest hash.

## Project Structure

```text
src/strike_a_pose/runlog.py              # new
src/strike_a_pose/eval/__init__.py       # new
src/strike_a_pose/eval/metrics.py        # new
src/strike_a_pose/eval/pca.py            # new
src/strike_a_pose/eval/config.py         # new
src/strike_a_pose/eval/baselines.py      # new
src/strike_a_pose/eval/cli.py            # new
configs/baselines.yaml                   # new
configs/baselines_fixture.yaml           # new
splits/smpl_proc_v1.json                 # new (frozen full-dataset split)
docs/results/003-baselines-smpl_proc_v1/{metrics,run_info}.json  # new, copied from the run
tests/test_eval_metrics.py               # new
tests/test_eval_pca.py                   # new
tests/test_eval_config.py                # new
tests/test_eval_cli.py                   # new (fixture end-to-end)
tests/test_runlog.py                     # new
pyproject.toml, .gitignore, .github/workflows/ci.yml
README.md, specs/roadmap.md, specs/001-dataset-generation/tasks.md
```

## Complexity Tracking

None. (Cross-slice touches — 001 tasks.md T025 record, one additive CI step —
are recorded, not violations.)
