<!-- provenance: author=domattioli model=claude-opus-5-5 effort=high date=2026-10-07 skill=speckit-analyze repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
# Contract: `sap` command line

**Branch**: `001-kill-test-mvp` | **Date**: 2026-10-07 | **Plan**: [../plan.md](../plan.md)

One entry point, `sap`, installed by `pyproject.toml`. Every subcommand reads one configuration file ([config.md](config.md)) and writes under one output directory ([artifacts.md](artifacts.md)). Paths may be absolute or relative to the current directory.

## Environment

| Variable | Meaning | Precedence |
|---|---|---|
| `SAP_ASSET_ROOT` | directory of licensed assets (FR-021) | configuration `assets.root` wins when set; the environment variable is the default |
| `SAP_OUTPUT_DIR` | default for `--out` | `--out` wins |
| `SAP_DEVICE` | `auto`, `cpu`, `cuda` | `--device` wins; `auto` picks CUDA when available, else CPU (FR-028) |

## Subcommands

| Command | Reads | Writes | Notes |
|---|---|---|---|
| `sap info [--strict]` | configuration (optional), environment | stdout | versions, device, asset presence per key; `--strict` exits 5 when an installed version differs from `constraints.txt` |
| `sap generate --config C --out O [--resume]` | configuration, pose asset when `pose.source: amass`, body asset when `body.model: smplx` | `data/` | resumes per shard; exit 4 when fewer than `data.min_unflagged` unflagged calibration or test bodies remain, naming both counts (SC-002) |
| `sap train --config C --out O [--resume] [--time-budget T]` | `data/` | `train/` | checkpoints per epoch and every `train.checkpoint_every` steps |
| `sap predict --config C --out O [--resume]` | `data/`, `train/` | `predict/` | one file per cell and split (cal, test) |
| `sap calibrate --config C --out O` | `predict/` (cal split) | `calibrate/` | refuses `n_cal < calibrate.min_cal` with exit 4 and a message naming the minimum (FR-011) |
| `sap evaluate --config C --out O` | `predict/` (test split), `calibrate/` | `evaluate/` | `results.csv` with band flags (FR-012, FR-013) |
| `sap verdict --config C --out O` | `evaluate/` | `verdict/` | FR-014 rule with its constants held in code; prints the verdict line; a band miss is KILL with exit 0; exit 4 only on a non-finite ratio, a missing compared cell, or an SC-004 violation, after writing the verdict files; exit 2 when the configuration differs from FR-014 |
| `sap report --config C --out O` | `evaluate/`, `verdict/`, `real/` when present | `report/` | `results.md` and plots (FR-015, FR-016) |
| `sap run --config C --out O [--resume] [--time-budget T] [--seed-check O_ref]` | as above | all of the above | stages in order; a KILL (exit 0 from `verdict`) continues to `report`; an exit 4 stops the run after the verdict files are written; `--seed-check` makes this run the second run and compares its manifests byte for byte and its metrics within SC-005 tolerance against a reference output |
| `sap real-eval --config C --out O --dataset {bodym,ssp3d} [--mask-source {provided,sam2}]` | `train/`, `calibrate/`, dataset assets, SAM 2 checkpoint when `sam2` | `real/<dataset>/` | reported only (FR-020) |
| `sap verify --out O` | `predict/`, `evaluate/`, `verdict/` | stdout | recomputes evaluate and verdict from predict outputs; exit 6 on any difference (SC-003) |

Common options: `--device {auto,cpu,cuda}`, `--log-level`, `--time-budget` as `<hours>h` or `<minutes>m` (the run stops cleanly at the next checkpoint when the budget would be exceeded by the next unit of work; the partial stage is resumable). `--set key=value` overrides one configuration key; it is repeatable, one key per `--set`, and the value is parsed as YAML (`--set evaluate.views=[1,4]` sets a list); an override of an FR-014 fixed key (`verdict.*`, `evaluate.band`) exits 2.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | configuration error (unknown key, bad type, missing required key); the message names the key |
| 3 | missing asset; the message names the asset file and the configuration key (FR-021) |
| 4 | stage refused (calibration set too small, too few unflagged bodies after generation) or a non-finite prediction; for `sap verdict`, a non-finite ratio, a missing compared cell, or an SC-004 violation, after the verdict files are written; the message names the sample, the cell, or the limit. A KILL verdict is not an error and exits 0 |
| 5 | strict version check failed |
| 6 | verification found a difference |
| 7 | time budget reached before completion (partial, resumable); printed with the resume command |

## Stdout and stderr

Progress and the verdict line go to stdout. The verdict line format is fixed: `VERDICT: PASS|KILL median_ratio=<x.xxx> threshold=0.700 cells_in_band=<true|false> invalid=<true|false> sc004_violations=<n> (chest=<r> waist=<r> hip=<r> thigh=<r>) widths_v1_v4_cm=(chest=<w1>/<w4> waist=<w1>/<w4> hip=<w1>/<w4> thigh=<w1>/<w4>) config=<hash12> seed=<n>`, followed by `; comparison invalid for cells <ids>` exactly when `cells_in_band=false`, where `<ids>` is a comma-separated list of `<cell_id>:<measurement>` (for example `v4_n0:waist,v1_n0:hip`). Ratios and widths print with three decimals; a non-finite value prints as `nan` or `inf`. The same line is the first line of `verdict/verdict.md`. `scripts/check_tiny_run.py` matches it with this regex:

```text
^VERDICT: (PASS|KILL) median_ratio=\S+ threshold=0\.700 cells_in_band=(true|false) invalid=(true|false) sc004_violations=[0-9]+ \(chest=\S+ waist=\S+ hip=\S+ thigh=\S+\) widths_v1_v4_cm=\(chest=\S+/\S+ waist=\S+/\S+ hip=\S+/\S+ thigh=\S+/\S+\) config=[0-9a-f]{12} seed=[0-9]+(; comparison invalid for cells [a-z0-9_]+:[a-z]+(,[a-z0-9_]+:[a-z]+)*)?$
```

Errors go to stderr as one line that starts with `sap: error:`.
