<!-- provenance: author=domattioli model=claude-fable-5-1 effort=high date=2026-10-07 skill=speckit-plan repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
# Quickstart: Kill-Test MVP for Calibrated Multi-View Body-Measurement Uncertainty

**Branch**: `001-kill-test-mvp` | **Date**: 2026-10-07 | **Plan**: [plan.md](plan.md) | **CLI**: [contracts/cli.md](contracts/cli.md)

This file is the runbook for the implementation and for the operator. It assumes the tasks in `tasks.md` are complete. No README exists before the first verdict (constitution Principle IV).

## 1. CPU development environment (no GPU, no asset)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -c constraints.txt -e ".[dev]"
sap info                       # prints versions, device (cpu), asset root status
bash scripts/cpu_smoke.sh      # ruff, pytest with timing, tiny end-to-end run; fails over 10 minutes
```

`scripts/cpu_smoke.sh` is the pre-commit gate of constitution Principle VI. The suite runs with the network blocked and `SAP_ASSET_ROOT` unset; a test that needs either is a defect.

## 2. Tiny end-to-end run (FR-027)

```bash
sap run --config configs/tiny.yaml --out /tmp/sap-tiny
cat /tmp/sap-tiny/report/results.md
```

Expected within 10 minutes on 4 cores: `results.csv` with 45 rows (9 cells by 5 measurements), `verdict.md` with one verdict line, two plots, and `run_record.json`. The numbers carry no scientific meaning at this size; the columns and the verdict logic are what the run checks.

## 3. Licensed assets (operator step; never committed, never downloaded by the package)

Place the files under one directory and point `SAP_ASSET_ROOT` (or `assets.root` in the configuration) at it:

```text
<asset_root>/
├── smplx/SMPLX_NEUTRAL.npz              # SMPL-X site, research license
├── smpl/SMPL_MALE.pkl, SMPL_FEMALE.pkl   # SMPL site; chumpy-free files per the smplx README (SSP-3D ground truth only)
├── amass/<subset>/**/*.npz               # AMASS, SMPL-X format, one or more subsets
├── bodym/{train,testA,testB}/            # BodyM (CC BY-NC 4.0) as downloaded
├── ssp3d/                                # SSP-3D zip contents: images, silhouettes, labels.npz
└── sam2/<checkpoint>.pt                  # SAM 2 checkpoint (optional, extra [real])
```

`sap info` lists each asset as present or missing with the configuration key that names it. A run that needs a missing asset stops with that message (FR-021).

## 4. Full run on Kaggle (FR-029, SC-010)

1. Build the bundle on any machine: `bash scripts/build_kaggle_bundle.sh` writes `dist/kaggle-bundle/` with the wheel, `constraints.txt`, and `configs/`.
2. Create two private, owner-only Kaggle datasets: `sap-src` from `dist/kaggle-bundle/` and `sap-assets` from `<asset_root>` (the asset licenses forbid public datasets).
3. Create a notebook from `notebooks/kaggle_run.ipynb`, attach both datasets, enable a GPU, and run. The notebook prints `python --version`, `torch.__version__`, and the GPU name, installs the wheel with `-c constraints.txt`, sets `SAP_ASSET_ROOT=/kaggle/input/sap-assets` and `SAP_OUTPUT_DIR=/kaggle/working/out`, and runs `sap run --config configs/full.yaml --out $SAP_OUTPUT_DIR --resume --time-budget 8.5h`.
4. The run stops at a checkpoint before the budget ends. For the next session, add the previous notebook version's output as an input (Add Input, Your Work), and run again: the notebook copies `out/` from that input to `/kaggle/working/out` and `--resume` continues from the last shard, epoch, or cell. Fallback when output chaining is unavailable: save `out/` to a private dataset from inside the notebook and attach it.
5. Repeat until `report/results.md` and `verdict/verdict.md` exist. Expected: generation about 20 minutes, training under 2 hours, prediction 1 to 3 hours (research R13); three sessions at most.

## 5. Real-image rows (FR-017 to FR-020, reported only)

```bash
sap real-eval --config configs/full.yaml --out <out> --dataset bodym
sap real-eval --config configs/full.yaml --out <out> --dataset ssp3d --mask-source provided
sap real-eval --config configs/full.yaml --out <out> --dataset ssp3d --mask-source sam2   # needs extra [real] and the checkpoint
```

Each call writes `real/<dataset>/results.csv` (aggregate; committed) and `real/<dataset>/subjects.csv` (per subject; never committed).

## 6. Reproduce and verify (SC-003, SC-005)

```bash
sap verify --out <out>          # recomputes evaluate and verdict from predict outputs; fails on any difference
sap run --config configs/full.yaml --out <out2> --seed-check <out>   # second run; compares manifests byte for byte and metrics within tolerance
```

## 7. Record the gate decision

Copy `evaluate/results.csv`, `report/results.md`, `verdict/verdict.json`, `verdict/verdict.md`, the two plots, `run_record.json`, and the aggregate `real/*/results.csv` into `specs/001-kill-test-mvp/results/` and commit them with the clean-room statement in the pull request. Nothing else from `<out>` is committed.
