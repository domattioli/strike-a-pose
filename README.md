# strike-a-pose

Compact latent representations of human pose: encode, reconstruct and generate
poses. Successor to `Conv_AE-Human_Pose`, rebuilt as an importable Python
library driven by CLI commands and versioned configs, with reproducible runs.

## Status

MVP in progress (plan: [`specs/roadmap.md`](specs/roadmap.md)).

| Slice | What | State |
|---|---|---|
| 001 `dataset-generation` | Synthetic SMPL renders + labels + manifest (`strike-a-pose-data`) | implemented |
| 002 `data-pipeline` | Dataset loader, transforms, frozen split (`strike_a_pose.data`) | implemented |
| 005 `repo-foundation` | CI, README, packaging metadata | implemented |
| 003 `baselines-and-metrics` | Metrics, mean / PCA baselines, eval CLI (`strike_a_pose.eval`) | implemented |
| 004 `autoencoder-mvp` | Config-driven AE train/eval, CI smoke train | planned |

## Install

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/). PyTorch comes
from PyPI; on Linux that build pulls CUDA runtime wheels (several GB) but runs on
CPU without a GPU.

```bash
git clone https://github.com/domattioli/strike-a-pose
cd strike-a-pose
uv sync --locked          # exact locked environment, incl. dev tools
```

## Usage

Generating a dataset needs the licensed SMPL body model (never downloaded or
committed; see [`docs/data-assets.md`](docs/data-assets.md)). Validating does not.

```bash
# render 2000 procedural poses x 8 cameras into data/generated/smpl_proc_v1
uv run strike-a-pose-data generate --config configs/default.yaml --seed 0

# check a dataset directory against the layout contract (exit 0 = OK)
uv run strike-a-pose-data validate data/generated/smpl_proc_v1
uv run strike-a-pose-data validate tests/fixtures/smpl_fixture   # committed 16-sample fixture
```

Freeze a train/val/test split once per dataset (whole poses, all 8 camera views
together; refuses to overwrite) and commit the file under `splits/`. `val` is for
tuning; `test` is never tuned on.

```bash
uv run strike-a-pose-data split data/generated/smpl_proc_v1 --seed 0 \
    --out splits/smpl_proc_v1.json                  # default fractions 0.8 0.1 0.1
```

The committed `splits/smpl_proc_v1.json` (1600 / 200 / 200 poses) records the
SHA-256 of the dataset's `manifest.json`, and the loader refuses a mismatch. The
manifest includes the generating config, seed **and code commit**, so this split
is valid only for `smpl_proc_v1` generated with `configs/default.yaml`, seed 0, at
commit `406b68f` (default run: ≈ 5 min on 4 CPU cores, 16,000 samples, 351 MB).

Load samples in PyTorch (sample dict, units and normalization:
[`specs/002-data-pipeline/contracts/sample-contract.md`](specs/002-data-pipeline/contracts/sample-contract.md)):

```python
from torch.utils.data import DataLoader
from strike_a_pose.data import PoseImageDataset

ds = PoseImageDataset(
    "tests/fixtures/smpl_fixture", split="train", split_file="splits/smpl_fixture.json"
)
batch = next(iter(DataLoader(ds, batch_size=4)))
batch["image"].shape  # [4, 1, 256, 192], float32 in [0, 1]
```

Score trivial baselines (mean image, PCA images; mean pose, PCA joints) fitted on
`train` only, on `val` and `test`. Each run writes a directory under `runs/`
(git-ignored) with the resolved config, `run_info.json` (git commit, seed,
environment, dataset + split hashes), `metrics.json` and a reconstruction figure
(contract: [`specs/003-baselines-and-metrics/contracts/eval-run.md`](specs/003-baselines-and-metrics/contracts/eval-run.md)):

```bash
uv run strike-a-pose-eval baseline --config configs/baselines.yaml           # full dataset
uv run strike-a-pose-eval baseline --config configs/baselines_fixture.yaml   # 16-sample fixture
```

The metric functions (`image_mse`, `psnr`, `silhouette_iou`, `mpjpe`,
`keypoint_error_2d`) live in `strike_a_pose.eval` and accept NumPy arrays or torch
tensors. Baseline numbers for `smpl_proc_v1` are under [Results](#results).

Train the convolutional autoencoder (latent 64) from a config: it fits on `train`,
logs train/val loss per epoch to `history.csv`, checkpoints after every epoch, then
scores the final model next to the mean image and PCA k=64 (refit on train in the
same run) on `val` and `test` and writes a reconstruction figure
(contract: [`specs/004-autoencoder-mvp/contracts/train-run.md`](specs/004-autoencoder-mvp/contracts/train-run.md)):

```bash
uv run strike-a-pose-train --config configs/ae.yaml           # full dataset (~1.5 h, 4 vCPU)
uv run strike-a-pose-train --config configs/ae_fixture.yaml   # 16-sample fixture smoke (seconds)
uv run strike-a-pose-train --resume runs/<run-dir>            # continue an interrupted run
```

`--resume` restores model, optimizer, lr schedule, RNG states and history from the
last end-of-epoch `checkpoint.pt`; the finished run equals an uninterrupted one on
the same machine and thread count.

Exit codes and the on-disk layout are specified in
[`specs/001-dataset-generation/contracts/dataset-layout.md`](specs/001-dataset-generation/contracts/dataset-layout.md).

## Results

Baselines on `smpl_proc_v1` (fit on train, 12,800 images; val / test 1,600 images
each; images in [0, 1]; MPJPE root-relative). Context for the autoencoder (004),
not a bar to beat. Source: run `runs/baselines-smpl_proc_v1-20261020T022918Z` at
commit `aded3a4`, config [`configs/baselines.yaml`](configs/baselines.yaml); copy of
its [`metrics.json`](docs/results/003-baselines-smpl_proc_v1/metrics.json) and
[`run_info.json`](docs/results/003-baselines-smpl_proc_v1/run_info.json) committed.

| Baseline | val MSE | test MSE | val / test PSNR (dB) | val / test silhouette IoU |
|---|---|---|---|---|
| mean image | 0.01403 | 0.01385 | 18.65 / 18.71 | 0.416 / 0.420 |
| PCA k=64 (= AE latent size) | 0.00459 | 0.00452 | 23.73 / 23.79 | 0.644 / 0.646 |
| PCA k=128 | 0.00327 | 0.00321 | 25.19 / 25.25 | 0.701 / 0.704 |

| Baseline (3D joints) | val / test MPJPE (mm) | val / test 2D error (px) |
|---|---|---|
| mean pose | 253.4 / 250.3 | 18.35 / 18.17 |
| PCA k=8 | 94.0 / 93.6 | 6.83 / 6.78 |
| PCA k=16 | 34.2 / 34.6 | 2.50 / 2.52 |

All k values (image 8–128, pose 4–32) are in `metrics.json` and
[`specs/003-baselines-and-metrics/tasks.md`](specs/003-baselines-and-metrics/tasks.md).

## Development checks

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs these on
Python 3.11 and 3.13 for every push and pull request; run the same locally:

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked strike-a-pose-data validate tests/fixtures/smpl_fixture
uv run --locked strike-a-pose-eval baseline --config configs/baselines_fixture.yaml --run-root /tmp/runs
uv run --locked strike-a-pose-train --config configs/ae_fixture.yaml --run-root /tmp/runs
```

No SMPL/AMASS files are needed: tests use synthetic SMPL-format assets and the
committed fixture.

Work follows spec-kit (specify, clarify, plan, tasks, analyze, implement);
per-slice docs live in [`specs/`](specs/), project rules in
[`.specify/memory/constitution.md`](.specify/memory/constitution.md).

## License

[PolyForm Noncommercial 1.0.0](LICENSE) **plus an additional No-AI/ML-training
term**: the software may not be used as training, fine-tuning or evaluation
input for AI/ML models, or for code generation, distillation, embeddings or
retrieval corpora, without a separate written license. See [`LICENSE`](LICENSE)
for the full terms and the contact for commercial or AI-training licenses.

The SMPL body model and AMASS motion data are third-party assets under their
own non-commercial research licenses; obtain them yourself and follow their
terms ([`docs/data-assets.md`](docs/data-assets.md)).
