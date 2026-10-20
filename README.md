# strike-a-pose

Compact latent representations of human pose: encode, reconstruct and generate
poses. Successor to `Conv_AE-Human_Pose`, rebuilt as an importable Python
library driven by CLI commands and versioned configs, with reproducible runs.

## Status

MVP done (2026-10-20): the image autoencoder trains from a config, its loss
decreases and held-out reconstructions are recognizable (plan:
[`specs/roadmap.md`](specs/roadmap.md); numbers under [Results](#results)).
Post-MVP slice 006 (latent probe) done 2026-10-20 under provisional surrogate
rulings (operator sign-off 2026-10-21); no further slice is planned until the operator
answers the roadmap's open questions.

| Slice | What | State |
|---|---|---|
| 001 `dataset-generation` | Synthetic SMPL renders + labels + manifest (`strike-a-pose-data`) | implemented |
| 002 `data-pipeline` | Dataset loader, transforms, frozen split (`strike_a_pose.data`) | implemented |
| 005 `repo-foundation` | CI, README, packaging metadata | implemented |
| 003 `baselines-and-metrics` | Metrics, mean / PCA baselines, eval CLI (`strike_a_pose.eval`) | implemented |
| 004 `autoencoder-mvp` | Config-driven AE train/eval (`strike_a_pose.models`, `strike_a_pose.train`), CI smoke train | implemented |
| 006 `latent-probe` | Linear + MLP probes on the frozen AE latent: 3D joints, camera yaw, SMPL pose (`strike-a-pose-eval probe`) | implemented (post-MVP; surrogate-approved, operator signed off 2026-10-21) |

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
uv run strike-a-pose-train --config configs/ae.yaml           # full dataset (~85 min, 4 vCPU)
uv run strike-a-pose-train --config configs/ae_fixture.yaml   # 16-sample fixture smoke (seconds)
uv run strike-a-pose-train --resume runs/<run-dir>            # continue an interrupted run
```

`--resume` restores model, optimizer, lr schedule, RNG states and history from the
last end-of-epoch `checkpoint.pt`; the finished run equals an uninterrupted one on
the same machine and thread count. Long runs can be killed (e.g. a container
restart): launch them detached and resume the same run dir rather than starting
over:

```bash
mkdir -p runs
setsid nohup uv run --locked strike-a-pose-train --config configs/ae.yaml \
    > runs/ae.log 2>&1 < /dev/null &        # prints the run dir + resume command
tail -3 runs/ae.log                          # progress, one line per epoch
setsid nohup uv run --locked strike-a-pose-train --resume runs/<run-dir> \
    > runs/ae-resume.log 2>&1 < /dev/null &  # after a kill (no metrics.json yet)
```

Never run two `--resume` processes on one run dir (there is no lock).

Runs that have a non-empty val split also keep `best.pt` (weights at the lowest val
loss, epoch logged as `best_val_epoch`) next to `checkpoint.pt`; it is informational —
the final weights are the ones evaluated.

Probe what the frozen encoder of a trained AE knows (spec 006): every image is encoded
once (latents cached under `runs/probe-cache/`, keyed by checkpoint, manifest and split
hashes), then closed-form ridge and small MLP probes are fit on train latents to
predict root-relative 3D joints (world and camera frame), the camera (8 ring yaws) and
the SMPL pose; one hyper-parameter per probe is chosen on val, test is scored once.
Mean-pose / majority / pose-PCA baselines and an untrained-encoder control sit next to
them in `metrics.json` (contract:
[`specs/006-latent-probe/contracts/probe-run.md`](specs/006-latent-probe/contracts/probe-run.md)):

```bash
uv run strike-a-pose-eval probe --config configs/probe.yaml        # full dataset (~7 min, 4 vCPU)
uv run strike-a-pose-eval probe --config configs/probe_fixture.yaml \
    --checkpoint runs/<ae-fixture-run>/checkpoint.pt               # fixture smoke (seconds)
```

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

Autoencoder (004) on `smpl_proc_v1`: `ConvAutoencoder` (latent 64, 3.1 M
parameters) trained 60 epochs on train with config
[`configs/ae.yaml`](configs/ae.yaml); final-epoch weights scored on val and test,
with the mean image and PCA k=64 refit on train in the same run (identical to the
003 numbers above). Source: run `runs/ae-smpl_proc_v1-20261020T030038Z` at commit
`f9619c6` (clean; resumed across two container restarts with unchanged code); copies of
[`metrics.json`](docs/results/004-ae-smpl_proc_v1/metrics.json),
[`run_info.json`](docs/results/004-ae-smpl_proc_v1/run_info.json),
[`history.csv`](docs/results/004-ae-smpl_proc_v1/history.csv) and the figures in
[`docs/results/004-ae-smpl_proc_v1/`](docs/results/004-ae-smpl_proc_v1/).

| Model | val MSE | test MSE | val / test PSNR (dB) | val / test silhouette IoU |
|---|---|---|---|---|
| **autoencoder (latent 64)** | **0.00221** | **0.00219** | **27.15 / 27.13** | **0.879 / 0.881** |
| mean image | 0.01403 | 0.01385 | 18.65 / 18.71 | 0.416 / 0.420 |
| PCA k=64 | 0.00459 | 0.00452 | 23.73 / 23.79 | 0.644 / 0.646 |

Training took 84 min of compute on 4 vCPU (78 s/epoch + data loading and
evaluation). Train MSE fell 0.0352 → 0.00106; val MSE fell 0.0100 → its minimum
0.00217 at epoch 34 and ended at 0.00221: val flattens from about epoch 30 while
train keeps falling, i.e. mild overfitting
([loss curve](docs/results/004-ae-smpl_proc_v1/loss_curve.png)). In the
[val reconstructions](docs/results/004-ae-smpl_proc_v1/recon_val.png) all six
poses shown are recognizable (body orientation, arm and leg angles), but the images are
blurred: faces and hands lose all detail (fingers vanish), feet and wrists are
soft, and crossing or touching limbs merge.

Latent probe (006) on `smpl_proc_v1`: the frozen 004 encoder (final-epoch weights,
latent 64) encodes every image once; probes are fit on the 12,800 train latents, one
hyper-parameter (ridge λ, MLP weight decay) is chosen on val, test is scored once.
The control is the same probe on an **untrained** encoder of the same architecture
(seeded random weights). Source: run `runs/probe-smpl_proc_v1-20261020T060632Z` at
commit `ae32db9` (clean; 450 s on 4 vCPU, encoding 80 s), config
[`configs/probe.yaml`](configs/probe.yaml); copies of
[`metrics.json`](docs/results/006-probe-smpl_proc_v1/metrics.json),
[`run_info.json`](docs/results/006-probe-smpl_proc_v1/run_info.json) and the figures in
[`docs/results/006-probe-smpl_proc_v1/`](docs/results/006-probe-smpl_proc_v1/).

| Root-relative 3D joints, MPJPE mm (val / test) | world frame (as 003) | camera frame |
|---|---|---|
| **MLP probe** (64 → 256 → 256) | **172.0 / 172.8** | **161.1 / 159.6** |
| ridge (linear) probe | 214.5 / 212.5 | 234.3 / 231.8 |
| MLP probe, untrained encoder | 214.6 / 214.5 | 235.4 / 234.5 |
| ridge probe, untrained encoder | 228.4 / 226.7 | 268.3 / 266.7 |
| mean pose (fit on train) | 253.4 / 250.3 | 313.1 / 308.6 |
| *oracle:* pose PCA k=8 of the **true** joints | *94.0 / 93.6* | *143.3 / 139.8* |
| *oracle:* pose PCA k=16 of the **true** joints | *34.2 / 34.6* | *44.1 / 45.2* |

The PCA rows are not image-based: they compress the true joints themselves (an upper
bound on what a k-dimensional linear code of the joints can reach); the probes see only
the image latent. World-frame mean-pose and PCA numbers equal the 003 run exactly.
2D error of the projected prediction (true root and camera): MLP 11.5 px, ridge 14.8 px,
mean pose 18.4 px (val, world frame).

| Camera (8 ring yaws), accuracy (val / test) | accuracy | val errors by offset 45° / 90° / 135° / 180° |
|---|---|---|
| **MLP probe** | **0.965 / 0.959** | 17 / 11 / 6 / 22 |
| ridge (linear) probe | 0.826 / 0.819 | 142 / 29 / 48 / 60 |
| MLP probe, untrained encoder | 0.685 / 0.670 | 161 / 133 / 105 / 105 |
| ridge probe, untrained encoder | 0.602 / 0.599 | 209 / 187 / 135 / 106 |
| majority class | 0.125 / 0.125 | — |

SMPL pose (geodesic joint-rotation error, joints 1–21, val / test): MLP 23.4° / 24.1°,
ridge 24.9° / 25.5°, untrained-encoder MLP 25.0° / 25.7°, mean pose 26.9° / 27.5°.

What the latent encodes (lead's reading of this one run): the **view** is in it and
almost fully recoverable — 96 % camera accuracy with an MLP; the remaining errors are
mostly front ↔ back flips (`Front_0` → `Back_180` 9, `Back_180` → `Front_0` 12 of 200
each on val; [confusion](docs/results/006-probe-smpl_proc_v1/yaw_confusion_val.png)),
which a grey silhouette-like render barely distinguishes. Much of that is linear
(ridge 83 %, its errors mostly 45° neighbours). **Body configuration is only coarsely
present:** the best probe removes ≈ 32 % (world) / 48 % (camera frame) of the mean-pose
error, about as good as a 4-component PCA of the true joints (170 mm in 003), far from
the 8-component code (94 mm). The
[skeleton figure](docs/results/006-probe-smpl_proc_v1/skeleton_val.png) shows the
pattern: torso orientation and outstretched arms are right, bent or raised legs and
bent arms are pulled toward the mean pose. Training the encoder clearly adds pose
information over random conv features (MLP 172 vs 215 mm; yaw 96 % vs 69 %), but a
pixel-MSE autoencoder spends its 64 numbers on what dominates pixel error (silhouette
extent and orientation), not on limb angles, and what joint information there is is
mostly non-linear (ridge 214 mm). Caveat: the MLP weight-decay grid picked its largest
value (1e-3) for most targets, so slightly stronger regularization might help; the grid
was fixed a priori and not widened after test was seen.

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
uv run --locked strike-a-pose-eval probe --config configs/probe_fixture.yaml --run-root /tmp/runs \
    --checkpoint "$(ls -d /tmp/runs/ae-fixture-* | tail -1)/checkpoint.pt"
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
