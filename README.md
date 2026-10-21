# strike-a-pose

Compact latent representations of human pose: encode, reconstruct and generate
poses. Successor to `Conv_AE-Human_Pose`, rebuilt as an importable Python
library driven by CLI commands and versioned configs, with reproducible runs.

## Status

MVP done (2026-10-20): the image autoencoder trains from a config, its loss
decreases and held-out reconstructions are recognizable (plan:
[`specs/roadmap.md`](specs/roadmap.md); numbers under [Results](#results)).
Post-MVP slice 006 (latent probe) done 2026-10-20 under provisional surrogate
rulings (operator sign-off 2026-10-21). Slice 007 (split content hash) implements the
operator-approved roadmap question #14, slice 008 (best-val weight selection) the
operator's choice B for question #13 (both 2026-10-20), and slice 009 (per-region image
metrics, 2026-10-21) the operator's choice (i) for question #16, and slice 010 (dataset
realism options, 2026-10-21) the operator's decision on question #15 (all four approved at
surrogate scope, operator sign-off pending); the remaining open questions are listed in
the roadmap.

| Slice | What | State |
|---|---|---|
| 001 `dataset-generation` | Synthetic SMPL renders + labels + manifest (`strike-a-pose-data`) | implemented |
| 002 `data-pipeline` | Dataset loader, transforms, frozen split (`strike_a_pose.data`) | implemented |
| 005 `repo-foundation` | CI, README, packaging metadata | implemented |
| 003 `baselines-and-metrics` | Metrics, mean / PCA baselines, eval CLI (`strike_a_pose.eval`) | implemented |
| 004 `autoencoder-mvp` | Config-driven AE train/eval (`strike_a_pose.models`, `strike_a_pose.train`), CI smoke train | implemented |
| 006 `latent-probe` | Linear + MLP probes on the frozen AE latent: 3D joints, camera yaw, SMPL pose (`strike-a-pose-eval probe`) | implemented (post-MVP; surrogate-approved, operator signed off 2026-10-21) |
| 007 `split-content-hash` | Split files pin the manifest content, not the generating commit (`split_version` 2); warning on a commit mismatch | implemented (roadmap #14; follow-up F1: rewrite `splits/smpl_proc_v1.json`, needs SMPL) |
| 008 `best-val-selection` | Training runs score the weights of the epoch with the lowest val loss (`best.pt`) by default (`select_weights`), the final weights when val is empty | implemented (roadmap #13, operator choice B) |
| 009 `region-metrics` | Every image model also reports its MSE in hands / feet / head regions around the true 2D keypoints (`region_mask`, `masked_mse`) | implemented (roadmap #16, operator choice (i)) |
| 010 `dataset-realism` | Opt-in generator options: arms-down table, random body shape, camera ring jitter with an in-frame check; configs for the chain `smpl_proc_v2` → `v5` | implemented (roadmap #15); datasets and runs wait for SMPL |

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

Realism options (slice [010](specs/010-dataset-realism/spec.md); each off by default and then
left out of the manifest, so `smpl_proc_v1` regenerates byte for byte):
`ranges_version: predecessor-v2-armsdown` lowers the arms (∓65° at the shoulders after
sparsification; every other angle drawn as with the default table), `body_shape: {std, clip}`
gives each pose random SMPL betas N(0, std²) clipped at ±clip (shared by its 8 views), and
`camera_jitter` renders each sample from a camera near its ring camera (height ±0.3 m,
0–0.5 m farther, aimed at the ring's look-at point and tilted 0–5° down), kept only if every
mesh vertex projects 2 px inside the image, else the ring camera; labels record the deltas in
`camera.jitter`, while `camera_name` / `yaw_deg` stay the ring camera's. More poses is
`num_poses`. The committed chain `configs/smpl_proc_v2` … `v5` adds one change per version
(v2 arms-down, v3 + shape, v4 + jitter, v5 + 4,000 poses); each directory holds the
`generate`, `baselines`, `ae` and `probe` configs. The run procedure (needs SMPL; after F1 and
a `smpl_proc_v1` re-baseline) is in [`specs/010-dataset-realism/tasks.md`](specs/010-dataset-realism/tasks.md).
For 4,000 poses the 003 image PCA would need an estimated 25–30 GB (≥ 32 GB machine), so
`smpl_proc_v5` runs its baselines with `image_pca_ks: []` (mean image and pose baselines only)
and its AE with `context_pca_k: null`.

Freeze a train/val/test split once per dataset (whole poses, all 8 camera views
together; refuses to overwrite) and commit the file under `splits/`. `val` is for
tuning; `test` is never tuned on.

```bash
uv run strike-a-pose-data split data/generated/smpl_proc_v1 --seed 0 \
    --out splits/smpl_proc_v1.json      # default fractions 0.8 0.1 0.1; split_version 2
```

A split file records the SHA-256 of the dataset's manifest *content*: `manifest.json`
without its generating `code` commit (spec
[007](specs/007-split-content-hash/spec.md)). The loader refuses a dataset with other
content (config, seed, labels, …) and accepts the same dataset regenerated at another
commit with a `SplitCommitWarning` naming both commits; the split keeps the generating
commit for information (`manifest_code`; rewriting the split with `--overwrite`
refreshes it). The 004 `--resume` check and the 006 latent cache still key on the raw
manifest bytes (the manifest does not hash the images), so a regenerated dataset does
not resume an older run or reuse its cached latents.

The committed `splits/smpl_proc_v1.json` (1600 / 200 / 200 poses) is still a
`split_version` 1 file: it pins the raw manifest bytes, code commit included, so it
is valid only for `smpl_proc_v1` generated with `configs/default.yaml`, seed 0, at
commit `406b68f` (default run: ≈ 5 min on 4 CPU cores, 16,000 samples, 351 MB). After
regenerating the dataset, rewrite it with the command above plus `--overwrite` (same
pose lists); the full procedure with its checks is follow-up F1 in
[`docs/HANDOFF.md`](docs/HANDOFF.md).

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

Besides the whole-image MSE, PSNR and silhouette IoU, every image model (here the mean
image and each PCA k; in a training run also the autoencoder) is scored by its MSE in
three regions built from the true 2D keypoints (spec
[009](specs/009-region-metrics/spec.md)): `hands` (21-px squares on both hand joints),
`feet` (17-px squares on both ankle and both foot joints) and `head` (a 29-px square on
the head joint; SMPL-24 has no face joint). A region is the union of its visible joints'
squares, clipped to the image, and every pixel in it counts, body and background; joints
outside the image are dropped, and `samples` counts the images whose region is
non-empty. Each region holds its own mix of body and background pixels, so compare
models within a region, not regions with each other or with the whole-image MSE
(`metrics.json`: `regions` in every image entry, the definitions at the top level).

The metric functions (`image_mse`, `psnr`, `silhouette_iou`, `mpjpe`,
`keypoint_error_2d`, and the region helpers `region_mask` and `masked_mse`) live in
`strike_a_pose.eval` and accept NumPy arrays or torch tensors. Baseline numbers for
`smpl_proc_v1` are under [Results](#results).

Train the convolutional autoencoder (latent 64) from a config: it fits on `train`,
logs train/val loss per epoch to `history.csv`, checkpoints after every epoch, then
scores the selected weights next to the mean image and PCA k=64 (refit on train in the
same run) on `val` and `test`, whole-image and per region (hands, feet, head, as for the
baselines), and writes a reconstruction figure with them. Which
weights are scored is set by `select_weights` (spec
[008](specs/008-best-val-selection/spec.md)): `best_val` (default) scores the epoch with
the lowest val loss, kept as `best.pt`, and falls back to the final weights when val is
empty (as in the fixture) or never finite; `final` scores the last epoch. Training is
the same either way; `metrics.json` names the scored weights in `selection`
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

Runs that have a non-empty val split keep `best.pt` (weights at the lowest val loss,
epoch logged as `best_val_epoch`) next to `checkpoint.pt` (the resume state of the last
epoch); under `select_weights: best_val` it is the scored file. A resume rewrites
`best.pt` from the checkpoint after a kill between the two writes and refuses (exit 2)
when the selected epoch's weights are lost; run directories started before slice 008
resume with `final`, the rule they were started under.

Probe what the frozen encoder of a trained AE knows (spec 006): every image is encoded
once (latents cached under `runs/probe-cache/`, keyed by checkpoint, manifest and split
hashes), then closed-form ridge and small MLP probes are fit on train latents to
predict root-relative 3D joints (world and camera frame), the camera (8 ring yaws) and
the SMPL pose; one hyper-parameter per probe is chosen on val, test is scored once.
Mean-pose / majority / pose-PCA baselines and an untrained-encoder control sit next to
them in `metrics.json` (contract:
[`specs/006-latent-probe/contracts/probe-run.md`](specs/006-latent-probe/contracts/probe-run.md)):

```bash
uv run strike-a-pose-eval probe --config configs/probe.yaml \
    --checkpoint runs/<ae-run>/best.pt                             # full dataset (~7 min, 4 vCPU)
uv run strike-a-pose-eval probe --config configs/probe_fixture.yaml \
    --checkpoint runs/<ae-fixture-run>/checkpoint.pt               # fixture smoke (seconds)
```

Probe the scored weights of the train run, the file its `metrics.json` names in
`selection.checkpoint`: `best.pt` for a full run, `checkpoint.pt` for the fixture
(empty val). `configs/probe.yaml` names a placeholder run directory until the next full
AE run exists; set it to that run's `best.pt` or pass `--checkpoint`.

Exit codes and the on-disk layout are specified in
[`specs/001-dataset-generation/contracts/dataset-layout.md`](specs/001-dataset-generation/contracts/dataset-layout.md).

## Results

The published 003 / 004 / 006 runs below predate the per-region image metrics (slice
009); per-region numbers come with the next full run.

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
[`config.yaml`](docs/results/004-ae-smpl_proc_v1/config.yaml) (the run's resolved config;
`configs/ae.yaml` has since gained `select_weights`); final-epoch weights scored on val
and test, with the mean image and PCA k=64 refit on train in the same run (identical to
the 003 numbers above). Source: run `runs/ae-smpl_proc_v1-20261020T030038Z` at commit
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
([loss curve](docs/results/004-ae-smpl_proc_v1/loss_curve.png)). These published 004
numbers, and the 006 probe numbers below, come from the final-epoch weights of a run made
before slice 008; runs from 008 on report the best-val weights (`select_weights:
best_val`; here the val minimum was epoch 34, 1.8 % below the final val MSE, ≈ 0.08 dB).
That gain is known on val only, the split that selects the epoch, so best-val numbers of
a later run and the final-weight numbers here are compared with that caveat. In the
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
[`config.yaml`](docs/results/006-probe-smpl_proc_v1/config.yaml) (the run's resolved
config; `configs/probe.yaml` now points at the next AE run's `best.pt`); copies of
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
