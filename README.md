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
| 003 `baselines-and-metrics` | Metrics, mean-pose / PCA baselines | planned |
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

Exit codes and the on-disk layout are specified in
[`specs/001-dataset-generation/contracts/dataset-layout.md`](specs/001-dataset-generation/contracts/dataset-layout.md).

## Development checks

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs these on
Python 3.11 and 3.13 for every push and pull request; run the same locally:

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked strike-a-pose-data validate tests/fixtures/smpl_fixture
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
