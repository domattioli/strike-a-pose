# Implementation Plan: Data Pipeline (loader, transforms, frozen split)

**Branch**: `claude/charming-wozniak-5ouldf` (slice `002-data-pipeline`) | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)
**Input**: `specs/002-data-pipeline/spec.md` (approved, autonomous run 2026-10-20)

## Summary

New package `strike_a_pose.data` with three modules: `transforms` (image and
keypoint normalization, NumPy/torch-generic), `split` (pose-level seeded split,
split-file I/O and checks; NumPy only), `dataset` (`PoseImageDataset`, manifest
checks, item reading). The existing `strike-a-pose-data` CLI gains a `split`
subcommand. PyTorch is added as a locked dependency. The fixture split is
committed under `splits/`.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.11 + 3.13)
**Primary Dependencies**: + `torch>=2.5` (locked 2.14.1, PyPI; research R1). Reason: the
`Dataset`/`DataLoader` API and tensors for 003/004 (constitution stack). License:
BSD-3 (compatible). Existing numpy, pillow reused.
**Storage**: datasets read-only (001 layout v1); split files JSON in `splits/` (committed)
**Testing**: pytest; committed fixture (2 poses × 8 cams) + temp copies corrupted
per error case + synthetic manifests (no files) for split statistics
**Target Platform**: CPU (Linux CI, Linux/macOS dev)
**Performance Goals**: fixture loads + batches < 5 s (SC-001); CI leg < 5 min (SC-004)
**Constraints**: no change to 001 behaviour or layout; split file byte-deterministic

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | All logic in `src/strike_a_pose/data/`; CLI is a thin wrapper. |
| II CLI + config | PASS | Split is a CLI with explicit `--seed`, `--out`, `--fractions`; no hidden paths (split file path explicit in the API too). Loader has no run of its own; 004's config names dataset dir + split file. |
| III Reproducibility | PASS | Split deterministic from (manifest bytes, seed, fractions, algorithm); manifest SHA-256 recorded and checked on load; file committed. |
| IV Test-first | PASS | Transforms, split, dataset, CLI tests written with the code (tasks order tests first per module). Smoke train remains 004. |
| V Honest evaluation | PASS | Held-out split defined once (frozen: CLI refuses overwrite), pose-level to avoid leakage; test split documented as never tuned on. Units stated in contract. |
| VI Simplicity | PASS | 3 modules, 1 subcommand, 1 new dependency with stated reason; no caching, no registry, no augmentation. |
| Tech: pinned deps | PASS | `uv.lock` updated, `uv sync --locked` in CI. |
| Tech: license-compatible deps | PASS | PyTorch BSD-3; NVIDIA runtime wheels are redistributable runtime deps of torch (not shipped by us). |

## Design

### `strike_a_pose/data/transforms.py` (no torch import at module level needed for keypoint helpers)

- `image_to_tensor(img: np.ndarray) -> torch.Tensor` — uint8 `[H, W]` → float32 `[1, H, W]` `/255`. Rejects non-uint8 / non-2D.
- `tensor_to_image(t: torch.Tensor) -> np.ndarray` — `[1, H, W]` or `[H, W]` → uint8 via clamp + `round(x·255)`.
- `mask_to_tensor(mask: np.ndarray) -> torch.Tensor` — `[H, W]` → bool `[1, H, W]` (`> 127`).
- `root_relative(j, root=0)`, `add_root(j_rel, root_pos)`.
- `normalize_keypoints_2d(kp, width, height)`, `denormalize_keypoints_2d(kp_n, width, height)`.
  Generic: only slicing and arithmetic, so NumPy and torch both work.

### `strike_a_pose/data/split.py` (NumPy + 001 io only)

- `SPLITS = ("train", "val", "test")`, `SPLIT_VERSION = 1`, `ALGORITHM = "pose-permutation-v1"`,
  `DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)`.
- `class SplitError(ValueError)`.
- `parse_sample_id(sid) -> (pose_index, camera_name)` (raises `ValueError`).
- `manifest_sha256(dataset_dir) -> str`.
- `make_split(manifest, manifest_sha256, seed, fractions=DEFAULT_FRACTIONS) -> dict` — pure;
  `SplitError` on bad fractions / train empty.
- `write_split(split, path, overwrite=False)` — `FileExistsError` if exists and not overwrite;
  creates parent dir; uses 001 `write_json`.
- `load_split(path) -> dict` — version + keys + disjointness; `SplitError`.
- `check_split_for(split, manifest_sha256, pose_indices) -> list[str]` — hash and coverage problems.

### `strike_a_pose/data/dataset.py`

- `class DatasetError(ValueError)`.
- `check_manifest(dataset_dir) -> (manifest, entries, problems)` → used by the constructor;
  constructor raises `DatasetError("invalid dataset <dir>:\n  - ...")` listing problems.
- `PoseImageDataset(dataset_dir, split=None, split_file=None, transform=None, verify=False)`
  per contract; `__getitem__` reads PNGs with PIL (`convert("L")`), labels JSON, runs
  per-item checks, builds the dict (float32 tensors via `torch.as_tensor(np.asarray(.., float32))`).

### `strike_a_pose/data/__init__.py`

Re-exports `PoseImageDataset`, `DatasetError`, split + transform functions.

### CLI (`datagen/cli.py`)

`split` subparser per contract; imports `strike_a_pose.data.split` (torch-free) lazily in the
`split` branch. Flow: `validate_dataset` (exit 1 on problems) → `make_split` (exit 2 on
`SplitError`) → `write_split` (exit 4 on `FileExistsError`) → print summary.

### 001 touch (minimal)

`datagen/io.py`: rename `_check_labels` → `check_labels` (public, reused by the loader);
one call site. No behaviour change.

### Dependency

`uv add 'torch>=2.5'` (PyPI; R1). CI workflow unchanged (`uv sync --locked` already).
Measure install time on CI.

## Project Structure

```text
src/strike_a_pose/data/__init__.py      # new
src/strike_a_pose/data/transforms.py    # new
src/strike_a_pose/data/split.py         # new
src/strike_a_pose/data/dataset.py       # new
src/strike_a_pose/datagen/cli.py        # + split subcommand
src/strike_a_pose/datagen/io.py         # _check_labels → check_labels
splits/smpl_fixture.json                # new, committed fixture split
tests/test_data_transforms.py           # new
tests/test_data_split.py                # new
tests/test_data_dataset.py              # new
tests/test_cli.py                       # + split CLI cases
pyproject.toml, uv.lock                 # + torch
README.md, specs/roadmap.md             # status, usage, changelog
```

## Complexity Tracking

None. (New dependency `torch` justified above; PyPI vs CPU index trade-off in research R1.)
