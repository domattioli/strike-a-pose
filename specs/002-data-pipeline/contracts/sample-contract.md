# Contract: Data Pipeline v1 (sample dict + split file)

Consumed by slices 003 (metrics, baselines) and 004 (AE train/eval). Input
datasets follow [001 layout contract v1](../../001-dataset-generation/contracts/dataset-layout.md).
Breaking changes to the sample dict bump this contract's version; breaking
changes to the split file bump `split_version` (now 2: slice
[007](../../007-split-content-hash/spec.md), 2026-10-20; the sample dict is unchanged).

## Python API (`strike_a_pose.data`)

```python
PoseImageDataset(
    dataset_dir: str | Path,
    split: str | None = None,          # "train" | "val" | "test"; needs split_file
    split_file: str | Path | None = None,
    transform: Callable[[dict], dict] | None = None,  # applied to each item
    verify: bool = False,              # True: run the full 001 validator on load
)
```

Attributes: `dataset_dir`, `manifest` (dict), `manifest_sha256` (str, SHA-256 of
the raw `manifest.json` bytes), `manifest_content_sha256` (str, the manifest content
hash, see *Split file*; slice 007), `sample_ids` (list[str], manifest order filtered
to the split), `cameras` (list[str], ring order), `image_size` (`(H, W)`), `split`
(str | None).

Errors: `DatasetError` (subclass of `ValueError`) on any construction or
per-item check failure; its message lists every problem found (construction) or
names the sample (item). `ValueError` for bad arguments (unknown split name,
`split` without `split_file`).

Warning: `SplitCommitWarning` (subclass of `UserWarning`, `warnings.warn`) at
construction when a `split_version` 2 file matches the dataset's manifest content
but records other generating code (`manifest_code` ≠ the manifest's `code`); the
dataset loads. Make it an error with `warnings.simplefilter("error",
SplitCommitWarning)`.

Hash helpers (`strike_a_pose.data`): `manifest_sha256(dataset_dir)` (raw bytes) and
`manifest_content_sha256(manifest: dict)` (content hash of a parsed manifest).

## Sample dict (one item; `H, W` = manifest `config.image`, `J` = 24)

| Key | Type / dtype | Shape | Units / meaning |
|---|---|---|---|
| `image` | tensor float32 | `[1, H, W]` | grey level `uint8 / 255`, in `[0, 1]`; 1 channel (`PIL convert("L")`, exact since renders have R = G = B) |
| `mask` | tensor bool | `[1, H, W]` | True = body (`mask_png > 127`) |
| `keypoints_2d` | tensor float32 | `[J, 2]` | pixels, (u right, v down), image origin top-left |
| `visibility` | tensor bool | `[J]` | in front of camera and inside image (001) |
| `joints_3d_cam` | tensor float32 | `[J, 3]` | metres, camera frame (OpenCV: x right, y down, z forward) |
| `joints_3d_world` | tensor float32 | `[J, 3]` | metres, world frame (+Y up) |
| `pose` | tensor float32 | `[72]` | SMPL axis-angle, radians, root zeroed, hands zeroed |
| `betas` | tensor float32 | `[num_betas]` | zeros (mean shape) |
| `camera` | dict | | |
| `camera.K` | tensor float32 | `[3, 3]` | intrinsics, pixels |
| `camera.R` | tensor float32 | `[3, 3]` | world → camera rotation |
| `camera.t` | tensor float32 | `[3]` | world → camera translation, metres |
| `camera.yaw_deg` | tensor float32 | `[]` | ring yaw, degrees |
| `sample_id` | str | | `f"{pose_index:05d}_{camera_name}"` |
| `pose_index` | int | | pose draw index (split unit) |
| `camera_name` | str | | e.g. `Front_0` |
| `camera_index` | int | | position of `camera_name` in the manifest camera ring |

Joint order: `strike_a_pose.datagen.skeleton.SMPL24_JOINT_NAMES` (root = 0, pelvis).
Invariants: `joints_3d_cam = joints_3d_world @ R.T + t` (1e-4 m); projecting
`joints_3d_cam` with `K` gives `keypoints_2d` (1 px) — the 001 SC-003 invariant.

`torch.utils.data.default_collate` batches items: tensors gain a leading batch
dim, `camera` stays a dict of batched tensors, strings become lists, ints become
int64 tensors.

## Normalization helpers (`strike_a_pose.data.transforms`)

All keypoint helpers work on NumPy arrays and torch tensors with any leading
batch dims (`[..., J, D]`).

| Function | Maps | Inverse |
|---|---|---|
| `image_to_tensor(img_u8 [H, W])` | → float32 `[1, H, W]`, `/ 255` | `tensor_to_image(t)` → uint8 `[H, W]` (clamp to [0,1], `round(x·255)`) — exact round trip for uint8 input |
| `root_relative(j, root=0)` | `j − j[..., root, :]` (root joint → 0) | `add_root(j_rel, root_pos)` with `root_pos [..., 3]` |
| `normalize_keypoints_2d(kp, width, height)` | `(2u/W − 1, 2v/H − 1)`: image edges → ±1, centre → 0 | `denormalize_keypoints_2d(kp_n, width, height)` |

`normalize_keypoints_2d` scales x and y separately (W ≠ H), so distances in
normalized space are anisotropic; compute pixel-space errors after denormalizing.

## Split file (`split_version` 2)

JSON written with the 001 serializer (sorted keys, indent 2, trailing newline).

```json
{
  "algorithm": "pose-permutation-v1",
  "counts": {"test": {"poses": 1, "samples": 8}, "train": {...}, "val": {...}},
  "dataset_name": "smpl_fixture",
  "fractions": {"test": 0.5, "train": 0.5, "val": 0.0},
  "manifest_code": {"commit": "ab95629e6155d38a176d6a54492cc65a1be3a1fa", "dirty": true},
  "manifest_content_sha256": "<sha256 of the manifest without its code key, see below>",
  "poses": {"test": [0], "train": [1], "val": []},
  "seed": 0,
  "split_version": 2,
  "unit": "pose_index"
}
```

- Manifest content hash: `sha256(dumps(m′))`, `m′` = the parsed manifest without its
  top-level `code` key, `dumps` = the 001 serializer (`strike_a_pose.datagen.io.dumps`:
  floats rounded to 6 decimals, sorted keys, indent 2, trailing newline). For a
  manifest written by 001 this is the SHA-256 of the `manifest.json` 001 would write
  without `code`; it does not depend on on-disk formatting, and float differences
  below 1e-6 do not show. A dataset regenerated with the same config + seed at another
  commit has the same content hash. Committed fixture:
  `7843ff453af51734c6d87ce2c473bb623d1368ae43beaa5bb26718e7daa9f769`.
- `manifest_code`: the manifest's `code` object (`{"commit": <sha | "unknown">,
  "dirty": bool | null}`), copied verbatim; informational. Rewriting the split with
  `--overwrite` refreshes it.
- Unit = pose: every camera view of a pose is in the same split.
- Algorithm `pose-permutation-v1`: `P` = sorted distinct pose indices (N);
  `perm = numpy.random.default_rng(seed).permutation(P)`;
  `n_s = round(f_s·N)` (Python `round`, half to even), raised to 1 if `f_s > 0`,
  for `s` in (test, val);
  `test = perm[:n_test]`, `val = perm[n_test : n_test+n_val]`, `train` = rest
  (must be ≥ 1 pose); each list stored sorted.
- Fractions: three non-negative numbers summing to 1 (±1e-6), `train > 0`.
- Loading (`load_split`) checks: the file has a `split_version` of 1 or 2 (else
  `SplitError`, unsupported), exactly that version's keys, `unit`, pose lists disjoint
  (v2: `manifest_code` is an object). Against a dataset (`PoseImageDataset`):
  `manifest_content_sha256` equal to the dataset's (else `DatasetError`), union of the
  pose lists = the dataset's poses (else `DatasetError`); then, if `manifest_code` ≠
  the dataset manifest's `code` (exact JSON equality: `commit` or `dirty` differ), a
  `SplitCommitWarning` naming both, and the split is used.

### Legacy `split_version` 1 (read only)

Files written before slice 007 (2026-10-20) have `"split_version": 1` and
`"manifest_sha256"` (SHA-256 of the raw `manifest.json` bytes) in place of
`manifest_content_sha256` and `manifest_code`; otherwise the same keys. They still
load, with their original strict check: the dataset's raw manifest hash must equal
`manifest_sha256`, so any byte difference, including only the generating `code`,
refuses (no warning path). The refusal says how to migrate: rewrite the file with the
CLI below and `--overwrite`, which writes `split_version` 2 with the same pose lists for
the same dataset, seed and fractions. The committed `splits/smpl_proc_v1.json` is
still v1 (valid only for `smpl_proc_v1` generated at commit `406b68f`) until slice 007
follow-up F1 rewrites it. v1 files are no longer written.

## CLI

```text
strike-a-pose-data split DATASET_DIR --seed INT --out PATH
                         [--fractions TRAIN VAL TEST] [--overwrite]
```

Runs the full 001 validator first and writes `split_version` 2. Exit codes: 0
written; 1 dataset invalid (problems listed on stderr); 2 bad fractions / too few
poses; 4 `--out` exists and `--overwrite` not given (frozen split). `--overwrite`
replaces an existing file of either version. Committed split files live in
`splits/` (datasets themselves are git-ignored).
