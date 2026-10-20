# Contract: Data Pipeline v1 (sample dict + split file)

Consumed by slices 003 (metrics, baselines) and 004 (AE train/eval). Input
datasets follow [001 layout contract v1](../../001-dataset-generation/contracts/dataset-layout.md).
Breaking changes to the sample dict bump this contract's version; breaking
changes to the split file bump `split_version`.

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

Attributes: `dataset_dir`, `manifest` (dict), `manifest_sha256` (str),
`sample_ids` (list[str], manifest order filtered to the split), `cameras`
(list[str], ring order), `image_size` (`(H, W)`), `split` (str | None).

Errors: `DatasetError` (subclass of `ValueError`) on any construction or
per-item check failure; its message lists every problem found (construction) or
names the sample (item). `ValueError` for bad arguments (unknown split name,
`split` without `split_file`).

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

## Split file (`split_version` 1)

JSON written with the 001 serializer (sorted keys, indent 2, trailing newline).

```json
{
  "algorithm": "pose-permutation-v1",
  "counts": {"test": {"poses": 1, "samples": 8}, "train": {...}, "val": {...}},
  "dataset_name": "smpl_fixture",
  "fractions": {"test": 0.5, "train": 0.5, "val": 0.0},
  "manifest_sha256": "<sha256 of manifest.json bytes>",
  "poses": {"test": [1], "train": [0], "val": []},
  "seed": 0,
  "split_version": 1,
  "unit": "pose_index"
}
```

- Unit = pose: every camera view of a pose is in the same split.
- Algorithm `pose-permutation-v1`: `P` = sorted distinct pose indices (N);
  `perm = numpy.random.default_rng(seed).permutation(P)`;
  `n_s = round(f_s·N)` (Python `round`, half to even), raised to 1 if `f_s > 0`,
  for `s` in (test, val);
  `test = perm[:n_test]`, `val = perm[n_test : n_test+n_val]`, `train` = rest
  (must be ≥ 1 pose); each list stored sorted.
- Fractions: three non-negative numbers summing to 1 (±1e-6), `train > 0`.
- Loading (`load_split`) checks: `split_version == 1`, keys, pose lists disjoint,
  and, against a dataset, `manifest_sha256` equal and union = the dataset's poses.

## CLI

```text
strike-a-pose-data split DATASET_DIR --seed INT --out PATH
                         [--fractions TRAIN VAL TEST] [--overwrite]
```

Runs the full 001 validator first. Exit codes: 0 written; 1 dataset invalid
(problems listed on stderr); 2 bad fractions / too few poses; 4 `--out` exists
and `--overwrite` not given (frozen split). Committed split files live in
`splits/` (datasets themselves are git-ignored).
