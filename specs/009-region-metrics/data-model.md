# Data Model: 009 Per-Region Image Metrics

Normative contracts: [`../003-baselines-and-metrics/contracts/eval-run.md`](../003-baselines-and-metrics/contracts/eval-run.md)
(metrics API, scoring seam, `metrics.json` `metrics_version` 1) and
[`../004-autoencoder-mvp/contracts/train-run.md`](../004-autoencoder-mvp/contracts/train-run.md)
(`metrics.json` `train_version` 2).

## Entities

**Region** (`strike_a_pose.eval.metrics.REGIONS`, code constant, no config key):

| Name | Joints (SMPL-24 index) | `half_size_px` h | Square side |
|---|---|---|---|
| `hands` | `left_hand` 22, `right_hand` 23 | 10 | 21 px |
| `feet` | `left_ankle` 7, `right_ankle` 8, `left_foot` 10, `right_foot` 11 | 8 | 17 px |
| `head` | `head` 15 | 14 | 29 px |

**Region mask** (`region_mask`, per sample): bool `[H, W]`; pixel (r, c) is in the region
iff some region joint is visible and `|r − ⌊v⌋| ≤ h` and `|c − ⌊u⌋| ≤ h`; empty when no
region joint is visible.

**Region labels** (the optional `labels` group of `score_images`): `(keypoints [n, J, 2]
px (u right, v down), visible [n, J] bool, (H, W))` of the scored rows. 003 holds the
keypoints as float64 (`load_arrays` `kp2d`), 004 as float32 (`load_split` `kp2d`, kept
with `visible` whenever the masks are kept).

**Region record** (`metrics.json`, per split with samples, image model and region):

| Key | Value |
|---|---|
| `image_mse` | float64 mean over the split's samples with a non-empty region of the per-sample MSE over the region's pixels (body and background); null when `samples` is 0 |
| `samples` | number of those samples |

**Region definitions** (`metrics.json` top level, `region_definitions()`):
`{"<region>": {"joints": [SMPL-24 joint names], "half_size_px": h}}` for the three
regions, in both the 003 and the 004 file, also when no split has samples.

**`units.regions`** (both files): "per-pixel mean squared error (images in [0, 1]) over
every pixel of the region, body and background, per image; mean over the images whose
region is non-empty (samples); a region (top-level regions) is the union of the squares
of side 2*half_size_px+1 px centred on the pixels of its visible joints, clipped to the
image".

## `metrics.json` (additions only)

```json
{
  "regions": {
    "feet": {"half_size_px": 8, "joints": ["left_ankle", "right_ankle", "left_foot", "right_foot"]},
    "hands": {"half_size_px": 10, "joints": ["left_hand", "right_hand"]},
    "head": {"half_size_px": 14, "joints": ["head"]}
  },
  "units": {"…": "…", "regions": "…"},
  "splits": {
    "test": {"samples": 1600, "poses": 200, "image": {
      "mean_image": {"image_mse": 0.0, "psnr_db": 0.0, "silhouette_iou": 0.0,
                     "regions": {"feet": {"image_mse": 0.0, "samples": 1600},
                                 "hands": {"image_mse": 0.0, "samples": 1600},
                                 "head": {"image_mse": 0.0, "samples": 1600}}}
    }}
  }
}
```

Unchanged: every existing key and value, `metrics_version` 1, `train_version` 2, pose
entries (003), empty splits `{"samples": 0, "poses": 0}`, the writer (6 significant
digits, NaN → null, sorted keys).
