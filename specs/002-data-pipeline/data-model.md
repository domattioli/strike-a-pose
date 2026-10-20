# Data Model: 002 Data Pipeline

Normative contract: [`contracts/sample-contract.md`](contracts/sample-contract.md).
Input: 001 dataset layout v1 ([`../001-dataset-generation/data-model.md`](../001-dataset-generation/data-model.md)).

## Entities

**DatasetIndex** (internal, built on construction from `manifest.json`):
`entries: list[(sample_id, pose_index, camera_name, camera_index, image, mask, labels)]`
in manifest order (pose-major, camera ring order); `cameras: list[str]`;
`image_size: (H, W)` from `manifest.config.image`.

**Sample** (item dict): image `[1,H,W]` float32, mask `[1,H,W]` bool, labels as
tensors in source units, `camera` sub-dict, ids. See contract.

**Split** (JSON, `split_version` 2 since slice 007, 2026-10-20): `dataset_name`,
`manifest_content_sha256` (SHA-256 of the manifest without `code`, 001 serializer),
`manifest_code` (the manifest's `code`, informational), `seed`, `unit` =
`"pose_index"`, `algorithm` = `"pose-permutation-v1"`, `fractions` {train, val, test},
`poses` {train, val, test: sorted int lists}, `counts` {split: {poses, samples}}.
Legacy `split_version` 1 (read only): `manifest_sha256` (raw `manifest.json` bytes) in
place of `manifest_content_sha256` + `manifest_code`.

## Construction checks (`DatasetError` on failure)

| Check | Source |
|---|---|
| `manifest.json` readable JSON | file |
| manifest key set = 001 `MANIFEST_KEYS`; `schema_version == 1` | 001 io |
| `complete is True` | 001 state machine |
| `counts.samples == len(samples) == counts.poses × counts.cameras` | manifest |
| `skeleton.name == "smpl24"`, 24 joints | 001 skeleton |
| sample ids unique, form `<int>_<camera in ring>` | 001 data model |
| image / mask / labels files exist | manifest paths |
| (verify=True) 001 `validate_dataset` returns no problems | 001 io |
| (split) split file valid; v2: `manifest_content_sha256` equal (other `manifest_code` → `SplitCommitWarning`, not an error); v1: raw `manifest_sha256` equal; union of split poses = dataset poses | split (amended by slice 007) |

## Per-item checks (`DatasetError` naming the sample)

001 `check_labels` (label key set, array shapes — pose 72, joints 24×3,
keypoints 24×2, visibility 24 — and the SC-003 camera/projection invariant);
`sample_id`, `pose_index`, `camera_name` match the manifest entry; image and
mask are `H × W`. (`check_labels` is 001's existing `_check_labels`, made
public; no behaviour change.)

## State

Stateless reader: no caching; safe with `DataLoader(num_workers > 0)` (the
dataset object holds only paths and small lists).
