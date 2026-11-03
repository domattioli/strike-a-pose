# Data Model: 007 Split Content Hash

Normative contract: [`../002-data-pipeline/contracts/sample-contract.md`](../002-data-pipeline/contracts/sample-contract.md)
(split file section). Input: 001 dataset layout v1 (manifest unchanged).

## Entities

**Manifest content hash** (str, 64 hex): `sha256(dumps(manifest without "code"))`, with
`dumps` the 001 serializer. Exposed as `PoseImageDataset.manifest_content_sha256` and
`strike_a_pose.data.manifest_content_sha256(manifest)`. Fixture:
`7843ff453af51734c6d87ce2c473bb623d1368ae43beaa5bb26718e7daa9f769`.

**Raw manifest hash** (unchanged): `sha256(manifest.json bytes)`; includes `code`.
`PoseImageDataset.manifest_sha256`; keys the 004 resume check and the 006 latent cache.

**Split file v2** (JSON, written by `strike-a-pose-data split`):

| Key | Value |
|---|---|
| `split_version` | `2` |
| `algorithm` | `"pose-permutation-v1"` (unchanged) |
| `unit` | `"pose_index"` |
| `dataset_name` | manifest `dataset_name` |
| `manifest_content_sha256` | content hash of the manifest split |
| `manifest_code` | that manifest's `code` object, verbatim (`{"commit", "dirty"}`); informational |
| `seed`, `fractions`, `poses`, `counts` | as in v1 |

**Split file v1** (legacy, read only): the 002 format with `manifest_sha256` (raw) in
place of `manifest_content_sha256` + `manifest_code`.

**`SplitCommitWarning`** (`UserWarning` subclass): content equal, `manifest_code` ≠ the
dataset manifest's `code`.

## Loading a split against a dataset (`PoseImageDataset(..., split_file=...)`)

| Split version | Check | Outcome |
|---|---|---|
| any | file readable, `split_version` ∈ {1, 2}, key set of that version, `unit`, pose lists disjoint (v2: `manifest_code` an object) | `DatasetError` (from `SplitError`) otherwise |
| 2 | `manifest_content_sha256` = dataset content hash | `DatasetError` otherwise |
| 1 | `manifest_sha256` = dataset raw hash | `DatasetError` otherwise (with the v2 migration hint) |
| any | union of split poses = dataset poses | `DatasetError` otherwise |
| 2 | `manifest_code` = dataset manifest `code` (exact JSON equality) | `SplitCommitWarning` otherwise, then load |

## Run records

`run_info.json` of 003 / 004 / 006: `dataset` = {`dir`, `manifest_sha256`,
`manifest_content_sha256`}. `metrics.json` unchanged.
