# Implementation Plan: Split Content Hash (split file v2)

**Branch**: `claude/charming-wozniak-5ouldf` (slice `007-split-content-hash`) | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)
**Input**: `specs/007-split-content-hash/spec.md` (approved at surrogate scope, surrogate rulings 2026-10-20; operator sign-off pending)

## Summary

The frozen split stops pinning the raw `manifest.json` bytes (which include the
generating `code` commit) and pins the manifest *content* instead: SHA-256 of the 001
serializer's output for the manifest without its `code` key. New split files are
`split_version` 2 with `manifest_content_sha256` and an informational `manifest_code`;
a dataset with the same content but other generating code loads with a
`SplitCommitWarning`; other content is refused as before. `split_version` 1 files stay
readable with their strict raw-bytes check (the committed `splits/smpl_proc_v1.json`
until follow-up F1). The fixture split is rewritten as v2. 003 / 004 / 006 `run_info.json`
gain the content hash next to the raw one; the 004 resume check and the 006 latent
cache keep the raw hash (no code change there).

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.11 + 3.13); 3.11-compatible APIs only.
**Primary Dependencies**: none new (stdlib `hashlib`, `json`, `warnings`; existing 001
`datagen.io.dumps`). `pyproject.toml` / `uv.lock` untouched.
**Storage**: split files JSON in `splits/` (committed); one new committed test asset
`tests/fixtures/smpl_fixture_split_v1.json` (578 bytes, the pre-007 fixture split).
**Testing**: pytest — content-hash units on synthetic manifests and the committed
fixture, split v1/v2 file I/O, loader warning / refusal on fixture copies, one
end-to-end regeneration with the synthetic body model (`small_cfg`, 16 tiny samples),
CLI rewrite of v1, run_info fields of the three fixture CLIs, one 004 resume refusal.
**Performance Goals**: content hash of a 16,000-sample manifest (4.5 MB) ≈ 0.09 s
(research R5), i.e. < 0.3 s per run for three split datasets; CI leg stays < 5 min.
**Constraints**: no change to 001 generation, the manifest format, the split algorithm
(`pose-permutation-v1`) or the sample dict; `splits/smpl_proc_v1.json` and
`docs/results/` byte-unchanged; no numerical result changes (SC-004).

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | All logic in `strike_a_pose.data.split` / `dataset`; the CLI stays a thin wrapper. |
| II CLI + config | PASS | Split CLI unchanged in shape (`--seed`, `--out`, `--fractions`, `--overwrite`); no new hidden paths. |
| III Reproducibility | PASS | The data version is pinned by content (independent of the generating commit) and recorded in every run's `run_info.json`; the generating commit stays recorded (`manifest_code`) and a mismatch is reported; pixel-dependent reuse (resume, latent cache) keeps the stricter raw key; split bytes deterministic. |
| IV Test-first | PASS | FR-010 tests written before the code (tasks T003–T006 precede T007–T010); they fail on the current code. |
| V Honest evaluation | PASS | Same held-out poses: pose assignment unchanged, checked for the fixture (tests) and for `smpl_proc_v1` (F1 procedure); no number changes (SC-004 diff). |
| VI Simplicity | PASS | Two functions, one warning class, one version branch in the loader; no new dependency, no change to 004 / 006 logic beyond one additive `run_info` key each. |

## Design

### `strike_a_pose/data/split.py`

- Constants: `SPLIT_VERSION = 2` (written), `LEGACY_SPLIT_VERSION = 1` (read only),
  `SPLIT_KEYS: dict[int, frozenset[str]]` = common keys (`split_version`, `algorithm`,
  `unit`, `dataset_name`, `seed`, `fractions`, `poses`, `counts`) + `manifest_sha256`
  (v1) or + `manifest_content_sha256`, `manifest_code` (v2).
- `class SplitCommitWarning(UserWarning)`.
- `manifest_sha256(dataset_dir)` unchanged (raw bytes).
- `manifest_content_sha256(manifest: dict) -> str` =
  `sha256(dumps({k: v for k, v in manifest.items() if k != "code"}))`.
- `make_split(manifest, seed, fractions=DEFAULT_FRACTIONS) -> dict` — v2; the hash
  parameter is dropped; `manifest_content_sha256(manifest)` and `dict(manifest["code"])`
  are recorded. Assignment code unchanged.
- `write_split` unchanged.
- `load_split(path)`: readable JSON object with a `split_version` key, else "not a split
  file"; `split_version` an int (not bool) in `SPLIT_KEYS`, else "unsupported"; key set =
  `SPLIT_KEYS[version]`, else "keys mismatch"; `unit`; v2 `manifest_code` must be an
  object; pose-list checks unchanged.
- `check_split_for(split, raw_sha256, content_sha256, pose_indices) -> list[str]` —
  v1: raw hash equality (message keeps "split was made for manifest …" and adds the
  migration hint); v2: content hash equality ("split was made for manifest content …");
  coverage unchanged.
- `check_split_code(split, manifest_code) -> str | None` — v2 only: the warning text when
  `split["manifest_code"] != manifest_code` (both objects as compact JSON, "manifest
  content is identical", "rewriting the split with `strike-a-pose-data split …
  --overwrite` refreshes manifest_code"); `None` for v1 or equal code.

### `strike_a_pose/data/dataset.py`

`PoseImageDataset.__init__`: `self.manifest_content_sha256 = manifest_content_sha256(m)`
next to `self.manifest_sha256`; split check passes both; after the checks pass,
`check_split_code(sp, m["code"])` → `warnings.warn(f"split file {split_file}, dataset
{dir}: {note}", SplitCommitWarning, stacklevel=1)` (explicit, as ruff B028 asks; it is
the default), so the warning is attributed to one line of the data package and the
default filter prints a given split + dataset message once per process (research R2).

### `strike_a_pose/data/__init__.py`

Also export `SplitCommitWarning`, `manifest_content_sha256`.

### CLI (`datagen/cli.py`)

`make_split(load_manifest(dir), seed, fractions)`; `manifest_sha256` import dropped. Exit
codes and output unchanged.

### Run records (additive)

`eval/baselines.py`, `train/trainer.py`, `eval/probe.py`: `run_info.json` `dataset` gains
`"manifest_content_sha256": <train_ds>.manifest_content_sha256`. Not added to the probe's
`hashes` dict (that dict is the latent-cache key and `metrics.json` source, both
unchanged). `metrics.json` unchanged everywhere.

### Fixture split migration

`git show HEAD:splits/smpl_fixture.json > tests/fixtures/smpl_fixture_split_v1.json`
(verbatim v1 bytes), then `strike-a-pose-data split tests/fixtures/smpl_fixture --seed 0
--fractions 0.5 0 0.5 --out splits/smpl_fixture.json --overwrite`; pose lists compared
with the v1 file.

### Test changes (callers of `make_split` and hash assertions)

`tests/test_data_split.py` (synthetic manifests gain a `code`), `test_data_dataset.py`,
`test_cli.py`, `test_train.py`, `test_probe.py`, `test_eval_cli.py`: new `make_split`
signature; `manifest_sha256` assertions on split files become content-hash assertions;
`run_info` assertions gain the content hash.

## Project Structure

### Documentation (this feature)

```text
specs/007-split-content-hash/
├── spec.md
├── plan.md            # this file
├── research.md
├── data-model.md
└── tasks.md
```

Normative contract text stays in the slices that own it (updated in place):
`specs/002-data-pipeline/contracts/sample-contract.md` (split file v2 + v1 legacy, API),
`specs/003-baselines-and-metrics/contracts/eval-run.md`,
`specs/004-autoencoder-mvp/contracts/train-run.md`,
`specs/006-latent-probe/contracts/probe-run.md` (run_info field; raw-hash keys).

### Source Code (repository root)

```text
src/strike_a_pose/data/split.py          # v2 writer, v1/v2 reader, content hash, warning
src/strike_a_pose/data/dataset.py        # content-hash attribute, warning emission
src/strike_a_pose/data/__init__.py       # exports
src/strike_a_pose/datagen/cli.py         # make_split call
src/strike_a_pose/eval/baselines.py      # run_info + content hash
src/strike_a_pose/train/trainer.py       # run_info + content hash
src/strike_a_pose/eval/probe.py          # run_info + content hash
splits/smpl_fixture.json                 # rewritten as v2
tests/fixtures/smpl_fixture_split_v1.json  # new: verbatim pre-007 v1 fixture split
tests/test_data_split.py, tests/test_data_dataset.py, tests/test_cli.py,
tests/test_train.py, tests/test_probe.py, tests/test_eval_cli.py
README.md, specs/roadmap.md, docs/HANDOFF.md, specs/002-data-pipeline/{spec,data-model}.md
```

**Structure Decision**: existing single package; no new module.

## Complexity Tracking

None.
