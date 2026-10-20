# Research: 007 Split Content Hash

## R1 — Canonical serialization and idempotence

The content hash re-serializes the *parsed* manifest with the 001 serializer
`datagen.io.dumps` (`_round` → floats rounded to 6 decimals, `-0.0` → `0.0`;
`json.dumps(sort_keys=True, indent=2)` + `"\n"`, ASCII-escaped). For a manifest that
001 wrote, this reproduces the file bytes exactly:

- floats were already rounded and written with `repr` (shortest round-trip form), so
  `json.loads` returns the same float and `round(x, 6)` of an already-rounded float is
  the float itself; ints, bools, `null` and strings round-trip unchanged;
- keys are re-sorted the same way; indentation and the trailing newline are
  regenerated identically.

Checked on the committed fixture (`dumps(load_manifest(fixture)) == manifest.json`
bytes) and on a 16,000-sample manifest built from it (R5). Hence the content hash is
the SHA-256 of the `manifest.json` that 001 would write without the `code` key: one
definition, checkable with a few lines of Python, and independent of on-disk
formatting for manifests edited by other tools. The 6-decimal rounding means float
changes below 1e-6 are invisible to the hash; 001 never writes finer floats.
Alternatives rejected: a compact canonical JSON (a second serializer to keep in sync,
and no longer "the file 001 would write"); hashing the raw bytes with the `code` block
cut out textually (fragile, depends on formatting).

Fixture content hash: `7843ff453af51734c6d87ce2c473bb623d1368ae43beaa5bb26718e7daa9f769`
(real `dumps`; equals the spec-time NumPy-free computation). Raw hash unchanged:
`06787bb039f7d64a69248d7b58147fea3f8708fc28508c89a789f1dd6b52574e`.

## R2 — Warning mechanics (Python 3.11 + 3.13)

`warnings.warn(message, SplitCommitWarning)` with the default `stacklevel=1`:

- `SplitCommitWarning(UserWarning)` is shown by Python's default filters (the
  `"default"` action: first occurrence per location); unlike `DeprecationWarning` it is
  not hidden outside `__main__`.
- The location is the single `warnings.warn` line in `data/dataset.py`, and the
  registry key is (message, category, line), so the identical message of the three
  split datasets a run builds is printed once per process; a different split or
  dataset prints again. A caller-attributed `stacklevel=2` would print once per call
  site instead (the 004 trainer builds its datasets on three lines).
- Users who want the strict behaviour back can turn the category into an error:
  `warnings.simplefilter("error", SplitCommitWarning)`.
- Only 3.11 APIs: no `skip_file_prefixes` (3.12+). Tests use `pytest.warns` and
  `warnings.catch_warnings()` + `simplefilter("error", SplitCommitWarning)`; changing
  the filters invalidates the once-per-location registries (pytest also wraps every
  test in `catch_warnings`), so test order does not matter.

## R3 — split_version 1 stays readable, strict

A v1 file records only the raw hash of the manifest it was made from. The content
hash cannot be derived from it (the manifest is not in git, and the generating `code`
is not in the v1 file), so a v1 file cannot tell a regenerated-identical dataset from a
different one. The approved proposal asks to refuse content mismatches; therefore v1
keeps its exact 002 behaviour (refuse any byte difference) and gains only a migration
hint in the refusal. Reading v1 costs one key set and one branch; it keeps
`splits/smpl_proc_v1.json` usable for the dataset it was made for until F1. Writing v1
is dropped (v2 is strictly more useful; one writer).

## R4 — Resume and latent cache keep the raw hash (surrogate ruling Q1)

Both reuse state computed from *pixels* (model weights mid-training; encoder latents).
The manifest pins labels (`labels_sha256`) but not images: PNG hashes live in
`checksums.json`, and 001 (spec US2) tolerates non-bit-exact renders across machines.
So equal manifest content at another commit does not prove equal images: after a
renderer change, a content-keyed resume would mix two image sets in one training run,
and a content-keyed cache would return latents of the old images, both silently. The
raw hash includes the generating commit and is the conservative key. The benefit of
switching would be small: `data/` and `runs/` are created and lost together in this
environment, so a resume or cache hit across a regeneration does not happen in
practice. Zero code change in 004 / 006; their contracts say so explicitly.

## R5 — Cost

Content hash of a 16,000-sample manifest (4.5 MB; the fixture manifest with 2,000 × 8
synthetic sample entries): `json.loads` 0.016 s, `dumps` + SHA-256 0.086 s (4 vCPU
container, Python 3.13). A run builds three split datasets: < 0.3 s in total.

## R6 — What else could differ between regenerations

`generate.py` builds the manifest from the config (`cfg.to_dict()`), seed,
`git_commit()`, the SMPL file name + SHA-256, the pose-source description, skeleton,
cameras, counts and per-sample paths + `labels_sha256`. Only `git_commit()` depends on
the commit by value; timestamps, host and argv go to the dataset's `run_info.json`.
`--output-root` changes the output directory, not `config.output_root` in the manifest.
001's `test_reproducible_us2` already shows byte-identical manifests for two
generations at one commit. No datagen source, default config or lockfile change since
`406b68f` (`git diff 406b68f..HEAD -- src/strike_a_pose/datagen configs/default.yaml
uv.lock` is empty), so F1 is expected to reproduce the `406b68f` content.
