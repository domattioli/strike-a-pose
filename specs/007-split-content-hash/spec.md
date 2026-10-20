# Feature Specification: Split Content Hash (split file v2)

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `007-split-content-hash`)
**Created**: 2026-10-20
**Status**: Implemented 2026-10-20 (verification: [tasks.md](tasks.md) T016; follow-up F1 open, needs SMPL).
Approved at surrogate scope (surrogate rulings 2026-10-20; operator sign-off pending)
**Input**: Operator instruction for this session: "Implement approved roadmap question
#14 (split-hash contract change) as a small spec-kit slice. Update 002's contract and
its tests." Roadmap open question #14: "Split-hash contract change. Proposal: hash the
manifest *without* `code.commit` / `code.dirty`, store the commit informationally, and
warn instead of refusing on a commit mismatch (today a regenerated-but-identical
dataset at another commit is rejected by the frozen split)." Disposition, operator
"approve all" ([`../roadmap.md`](../roadmap.md) changelog; that entry is labelled
2026-10-21 but was written on 2026-10-20 UTC): "#14 approved — implement the split-hash
proposal (hash manifest without `code.commit`/`code.dirty`, commit informational, warn on
commit mismatch, refuse on content mismatch)".

Settled context (operator decision above; code facts checked in this session):

- A slice-001 dataset is byte-reproducible for the same config + seed except the
  manifest's `code` object `{"commit": <sha | "unknown">, "dirty": bool | null}`
  (`datagen.generate.git_commit`). `manifest.json` carries no timestamps or host info
  (001 layout contract; those live in the dataset's own `run_info.json`) and is written
  by the 001 serializer `datagen.io.dumps` (sorted keys, indent 2, floats rounded to 6
  decimals, `-0.0` → `0.0`, trailing newline). Sample entries carry `labels_sha256`, so
  the manifest pins every label file; image and mask PNGs are hashed only in
  `checksums.json`, not in the manifest.
- Today (`split_version` 1, slice 002) the split file records `manifest_sha256` =
  SHA-256 of the raw `manifest.json` bytes, which include `code`; the loader refuses any
  difference, so a dataset regenerated at another commit is rejected.
  `PoseImageDataset.manifest_sha256` (raw) goes into the 003 / 004 / 006 `run_info.json`
  and `metrics.json` (`dataset.manifest_sha256`); 004 `--resume` refuses when it changed
  (`train/trainer.py`, `resume_training`); the 006 latent cache is keyed by the
  checkpoint, raw-manifest and split-file SHA-256s (`eval/probe.py`, `cache_path`).
- Committed split files, both `split_version` 1: `splits/smpl_fixture.json` (dataset =
  the committed fixture `tests/fixtures/smpl_fixture`, manifest `code` =
  `{"commit": "ab95629e…", "dirty": true}`, raw hash `06787bb0…`) and
  `splits/smpl_proc_v1.json` (dataset not in git; raw hash `9000c915…` matches only
  `smpl_proc_v1` generated at `406b68f`).
- Environment: the licensed SMPL model (`assets/smpl/SMPL_NEUTRAL.npz`, private repo
  `domattioli/smpl-assets`) is not obtainable in this session, so `smpl_proc_v1` cannot
  be regenerated and its content hash cannot be computed here. No datagen source,
  default config or lockfile change since `406b68f` (`git diff 406b68f..HEAD --
  src/strike_a_pose/datagen configs/default.yaml uv.lock` is empty), so a regeneration
  is expected to reproduce the `406b68f` content.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reuse a frozen split after regenerating the dataset (Priority: P1)

The researcher regenerates a dataset with the same config + seed at another commit
(for example in a fresh container) and loads it with the committed split file. It
loads, and a warning says the generating commit differs. A dataset whose content
differs is still refused.

**Why this priority**: this is roadmap question #14; today the frozen split rejects a
regenerated-but-identical dataset, so every fresh container needs a new split file.

**Independent Test**: generate a tiny dataset with the synthetic body model (as the
001 tests do), write a v2 split, regenerate with the same config + seed but another
`code`, load it with the split (one `SplitCommitWarning`, same samples per split);
regenerate with another seed (refused).

**Acceptance Scenarios**:

1. **Given** a v2 split file and a dataset whose manifest equals the split's source
   manifest except `code`, **When** the dataset is loaded with the split, **Then** it
   loads with exactly the split's poses and emits a `SplitCommitWarning` naming both
   `code` objects.
2. **Given** a v2 split file and the dataset it was made from (same `code`), **When**
   it is loaded, **Then** no warning is emitted.
3. **Given** a v2 split file and a dataset whose manifest differs in any field other
   than `code`, **When** it is loaded, **Then** construction fails with `DatasetError`
   naming both content hashes.
4. **Given** a 003 / 004 / 006 run, **When** it writes `run_info.json`, **Then**
   `dataset` holds the manifest content hash next to the raw manifest hash.

### User Story 2 - Write v2 split files; migrate the fixture split (Priority: P1)

`strike-a-pose-data split` writes `split_version` 2. The committed fixture split is
rewritten once as v2; which poses are in which split does not change.

**Why this priority**: v2 files are what US1 loads; the fixture split is what CI and
every fixture test use.

**Independent Test**: run the CLI on the fixture with the committed arguments and
compare bytes with the committed v2 file; compare its pose lists with the pre-007 v1
file.

**Acceptance Scenarios**:

1. **Given** a valid dataset and a seed, **When** the split command runs, **Then** it
   writes a v2 file (content hash, manifest `code`, dataset name, seed, fractions,
   algorithm, per-split pose lists and counts), byte-identical for the same manifest +
   seed + fractions.
2. **Given** an existing v1 file at `--out` for the same dataset, seed and fractions,
   **When** the command runs with `--overwrite`, **Then** the new v2 file has the same
   algorithm, seed, fractions, pose lists and counts.
3. **Given** an existing file at `--out`, **When** the command runs without
   `--overwrite`, **Then** it refuses (exit 4) and writes nothing (unchanged).

### User Story 3 - Legacy v1 split files keep working (Priority: P2)

`splits/smpl_proc_v1.json` stays `split_version` 1 until someone with the SMPL model
regenerates `smpl_proc_v1` and rewrites the split (follow-up F1). Until then it keeps
its exact 002 behaviour.

**Why this priority**: the repository must stay usable although SMPL is unavailable
in this session.

**Independent Test**: the pre-007 fixture split (kept verbatim as a test asset) loads
with the fixture; a fixture copy with only `code` changed is refused with the
migration hint; `splits/smpl_proc_v1.json` passes `load_split`.

**Acceptance Scenarios**:

1. **Given** a v1 split and the dataset it was made for (identical manifest bytes),
   **When** it is loaded, **Then** it loads with no warning.
2. **Given** a v1 split and a dataset whose manifest bytes differ in any way (even only
   in `code`), **When** it is loaded, **Then** `DatasetError` says that v1 pins the raw
   manifest bytes and that the split command with `--overwrite` rewrites it as v2.
3. **Given** a split file with a `split_version` other than 1 or 2, **When** it is
   loaded, **Then** `SplitError` ("unsupported").

### Edge Cases

- Same commit, only `dirty` differs → warning (the code may differ by uncommitted
  changes).
- `code.commit` `"unknown"` / `dirty` `null` (no git at generation time): compared as
  plain values; both unknown → no warning (content equality is the check that matters).
- Same values, different on-disk formatting (whitespace, key order) → same content
  hash (a v1 split refuses: the raw bytes differ).
- Float changes below 1e-6 are invisible to the content hash (001 writes 6-decimal
  floats).
- Config paths (`smpl_model_path`, `output_root`, `amass_root`) are content: a dataset
  generated from a config naming another SMPL path is refused even if the SMPL file's
  SHA-256 is the same. (`--output-root` does not enter the manifest.)
- Pixels: equal manifest content does not prove equal images (a renderer change at
  another commit; 001 tolerates non-bit-exact renders across machines). The split is
  pose-level, so its validity is unaffected; the warning flags the other commit. State
  computed from pixels keeps the raw-bytes key (clarification 8).
- A v2 file carrying v1 keys (or the reverse), extra or missing keys, or a non-object
  `manifest_code` → `SplitError`.
- Rewriting a split with another seed or other fractions changes its pose lists; that
  stays allowed with `--overwrite`, as today. The F1 procedure checks the pose lists
  explicitly.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (content hash): `strike_a_pose.data.split` MUST compute the manifest
  content hash = SHA-256 (hex) of `datagen.io.dumps(m′)`, where `m′` is the parsed
  manifest object without its top-level `code` key; every other key (`schema_version`,
  `dataset_name`, `complete`, `seed`, `config`, `inputs`, `skeleton`, `cameras`,
  `counts`, `samples`) is hashed unchanged. Computed from the parsed JSON, it does not
  depend on on-disk formatting. For a manifest written by 001, `dumps` of the parsed
  manifest reproduces the file bytes exactly (idempotent; true for the committed
  fixture), so the content hash equals the SHA-256 of the `manifest.json` 001 would
  write without `code`. Committed fixture:
  `7843ff453af51734c6d87ce2c473bb623d1368ae43beaa5bb26718e7daa9f769` (computed while
  writing this spec with a NumPy-free replica of `dumps`; the implementation must
  reproduce it, not update it).
- **FR-002** (split file v2): new split files MUST be `split_version` 2 with exactly
  the keys `split_version`, `algorithm` (`"pose-permutation-v1"`, unchanged), `unit`
  (`"pose_index"`), `dataset_name`, `manifest_content_sha256` (FR-001), `manifest_code`
  (the manifest's `code` object, verbatim; informational), `seed`, `fractions`, `poses`,
  `counts`, written with `dumps`. v2 has no raw manifest hash. The writer MUST derive
  `manifest_content_sha256` and `manifest_code` from the manifest it splits (no
  caller-supplied hash). Same manifest + seed + fractions → identical bytes; the pose
  assignment is the unchanged `pose-permutation-v1` algorithm.
- **FR-003** (CLI): `strike-a-pose-data split` MUST write v2 only. Its arguments, the
  full 001 validation before writing, its exit codes (0 / 1 / 2 / 4) and the
  frozen-file refusal are unchanged; `--overwrite` replaces an existing file of either
  version.
- **FR-004** (loading v2): against a dataset, a v2 split MUST (a) refuse with
  `DatasetError` when `manifest_content_sha256` differs from the dataset's content
  hash, naming both (first 12 hex digits) and the split's dataset name; (b) refuse when
  its poses do not cover exactly the dataset's poses (unchanged); (c) otherwise, when
  `manifest_code` differs from the dataset manifest's `code` (exact JSON equality: any
  difference in `commit` or `dirty`), emit a `SplitCommitWarning` (subclass of
  `UserWarning`, via `warnings.warn`) naming both `code` objects, saying the content is
  identical and that rewriting the split with `--overwrite` refreshes `manifest_code`,
  then proceed; (d) emit nothing when they are equal. Under Python's default warning
  filter a given split + dataset pair prints the warning once per process.
- **FR-005** (loading v1, legacy): `split_version` 1 files MUST keep loading with their
  v1 key set and their unchanged strict semantics: the dataset's raw manifest SHA-256
  must equal `manifest_sha256`; any byte difference refuses, including a `code`-only
  one (no warning path, no new leniency). The refusal MUST add a hint: v1 pins the raw
  manifest bytes including the generating commit, and `strike-a-pose-data split …
  --overwrite` rewrites the file as v2. Any other `split_version`, or a key set that
  matches neither version, → `SplitError`. v1 files are no longer written.
- **FR-006** (dataset API): `PoseImageDataset` MUST expose `manifest_content_sha256`
  next to the unchanged `manifest_sha256` (raw bytes); FR-004 / FR-005 apply at
  construction. `SplitCommitWarning` and the content-hash function are exported from
  `strike_a_pose.data`.
- **FR-007** (run records): the `run_info.json` written by 003 `baseline`, 004 `train`
  and 006 `probe` MUST record `dataset.manifest_content_sha256` next to
  `dataset.manifest_sha256` (additive); `metrics.json` is unchanged. The 004 `--resume`
  check and the 006 latent-cache key keep the raw `manifest_sha256`, which includes the
  generating commit (no code change; clarification 8): a dataset regenerated at another
  commit does not resume and misses the cache, by design. The 004 and 006 contracts say
  so explicitly.
- **FR-008** (fixture split): `splits/smpl_fixture.json` MUST be rewritten as v2 by the
  CLI (`--seed 0 --fractions 0.5 0 0.5 --overwrite`) with algorithm, seed, fractions,
  pose lists (train `[1]`, val `[]`, test `[0]`) and counts identical to the v1 file.
  The pre-007 v1 bytes MUST be kept verbatim as the test asset
  `tests/fixtures/smpl_fixture_split_v1.json`.
- **FR-009** (`smpl_proc_v1`): `splits/smpl_proc_v1.json` MUST stay byte-unchanged
  (v1) in this slice. Its migration is follow-up F1 (below), documented with exact
  commands in 007 `tasks.md`, `docs/HANDOFF.md` and README.
- **FR-010** (tests): pytest MUST cover
  (a) the content hash: invariant to `code` (`commit`, `dirty`, `"unknown"` / `null`),
  changed by any other top-level field and by one nested value (a `labels_sha256`),
  insensitive to formatting;
  (b) the committed fixture: content hash = the FR-001 constant; `dumps` of the parsed
  manifest = the file bytes;
  (c) v2 write / load round trip: exact key set, `manifest_code` = the manifest's
  `code`, deterministic bytes, same pose lists as before for the same inputs;
  (d) `load_split`: v1 and v2 key sets accepted; mixed or extra keys and versions other
  than 1 / 2 rejected;
  (e) v2 loading: committed fixture + committed split → no warning; `code.commit`
  changed → `SplitCommitWarning` and the same `sample_ids` per split; only `dirty`
  changed → warning; content changed (`seed`, one `labels_sha256`) → `DatasetError`;
  (f) US1 end to end with the synthetic body model (`small_cfg`): generate, split,
  regenerate with the same config + seed and a monkeypatched `git_commit` → loads with
  the warning; another seed → refused;
  (g) v1 legacy with the preserved v1 fixture split: loads with the fixture, no
  warning; a fixture copy with only `code` changed → refused, with the hint;
  `splits/smpl_proc_v1.json` passes `load_split`;
  (h) CLI: the committed fixture split regenerates byte-identically (002 SC-002);
  `--overwrite` on a copy of the preserved v1 file → v2 with identical algorithm, seed,
  fractions, pose lists and counts;
  (i) 003 / 004 / 006 fixture runs: `run_info.json` `dataset.manifest_content_sha256`
  = the fixture's content hash;
  (j) resume / cache (clarification 8): one 004 test pins that `--resume` still refuses
  (exit 1) after a `code`-only change of the dataset manifest; the 006 cache code is
  untouched and keeps its tests.
- **FR-011** (documentation, same slice): see *Documentation scope*.
- **FR-012** (constraints): no new dependency; Python 3.11-compatible code (CI runs 3.11
  and 3.13; e.g. no `warnings.warn(skip_file_prefixes=…)`, which is 3.12+); no change to
  001 generation, the manifest format, the split algorithm or the sample dict;
  `docs/results/` untouched; nothing under `data/`, `runs/` or `assets/` committed.

### Documentation scope (FR-011)

- `specs/002-data-pipeline/contracts/sample-contract.md`: split-file section rewritten
  for `split_version` 2 (fields, content-hash definition, loading rules, warning) with a
  v1 legacy note; the Python API gains `manifest_content_sha256` and
  `SplitCommitWarning`.
- `specs/002-data-pipeline/spec.md`: FR-007, FR-008, US2 scenario 3 and clarification 5
  ("No code commit in the file") get dated notes "amended 2026-10-20 by slice 007".
- `specs/002-data-pipeline/data-model.md`: Split entity (v2, v1 legacy) and the split
  construction check.
- `specs/003-baselines-and-metrics/contracts/eval-run.md`,
  `specs/004-autoencoder-mvp/contracts/train-run.md`,
  `specs/006-latent-probe/contracts/probe-run.md`: `run_info.json` `dataset` fields
  (additive); the 004 resume and 006 cache text says they key on the raw manifest hash
  (which includes the generating commit), so a dataset regenerated at another commit
  does not resume and misses the cache, by design.
- `README.md`: the split paragraph (today: valid only at commit `406b68f`) and the
  split usage lines: v2 semantics, the warning, `smpl_proc_v1.json` still v1 until F1.
- `specs/roadmap.md`: strike #14 (done in 007); add 007 to the post-MVP table and the
  status line (the "exactly one post-MVP slice" ruling is superseded for this slice by
  the operator's #14 approval); a changelog entry dated 2026-10-20, appended at the end,
  which also records the #12 re-test of this session (`download.pytorch.org` still
  refused: HTTP 403 at the proxy CONNECT, 3 attempts; torch stays on PyPI).
- `docs/HANDOFF.md`: next-work item 1 becomes F1 with the commands below; item 2 (#12)
  records the re-test; #13 / #15 / #16 and the SMPL-asset access problem were put to
  the operator in this session's closing message and nothing on them was started; the
  environment note "Until #14 lands …" is updated.
- Not edited: 001–006 `plan.md` / `tasks.md` (historical records).

### Key Entities

- **Manifest content hash**: FR-001.
- **Split file v2**: FR-002; normative text in the 002 contract.
- **Split file v1 (legacy)**: the 002 format; read-only, strict raw-bytes check (FR-005).
- **`SplitCommitWarning`**: FR-004 (c).

## Success Criteria *(mandatory)*

- **SC-001**: ruff, ruff format, pytest and the CI smoke steps are green locally and on
  CI (Python 3.11 + 3.13); per-leg CI wall time stays < 5 min (005 SC-003).
- **SC-002**: `strike-a-pose-data split tests/fixtures/smpl_fixture --seed 0 --fractions
  0.5 0 0.5` reproduces the committed v2 `splits/smpl_fixture.json` byte for byte, and
  its pose lists equal the pre-007 v1 file's.
- **SC-003**: a dataset that differs from the split's source only in `code` loads with
  a `SplitCommitWarning` and the same samples per split (fixture copy and synthetic
  regeneration); any content difference is refused.
- **SC-004**: no numerical result changes: the fixture 003 / 004 / 006 `metrics.json`
  before and after 007 differ only in `dataset.split_file_sha256` (the fixture split
  file's bytes change with the migration; the probe is compared on the same checkpoint
  file, since a checkpoint stores its resolved config, run root included);
  `docs/results/` and `splits/smpl_proc_v1.json` are byte-unchanged.

## Assumptions

- `code` is the only manifest field that differs between regenerations of the same
  config + seed (001 SC-002; `test_reproducible_us2` shows byte-identical manifests at
  one commit, and `git_commit` is the only commit-dependent manifest input).
- A pose-level split's validity depends only on the pose set and the pose identities,
  which the manifest content pins (sample ids, `labels_sha256`); pixels are outside its
  scope.
- SMPL stays unavailable for this session; F1 runs in a later session.

## Follow-up F1 — migrate `splits/smpl_proc_v1.json` to v2 (needs SMPL)

Blocked in this session (SMPL model not obtainable). With `assets/smpl/SMPL_NEUTRAL.npz`
in place (never committed), at any commit that contains 007:

```bash
uv sync --locked            # the locked environment (uv.lock unchanged since 406b68f)

# the SMPL file must be the one the 2026-10-20 datasets were generated from
# (recorded in the committed fixture manifest: inputs.smpl_model.sha256 = f408141b…)
uv run --locked python - <<'EOF'
import hashlib, json
want = json.load(open("tests/fixtures/smpl_fixture/manifest.json"))["inputs"]["smpl_model"]["sha256"]
got = hashlib.sha256(open("assets/smpl/SMPL_NEUTRAL.npz", "rb").read()).hexdigest()
assert got == want, f"SMPL file {got[:12]}… is not the recorded {want[:12]}…: stop here"
print("SMPL file matches", want[:12])
EOF

uv run --locked strike-a-pose-data generate --config configs/default.yaml --seed 0  # ≈ 5 min, 351 MB

# provenance check: same content as the dataset behind the published 003/004/006 runs?
uv run --locked python - <<'EOF'
import hashlib
from strike_a_pose.datagen.io import dumps, load_manifest
m = load_manifest("data/generated/smpl_proc_v1")
for dirty in (False, True):
    m["code"] = {"commit": "406b68f990c403dd34747dfc41914407d9ec2ea9", "dirty": dirty}
    print(dirty, hashlib.sha256(dumps(m)).hexdigest())
EOF
# identical content iff one line prints 9000c915c0b9e5db495fc188821d97027c18b80a1b5859ebbee921ed745c9588

uv run --locked strike-a-pose-data split data/generated/smpl_proc_v1 --seed 0 \
    --out splits/smpl_proc_v1.json --overwrite        # default fractions 0.8 0.1 0.1

# the pose assignment must not change (same seed, fractions, algorithm and pose set)
uv run --locked python - <<'EOF'
import json, subprocess
old = json.loads(subprocess.check_output(["git", "show", "HEAD:splits/smpl_proc_v1.json"]))
new = json.loads(open("splits/smpl_proc_v1.json").read())
keys = ("algorithm", "unit", "dataset_name", "seed", "fractions", "poses", "counts")
assert [k for k in keys if old[k] != new[k]] == [], "pose assignment changed: do not commit"
assert (old["split_version"], new["split_version"]) == (1, 2)
print("pose assignment unchanged; split_version 1 -> 2")
EOF
```

Then commit only `splits/smpl_proc_v1.json`; update README (drop the "still v1" note),
`docs/HANDOFF.md` and the roadmap changelog (record the provenance-check outcome); tick
F1 in 007 `tasks.md`. If the provenance check fails although the SMPL file and the
environment match, still rewrite the split as v2 after the pose-list check, and record a
reproducibility caveat in the roadmap changelog, the README "Results" section and
`docs/HANDOFF.md`: the published 003 / 004 / 006 numbers come from the `406b68f` content
(raw manifest `9000c915…`), which the regenerated dataset does not reproduce; the
operator rules on it (clarification 11).

## Clarifications

### Session 2026-10-20 (third session that day; autonomous run, Fable surrogate scope)

Defaults chosen by the lead within the surrogate's scope are marked *(lead default)*; the
two questions sent to the surrogate in one batch, and the dates, carry *(surrogate ruling
2026-10-20)*. The surrogate accepted every lead default, including the four deviations
from the orchestrator's initial leaning (field name `manifest_code`; `make_split`
deriving hash and `code` from the manifest; the SC-004 wording; raw-hash resume and
cache). Each answer is provisional and overturnable by the operator.

1. Q: Slice number and slug? → A: `007-split-content-hash` (next free number; names the
   change). *(lead default)*
2. Q: What exactly is hashed ("content")? → A: the whole manifest minus its top-level
   `code` key, serialized with the 001 serializer `dumps` (FR-001). Dropping the whole
   `code` object rather than its two fields keeps the definition short and also covers
   any provenance field added under `code` later. Reusing the 001 serializer (not a new
   compact form) makes the hash the SHA-256 of the manifest 001 would write without
   `code`, checkable with a few lines of Python. *(lead default)*
3. Q: Field names in the v2 file? → A: `manifest_content_sha256` (says what it hashes;
   the name `manifest_sha256` keeps its raw-bytes meaning in v1 files, the dataset API
   and the run records) and `manifest_code` rather than `code`: 002 clarification 5 kept
   the split writer's own commit out of the file, and a bare `code` key would read as
   that; `manifest_code` is the dataset's generating code, copied from its manifest. No
   raw hash in v2: it would pin the commit again. *(lead default)*
4. Q: When does the loader warn? → A: content equal and `manifest_code` ≠ the dataset's
   `code` by exact JSON equality (`commit` or `dirty` differ; `"unknown"` and `null` are
   plain values). Equal `code` with `dirty: true` on both sides does not warn: a commit
   cannot tell two dirty trees apart, and warning there would fire on every fixture use
   (the fixture was generated from a dirty tree); content equality is the real check.
   *(lead default)* The warning text adds that rewriting the split with `--overwrite`
   refreshes `manifest_code`. *(surrogate ruling 2026-10-20)*
5. Q: v1 files? → A: read-only legacy with the unchanged strict raw-bytes check. No
   leniency: a v1 file does not record the generating `code`, so content equality cannot
   be established from it. No warning when it matches; a migration hint when it does
   not. *(lead default)*
6. Q: CLI behaviour? → A: writes v2 only; `--overwrite` replaces v1 and v2 alike (no
   special case; F1 checks the pose lists). `make_split` derives the content hash and
   `manifest_code` from the manifest it is given, so a caller cannot pair a split with a
   wrong hash; its hash parameter is dropped and its callers (CLI, tests) updated.
   *(lead default)*
7. Q: Which run records change? → A: the 003 / 004 / 006 `run_info.json` gains
   `dataset.manifest_content_sha256` (provenance: the data version independent of the
   generating commit); `metrics.json` keeps its format. *(lead default)*
8. Q: 004 `--resume` and the 006 latent cache: keep the raw `manifest_sha256` or switch
   to the content hash? → A: keep the raw hash; no code change (batch Q1, option (a)).
   Both reuse state computed from pixels, and the manifest does not hash pixels (labels
   only, via `labels_sha256`; PNG hashes live in `checksums.json`, and 001 tolerates
   non-bit-exact renders across machines). Equal content at another commit therefore
   does not prove equal images: after a renderer change, a content-keyed resume would
   mix two image sets in one training run and a content-keyed cache would serve stale
   latents, both silently. The raw hash includes the generating commit and stays the
   conservative key; switching would gain little, because `data/` and `runs/` are
   created and lost together. The 004 and 006 contracts state that they key on the raw
   manifest hash, so a dataset regenerated at another commit does not resume and misses
   the cache, by design; one 004 test pins the refusal (FR-010 (j)). Rejected: (b)
   content hash; (c) content hash plus the SHA-256 of `checksums.json` (a new identity
   and more code). *(surrogate ruling 2026-10-20)*
9. Q: Fixture split and the legacy test asset? → A: FR-008. The v1 bytes are kept so
   that the legacy path `splits/smpl_proc_v1.json` relies on is tested on a real
   002-written v1 file. *(lead default)*
10. Q: `smpl_proc_v1`? → A: not migratable here (no SMPL); follow-up F1 with exact
    commands; the committed v1 file keeps working for the `406b68f` dataset.
    *(lead default; forced by the environment)*
11. Q: F1: what if the regenerated `smpl_proc_v1` fails the provenance check (its
    content differs from the `406b68f` dataset behind the published 003 / 004 / 006
    numbers)? → A: batch Q2, option (a) with two additions. Before the provenance check,
    F1 checks that the SMPL file's SHA-256 equals the recorded one (`f408141b…`, the
    committed fixture manifest's `inputs.smpl_model.sha256`) and runs in the locked
    environment (`uv sync --locked`), so a mismatch is not misattributed. If the check
    still fails, the split is still rewritten as v2 after the pose-list check, and the
    finding is recorded as a reproducibility caveat in the roadmap changelog, the README
    "Results" section and `docs/HANDOFF.md`, for the operator to rule on. Rejected: keep
    the v1 file and ask first. *(surrogate ruling 2026-10-20)*
12. Q: Dates? → A: the session clock, 2026-10-20 (every commit of this session is
    2026-10-20 UTC; the "2026-10-21" label on the roadmap's "approve all" entry came
    from the previous session and is left as written). 007 records are dated
    2026-10-20; its roadmap changelog entry is appended at the end (insertion order).
    *(surrogate ruling 2026-10-20)*
