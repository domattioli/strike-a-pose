# Tasks: Split Content Hash (slice 007)

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md); normative contract
[`../002-data-pipeline/contracts/sample-contract.md`](../002-data-pipeline/contracts/sample-contract.md)
(split file section) + the 003 / 004 / 006 run contracts.
**Tests**: required (constitution IV; spec FR-010), written before the code.

## Phase 0: Baseline

- [x] T001 SC-004 baseline before any code change (at `ae99e6f`): fixture `baseline`,
  `train`, `probe` smoke commands (README "Development checks") with `--run-root
  <scratchpad>/runs-before`; keep their `metrics.json` / `run_info.json`. → done
  (3 runs, 15 s); pytest at `ae99e6f`: 230 passed in 29 s.

## Phase 1: Tests first (FR-010)

- [x] T002 Test asset: `git show HEAD:splits/smpl_fixture.json >
  tests/fixtures/smpl_fixture_split_v1.json` (verbatim pre-007 v1 bytes); check it is
  not git-ignored (`git check-ignore`). → 578 bytes, SHA-256 `af3e0f1a…` (= the v1
  fixture split's), not ignored.
- [x] T003 [P] `tests/test_data_split.py`: synthetic manifests gain `code`; new
  `make_split(manifest, seed, fractions)`; (a) content hash invariant to `code`
  (commit / dirty / `"unknown"` + `null` / `code` absent), changed by a top-level field
  (`seed`, `dataset_name`) and a nested one (`samples[0].labels_sha256`); (c) v2 key set,
  `split_version` 2, `manifest_code` = the manifest's `code` (a copy), deterministic
  bytes, pose lists as before; (d) `load_split`: v1 asset and v2 accepted; extra key,
  v1 key set with `split_version` 2 (and the reverse), versions 0 / 3 / `"2"` / `true`,
  non-object `manifest_code`, missing `split_version` rejected; `check_split_for` v2
  content message and v1 raw message + `--overwrite` hint; `check_split_code` for
  equal / commit / dirty / unknown code and for a v1 split; (g) `splits/smpl_proc_v1.json`
  passes `load_split` as v1.
- [x] T004 [P] `tests/test_data_dataset.py`: (b) fixture content hash = the pinned
  constant, `dumps` of the parsed manifest = the file bytes, raw hash unchanged; (e)
  committed fixture + committed split → no `SplitCommitWarning` (filter set to error);
  fixture copies with `code.commit` changed / only `dirty` changed / `code` unknown →
  one warning (text names both codes, "identical", `--overwrite`), same `sample_ids` per
  split; formatting-only rewrite → v2 silent, v1 asset refuses; content edits (`seed`,
  one `labels_sha256`) → `DatasetError` "split was made for manifest content"; (g) v1
  asset loads with the fixture silently and refuses a `code`-only copy with the
  migration hint; (f) end to end with `small_cfg`: generate (seed 5), split, regenerate
  with a monkeypatched `generate.git_commit` → warning + same split samples; seed 6 →
  refused; update the remaining `make_split` callers.
- [x] T005 [P] `tests/test_cli.py`: (h) the existing byte-equality test now pins the
  committed v2 file; new: `--overwrite` on a copy of the v1 asset → v2 file equal to the
  committed v2 split, with algorithm / unit / dataset name / seed / fractions / poses /
  counts equal to the v1 file's.
- [x] T006 [P] `tests/test_eval_cli.py`, `tests/test_train.py`, `tests/test_probe.py`:
  (i) `run_info.json` `dataset.manifest_content_sha256` = the fixture's content hash;
  `make_split` callers; (j) `test_train.py`: a fixture-copy run stopped after epoch 1,
  then the copy's `code.commit` changed → `--resume` exits 1 ("dataset manifest
  changed"), with the `SplitCommitWarning` raised while re-opening the splits.
- [x] T007 Run the new and changed tests on the unchanged code: they must fail (record
  the count). → on the pre-007 code: 5 of the 6 changed modules fail at collection
  (`SplitCommitWarning` / `manifest_content_sha256` not importable), and in
  `test_cli.py` the new v1 → v2 test fails (the CLI still wrote `split_version` 1);
  the other 5 CLI tests pass.

## Phase 2: Implementation

- [x] T008 `src/strike_a_pose/data/split.py`: `SPLIT_VERSION = 2`,
  `LEGACY_SPLIT_VERSION = 1`, `SPLIT_KEYS` per version, `SplitCommitWarning`,
  `manifest_content_sha256`, v2 `make_split`, v1/v2 `load_split`, `check_split_for`
  (raw / content), `check_split_code` (plan "Design").
- [x] T009 `src/strike_a_pose/data/dataset.py` (content-hash attribute, both hashes to the
  split check, `warnings.warn(..., SplitCommitWarning, stacklevel=1)`) and
  `src/strike_a_pose/data/__init__.py` (exports). → ruff B028 wants an explicit
  `stacklevel`; 1 (the default) is spelled out. Checked: three split datasets on three
  lines of one process print the warning once.
- [x] T010 `src/strike_a_pose/datagen/cli.py`: `make_split(load_manifest(dir), seed,
  fractions)`.
- [x] T011 Additive `run_info.json` `dataset.manifest_content_sha256` in
  `eval/baselines.py`, `train/trainer.py`, `eval/probe.py` (not in the probe's `hashes`
  dict: cache key and `metrics.json` unchanged).
- [x] T012 Rewrite `splits/smpl_fixture.json` with the CLI (`--seed 0 --fractions 0.5 0
  0.5 --overwrite`); pose lists / counts / seed / fractions / algorithm equal to the v1
  asset's (FR-008). → equal (train `[1]`, val `[]`, test `[0]`); `manifest_content_sha256`
  `7843ff45…`, `manifest_code` `{"commit": "ab95629e…", "dirty": true}`; new file
  SHA-256 `62cc396c…`.

## Phase 3: Contracts and docs

- [x] T013 Contracts in place: 002 `sample-contract.md` (split file v2, v1 legacy,
  API + warning), 003 `eval-run.md`, 004 `train-run.md`, 006 `probe-run.md` (additive
  run_info field; 004 resume and 006 cache key on the raw manifest hash). → in the plan
  commit.
- [x] T014 002 `spec.md` dated amendment notes (FR-007, FR-008, US2 scenario 3,
  clarification 5: "amended 2026-10-20 by slice 007") and 002 `data-model.md` (Split
  entity v2 + v1 legacy, split construction check).
- [x] T015 README (split paragraph and usage: v2 semantics, warning, `smpl_proc_v1.json`
  still v1 until F1), `specs/roadmap.md` (strike #14, 007 row + status line, changelog
  entry 2026-10-20 appended at the end incl. the #12 re-test), `docs/HANDOFF.md` (F1
  with its commands, #12 re-test, #13 / #15 / #16 and SMPL access put to the operator,
  environment note).

## Phase 4: Verify and commit

- [x] T016 Local checks: `uv run --locked ruff check .`, `ruff format --check .`,
  `pytest` (count + wall time), `strike-a-pose-data validate tests/fixtures/smpl_fixture`,
  the three fixture smoke commands into `<scratchpad>/runs-after`; SC-004 diff of the
  three `metrics.json` against T001 (probe compared on the T001 checkpoint as well);
  `git diff --stat ae99e6f -- splits/smpl_proc_v1.json docs/results` empty.
  → at `187fa3b` + docs: `ruff check .` clean; `ruff format --check .` 106 files
  formatted; `pytest` 249 passed in 28 s (230 at `ae99e6f`; +19), and 249 passed with
  `-W error::strike_a_pose.data.split.SplitCommitWarning` (no unexpected warning);
  `validate tests/fixtures/smpl_fixture` OK; the three smoke commands into
  `runs-after` OK (10.7 s). SC-004: 003 and 004 `metrics.json` differ from T001 only in
  `dataset.split_file_sha256` (`af3e0f1a…` → `62cc396c…`); 006 on the T001 checkpoint
  likewise; 006 on the retrained checkpoint also in `encoder.checkpoint_sha256` (the
  weights are bit-identical; only the stored `config.run_root` differs). `run_info.json`
  gains `dataset.manifest_content_sha256` = `7843ff45…`. Untouched since `ae99e6f`:
  `splits/smpl_proc_v1.json`, `docs/results/`, `pyproject.toml`, `uv.lock`,
  `datagen/generate.py`, `datagen/io.py`. F1 block: identical in spec, this file and
  HANDOFF, `bash -n` clean, its Python snippets parse; the provenance-check method,
  tried on the fixture (`code` changed, then restored), reproduces the raw
  `06787bb0…`.
- [x] T017 Commits (`spec`, `plan`, `feat`, `docs`), no push (the orchestrator reviews,
  pushes and watches CI). → `09d2948` spec, `6c3de15` plan, `187fa3b` feat, docs = the
  commit holding this record.

## Follow-up (blocked: SMPL model not obtainable in this session)

- [ ] F1 Migrate `splits/smpl_proc_v1.json` to `split_version` 2 (spec "Follow-up F1",
  clarification 11). With `assets/smpl/SMPL_NEUTRAL.npz` in place (never committed), at
  any commit containing 007, run the commands under "F1 commands" below.

  Then follow the outcome rule after the block (amended 2026-10-21 by slice 012: reproduced,
  reproduced with an SMPL-file note, or continue with a caveat naming the part that differs).

### F1 commands

Verbatim from [spec.md](spec.md) ("Follow-up F1", amended 2026-10-21 by slice 012):

```bash
uv sync --locked            # the locked environment (uv.lock unchanged since 406b68f)

# the SMPL file the 2026-10-20 datasets were generated from (fixture manifest:
# inputs.smpl_model.sha256 = f408141b…); a mismatch is reported, generation continues
uv run --locked python - <<'EOF'
import hashlib, json
want = json.load(open("tests/fixtures/smpl_fixture/manifest.json"))["inputs"]["smpl_model"]["sha256"]
got = hashlib.sha256(open("assets/smpl/SMPL_NEUTRAL.npz", "rb").read()).hexdigest()
print("SMPL file matches" if got == want else f"SMPL FILE DIFFERS: {got[:12]}… is not {want[:12]}…")
EOF

uv run --locked strike-a-pose-data generate --config configs/default.yaml --seed 0  # ≈ 5 min, 351 MB

# manifest diagnostics (config, inputs, sample list, labels): same content as the dataset behind
# the published 003/004/006 runs? check 1 as generated; check 2 with the recorded SMPL file hash
uv run --locked python - <<'EOF'
import hashlib
from strike_a_pose.datagen.io import dumps, load_manifest
m = load_manifest("data/generated/smpl_proc_v1")
recorded = "f408141b4cd5b6ca6466e4e218aeb88a11c60f598114951b1d6d1f8078c1da16"
for check, smpl in ((1, m["inputs"]["smpl_model"]["sha256"]), (2, recorded)):
    m["inputs"]["smpl_model"]["sha256"] = smpl
    for dirty in (False, True):
        m["code"] = {"commit": "406b68f990c403dd34747dfc41914407d9ec2ea9", "dirty": dirty}
        print(check, dirty, hashlib.sha256(dumps(m)).hexdigest())
EOF
# 9000c915c0b9e5db495fc188821d97027c18b80a1b5859ebbee921ed745c9588 on a check-1 line: identical
# manifest content; only on a check-2 line: identical except the SMPL file's bytes (a
# re-converted file with the same arrays; note it in one line); on neither: labels differ

uv run --locked strike-a-pose-data split data/generated/smpl_proc_v1 --seed 0 \
    --out splits/smpl_proc_v1.json --overwrite        # default fractions 0.8 0.1 0.1

# the same seed, fractions and algorithm give the same pose lists (this proves the split
# procedure, not the dataset's identity)
uv run --locked python - <<'EOF'
import json, subprocess
old = json.loads(subprocess.check_output(["git", "show", "HEAD:splits/smpl_proc_v1.json"]))
new = json.loads(open("splits/smpl_proc_v1.json").read())
keys = ("algorithm", "unit", "dataset_name", "seed", "fractions", "poses", "counts")
assert [k for k in keys if old[k] != new[k]] == [], "pose assignment changed: do not commit"
assert (old["split_version"], new["split_version"]) == (1, 2)
print("pose assignment unchanged; split_version 1 -> 2")
EOF

# pixel diagnostic: the 003 baselines (deterministic, ≈ 3 min; also R0's baselines) against
# every number of the published run (relative difference <= 1e-5; strings and newer keys skipped)
uv run --locked strike-a-pose-eval baseline --config configs/baselines.yaml
uv run --locked python - <<'EOF'
import json, pathlib
old = json.load(open("docs/results/003-baselines-smpl_proc_v1/metrics.json"))
run = sorted(pathlib.Path("runs").glob("baselines-smpl_proc_v1-2*"))[-1]
new = json.load(open(run / "metrics.json"))
bad = []
def walk(a, b, path):
    if isinstance(a, dict):
        for k in a:
            walk(a[k], b.get(k) if isinstance(b, dict) else None, f"{path}.{k}")
    elif isinstance(a, int | float) and not isinstance(a, bool):
        if not isinstance(b, int | float) or abs(b - a) > 1e-5 * abs(a):
            bad.append((path, a, b))
walk(old, new, "")
print(run, "pixels reproduced" if not bad else f"{len(bad)} NUMBERS DIFFER, e.g. {bad[:3]}")
EOF
```

Reproduced = manifest check 1 (or check 2, with a one-line note) and "pixels reproduced": commit
only `splits/smpl_proc_v1.json` (and the baselines run's records, campaign procedure); update
README (drop the "still v1" note), `docs/HANDOFF.md` and the roadmap changelog (record the
outcome); tick F1 in 007 `tasks.md`. Otherwise (operator decision 2026-10-21): keep the
rewritten split, run the campaign, and record in README "Results", the roadmap changelog and
`docs/HANDOFF.md` that the published 003 / 004 / 006 numbers come from the `406b68f` dataset
(raw manifest `9000c915…`), which no longer reproduces, naming the part that differs: the SMPL
file (step 1), the labels or config (neither manifest check), the pixels (the baselines);
report it first and prominently in the next operator message.

## Dependencies

T001 first. T002 → T003–T006 (parallel) → T007 → T008 → T009 → T010, T011 → T012 →
T014, T015 → T016 → T017. T013 is done in the plan commit. F1 waits for the SMPL model.

## Analyze pass (lead, 2026-10-20)

Coverage: FR-001 → T003 (a), T004 (b), T008 · FR-002 → T003 (c), T008 · FR-003 →
T005, T010, T012 · FR-004 → T003, T004 (e, f), T009 · FR-005 → T003 (d), T004 (g),
T008 · FR-006 → T004, T009 · FR-007 → T006 (i, j), T011, T013 · FR-008 → T002, T005,
T012 · FR-009 → T016 (byte check), F1, T015 · FR-010 → T003–T007 · FR-011 → T013–T015 ·
FR-012 → T016 (diff review: no `pyproject.toml` / `uv.lock` / datagen change).
SC-001 → T016 (+ CI after the orchestrator's push) · SC-002 → T005, T012 · SC-003 →
T004 · SC-004 → T001, T016. Edge cases: dirty-only (T004), unknown (T003, T004),
formatting (T004), config paths (content by definition; no test needed beyond (a)),
pixels (contracts, T013), mixed key sets (T003), overwrite (T005).

Findings and resolutions:
1. SC-004 ambiguity: the probe's `metrics.json` holds the checkpoint's SHA-256, and a
   004 checkpoint stores its resolved config (run root included), so a retrained
   fixture checkpoint under another run root has other bytes → the spec's SC-004 says
   the probe is compared on the same checkpoint file; T016 runs the probe on the T001
   checkpoint too.
2. FR-004 "once per process" depends on where the warning is attributed → plan /
   research R2 fix `stacklevel=1` (one line in the data package); the 004 trainer opens
   its datasets on three lines, so a caller-attributed warning would print three
   times.
3. `split_version` is compared as a dict key; JSON `true` equals `1` in Python → the
   loader requires a real int (T003 (d) covers `true`).
4. The 006 `hashes` dict feeds the cache key, the cache meta and `metrics.json`; adding
   the content hash there would change all three → T011 adds it to `run_info.json`
   only; the SC-004 diff would catch a slip.
5. Consistency: the F1 command block appears in the spec, this file, README (short
   form) and HANDOFF → the spec is the source; tasks and HANDOFF copy it verbatim
   (T015 check).
6. Terminology: "content hash" always means FR-001's hash and the key
   `manifest_content_sha256`; "raw hash" the bytes hash and the key `manifest_sha256`
   (v1 split files, dataset attribute, run records, 004 resume, 006 cache).

Found during implementation:

7. ruff B028 rejects `warnings.warn` without an explicit `stacklevel` → `stacklevel=1`
   is spelled out (same behaviour as the default); plan and research R2 updated.
8. A heredoc inside an indented list item breaks when copied from the raw file (the
   `EOF` terminator and the Python lines gain leading spaces) → the F1 block lives in
   unindented "F1 commands" sections here and in HANDOFF, identical to the spec's
   (checked by script).
