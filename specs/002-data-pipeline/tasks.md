# Tasks: Data Pipeline (002)

**Input**: plan.md, contracts/sample-contract.md, data-model.md, research.md.
Tests are written with (before) each module (constitution IV).

## Phase 1: Setup

- [ ] T001 Add `torch>=2.5` with `uv add` (PyPI, research R1); `uv lock --check` and `uv sync --locked` pass; confirm `import torch` on CPU.
- [ ] T002 `datagen/io.py`: rename `_check_labels` → `check_labels` (+ call site); existing tests stay green.

## Phase 2: US3 — normalization helpers (P2, but a dependency of US1)

- [ ] T003 [P] [US3] `tests/test_data_transforms.py`: uint8 image round trip exact; range/shape/dtype; rejects bad input; mask → bool; `root_relative` zeroes root and `add_root` inverts, NumPy + torch + batched; 2D normalize: (0,0)→(−1,−1), (W,H)→(1,1), centre→0; round trip NumPy + torch.
- [ ] T004 [US3] `src/strike_a_pose/data/transforms.py` per plan.

## Phase 3: US2 — frozen split (P1)

- [ ] T005 [P] [US2] `tests/test_data_split.py` (synthetic manifests, 100 poses × 8 cams, no files): determinism (same inputs → identical dict and bytes), seed sensitivity, disjoint + cover, counts rule (`round`, ≥1 if f>0, train ≥1), bad fractions → `SplitError`, `parse_sample_id` incl. camera names with `_`, `write_split` refuses existing file, `load_split` rejects overlap / bad version, `check_split_for` reports hash and coverage mismatch.
- [ ] T006 [US2] `src/strike_a_pose/data/split.py` per plan/contract.
- [ ] T007 [US2] CLI `split` subcommand in `datagen/cli.py` per contract; tests in `tests/test_cli.py`: writes + exit 0, second run exit 4, `--overwrite` exit 0, corrupt dataset exit 1, bad fractions exit 2, fixture split equals committed `splits/smpl_fixture.json`.
- [ ] T008 [US2] Generate + commit `splits/smpl_fixture.json` (`--seed 0 --fractions 0.5 0 0.5`).

## Phase 4: US1 — dataset loader (P1)

- [ ] T009 [P] [US1] `tests/test_data_dataset.py` on the fixture: len 16; contract keys, shapes, dtypes; image in [0,1] and equals PNG/255; mask equals `image > 0`; labels equal JSON; camera_index matches ring; `default_collate` batch via `DataLoader(batch_size=4)`; transform applied; split train/test sample sets (8 each, all 8 cameras of one pose, disjoint); `verify=True` passes; SC-001 timing < 5 s.
- [ ] T010 [P] [US1] Error-path tests on temp copies of the fixture: missing dir/manifest, `complete: false`, schema_version 2, missing PNG listed, counts mismatch, unknown camera in id, label sha mismatch only caught with `verify=True`, malformed label shape on item read, wrong image size on item read, split without split_file, unknown split name, split file for another manifest (hash mismatch).
- [ ] T011 [US1] `src/strike_a_pose/data/dataset.py` + `data/__init__.py` per plan/contract.

## Phase 5: Verify + record

- [ ] T012 Local: `uv sync --locked`, `ruff check .`, `ruff format --check .`, `pytest` (3.13; and 3.11 venv), `strike-a-pose-data validate tests/fixtures/smpl_fixture`.
- [ ] T013 Push; CI run for head SHA green on both legs; record per-leg wall time vs SC-004.
- [ ] T014 README (status row, usage: split + loader snippet), roadmap changelog, this file's verification record.

## Dependencies

T001 → all code tasks. T002 → T011. T003 → T004. T005 → T006 → T007 → T008.
T004, T006 → T011; T009/T010 → T011 (tests first). T008 → T009 (split tests use
the committed file). All → T012 → T013 → T014.

## Analyze pass (lead, 2026-10-20)

Coverage: FR-001 → T011 · FR-002 → T010, T011 · FR-003 → T009, T011 · FR-004,
FR-005 → T003, T004 · FR-006, FR-007 → T005–T008 · FR-008 → T009–T011 · FR-009 →
T001, T013 · FR-010 → T003, T005, T007, T009, T010. SC-001 → T009 · SC-002 → T005,
T007 · SC-003 → T005, T009 · SC-004 → T012, T013. Every spec edge case maps to a
T010 or T005/T007 test.

Findings and resolutions:
1. Ambiguity: `round(f·N)` at .5 — fixed: contract now states Python `round`
   (half to even).
2. Constitution II ("entry points driven by a versioned config file"): `split` is
   a data-prep command, not a train/eval/inference run, and takes only seed,
   fractions and paths; all of them are recorded inside the committed output
   file, so the artifact fully describes itself. No config file added (VI).
3. Constitution III (seeds logged): the split seed is recorded in the split file;
   the loader itself is deterministic (no randomness; shuffling/worker seeding is
   004's run config).
4. Constitution IV: the loader's DataLoader smoke (T009) runs in CI; smoke *train*
   stays with 004 (005 placeholder unchanged).
5. Constitution V: "held-out split defined once and never tuned on" → split CLI
   refuses overwrite; `test` documented as never-tuned, `val` for tuning (README).
6. Constitution VI / new dependency: `torch` reason stated in plan; the PyPI vs
   CPU-index trade-off (R1) is an environment constraint, raised as an operator
   open question rather than a hidden choice.
7. Consistency: spec clarification 8 exit codes = contract CLI section; data-model
   per-item checks = plan `__getitem__` (001 `check_labels` + id match + size).
8. Spec says datasets follow layout v1; loader rejects other schema versions
   (T010 covers schema_version 2).
No constitution violations; Complexity Tracking empty.
