# Tasks: Realism Ablations (amendment to 010)

Input: [spec.md](spec.md), [plan.md](plan.md). Tests first in every stage (red run noted), then
code; each stage ends green (`uv run --locked ruff check .`, `uv run --locked ruff format
--check .`, `uv run --locked pytest -q`) and is committed with explicit paths.

## Stage 1 — spec (`spec(014)`)

- [x] T001 spec.md, plan.md, tasks.md from the operator decisions of 2026-10-21 (no
  clarification batch).

## Stage 2 — configs and rename (`feat(014): independent realism datasets + table rename`)

- [ ] T002 Tests (red first), `tests/test_realism.py`: `V2 = "armsdown-v1"`; the old name
  rejected; the series tests rewritten (each v2x against v1, no guard exemption, the directory
  set is exactly v2a–v2d; v2d's PCA keys); `test_series_configs_generate_on_synthetic_model` over
  v2a–v2d.
- [ ] T003 `procedural.py` version string; `configs/smpl_proc_v2{a,b,c,d}/`; remove
  `configs/smpl_proc_v{2,3,4,5}/`.

## Stage 3 — intersection check (`feat(014): intersection check`)

- [ ] T004 Tests (red first), `tests/test_intersection.py` (spec FR-010 (c)–(e)).
- [ ] T005 `datagen/intersection.py`; `scripts/intersection_check.py`.
- [ ] T006 Rates on the synthetic body for `configs/default.yaml` vs
  `configs/smpl_proc_v2a/generate.yaml` (and the widened torso), recorded below.

## Stage 4 — docs (`docs(014)`)

- [ ] T007 010 spec / tasks / data-model / research notes; 001 spec / data-model / research R9.
- [ ] T008 README, roadmap, HANDOFF (state, sign-off, next work), campaign procedure in HANDOFF
  and 012 `tasks.md` (identical, script check, `bash -n`).
- [ ] T009 Verification record.

## Verification record
