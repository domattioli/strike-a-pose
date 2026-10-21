# Tasks: Realism Ablations (amendment to 010)

Input: [spec.md](spec.md), [plan.md](plan.md). Tests first in every stage (red run noted), then
code; each stage ends green (`uv run --locked ruff check .`, `uv run --locked ruff format
--check .`, `uv run --locked pytest -q`) and is committed with explicit paths.

## Stage 1 — spec (`spec(014)`)

- [x] T001 spec.md, plan.md, tasks.md from the operator decisions of 2026-10-21 (no
  clarification batch).

## Stage 2 — configs and rename (`feat(014): independent realism datasets + table rename`)

- [x] T002 Tests (red first), `tests/test_realism.py`: `V2 = "armsdown-v1"`; the old name
  rejected; the series tests rewritten (each v2x against v1, no guard exemption, the directory
  set is exactly v2a–v2d; v2d's PCA keys); `test_series_configs_generate_on_synthetic_model` over
  v2a–v2d.
- [x] T003 `procedural.py` version string; `configs/smpl_proc_v2{a,b,c,d}/`; remove
  `configs/smpl_proc_v{2,3,4,5}/`.
  Red run: 10 failed (old name, no v2a–v2d directories). Stage checks: ruff and format clean;
  pytest 447 passed in 1017 s (446 in 238 s before; the machine was shared with a live
  training run, so wall times of this slice are not comparable with earlier slices').

## Stage 3 — intersection check (`feat(014): intersection check`)

- [x] T004 Tests (red first), `tests/test_intersection.py` (spec FR-010 (c)–(e)).
- [x] T005 `datagen/intersection.py`; `scripts/intersection_check.py`.
- [x] T006 Rates on the synthetic body for `configs/default.yaml` vs
  `configs/smpl_proc_v2a/generate.yaml` (and the widened torso), recorded below. Red run:
  collection error (no `datagen.intersection`); then 5 failures that were test design, not
  code: the constructed pose (left shoulder z −135°) misses the plain synthetic body's 6-cm torso
  cubes by 1 mm after the 1 cm margin, so the constructed-pose tests use the widened-torso body
  (torso cubes scaled 5 / 2.2 / 3 in x / y / z: 0.30 × 0.13 × 0.18 m per segment) and the
  script test no longer assumes a PASS at 5 %. 12 tests.

  Rates (`scripts/intersection_check.py --seed 0`, all 2,000 poses, margin 0.01 m):

  | Body | `configs/default.yaml` (v1 table) | `configs/smpl_proc_v2a/generate.yaml` (`armsdown-v1`) |
  |---|---|---|
  | synthetic (6-cm cube per joint) | 0 / 2000 = 0.00 %: PASS | 1 / 2000 = 0.05 %: PASS |
  | synthetic, widened torso | 13 / 2000 = 0.65 %: PASS | 154 / 2000 = 7.70 %: FAIL |

  Caveat: the synthetic arms are 6-cm cubes at the joints only (no limb surface between
  joints) and its torso is either far too thin (plain) or an arbitrary box (widened); the
  numbers show that the count works and that the arms-down table raises it ≈ 12× on a
  torso-sized body, not what real SMPL will give. The 010 joint-level proxy (clarification 6)
  predicted ≈ 1–2 % per arm; the v2a step measures it with SMPL.
  Stage checks: ruff and format clean; pytest 460 passed in 1035 s (shared machine).

## Stage 4 — docs (`docs(014)`)

- [x] T007 010 spec / tasks / data-model / research notes; 001 spec / data-model / research R9.
- [x] T008 README, roadmap, HANDOFF (state, sign-off, next work), campaign procedure in HANDOFF
  and 012 `tasks.md` (identical, script check, `bash -n`).
- [x] T009 Verification record.

## Verification record

- Commits: `471659f` spec, `fc96934` configs + rename, `95ff45d` `fix(012)` (lock liveness for
  `python -m strike_a_pose.train.cli` launches; coordinator request mid-slice, found in live
  use), `7c80d46` intersection check, docs = the commit holding this record.
- Checks per commit (ruff check, ruff format --check clean each time): spec commit — the suite
  at `e131cc7` with identical code, 446 passed in 238 s, run just before (docs-only commit);
  stage 2: 447 in 1017 s; fix: 448 in 1032 s; stage 3: 460 in 1035 s; docs: 460 in 1591 s. The
  machine was shared with a live training run (wall times 7–40× the idle ≈ 40 s; the last run
  was pinned to core 3 by the coordinator); later checks run as `taskset -c 3 env
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked …`.
- SC-003: the campaign procedure in `docs/HANDOFF.md` and 012 `tasks.md` identical (script);
  every `bash` block of HANDOFF passes `bash -n`.
- FR-011: generator code changed only in the version string; `docs/results/`, `splits/`,
  `pyproject.toml`, `uv.lock`, `.github/` untouched; no dataset with the old table name existed
  (`grep` over `splits/`, `docs/results/`, `data/`, history).
- Not done here (needs SMPL): the gate on the real body, every campaign step.
