# Tasks: 013 Capsule Body

Stages per [plan.md](plan.md). `[x]` done on `wip-013`.

## Stage 1 — spec

- [x] T001 Spec with surrogate rulings Q1–Q3, plan, research, data model, tasks; analyze (below).

## Stage 2 — body builder (tests first)

- [x] T002 `tests/test_capsule.py`: FR-011 (a) skeleton, (b) arrays, (c) zero pose,
  (d) kinematics, (e) closed capsules + edge bound, (f) shape, (g) digest pin + CLI,
  (h) full-size render sanity vs the fixture's SMPL masks + procedural poses, (i) end to end.
- [x] T003 `datagen/capsule.py`: `rest_skeleton`, `section_frame`, `capsule_mesh`,
  `build_capsule_body`, `capsule_body_bytes`, `build_capsule_bytes`, `CAPSULE_V1_SHA256`.
- [x] T004 `datagen/cli.py`: `make-capsule-body [--out] [--fixture] [--overwrite]` (exit 0 / 1 /
  3 / 4).
- [x] T005 ruff, ruff format, full pytest green.

## Stage 3 — after slice 012 and the 010 amendments land (orchestrator sends the brief)

- [x] T006 Twin configs (26 files): `configs/capsule_proc_v1/{generate,baselines,ae,ae_seed1,
  ae_mask,ae_vae,ae_vae_beta0.1,ae_vae_beta0.001,ae_reg,probe}.yaml` and
  `configs/capsule_proc_v2{a,b,c,d}/{generate,baselines,ae,probe}.yaml`, each its SMPL
  counterpart with only names, paths and the body file changed; tests FR-011 (j)
  (`test_capsule_configs_are_twins_of_the_smpl_configs`, generation on the capsule body).
  `splits/capsule_proc_v1.json` committed.
- [x] T007 Docs: `docs/data-assets.md`, 001 contract CLI + note, 001 spec dated note, README
  (status row + Results pointer), roadmap (row + changelog), HANDOFF ("Interim capsule
  campaign", state, sign-off list), `docs/results/013-capsule/README.md` (FR-009, FR-010).
- [x] T008 Verification (orchestrator, live, 2026-10-21): body built, SHA-256 `bc1f3eb7…`
  equals the pin; `data/generated/capsule_proc_v1` generated; its manifest `config` equals
  `load_config("configs/capsule_proc_v1/generate.yaml").to_dict()` (checked); the frozen
  `splits/capsule_proc_v1.json` has `splits/smpl_proc_v1.json`'s pose lists (checked; SC-004).
  SC-005: run `runs/baselines-capsule_proc_v1-20261021T110723Z`: pose baselines equal the
  published `smpl_proc_v1` ones to ≤ 1e-4; image baselines val mean-image MSE 0.01509,
  PCA-64 0.00509 (infrastructure numbers). R0 `runs/ae-capsule_proc_v1-20261021T111109Z`
  training. Checks: ruff, ruff format, full pytest pinned to one core: 473 passed, 2 failed
  (`test_train_guards.py` lock-liveness tests: they read `/proc/<pid>/cmdline` right after
  `Popen`; on a single pinned core the child has not exec'd yet; they pass unpinned; 012 test
  race, not 013).

## Analyze (2026-10-21)

- Coverage: FR-001–FR-007, FR-011 (a)–(i), FR-012 → T002–T005; FR-008–FR-010, FR-011 (j) →
  T006–T007; SC-001/002 → T005 + CI; SC-003 → T002 (h); SC-004/005 → T008; SC-006 → diff review.
- Consistency: the spec's lead default 2 (literal table) is superseded by ruling Q3 (b);
  FR-001 and the CLI (`--fixture`) follow the ruling. Configs follow the post-012 / post-010
  SMPL configs, hence stage 3.
- Risk: numpy 2.4.6 (CI 3.11) bytes unverified offline; fallback in spec Assumptions.
