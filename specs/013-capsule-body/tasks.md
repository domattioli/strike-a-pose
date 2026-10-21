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

- [ ] T006 Twin configs `configs/capsule_proc_{v1,v2a,v2b,v2c,v2d}/` (FR-008) + test FR-011 (j).
- [ ] T007 Docs: `docs/data-assets.md`, 001 contract CLI + note, 001 spec dated note, README,
  roadmap, HANDOFF capsule procedure (FR-009, FR-010).
- [ ] T008 Verification: SC-004 (full `capsule_proc_v1` build, generate, validate, split pose
  lists) and SC-005 (pose baselines); timings recorded here.

## Analyze (2026-10-21)

- Coverage: FR-001–FR-007, FR-011 (a)–(i), FR-012 → T002–T005; FR-008–FR-010, FR-011 (j) →
  T006–T007; SC-001/002 → T005 + CI; SC-003 → T002 (h); SC-004/005 → T008; SC-006 → diff review.
- Consistency: the spec's lead default 2 (literal table) is superseded by ruling Q3 (b);
  FR-001 and the CLI (`--fixture`) follow the ruling. Configs follow the post-012 / post-010
  SMPL configs, hence stage 3.
- Risk: numpy 2.4.6 (CI 3.11) bytes unverified offline; fallback in spec Assumptions.
