# Tasks: Repository Foundation (005)

**Input**: plan.md. No new tests (the slice adds no code); verification = CI
commands run locally on 3.11 + 3.13 and the GitHub Actions run.

## Phase 1: US1 — CI on every push (P1)

- [x] T001 [US1] `.github/workflows/ci.yml` exactly per plan "Design" (triggers, concurrency, read-only permissions, 3.11/3.13 matrix no fail-fast, timeout 10, checkout@v5, setup-uv@v7 with cache, `uv sync --locked`, ruff check, ruff format --check, pytest, validate fixture, commented slice-004 smoke-train placeholder).
- [x] T002 [US1] Validate YAML parses (`python -c "import yaml; yaml.safe_load(...)"`) and has the expected keys.

## Phase 2: US2 — README + metadata (P2)

- [x] T003 [P] [US2] `README.md` per FR-007.
- [x] T004 [P] [US2] `pyproject.toml`: `readme`, `authors`, `[project.urls]` (FR-008); `uv lock --check` still passes.

## Phase 3: Verify + record

- [x] T005 Run locally, Python 3.11 and 3.13 (separate venvs): `uv sync --locked`, `ruff check .`, `ruff format --check .`, `pytest`, `strike-a-pose-data validate tests/fixtures/smpl_fixture` — all green.
- [ ] T006 Push; confirm GitHub Actions run for the head SHA is green on both legs (SC-001) and each leg < 5 min (SC-003); fix and re-push until green.
- [ ] T007 Roadmap changelog entry for slice 005.

## Dependencies

T001 → T002; T003, T004 parallel; T001–T004 → T005 → T006 → T007.

## Analyze pass (lead, 2026-10-20)

Coverage: FR-001/002/003/004/005/006 → T001 (+T002) · FR-007 → T003 · FR-008 →
T004 · SC-001/SC-003 → T006 · SC-002 → T005. US1 edge cases: lock drift →
`--locked` (T001); concurrency → T001; missing/corrupt fixture → validate exits 1
(existing CLI, T001); Python provisioning → setup-uv (T001).

Findings and resolutions:
1. Constitution workflow says CI = "lint + unit tests + smoke train"; smoke
   train cannot exist before slice 004 → placeholder step (FR-006), recorded in
   roadmap so 004 owns adding it. Not a violation (nothing to smoke-test).
2. Roadmap "Done when: CI green on an empty package" is superseded: the package
   is no longer empty (001 landed); criterion applied to the current package.
3. `uv run` without `--locked` could silently re-lock; every CI `uv` call uses
   `--locked`.
4. Push + PR triggers double-run on PR branches; accepted (spec Clarification 4).
5. No ambiguity or conflict with slice 001 artifacts; 001 plan already defers CI
   to 005.
