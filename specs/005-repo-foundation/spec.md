# Feature Specification: Repository Foundation (CI + scaffolding)

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `005-repo-foundation`)
**Created**: 2026-10-20
**Status**: Approved (autonomous run, operator delegation 2026-10-20); Operator sign-off 2026-10-21 ("approve all")
**Input**: Roadmap slice 005 + lead brief 2026-10-20: "GitHub Actions CI on push +
pull_request; Python 3.11 and 3.13 matrix; install with uv from the lockfile; run
`ruff check`, `ruff format --check`, `pytest`; must pass without SMPL/AMASS; smoke
step `strike-a-pose-data validate tests/fixtures/smpl_fixture`; fill scaffolding
gaps (README, pyproject metadata); leave a hook for slice 004's smoke train; keep
minimal."

Context: `pyproject.toml`, `uv.lock`, ruff and pytest config landed with slice
001 (roadmap changelog 2026-10-17). What is missing is the CI workflow, a README
and a few metadata fields.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Every push is checked automatically (Priority: P1)

A contributor (human or agent) pushes a commit or opens a pull request; GitHub
Actions installs the exact locked environment on each supported Python and runs
lint, format check, the test suite and a dataset-validation smoke step. The
result is visible on the commit / PR.

**Why this priority**: Constitution Development Workflow: "CI (lint + unit
tests + smoke train) must be green before review." Nothing enforces this today.

**Independent Test**: Push a commit; the CI run for that SHA reports success on
both Python versions. Push a commit with a lint error; the run fails.

**Acceptance Scenarios**:

1. **Given** a clean commit, **When** it is pushed to any branch, **Then** a CI
   run starts and passes on Python 3.11 and 3.13.
2. **Given** a pull request, **When** it is opened or updated, **Then** the same
   CI run executes against the PR merge ref.
3. **Given** `pyproject.toml` changed without regenerating `uv.lock`, **When**
   CI installs, **Then** the install step fails (lockfile is authoritative).
4. **Given** the runner has no SMPL/AMASS assets, **When** CI runs, **Then** all
   steps pass (tests use synthetic assets + the committed fixture).

### User Story 2 - A newcomer can install and run the CLI from the README (Priority: P2)

A reader lands on the repo, reads `README.md`, and learns what the project is,
how to install it with uv, how to run the dataset CLI and tests, where licensed
assets go, and the license terms.

**Independent Test**: Follow the README commands on a fresh clone without
SMPL; `uv sync`, `uv run pytest`, and `uv run strike-a-pose-data validate
tests/fixtures/smpl_fixture` succeed.

**Acceptance Scenarios**:

1. **Given** a fresh clone, **When** the README install + test commands are run,
   **Then** they succeed without licensed assets.

### Edge Cases

- Lockfile drift (pyproject edited, lock not) → install fails (`--locked`).
- Two quick pushes to the same ref → the older in-progress run is cancelled.
- Fixture directory missing or corrupt → validate smoke step fails (exit 1).
- A Python version in the matrix is not on the runner → uv provisions it.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `.github/workflows/ci.yml` MUST run on `push` (all branches) and
  `pull_request`.
- **FR-002**: The workflow MUST run a matrix over Python 3.11 and 3.13 on
  `ubuntu-latest`, without fail-fast so both results are always reported.
- **FR-003**: Dependencies MUST be installed with uv from `uv.lock` in locked
  mode (fail on drift); tools (ruff, pytest) come from the locked dev group.
- **FR-004**: Steps MUST run, in order: `ruff check .`, `ruff format --check .`,
  `pytest`, `strike-a-pose-data validate tests/fixtures/smpl_fixture`.
- **FR-005**: The workflow MUST NOT need secrets, licensed assets, or network
  beyond package/Python downloads; token permissions are read-only.
- **FR-006**: The workflow MUST carry a clearly marked placeholder where slice
  004 adds its CPU smoke-train step (constitution IV).
- **FR-007**: `README.md` MUST cover: project summary, status, install (uv),
  CLI usage (generate, validate), running checks locally (same commands as CI),
  licensed assets pointer (`docs/data-assets.md`), and a license note
  (PolyForm Noncommercial 1.0.0 + No-AI/ML-training term; SMPL/AMASS have their
  own non-commercial licenses).
- **FR-008**: `pyproject.toml` MUST declare `readme`, `authors`, and project
  URLs; the change MUST NOT invalidate `uv.lock`.

### Key Entities

- **CI workflow**: `.github/workflows/ci.yml`, one job `test` × Python matrix.

## Success Criteria *(mandatory)*

- **SC-001**: CI run for the slice's pushed head commit is green on both Python
  versions.
- **SC-002**: The same commands run locally (both Pythons) are green.
- **SC-003**: Workflow wall time per matrix leg < 5 minutes (keeps headroom for
  the ~2-minute smoke train budget of constitution IV).

## Assumptions

- GitHub-hosted runners are available for `domattioli/strike-a-pose`.
- Only Linux CI; macOS/Windows deferred (constitution: Linux/macOS CPU targets,
  macOS can be added when a platform-specific bug appears).

## Clarifications

### Session 2026-10-20 (autonomous run, defaults chosen by lead)

1. Q: Runner OS matrix? → A: `ubuntu-latest` only (constitution VI; macOS later if needed).
2. Q: How is Python provisioned? → A: `astral-sh/setup-uv` with `python-version`
   from the matrix; uv installs it if absent. No separate `actions/setup-python`.
3. Q: Run ruff via a dedicated action or from the lock? → A: From the lock
   (`uv run`), so CI and local use the same pinned ruff.
4. Q: Push trigger scope? → A: all branches (agents work on `claude/*` branches;
   PR runs add merge-ref coverage). Duplicate push+PR runs accepted.
5. Q: Concurrency? → A: one run per workflow+ref; newer push cancels older run.
6. Q: Smoke-train hook form? → A: commented placeholder step in `ci.yml` naming
   slice 004; no stub command (nothing to run yet).
7. Q: Cache? → A: setup-uv cache keyed on `uv.lock` (cheap, built in).
8. Q: Coverage / type-checking / pre-commit? → A: out of scope (not required by
   constitution; add when a slice needs them).
