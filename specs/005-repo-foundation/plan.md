# Implementation Plan: Repository Foundation (CI + scaffolding)

**Branch**: `claude/charming-wozniak-5ouldf` (slice `005-repo-foundation`) | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)
**Input**: `specs/005-repo-foundation/spec.md` (approved, autonomous run 2026-10-20)

## Summary

Add one GitHub Actions workflow that, on every push and pull request, installs
the locked environment with uv on Python 3.11 and 3.13 and runs the exact
commands a developer runs locally: `ruff check`, `ruff format --check`,
`pytest`, and a `validate` smoke step on the committed fixture. Add a README and
three pyproject metadata fields. No code or dependency changes.

## Technical Context

**Language/Version**: Python ≥ 3.11; CI matrix 3.11 (floor) + 3.13 (container/dev)
**Primary Dependencies**: none new. CI actions: `actions/checkout@v5`, `astral-sh/setup-uv@v7` (both Node 24 majors; setup-uv provisions Python and caches on `uv.lock`)
**Testing**: existing pytest suite (42 tests, CPU, synthetic assets + committed fixture)
**Target Platform**: `ubuntu-latest` GitHub-hosted runner
**Constraints**: no secrets, no licensed assets; `permissions: contents: read`; per-leg < 5 min (SC-003)

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | No logic added; CI calls the package CLI. |
| II CLI + config | PASS | Smoke step uses the existing CLI. |
| III Reproducibility | PASS | `uv sync --locked` installs the pinned lock; drift fails CI. |
| IV Test-first | PASS / hook | Runs the full suite; smoke-train step is a marked placeholder for slice 004 (no training code exists yet). |
| V Honest evaluation | N/A | No models. |
| VI Simplicity | PASS | One workflow, one job, two actions, no new deps, Linux only. |
| Tech: no secrets | PASS | Workflow uses no secrets; read-only token. |
| Tech: pinned deps | PASS | Lockfile enforced in CI. |

## Design

`.github/workflows/ci.yml`:

- `on: push` (all branches) + `pull_request`; `concurrency` group
  `${{ github.workflow }}-${{ github.ref }}`, `cancel-in-progress: true`.
- job `test`, `runs-on: ubuntu-latest`, `timeout-minutes: 10`,
  `strategy: { fail-fast: false, matrix: { python-version: ["3.11", "3.13"] } }`.
- steps: checkout → setup-uv (`python-version: ${{ matrix.python-version }}`,
  `enable-cache: true`) → `uv sync --locked` → `uv run --locked ruff check .` →
  `uv run --locked ruff format --check .` → `uv run --locked pytest` →
  `uv run --locked strike-a-pose-data validate tests/fixtures/smpl_fixture` →
  commented placeholder "Smoke train (slice 004)".
- `UV_PYTHON` is set by setup-uv from `python-version`, so every `uv` call
  uses the matrix interpreter.

`README.md`: summary, status (slices), install, CLI usage, checks, licensed
assets, license note, repo layout pointer to `specs/`.

`pyproject.toml`: add `readme = "README.md"`, `authors`, `[project.urls]`.
These are not part of `uv.lock` resolution input; verify with `uv lock --check`.

## Project Structure

```text
.github/workflows/ci.yml     # new
README.md                    # new
pyproject.toml               # + readme, authors, urls
specs/005-repo-foundation/   # spec, plan, tasks
specs/roadmap.md             # changelog
```

## Complexity Tracking

None.
