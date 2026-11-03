<!-- provenance: author=domattioli date=2026-10-17 skill=speckit-constitution repo=strike-a-pose -->
<!--
Sync Impact Report
- Version: template → 1.0.0 (initial ratification)
- Principles added: I–VI (all new)
- Sections added: Technical Constraints, Development Workflow, Governance
- Templates: plan/spec/tasks templates compatible; plan "Constitution Check" gates = principles I–VI
- Deferred TODOs: dataset choice and license deferred to MVP spec
-->
# strike-a-pose Constitution

strike-a-pose is the successor to `Conv_AE-Human_Pose`: a Python + PyTorch
library for learning compact latent representations of human pose
(encode, reconstruct, and generate poses). The predecessor's notebooks are
reference material, not code to port verbatim.

## Core Principles

### I. Library-First, Notebooks Last
All model, data, and training logic lives in an importable package
(`src/strike_a_pose/`). Notebooks may only call the package for exploration
and figures; no logic that a result depends on may live only in a notebook.

### II. CLI-Driven, Config-Defined Runs
Every training, evaluation, and inference entry point is a CLI command driven
by a versioned config file (YAML/TOML). A run is fully described by
config + code commit + data version; no hidden notebook state or hard-coded
paths.

### III. Reproducibility (NON-NEGOTIABLE)
Seeds are set and logged for Python, NumPy, and PyTorch. Every run writes its
resolved config, git commit, environment summary, and metrics to its output
directory. A reported number that cannot be regenerated from those artifacts
is not a result.

### IV. Test-First for Core Logic
Data transforms, pose representations (joint ordering, normalization,
skeleton definitions), losses, and metrics have pytest unit tests written
before or alongside the implementation. Training includes a CPU smoke test
(a few steps on a tiny fixture dataset) that runs in CI in under ~2 minutes.

### V. Honest Evaluation
Each model is reported against a trivial baseline (e.g., mean pose, PCA) on a
held-out split defined once and never tuned on. Metrics use standard units
(e.g., MPJPE in mm or normalized units, stated explicitly). Claims in docs or
READMEs must cite the run that produced them.

### VI. Simplicity / MVP Discipline
Build the smallest thing that answers the current question. No speculative
abstractions, plugin systems, or multi-framework support. New dependencies
need a stated reason in the plan.

## Technical Constraints

- Python ≥ 3.11, PyTorch 2.x; dependencies pinned in `pyproject.toml` + lockfile.
- Must run on CPU for tests; GPU (CUDA) optional for training.
- Formatting/lint: `ruff`; type hints on public APIs.
- Datasets and checkpoints are never committed; data access is scripted and
  documented, with dataset license terms respected.
- No secrets committed (`*.env`, `*token*`, `*secret*`, `*.pem`).
- Repository license: PolyForm Noncommercial 1.0.0 + No-AI clause; new
  dependencies must be license-compatible.

## Development Workflow

- Work follows spec-kit: specify → clarify → plan → tasks → analyze → implement.
- Each feature's `spec.md` requires operator sign-off before plan/implement.
- Clarification questions are batched, never asked one at a time.
- `main` is protected; work on feature branches, one PR = one purpose.
  Operator merges; agents open PRs and do not merge them.
- CI (lint + unit tests + smoke train) must be green before review.

## Governance

This constitution supersedes other practices in this repo. Every plan's
Constitution Check gates on Principles I–VI; any violation must be justified
in the plan's Complexity Tracking table. Amendments require a PR that updates
this file, bumps the version (MAJOR: principle removed/redefined; MINOR:
principle/section added; PATCH: wording), and updates dependent templates.

**Version**: 1.0.0 | **Ratified**: 2026-10-17 | **Last Amended**: 2026-10-17
