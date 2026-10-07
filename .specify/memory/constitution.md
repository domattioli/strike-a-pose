<!-- provenance: author=domattioli model=claude-fable-5-1 effort=high date=2026-10-07 skill=speckit-constitution repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
<!--
Sync Impact Report (speckit-constitution, 2026-10-07)
- Version change: unfilled template (no version) -> 1.0.0
- Modified principles: the five template slots are filled and a sixth is added
  - slot 1 -> I. Clean-Room Provenance (NON-NEGOTIABLE)
  - slot 2 -> II. Public Data and License Compliance
  - slot 3 -> III. Coverage-First Evaluation
  - slot 4 -> IV. Kill-Test Gate Before Scope Grows
  - slot 5 -> V. Reproducibility
  - new    -> VI. CPU-Runnable Tests
- Added sections: Assets, Data, and Compute Constraints; Development Workflow and Quality Gates
- Removed sections: none
- Templates:
  - .specify/templates/plan-template.md: reviewed, no edit. The Constitution Check slot is filled per plan from this file.
  - .specify/templates/spec-template.md: reviewed, no edit. No mandatory section was added or removed.
  - .specify/templates/tasks-template.md: updated. The Tests rule and the per-story test headings now follow Principle VI (a CPU smoke test per new module is mandatory).
  - .specify/templates/commands/: absent (DomI ships the commands as speckit-* skills); nothing to check.
  - Runtime guidance docs (README, AGENTS.md): none exist; the operator ruled no README at the start.
- Follow-up TODOs: none. No placeholder was deferred.
- Note: the scaffold copy carried a provenance stamp from another repo (QuADMESH-RL, speckit-pipeline). That stamp is replaced by this file's own stamp.
-->
# strike-a-pose Constitution

## Core Principles

### I. Clean-Room Provenance (NON-NEGOTIABLE)

strike-a-pose is a clean-room rebuild of a 2022 idea. The rebuild starts from public literature and
fresh code only.

- No code, trained weights, configuration, data, or rendered output MUST enter this repository
  from `domattioli/Conv_AE-Human_Pose` or from the UVA / NFL-Biocore project (the excluded
  sources).
- Contributors and agents MUST NOT open, read, list, quote, or diff the excluded sources while
  working on this repository. A local checkout of an excluded source (for example a
  `conv_ae-human_pose` directory on a development machine) is off-limits.
- A module that implements a published method MUST name the public source in its docstring.
- Every pull request MUST state that it adds nothing from the excluded sources.

Rationale: clean-room status is what lets the rebuild be licensed under this repository's own terms.
One copied file voids it, and the loss cannot be undone after the fact.

### II. Public Data and License Compliance

- Real-image evaluation MUST use public datasets only. The named datasets are BodyM (front and
  side silhouettes with tape measurements) and SSP-3D.
- Licensed assets MUST NOT be committed, vendored, or redistributed. Licensed assets are the
  SMPL-X model files, BodyM, SSP-3D, segmentation-model weights, and any asset with a
  research-only or registration-gated license.
- Each contributor obtains licensed assets under their own agreement. Code MUST read them from an
  operator-configured path outside the repository. The ignore rules MUST exclude every asset
  directory.
- Artifacts derived from licensed assets (renders of licensed bodies, fitted parameters, per-subject
  measurement tables) MUST stay out of the repository and out of public releases unless the asset
  license permits the release.
- Evaluation outputs MUST be aggregate numbers. Real images and per-person data MUST NOT be
  re-published.
- The repository license is PolyForm Noncommercial 1.0.0 plus the licensor-added No-AI/ML-training
  restriction (`LICENSE`). Every new file falls under it. Third-party code MUST carry a license
  that permits inclusion, and its source and license MUST be recorded next to the code.

Rationale: the dataset and model licenses forbid redistribution. This repository asks others to
respect its own training restriction, so it respects theirs.

### III. Coverage-First Evaluation

- The primary metrics are empirical coverage of the prediction intervals at the stated nominal
  level and the median interval width in centimeters. Point accuracy (mean absolute error) is
  secondary. Point accuracy MUST be reported after the primary metrics and never instead of them.
- Every reported interval MUST come from a calibration step with a stated nominal level and a
  held-out calibration set. The calibration set MUST be disjoint from the training set and the
  test set.
- Every results table MUST state the nominal level, the number of calibration samples, the number
  of test samples, and the random seed.
- A width comparison between two conditions is valid only when both conditions reach the nominal
  coverage within the tolerance stated in the feature spec. A cell outside the tolerance MUST be
  flagged in the table.
- A metric that cannot be reproduced from a saved configuration and seed MUST NOT be reported.

Rationale: a narrow interval that misses the truth is worse than a wide honest one. The question
this project asks is whether more views shrink honest intervals.

### IV. Kill-Test Gate Before Scope Grows

- Work proceeds in gated increments. Each increment has a pass/kill criterion written in its spec
  before any model is trained.
- A criterion MUST NOT change after the result it judges has been computed. A new criterion needs a
  new spec revision and a fresh run.
- Until the current gate has a recorded verdict, scope MUST NOT grow: no new datasets, no new model
  families, no user interface, no packaging for distribution, and no README (operator ruling
  2026-10-07: no README to start).
- The first gate is feature 001: the 4-view median interval width MUST be at least 30% narrower
  than the 1-view width at matched coverage, or the approach is killed.
- A KILL verdict is a valid outcome. It MUST be recorded with its numbers. The next increment MUST
  change the hypothesis or stop the project.

Rationale: a criterion fixed before the run cannot be moved to fit the result. The MVP exists to
produce a decision.

### V. Reproducibility

- Every run MUST be fully specified by one configuration file and one integer seed.
- Synthetic data generation MUST be deterministic given the seed. The sample manifest (shape
  coefficients, camera placements, noise draws) MUST be identical on any machine.
- Dependency versions MUST be pinned. The CPU test environment and the training machine MUST use
  the same pins.
- Every results artifact MUST record the configuration hash, the seed, the code version, and the
  hardware class (CPU or GPU).
- Every results table MUST be written by code from saved per-sample outputs. Hand-edited numbers
  MUST NOT appear.

Rationale: the kill verdict is credible only when a second machine produces the same numbers.

### VI. CPU-Runnable Tests

- Every module MUST ship with a smoke test. The smoke test MUST run on a machine with no GPU, no
  network, and no licensed asset present.
- Smoke tests MUST use small stand-in data (a procedural placeholder body, a few samples, a small
  image size, a few optimization steps). Smoke tests MUST NOT depend on SMPL-X files, BodyM,
  SSP-3D, or downloaded weights.
- The full test suite MUST finish in 10 minutes or less on the reference development machine
  (4 CPU cores, 15 GB RAM). One test MUST finish in 60 seconds or less.
- Code MUST select the compute device at run time and fall back to CPU. Full-size training runs on
  a separate GPU machine from the same code and the same configuration schema.
- A change whose tests fail on CPU MUST NOT merge.

Rationale: the development container has no GPU. Without CPU tests, no change can be verified
before the expensive run.

## Assets, Data, and Compute Constraints

- Deliverable: a Python package with a command-line entry point. The development container runs
  Python 3.13.
- Reference development machine: 4 CPU cores, 15 GB RAM, no GPU. Full training happens on a
  separate, operator-provided GPU machine. Continuous integration MUST NOT assume GPU access.
- Asset location: one configuration key (or environment variable) names the asset root that holds
  the SMPL-X files, BodyM, SSP-3D, and segmentation weights. A missing asset MUST stop the run with
  a message that names the asset and the key. The package MUST NOT download licensed assets.
- Repository content limits: no datasets, no model weights, no rendered image sets, no per-subject
  tables. A committed test fixture MUST be free of licensed content and MUST be 100 kB or smaller.
- Run outputs go to an output directory outside version control. Only the text results tables
  (CSV and Markdown) and plots that support a gate decision are committed, under the feature's
  spec directory.
- Secrets and dataset credentials MUST NOT be committed.

## Development Workflow and Quality Gates

- Each feature follows the spec-kit flow: constitution, specify, clarify, plan, tasks, implement.
  The spec carries the pass/kill criterion before implementation starts (Principle IV).
- Work happens on numbered feature branches (`NNN-short-name`) created by the spec-kit script.
  `main` receives merges only. Force-push to `main` is forbidden.
- Pull request gate. Every pull request MUST: pass the CPU test suite; carry the clean-room
  statement (Principle I); add no licensed asset or derived artifact (Principle II); keep every
  results table code-generated (Principle V).
- Definition of done for a feature: the spec criteria are met or a KILL verdict is recorded; the
  results tables with seed, configuration hash, and code version are committed; every new module
  has a smoke test.
- Documentation until the first verdict: the feature spec directory and module docstrings. A
  docstring starts with a one-line summary. No README exists before the first recorded verdict.
- Results are reported as numbers with their baselines: coverage in percent with the nominal level,
  width in centimeters with the 1-view baseline, and the sample count.

## Governance

- This constitution supersedes every other practice document in this repository. A conflict
  resolves in favor of this file. An `AGENTS.md`, when one is added, MUST defer to this file and
  MUST NOT duplicate it.
- Amendment procedure: a pull request edits this file, states the version bump type with its
  rationale, and updates the Sync Impact Report comment. The repository owner ratifies by merging.
- Versioning policy (semantic): MAJOR for a principle removal or redefinition; MINOR for a new
  principle or section or materially expanded guidance; PATCH for wording and clarifications.
- Compliance review: every pull request review checks all six principles. The plan phase runs the
  Constitution Check gate. A violation needs a justified entry in the plan's Complexity Tracking
  table, or the change is rejected.
- Kill criteria are governed by Principle IV and cannot be amended after their result is computed.

**Version**: 1.0.0 | **Ratified**: 2026-10-07 | **Last Amended**: 2026-10-07
