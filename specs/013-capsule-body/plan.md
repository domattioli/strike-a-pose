# Implementation Plan: Capsule Body (013)

**Branch**: `wip-013` (rebased onto `claude/charming-wozniak-5ouldf` by the orchestrator) |
**Date**: 2026-10-21 | **Spec**: [spec.md](spec.md) (approved at surrogate scope)

## Summary

One new module `datagen/capsule.py` builds an SMPL-format body from the committed fixture's
rest skeleton (derived at build time, surrogate ruling Q3 (b)): 28 rigid capsules, 24 marker
vertices with a one-hot regressor, zero pose blends, three shape directions, written as an
uncompressed `.npz` whose SHA-256 is pinned. One CLI subcommand `make-capsule-body` writes it.
The generator, renderer, trainer and evaluators are untouched: a capsule dataset is a config
whose `smpl_model_path` names the file. Configs and README / roadmap / HANDOFF edits are a
later stage (after slice 012 and the 010 amendments land).

## Technical Context

Python ≥ 3.11 (CI 3.11 with numpy 2.4.6, 3.13 with numpy 2.5.4), NumPy only; pytest; no new
dependency. Build ≈ 0.05 s, 18.9 MB file; render ≈ 19 ms per 256 × 192 sample.

## Constitution Check

| Principle | Status | Note |
|---|---|---|
| I Library-first | PASS | logic in `datagen/capsule.py`; CLI is a thin wrapper |
| II CLI + config | PASS | `make-capsule-body`; datasets via generation configs (stage 3) |
| III Reproducibility | PASS | byte-deterministic file, digest pinned in code and test, checked by the CLI; manifests record it |
| IV Test-first | PASS | `tests/test_capsule.py` (FR-011 (a)–(i)); configs test (j) in stage 3 |
| V Honest evaluation | PASS | capsule numbers labelled infrastructure tests (FR-009); SC-005 cross-check |
| VI Simplicity | PASS | no generator change, no `builtin:` resolution, no resolution parameters |

## Stages

1. `spec(013)`: spec with the surrogate rulings, plan, research, data model, tasks.
2. `feat(013): capsule body builder`: module, CLI, tests (this branch, then stop).
3. (after 012 + 010 amendments) configs `configs/capsule_proc_{v1,v2a,v2b,v2c,v2d}/`, config
   test, docs (data-assets, 001 contract / spec note, README, roadmap, HANDOFF), verification
   SC-004 / SC-005.

## Complexity Tracking

None.
