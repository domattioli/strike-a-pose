# Implementation Plan: Recalibrated Limb–Torso Intersection Check

**Branch**: `claude/charming-wozniak-5ouldf` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

## Summary

Diagnosis (a): the 014 spine3 box reaches past the shoulders on real SMPL, so armpit skin of any
lowered arm counts as penetration. Fix in `datagen/intersection.py`: cap spine3's lateral
half-extent at the rest shoulder half-span − 5 cm and require ≥ 0.4 % of arm vertices inside per
pose; script reports the new parameters; the 5 % gate is meaningful again (v1 1.75 %, v2d 2.23 %,
v3 2.24 %, armsdown-v1 23 %). No pose-table, generator or eval change.

## Technical Context

Python ≥ 3.11, NumPy; pytest; ruff (line 100). No new dependency. Training queue running: tests
and one ≤ 1-min real-SMPL script run only. Slice 016 is being implemented in parallel: 017 touches
none of its files (`eval/roundtrip*`, `eval/cli.py`, `configs/roundtrip*`).

## Constitution Check

- I Library-first: logic in `datagen/intersection.py`, script thin. PASS.
- II CLI / config: script flags; defaults are module constants. PASS.
- III Reproducibility: deterministic; constants recorded in the report (`check_version`). PASS.
- IV Test-first: B1 tests red first; real-SMPL calibration test pins the rates. PASS.
- V Honest evaluation: 014 rates and their cause kept in research.md; residual real
  interpenetration reported, not hidden; tolerance not tuned post hoc. PASS.
- VI Simplicity: two constants and one cap; no mesh-level self-intersection. PASS.

## Files

- `src/strike_a_pose/datagen/intersection.py` (B1)
- `tests/test_intersection.py` (B1)
- `scripts/intersection_check.py` (B1)
- `specs/014-realism-ablations/spec.md` (one note, B2), `docs/HANDOFF.md` (B2)
