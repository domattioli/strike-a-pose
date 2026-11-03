# Implementation Plan: Realism Ablations (amendment to 010)

**Branch**: `claude/charming-wozniak-5ouldf` | **Date**: 2026-10-21 | **Spec**: [spec.md](spec.md)

## Summary

Configs and a rename, then one NumPy function plus a read-only script. The 010 chain
`configs/smpl_proc_v{2,3,4,5}/` becomes four independent single-change datasets
`configs/smpl_proc_v2{a,b,c,d}/`; the arms-down table is renamed `armsdown-v1`; a limb–torso
intersection count gates v2a's split at 5 % of poses. The generator is untouched except the
version string, so every v1 dataset still regenerates byte for byte.

## Technical Context

Python ≥ 3.11, NumPy, PyYAML; pytest; ruff (line 100). No new dependency. The script is a plain
file under `scripts/` (as `run_status.py`), run with `uv run --locked python
scripts/intersection_check.py`.

## Constitution Check

- I Library-first: the count lives in `datagen/intersection.py`; the script only samples poses
  and formats. II Config-driven: the script reads the dataset's generation config. III
  Reproducibility: the same pose and beta streams as `generate`; the body model's SHA-256 is
  printed. IV Test-first: red runs recorded in `tasks.md`. V Honest baselines: unchanged.

## Design

- `datagen/intersection.py`: `TORSO_JOINTS = (0, 3, 6, 9)`, `ARM_JOINTS = 16…23`,
  `MARGIN_M = 0.01`, `MAX_INTERSECTING_FRACTION = 0.05`; `kabsch(src, dst)`;
  `TorsoVolume(rest_verts, weights, margin_m)` (dominant joints, per-segment rest vertices,
  box centre and shrunk half-extents) with `.count(posed_verts)`; `survey(model, poses, betas)`
  → `Survey(counts, rest_count, fraction, …)`.
- Inside test of a point q in a segment's rest frame (centre c, half-extents h − margin):
  `((qx − cx)/hx)² + ((qz − cz)/hz)² ≤ 1` and `|qy − cy| ≤ hy`.
- `scripts/intersection_check.py`: load config (`datagen.config.load_config`), refuse AMASS, load
  the model (`SMPLModel.load`, `--smpl-model` overrides the config path), poses
  `sample_procedural_poses(N, seed, cfg.ranges_version)`, betas `sample_betas(…)` or zeros,
  `survey`, print / `--json`, exit codes of spec FR-006.
- Configs: copies of the v1 files with names and paths; test rewritten (FR-002).

## Stages and commits

1. `spec(014)`: spec, plan, tasks.
2. `feat(014): independent realism datasets + table rename`: tests first, configs, version
   string.
3. `feat(014): intersection check`: tests first, module, script.
4. `docs(014)`: 010 / 001 documents, README, roadmap, HANDOFF, 012 `tasks.md` campaign
   procedure, verification record.

## Risks

- The real SMPL weights' dominant joints might not match the assumed part sets: the script
  prints vertex counts per set and the rest-pose self-check, and refuses an intersecting rest
  pose.
- The synthetic torso is thin: the synthetic rates say nothing about SMPL; a widened-torso
  variant in the tests checks the direction (arms-down > v1), not the level.
