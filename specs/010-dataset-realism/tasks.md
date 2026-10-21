# Tasks: 010 Dataset Realism Options

Input: [spec.md](spec.md) (approved at surrogate scope, rulings 2026-10-21), [plan.md](plan.md),
[research.md](research.md), [data-model.md](data-model.md). Tests first in every stage (red run
noted), then code; each stage ends green (ruff check, ruff format --check, full pytest) and is
committed with explicit paths. Commands: `uv run --locked ruff check .`,
`uv run --locked ruff format --check .`, `uv run --locked pytest -q`.

## Stage 1 — plan (`plan(010)`)

- [x] T001 Spec: rulings folded in (Q1 (a), Q2 (a), Q3 (a), Q4 (b) → FR-014), status approved at
  surrogate scope.
- [x] T002 plan.md, research.md, data-model.md, tasks.md; analyze pass (below).

## Stage 2 — config surface (`feat(010): config surface`)

- [x] T003 "Before" reference: generate synthetic datasets with today's code (procedural
  `proc_cfg`-style, seed 0; AMASS `small_cfg`-style, seed 3) into the session scratchpad;
  keep them for T019. Done at `8d930ba` (datagen code as at `702d2f5`): `proc_small` (3 poses,
  64 × 48, seed 0), `proc_full` (fixture config with the synthetic model, seed 3),
  `amass_small` (seed 3).
- [x] T004 Tests (red first): FR-009 pins — `load_config("configs/fixture.yaml").to_dict()` ==
  fixture manifest `config`; `sample_procedural_poses(2, 3)` == fixture `euler_deg` and, at 6
  decimals, `pose`; rebuilt `inputs.pose_source` and `cameras` == the fixture's.
- [x] T005 `GenerationConfig.to_dict()` omit-when-default machinery (`ADDED_FIELD_DEFAULTS`,
  empty until stage 3) — no behaviour change for today's fields.
- [x] T006 FR-014: `image_pca_ks: []` valid in `eval/config.py`; `eval/baselines.py` skips the
  image PCA (`Baselines.image_pca` None, `fit.image_pca` null); tests (config accepted, fixture
  baseline run with `[]`: mean image + pose baselines, no `pca_k*`, figure with 2 columns);
  drop `{"image_pca_ks": []}` from the invalid-config list. Red run: the two FR-014 tests
  failed ("must be a non-empty list"); the pins passed (they pin today's behaviour). Stage
  checks: ruff and format clean; pytest 312 passed in 34.9 s (307 before).

## Stage 3 — arms-down table (`feat(010): arms-down table`)

- [x] T007 Tests (red first): v2 ranges and offsets in sampled `euler_deg` (`v == off or
  lo + off <= v < hi + off`), `pose` = axis-angle of the offset-included angles; v1 / v2 draws
  identical except z of 13 / 14 / 16 / 17; `ranges_table(v2)` rows carry `offset_deg`, v1 rows
  unchanged; config `ranges_version` validation (unknown → error; non-default with `amass` →
  error; default omitted from `to_dict`, v2 serialized); generated synthetic dataset names v2
  in manifest and labels, validates.
- [x] T008 `procedural.py` table registry + offsets; `config.py` field + validation;
  `generate.py` uses the version. Red run: collection error (no `RANGES_TABLES`). Stage checks:
  ruff and format clean; pytest 318 passed in 31.4 s.

## Stage 4 — random body shape (`feat(010): random body shape`)

- [x] T009 Tests (red first): `body_shape` validation (`std`/`clip` > 0, unknown keys, `{}` →
  defaults, `null` ≡ absent, needs `num_betas` ≥ 1), serialization; betas in `[-clip, clip]`,
  equal across a pose's views, different across poses, deterministic; poses identical with /
  without; zero-`shapedirs` model → images and joints equal the mean-shape generation;
  `shapedirs`-bearing synthetic variant → bone lengths change; `num_betas` above the model's
  components → `ConfigError` before anything is written.
- [x] T010 `datagen/shape.py` (`sample_betas`), `SMPLModel.num_shape_components`, `config.py`
  block, `generate.py` (check after loading the model; per-pose betas). Red run: collection
  error (no `datagen.shape`); one test expectation fixed (clip rate at clip 0.5 is 0.62, not
  < 0.5). Stage checks: ruff and format clean; pytest 323 passed in 33.6 s.

## Stage 5 — camera ring jitter (`feat(010): camera ring jitter`)

- [x] T011 Tests (red first): `camera_jitter` validation and serialization; `jittered_camera`
  with zero deltas == ring `R, t` bit for bit, yaw / name / K unchanged, centre and aim
  geometry; `in_frame`; generated dataset: deltas within ranges, accepted samples keep the
  margin (mask), impossible ranges → all `fallback` and labels equal the no-jitter labels
  apart from `camera.jitter`, rejection ranges exercise retries; manifest `cameras` unchanged;
  002 loader `camera_index` / `yaw_deg` = ring; validates; byte-reproducible; all options
  together; prefix property with all options on.
- [x] T012 `cameras.py` (`jittered_camera`, `in_frame`, `choose_camera`), `config.py` block,
  `generate.py` per-sample camera and `camera.jitter`. Red run: collection error (no
  `jittered_camera`). Stage checks: ruff and format clean; pytest 331 passed in 33.6 s.

## Stage 6 — dataset configs (`feat(010): dataset configs`)

- [ ] T013 `configs/smpl_proc_v{2,3,4,5}/{generate,baselines,ae,probe}.yaml` (FR-010; v5:
  `image_pca_ks: []`, `context_pca_k: null`).
- [ ] T014 Tests: each loads with its loader; consecutive `generate.yaml` differ in
  `dataset_name` + one key; run configs differ from v1's only in names / paths (+ v5 keys);
  each `generate.yaml` with the synthetic model, 2 poses, 64 × 48 validates.

## Stage 7 — docs (`docs(010)`)

- [ ] T015 001 spec (FR-002, FR-003, edge case, Q4, Q9 update, reopened arms-out answer),
  layout contract, data-model, research R5 / R9 — dated "amended 2026-10-21 by slice 010".
- [ ] T016 002 sample contract (`betas`, `camera.R/t`, `yaw_deg`); 003 research R2, 003
  contract + spec (FR-014); 006 probe contract note; 009 spec FR-001 note.
- [ ] T017 README (status table, Usage: options, series configs, run procedure, memory note);
  roadmap (#15 implemented by 010, row, status, changelog 2026-10-21 appended); HANDOFF (state,
  next work = F1 → v1 re-baseline → v2 → v3 → v4 → v5, the "#15 after F1" clause replaced).

## Verification record

- [ ] T018 Red runs noted per stage; final ruff / format / pytest (count, wall time).
- [x] T019 SC-002 before / after synthetic comparison (T003 datasets vs regenerated after
  stage 5): manifest, labels, checksums, images byte-identical. Done after stage 5 (code of
  `4c8db57` + the stage-5 changes): `proc_small` 74 files, `proc_full` 50, `amass_small` 50;
  manifest content (without `code`), `checksums.json`, every label and PNG identical: 0
  differences.

## Follow-up R — run procedure (needs SMPL; not part of "done")

Order: F1 (`docs/HANDOFF.md`) → `smpl_proc_v1` re-baseline with the same code (003 baselines,
`configs/ae.yaml`, probe) → `smpl_proc_v2` → `v3` → `v4` → `v5`. If F1's provenance check fails
after 010 landed, regenerate at a pre-010 commit (e.g. `702d2f5`) first: same failure →
environment; success → a 010 regression to fix before anything else.

```bash
uv sync --locked && mkdir -p runs
DS=smpl_proc_v2        # then smpl_proc_v3, smpl_proc_v4, smpl_proc_v5
uv run --locked strike-a-pose-data generate --config configs/$DS/generate.yaml --seed 0
uv run --locked strike-a-pose-data validate data/generated/$DS
# smpl_proc_v2 only: visual check (below) before the split
uv run --locked strike-a-pose-data split data/generated/$DS --seed 0 --out splits/$DS.json
uv run --locked strike-a-pose-eval baseline --config configs/$DS/baselines.yaml
setsid nohup uv run --locked strike-a-pose-train --config configs/$DS/ae.yaml \
    > runs/train-$DS.log 2>&1 &      # after a restart: strike-a-pose-train --resume RUN_DIR
uv run --locked strike-a-pose-eval probe --config configs/$DS/probe.yaml \
    --checkpoint runs/ae-$DS-<stamp>/best.pt
```

- Visual check (`smpl_proc_v2`): look at the `Front_0` and `Right_90` renders of poses 0–15;
  record arm placement and the number of renders with visible arm–body intersection in 001
  research R9. If intersections are frequent: delete the dataset, amend the table (new version
  string if a dataset with it was ever split or published), regenerate.
- Jittered datasets (`v4`, `v5`): fallback count and mean attempts from the labels:
  `python -c "import json,pathlib;j=[json.loads(p.read_text())['camera']['jitter'] for p in
  pathlib.Path('data/generated/$DS/samples').glob('*.json')];print(len(j),
  sum(x['fallback'] for x in j),sum(x['attempts'] for x in j)/len(j))"`.
- Commit per dataset: `splits/$DS.json`, `docs/results/{003-baselines,004-ae,006-probe}-$DS/`,
  README comparison table (one row per dataset, the v1 re-baseline first), roadmap changelog,
  HANDOFF.
- Time (4 vCPU): per 2,000-pose dataset ≈ 5 min generation (351 MB), ≈ 1 min validate + split,
  ≈ 3 min baselines, ≈ 85 min AE (2–3 h calendar with restarts), ≈ 8 min probe ≈ 100 min CPU;
  `smpl_proc_v5` ≈ 10 min (700 MB), ≈ 2 min, ≈ 3 min (no image PCA), ≈ 170 min, ≈ 16 min ≈
  200 min. Series ≈ 8.5 h CPU after the ≈ 1.6 h v1 re-baseline; about twice in calendar time.

## Analyze (2026-10-21)

- Coverage: FR-001 → T007 / T009 / T011 (each block lands with its feature: no stage accepts
  an option the generator ignores; the coordinator's stage list put all blocks in stage 2);
  FR-002 / FR-009 → T004 / T005 / T019; FR-003 → T007–T008; FR-004 → T009–T010; FR-005 →
  T011–T012; FR-006 / FR-007 → T009 / T011 (streams, prefix); FR-008 → T011 (loader) and the
  unchanged suite; FR-010 → T013–T014; FR-011 → T004–T014; FR-012 → T015–T017; FR-013 → every
  commit's file list; FR-014 → T006.
- Consistency: "ring camera" is the spec's term for the nominal camera throughout; the jitter
  defaults agree in spec FR-001 / FR-010, data-model and research R3; the arms-down numbers
  agree in spec clarification 6 and research R1. No open `[NEEDS CLARIFICATION]`.
- Risk noted: the draw-structure property requires that no table turns a zero-width range
  non-zero (research R1); the v2 table only narrows non-zero ranges.
