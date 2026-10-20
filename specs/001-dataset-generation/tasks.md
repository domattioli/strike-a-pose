# Tasks: Synthetic Pose Dataset Generation (001)

**Input**: plan.md, research.md, data-model.md, contracts/dataset-layout.md
**Tests**: required (constitution IV). All tests run without SMPL/AMASS.
Tier: all tasks are fully specified implementation (sonnet-tier); any open
design question → stop and escalate to the lead (opus).

## Phase 1: Setup

- [x] T001 Create `pyproject.toml` (name `strike-a-pose`, requires-python `>=3.11`, src layout, deps numpy/pillow/pyyaml with lower bounds, dev group pytest/ruff, script `strike-a-pose-data = strike_a_pose.datagen.cli:main`, ruff line-length 100 + rules E,F,I,UP,B, pytest `testpaths=["tests"]`); generate and commit `uv.lock`.
- [x] T002 [P] Add `data/` and `assets/` to `.gitignore` (FR-007); ensure `tests/fixtures/**/*.png|json` are not ignored.
- [x] T003 [P] Create package skeleton `src/strike_a_pose/__init__.py`, `src/strike_a_pose/datagen/__init__.py`.

## Phase 2: Foundational

- [x] T004 [P] `datagen/skeleton.py`: `SKELETON_NAME="smpl24"`, `SMPL24_JOINT_NAMES`, `SMPL24_PARENTS` per data-model.md; test `tests/test_skeleton.py` (24 names, unique, parents precede children).
- [x] T005 [P] `datagen/config.py`: frozen dataclasses `ImageConfig`, `CameraRingConfig`, `GenerationConfig` with defaults per data-model.md; `load_config(path) -> GenerationConfig` (unknown keys / bad values → `ConfigError`); `to_dict()` for the manifest. Test `tests/test_config.py`.
- [x] T006 [P] `datagen/assets.py`: `MissingAssetError(Exception)` with message naming the path and `docs/data-assets.md`; `require_file`, `require_dir`, `sha256_file`. 
- [x] T007 Synthetic test assets in `tests/conftest.py`: fixture `synthetic_smpl_npz(tmp_path)` (24 cubes of side 0.06 m centred on T-pose joint positions, 8 verts/12 tris each, one-hot weights, `J_regressor` = mean of each cube, zero `shapedirs` (V,3,10) and `posedirs` (V,3,207), `kintree_table` from parents with row0[0]=2**32-1); fixture `synthetic_amass(tmp_path)` (subsets `SubA`, `SubB`, each 2 files with `poses` (F,156) small random angles from a fixed rng, plus a `shape.npz` to be skipped).

## Phase 3: US1 — Generate a dataset from one command (P1)

- [x] T008 [P] [US1] `datagen/smpl.py`: `rodrigues`, `SMPLModel` (`load(path)` for `.npz`/`.pkl`, chumpy failure → `MissingAssetError` with conversion hint), `forward(pose72, betas) -> (verts, joints)` (standard LBS, research R2). Tests `tests/test_smpl.py`: rodrigues known rotations; zero pose → template + regressed joints; rotating joint 1 (left_hip) 90° about X rotates its cube about joint 1 and moves descendants, leaves others.
- [x] T009 [P] [US1] `datagen/amass.py`: `PoseRecord` dataclass; `enumerate_candidates(root, subsets, stride)`; `sample_poses(root, subsets, stride, n, seed) -> list[PoseRecord]` per research R3/R4; missing subset dir → `MissingAssetError`; n too large → `ValueError`. Tests `tests/test_amass.py` (sorted order, stride, shape.npz skipped, root zeroed, hands zero, same seed same result, different seed different).
- [x] T010 [P] [US1] `datagen/cameras.py`: `Camera` dataclass; `camera_ring(cfg, height, width)` per research R5; `project(cam, X_w) -> (X_c, uv)`; `visibility(X_c, uv, h, w)` per R6. Tests `tests/test_cameras.py`: names for 8; Front_0 at +Z; FrontRight_45 on −X side; R orthonormal det +1; origin projects to principal point; world +Y projects upward (smaller v).
- [x] T011 [P] [US1] `datagen/render.py`: `render(verts_w, faces, cam) -> (rgb uint8 H×W×3, mask uint8 H×W)` vectorized chunked z-buffer per research R1/R5 (pixel centres at +0.5, nearest depth wins, ties lowest face index; flat shade `round(255*(0.15+0.65*|n·view|))`, background 0). Tests `tests/test_render.py`: single triangle coverage matches analytic count within edge pixels; nearer triangle occludes farther; empty scene all-zero; deterministic repeat.
- [x] T012 [US1] `datagen/io.py`: `write_json` (sorted keys, indent 2, newline, 6-dp floats via recursive rounding), `build_labels(...)`, `write_manifest` (atomic temp+rename), `load_manifest`, `load_sample(dataset_dir, sample_id)`, `validate_dataset(dir) -> list[str]` per contract. Tests `tests/test_io.py`.
- [x] T013 [US1] `datagen/generate.py`: `generate_dataset(cfg, seed, output_root=None, overwrite=False, body_model=None) -> Path` per plan pipeline; seeds `random`/`numpy`; `git_commit()`; exceptions `OutputExistsError`. 
- [x] T014 [US1] `datagen/cli.py`: `main(argv=None) -> int` with `generate` and `validate` subcommands and exit codes per contract.
- [x] T015 [US1] `tests/test_generate.py` on synthetic assets (2 poses × 8 cams, 64×48 image, focal 80): N×C samples, all files, `validate_dataset` returns [], projected joints of visible keypoints fall inside mask dilated by 2 px (US1-2), output-exists → error, `--overwrite` works. `tests/test_cli.py`: missing SMPL → exit 3 and output dir not created; exists → exit 4; validate exit codes.
- [x] T016 [P] [US1] `configs/default.yaml`, `configs/fixture.yaml` (2 poses, stride 30, output under `tests/fixtures` via `--output-root`, dataset_name `smpl_fixture`).
- [x] T017 [P] [US1] `docs/data-assets.md` (FR-008): SMPL + AMASS registration URLs (smpl.is.tue.mpg.de, amass.is.tue.mpg.de), non-commercial license terms, which files (SMPL neutral; AMASS "SMPL+H G" sub-collections e.g. CMU/BMLmovi), expected paths, pkl→npz note, generate/validate commands.

## Phase 4: US2 — Reproduce exactly (P1)

- [x] T018 [US2] Test in `tests/test_generate.py`: two runs same seed → `manifest.json` and every label JSON byte-identical, and `checksums.json` identical; different seed → manifest differs; manifest records config, seed, commit, input sha256s, counts, `complete: true`.
- [x] T019 [US2] Test: simulated interruption (body model raising mid-run) leaves `complete: false`; `validate_dataset` rejects it.

## Phase 5: US3 — Fixture (P2)

- [x] T020 [US3] `tests/test_fixture.py`: if `tests/fixtures/smpl_fixture/manifest.json` exists, `validate_dataset` returns [] and runs < 5 s, sample count ≤ 16; else `pytest.skip` with reason "real-SMPL fixture not generated yet (operator task T022)".
- [x] T021 [US3] Schema-parity test: a synthetic dataset's labels keys equal the contract key set (guards fixture/full schema drift).
- [x] T022 [US3] (amended 2026-10-20: procedural source needs only SMPL; run by agent with operator-supplied SMPL): `strike-a-pose-data generate --config configs/fixture.yaml --seed 3 --output-root tests/fixtures`, then `strike-a-pose-data validate tests/fixtures/smpl_fixture`, `pytest`, commit `tests/fixtures/smpl_fixture/` (minus `run_info.json` host details are fine to keep).

## Phase 5b: Amendment 2026-10-20 (procedural pose source)

- [x] T026 `datagen/procedural.py`: `RANGES_DEG` (predecessor table, research R9), `RANGES_VERSION`, `sample_procedural_poses(n, seed)`; integer `[-A,B)`, k≥5 sparsification, scipy-`xyz` Euler → axis-angle; root zero; scipy test-only cross-check.
- [x] T027 Config `pose_source` (`procedural` default | `amass`); `amass_subsets` required only for `amass`; generate skips AMASS assets for procedural; manifest `inputs.pose_source` + label `pose_source.type`.
- [x] T028 Tests `tests/test_procedural.py` (ranges respected, axis-angle round trip, determinism, generate without AMASS) + config selection; existing AMASS tests run with `pose_source: amass`.
- [x] T029 `configs/{default,fixture}.yaml` → procedural; regenerate committed fixture; visually inspect renders.

## Phase 6: Polish

- [x] T023 `ruff check` + `ruff format --check` clean; full `pytest` green.
- [x] T024 Roadmap changelog entry for slice 001 (renderer deviation, status).
- [ ] T025 **OPERATOR**: first full default run; record wall time vs SC-004.

## Dependencies

T001–T003 → T004–T007 → T008–T011 (parallel) → T012 → T013 → T014 → T015–T021 → T023–T024. T022/T025 after merge-ready code + assets.

## Analyze pass (lead, 2026-10-17)

Coverage: FR-001 T013/T014 · FR-002 T008/T009 · FR-003 T010/T011 · FR-004 T012 ·
FR-005 T012/T013 · FR-006 T018 · FR-007 T002 · FR-008 T017 · FR-009 T020–T022 ·
FR-010 T004. Edge cases: missing assets T015 · output exists T015 · off-view
T010 (visibility) · interruption T019. SC-001 T015 · SC-002 T018 · SC-003
T012 (validate) · SC-004 T025 · SC-005 T020. Constitution I–VI: see plan.

Findings and resolutions:
1. Fixture dataset name inconsistent (plan "fixture" vs tasks "smpl_fixture") → plan fixed to `smpl_fixture`.
2. Spec input lists pyrender; plan uses a NumPy rasterizer → FR-003 still met; recorded as deviation (plan Complexity Tracking, research R1, roadmap changelog).
3. US2 "images within tolerance": with a deterministic renderer images are byte-identical on one platform; image hashes kept out of the manifest (`checksums.json`) so cross-machine float drift cannot break SC-002.
4. Constitution III asks for PyTorch seed; torch not used in this slice → noted N/A in plan.
5. FR-009 requires real SMPL renders, unavailable in the agent container → T022 operator task; test T020 skips until present. No synthetic data is committed as a fixture.
6. "Resumes or restarts cleanly": restart-only (`--overwrite`) chosen; interrupted manifest stays incomplete (T019).
