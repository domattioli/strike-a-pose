<!-- provenance: author=domattioli model=claude-opus-5-5 effort=max date=2026-10-07 skill=speckit-implement repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
# Research: Kill-Test MVP for Calibrated Multi-View Body-Measurement Uncertainty

**Date**: 2026-10-07 | **Branch**: `001-kill-test-mvp` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

Each entry gives the decision, the rationale, the alternatives considered, and the sources. A claim is tagged `[verified]` (checked on 2026-10-07 against the cited source), `[inferred]` (follows from a verified fact), or `[assumed]` (to confirm at implementation; the loader or test that confirms it is named).

## R1. Silhouette renderer: polygon-fill union with OpenCV, headless

**Decision**: `render.py` projects every triangle of the posed mesh with the pinhole camera and fills all projected triangles in one `cv2.fillPoly` call on a `uint8` canvas, with fixed-point sub-pixel coordinates (`shift=4`). Triangles with a vertex behind the camera are dropped. A mask that touches the image border marks the sample "partly outside the frame" (spec edge case).

**Rationale**: the silhouette of an opaque mesh is the union of the projections of all its triangles, so no depth test is needed. The call runs on CPU and on a Kaggle GPU node with no display, no OpenGL, no EGL, and no OSMesa. Integer rasterization is bit-identical across machines (constitution Principle V). Cost is about 5 ms per 20k-triangle body at 128 by 128 pixels `[inferred]`; the tiny-config test measures it.

**Alternatives considered**: pyrender (needs OSMesa or EGL; fails on Kaggle images and in containers without setup); PyTorch3D and nvdiffrast (CUDA builds, no CPU path on the Kaggle image, large installs); a pure-torch rasterizer (more code, slower on CPU).

**Sources**: OpenCV drawing functions, `fillPoly` and its `shift` parameter: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html

## R2. Realistic-pose source: AMASS motion-capture poses from the asset root; joint-limit sampler as the asset-free fallback

**Decision**: `pose.source: amass` for the full run. The loader reads AMASS `.npz` files under `<asset_root>/amass/<subset>/**`, concatenates frames, and draws frames with the seeded generator. The body pose is 21 joints by 3 axis-angle values; hand, jaw, and eye poses are zero. The root orientation is zero for both sources (an upright body): the loader discards AMASS root orientations, because AMASS stores them in its mocap world frame, which is Z-up while the body models are Y-up `[assumed; nothing depends on it once the root is discarded; T018's test asserts the discard]`, and the rig's uniform azimuth already supplies the viewing direction. `pose.source: limits` (tests, tiny config, and the fallback when the operator has no AMASS files) samples each joint's axis-angle uniformly inside the limit table of R3. Both sources feed the R3 filters. The drawn pose is stored in the manifest, so a run reproduces without re-reading AMASS.

**Rationale**: motion-capture poses are realistic by construction, the loader needs NumPy only, and AMASS and SMPL-X come from the same project (one registration flow). VPoser was rejected: its repository supports Python 3.11 to 3.12 only and requires `uv`, while both target machines run Python 3.13 `[verified: README of https://github.com/nghorbani/human_body_prior, fetched 2026-10-07]`.

**License**: AMASS is a non-commercial scientific research dataset, registration-gated, no redistribution `[assumed: the license page https://amass.is.tue.mpg.de/license.html is blocked by this container's egress proxy; the operator confirms the terms at download]`. The manifest with poses is a derived artifact and stays out of the repository (Principle II).

**File keys** `[assumed; the loader logs the keys it found and `tests/test_amass.py` covers both layouts]`: SMPL-X-format AMASS files carry `root_orient` (N by 3), `pose_body` (N by 63), `pose_hand`, `trans`, `betas`, `gender`; older SMPL-H-format files carry `poses` (N by 156). The loader accepts both.

**Alternatives considered**: VPoser sampling (rejected above); joint-limit sampling only (kept as the fallback; less natural).

**Sources**: AMASS https://amass.is.tue.mpg.de/ ; VPoser https://github.com/nghorbani/human_body_prior

## R3. Pose filters (FR-001): joint-angle magnitude limits and a capsule self-intersection proxy

**Decision**: a pose is rejected and redrawn when (a) any joint's rotation angle (the norm of its axis-angle vector) exceeds the limit table in the configuration (degrees: spine joints 35, neck 50, head 50, collars 20, shoulders 150, elbows 150, wrists 60, hips 120, knees 150, ankles 45), or (b) two non-adjacent bone capsules overlap by more than 1 cm. A capsule is one bone (joint to child joint) with radius equal to the median distance of the bone's vertices to the bone axis in the canonical pose. Adjacent pairs (bones that share a joint, and the pelvis or spine bones with the upper arms and thighs) are exempt. The rejection count per shard goes to the manifest summary.

**Rationale**: exact triangle-triangle tests over 20k faces per draw are too slow for 25,000 bodies. Capsules are deterministic, cheap, and catch the arm-through-torso and leg-through-leg cases that made the 2022 setup implausible.

**Alternatives considered**: exact mesh self-intersection with a bounding-volume hierarchy (slow; extra dependency); a learned pose prior as the filter (licensed; Python 3.13 problem, R2).

**Sources**: the limit values are rounded range-of-motion figures `[inferred]`; the table is configuration, not code, so the operator can change it without a release.

## R4. Stand-in body model for tests: a procedural capsule mannequin

**Decision**: `body.model: standin` is a pure-NumPy mannequin with the SMPL-X body joint names (22 joints: pelvis, left and right hip, knee, ankle, foot; spine1, spine2, spine3, neck, head; left and right collar, shoulder, elbow, wrist). Rest-pose joint positions scale with shape. Each bone is a capsule mesh of about 200 vertices (about 4,000 vertices in all) with rigid skinning (each vertex belongs to one bone) and forward kinematics from axis-angle rotations. The 10 shape coefficients map linearly to height, torso width, torso depth, hip width, thigh radius, arm radius, shoulder width, head radius, leg-length ratio, and waist indent. The canonical pose is zero rotation (arms 30 degrees from the torso).

**Rationale**: Principle VI forbids licensed assets in tests. The mannequin exercises every stage (pose filters, cameras, rendering, measurement, training, calibration) through the same `BodyModel` interface. Capsule cross-sections are circles, so the measurement tests have analytic expected values (2 pi r).

**Alternatives considered**: a committed SMPL-X-derived mesh (forbidden: derived artifact); random meshes (no joints, no measurements).

## R5. Body-model package: `smplx` for SMPL-X and for SMPL (SSP-3D ground truth)

**Decision**: `smplx==0.1.28` as the optional extra `[body]`. `SmplxBody` reads `<asset_root>/smplx/SMPLX_NEUTRAL.npz`. `SmplBody` reads `<asset_root>/smpl/SMPL_{MALE,FEMALE,NEUTRAL}.pkl` and serves SSP-3D ground truth only, because SSP-3D labels are SMPL shape coefficients, not SMPL-X. Part ids come from the argmax of the skinning weights (`lbs_weights`). Both models expose the joint names the measurement definitions need (pelvis, spine1 to spine3, neck, hips, knees, collars).

**Rationale**: the package is pure Python on torch and NumPy and installs on Python 3.13 `[verified: pip index versions smplx lists 0.1.28 for this interpreter, 2026-10-07]`. Its license is the SMPL-X research license (non-commercial scientific research, no redistribution) `[verified: https://smpl-x.is.tue.mpg.de/modellicense.html via search summary]`, so it is an install-time dependency and is never vendored (Principle II).

**Risk**: SMPL `.pkl` files from the SMPL site contain `chumpy` objects; the `smplx` README documents the cleaning step. Asset preparation is the operator's step (quickstart.md).

**Alternatives considered**: converting SSP-3D SMPL shapes to SMPL-X with the `smplx` transfer tool (needs both models plus an optimisation, adds error); dropping SSP-3D (loses the 1-view real check).

**Sources**: https://pypi.org/project/smplx ; https://smpl-x.is.tue.mpg.de/modellicense.html

## R6. Measurement definitions (FR-004), one rule set for synthetic and real meshes

**Decision**: all measurements are computed on the canonical-pose mesh for the sample's shape coefficients, in centimetres, with the vertical axis `y`. `P(h, S)` is the perimeter of the 2D convex hull of the intersection of the plane `y = h` with the triangles that have at least one vertex in part set `S`. Part sets use the skinning argmax. Searches over `h` step `measure.step_cm` (default 0.5 cm, the same step for ground truth and predictions). Joint heights `y(j)` come from the canonical-pose joints. The implementation is batched: `measure_batch` vectorizes the intersection over meshes and heights (torch, device-aware, chunked for memory) and computes each hull perimeter by the Cauchy projection formula over 64 directions (error under 0.1%).

| Measurement | Definition |
|---|---|
| height | `max_y(V) - min_y(V)` over all vertices |
| chest | `max` of `P(h, torso)` for `h` in `[y(spine2), y(spine3)]`, torso = pelvis, spine1, spine2, spine3 parts; collar parts are excluded and the range stops at spine3, because collar vertices sit at shoulder level and would move the maximum to the shoulders |
| waist | `min` of `P(h, torso)` for `h` in `[y(pelvis), y(spine2)]` |
| hip | `max` of `P(h, torso + left_hip + right_hip)` for `h` in `[y(pelvis) - 0.10 height, y(pelvis)]` |
| thigh | `P(h_t, left_hip part)` with `h_t = y(left_hip) - 0.30 (y(left_hip) - y(left_knee))` |

A slice with fewer than 3 intersection points gives `NaN` and flags the sample.

**Rationale**: a tape measure spans concavities, so the convex-hull perimeter is closer to a tape reading than the exact slice perimeter. Joint-anchored search ranges make the rule independent of the mesh topology (SMPL-X, SMPL, mannequin). The hip rule's hull of both thighs matches a tape wrapped around the buttocks and both legs.

**Alternatives considered**: fixed landmark vertex ids (topology-specific; needs a table per body model); exact slice perimeter (over-counts concavities); a linear map from shape coefficients to measurements for speed (not the default: the exact rule applies to ground truth and to predictions alike; R13 keeps a per-body linearization as the last mitigation).

**Known offset**: BodyM tape measurements follow a measurement protocol that differs from these rules; the real-image tables report mean signed error next to coverage (spec edge case).

## R7. Model: per-view conditional encoder, product-of-experts fusion, heteroscedastic shape head

**Decision**:
- Encoder (FR-006): a 5-block strided CNN (128 to 4 pixels, channels 32 to 256) on the silhouette, concatenated with a 64-d embedding of the camera placement given to the model (6D rotation representation plus translation scaled by 1/5 m), then an MLP to `mu_v` and `logvar_v` in a 16-d latent.
- Fusion (FR-007): product of experts with a `N(0, I)` prior expert: precision `T = 1 + sum_v exp(-logvar_v)`, mean `mu = (sum_v mu_v exp(-logvar_v)) / T`. Precision adds positive terms, so the fused variance never exceeds any contributing view's variance, and adding a view never widens it (SC-004). The per-view posterior is itself the product of the prior expert and the encoder expert of that view, so with one view the fused posterior equals the per-view posterior exactly (spec edge case).
- Decoder (FR-008): an MLP from `z` to the 10 shape coefficients with a heteroscedastic Gaussian head (mean and log-scale).
- Loss: Gaussian negative log-likelihood of the true shape coefficients under the decoded distribution for the joint posterior plus the mean of the same term over each single-view posterior (the MVAE sub-sampled objective), plus a KL term with a linear warm-up over the first 20% of steps and weight `kl_weight`.
- Training draw (FR-009): per sample, view count uniform in {1, 2, 3, 4} and placement noise uniform in [0, 5] degrees per view, so every evaluation cell is in-distribution.
- Prediction: `K = 32` latent samples, each decoded to shape coefficients sampled from the head, each mapped to measurements with R6; per sample and measurement, the median is the point estimate and the median absolute deviation (scaled by 1.4826) is the spread.

**Rationale**: the latent must express shape while poses are random and unknown to the model; a single view leaves shape ambiguous, more views remove ambiguity, and the PoE precision sum is the mechanism the kill test probes. The sub-sampled objective keeps single-view experts calibrated, which the 1-view cells need.

**Alternatives considered**: silhouette-reconstruction decoders (need a differentiable renderer; push pose into the latent); separate models per view count (confounds the comparison, FR-009); mixture-of-experts fusion (no monotone narrowing guarantee).

**Sources**: Wu and Goodman 2018, Multimodal Generative Models for Scalable Weakly-Supervised Learning (MVAE, product of experts, sub-sampled training): https://arxiv.org/abs/1802.05335 ; Zhou et al. 2019, On the Continuity of Rotation Representations in Neural Networks (6D representation): https://arxiv.org/abs/1812.07035 ; Kingma and Welling 2013, Auto-Encoding Variational Bayes: https://arxiv.org/abs/1312.6114

## R8. Calibration (FR-010): split conformal with normalized residual scores

**Decision**: per cell and measurement, the score of a calibration body is `|m_true - m_median| / max(spread, 0.1 cm)`. With `n` calibration scores and `alpha = 0.10`, `q_hat` is the `ceil((n + 1)(1 - alpha))`-th smallest score. The interval is `m_median +/- q_hat * spread`, lower bound clipped at 0 cm with the clip count recorded. Calibration refuses to run when `n < calibrate.min_cal` (default 200; the tiny config sets 32) (FR-011). The calibration and test sets are disjoint by body id and disjoint from training (R9).

**Rationale**: the finite-sample guarantee `P(coverage) >= 1 - alpha` holds for any score under exchangeability, and the normalized score lets interval width follow the model's own spread, so a narrower fused posterior gives a narrower calibrated interval (the quantity the kill rule compares).

**Alternatives considered**: absolute residual score (constant width per cell; hides the posterior narrowing); conformalized quantile regression (needs quantile heads; more training surface).

**Sources**: Lei, G'Sell, Rinaldo, Tibshirani, Wasserman 2018, Distribution-Free Predictive Inference for Regression (locally weighted conformal): https://arxiv.org/abs/1604.04173 ; Angelopoulos and Bates 2021, A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification: https://arxiv.org/abs/2107.07511

## R9. Determinism and reproducibility (FR-024, FR-025, SC-005)

**Decision**:
- `seeding.py` derives every generator from `numpy.random.SeedSequence([seed, stage_id, shard, body])`, so shards are independent and identical on any machine; PCG64 output is platform-independent. Implementation note (T007): a bare list lets distinct keys collide (`SeedSequence([1, 2])` and `SeedSequence([1, 2, 0])` give the same stream), so `rng_for(seed, *path)` encodes the path length first and each value as its 32-bit word count followed by its words; streams therefore differ from a literal reading of the list above, and no two (seed, path) keys share a stream.
- Generation is NumPy plus integer rasterization (R1): manifests and shards are byte-identical across machines.
- Splits are contiguous body-index ranges (train, then calibration, then test) written into the manifest; `tests/test_splits.py` asserts disjointness by body id; the last 5% of the train range is a loss-monitoring slice that takes no gradient step, and no model selection reads the calibration or test splits.
- Training sets `torch.manual_seed`, `torch.use_deterministic_algorithms(True, warn_only=True)`, cuDNN deterministic mode, `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment, and a single-process data loader with a seeded sampler; the tolerance in SC-005 covers residual kernel differences across hardware classes.
- The run record stores the configuration hash (SHA-256 of the canonical YAML dump), seed, package version plus git commit when available, hardware class and device name, and library versions.
- `sap verify` recomputes evaluation and verdict from the saved per-sample outputs and compares them with the stored tables (SC-003).

**Alternatives considered**: a single global RNG (order-dependent; breaks shard-level resume); storing only the seed (reproduction would need the exact pose files again).

## R10. Checkpoint and resume, and the Kaggle run (FR-029, SC-010)

**Kaggle facts**: a weekly GPU quota of about 30 hours that floats and is not stated in Kaggle's own docs; a session cap reported as 9 hours by some sources and as 12 hours for CPU and GPU sessions by others; one P100 (16 GB) or two T4 (2 by 16 GB) `[verified from the third-party pages cited below; Kaggle's documentation page is script-rendered and returned no text to this container]`. The Kaggle Python image pins Python 3.13 (`/usr/local/lib/python3.13`) on a September 2025 Colab base image and inherits torch from that base without pinning it `[verified: Dockerfile.tmpl of Kaggle/docker-python, fetched 2026-10-07]`.

**Decision**:
- Every stage writes to `<out>/<stage>/` and ends with a `DONE.json` marker holding the configuration hash and the input markers it consumed. `sap run --resume` skips a stage whose marker matches.
- Within a stage: generation resumes per shard (a shard file is written atomically through a temporary name); training checkpoints every `train.checkpoint_every` steps and at each epoch end with model, optimizer, scheduler, RNG states, epoch, and step; prediction, calibration, and evaluation cache per cell.
- `--time-budget 8.5h` makes the run stop at the next checkpoint before the budget ends, so a session never reaches the cap; the design bound of 8.5 hours sits under both reported caps and the operator's 9 hours.
- The next session attaches the previous notebook version's output as an input, copies it to the working directory, and runs with `--resume` `[assumed: Kaggle's notebook-output-as-data-source feature; the quickstart names the click path and the first full run confirms it]`.
- Licensed assets and the package wheel reach Kaggle as private, owner-only datasets (spec Assumptions); the notebook never downloads an asset.
- The notebook prints and records `python --version`, `torch.__version__`, and the GPU name at start.

**Alternatives considered**: one long uninterrupted run (exceeds the session cap); saving checkpoints to a Kaggle dataset through the API from inside the notebook (works; kept as the documented fallback when output chaining fails).

**Sources**: https://www.gmicloud.ai/en/blog/best-free-gpu-cloud-options-for-ai-startups-and-researchers ; https://gpuperhour.com/blog/free-cloud-gpus-and-credits ; https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/ ; https://github.com/Kaggle/docker-python

## R11. Dependencies and pins (FR-030, Principle V)

Versions available to Python 3.13 on 2026-10-07 `[verified: pip index versions on the development container]`:

| Package | Version seen | License | Role |
|---|---|---|---|
| numpy | 2.5.3 | BSD-3 | arrays, generation, measurement |
| torch | 2.14.1 (cp313 wheels exist from 2.6.0) | BSD-3 | model, training |
| opencv-python-headless | 4.14.0.94 (5.0.0.93 also listed; the 4.x line is pinned) | Apache-2.0 | rasterization, mask resizing |
| pyyaml | 6.0.3 | MIT | configuration |
| matplotlib | 3.11.2 | PSF-based | plots (Agg backend) |
| smplx | 0.1.28 | SMPL-X research license | optional extra `[body]`: SMPL-X and SMPL meshes |
| sam2 | 1.1.0 on PyPI, Meta's distribution `[verified by the cycle-1 analyzer on PyPI, 2026-10-07]`; needs python >= 3.10, torch >= 2.5.1 | Apache-2.0, checkpoints Apache-2.0 | optional extra `[real]`: SSP-3D masks |
| pytest | 9.1.1 | MIT | tests |
| ruff | current | MIT | lint |

**Pinning rule**: `pyproject.toml` carries lower bounds; `constraints.txt` carries exact versions for every package. torch is pinned to the version the Kaggle image ships at the first full run (recorded in the run record), and the CPU environment installs that same version from the PyTorch CPU wheel index. `sap info --strict` fails on a version that differs from the constraints file.

**Sources**: https://pypi.org/project/opencv-python-headless ; PyTorch 2.6 Python 3.13 support https://discuss.pytorch.org/t/unable-to-install-pytorch-on-python-3-13/212112 ; SAM 2 https://github.com/facebookresearch/sam2

## R12. Real datasets and masks (FR-017 to FR-020)

**BodyM** `[verified: https://registry.opendata.aws/bodym/ via search summary]`: 2,505 subjects, 8,978 frontal and lateral silhouettes, height, weight, gender, and 14 tape measurements in cm including height, chest girth, waist girth, hip girth, and thigh girth; splits train, Test-A (lab photographs), Test-B (less controlled photographs); license CC BY-NC 4.0. Layout `[assumed; the loader discovers CSV columns by header name and logs them; the project page https://adversarialbodysim.github.io/ is blocked by this container's egress proxy]`: per split, a frontal mask folder, a side mask folder, `measurements.csv`, `hwg_metadata.csv`, and `subject_to_photo_map.csv`. The evaluation uses Test-A and Test-B as separate rows.

**SSP-3D** `[verified: https://github.com/akashsengupta1997/SSP-3D, fetched 2026-10-07]`: MIT license; 311 images of tightly clothed sports persons with silhouettes, and `labels.npz` with filenames, SMPL pose and shape parameters, genders, 2D joints, camera translations, and bounding boxes; the zip is part of the repository; SMPL model files are needed separately (R5). Key names are not listed in the README `[assumed; the loader logs the keys]`.

**Masks (FR-019)**: `real.ssp3d.mask_source` is `provided` (the dataset silhouettes; default) or `sam2` (recomputed from the photograph with the SAM 2 image predictor and a box prompt from the dataset bounding box; checkpoint read from `<asset_root>/sam2/`). SAM 2 code and checkpoints are Apache-2.0, Python >= 3.10, torch >= 2.5.1 `[verified: https://github.com/facebookresearch/sam2 via search summary]`; the PyPI package `sam2` 1.1.0 is Meta's distribution `[verified by the cycle-1 analyzer on PyPI, 2026-10-07]`. A mask is usable when it covers at least 2% of the image and its largest connected component holds at least 90% of the mask area; otherwise the subject is skipped and counted. Status order: no mask gives `skipped_no_mask`; then a mask with two connected components that each hold at least 10% of the mask area is `skipped_multi_person`; then a mask that fails the usability rule is `skipped_unusable`; otherwise `ok`. The multi-person rule runs first because a two-person mask can also fail the usability rule (its largest component holds at most 90% of the mask area), and the status must not depend on which rule is checked first. BodyM silhouettes are used as provided.

**Nominal cameras**: BodyM front = azimuth 0 degrees, side = azimuth 90 degrees (the dataset's left-side view; the sign is confirmed on the first subject at implementation `[assumed]`); SSP-3D = azimuth 0 degrees; distance 3.0 m, height 1.2 m, aimed at a point 0.9 m above the floor (`real.nominal_camera.lookat_height_m`, about the bounding-box center of an average upright body, matching the synthetic rigs), no placement noise. Preprocessing: pad to square, resize to `camera.image_size` with area interpolation, threshold at 0.5.

**Limitation**: the camera distance of a real photograph is unknown, so the absolute scale (height) of a real silhouette is only weakly identifiable; the real-image rows are reported only (FR-020) and show mean signed error next to coverage.

## R13. Compute budget estimate (SC-010)

Slice count per mesh: the chest, waist, and hip searches of R6 at `measure.step_cm` = 0.5 cm over ranges of about 20 to 35 cm each, so one mesh needs about 130 slices plus the thigh slice and the height extremes.

| Stage | Estimate | Basis |
|---|---|---|
| Generation, 25,000 bodies, 4 views each, 128 by 128 | about 30 minutes on CPU | R1 cost plus SMPL-X forward passes and about 130 slices per body for ground truth `[inferred]` |
| Training, 30 epochs over 20,000 bodies | under 2 hours on a T4 | small CNN, 128 by 128 inputs `[inferred]` |
| Prediction, 9 cells by 5,000 bodies by 32 samples = 1,440,000 meshes at most (flagged bodies are skipped), about 130 slices each | 1 to 2.5 hours on the T4 with the batched `measure_batch` of R6; 9 to 45 hours on CPU alone | vectorized intersection over meshes and heights with the Cauchy perimeter; the CPU figure is the sequential NumPy cost of about 40 ms per mesh `[inferred]` |
| Calibration, evaluation, verdict, report | minutes | arithmetic on saved outputs |

Gate before the first full run (quickstart section 4 step 0; tasks T049 and T052): the tiny run's `timings.predict`, measured on the full run's hardware class by the notebook's gate cell, scaled by (full mesh evaluations / tiny mesh evaluations) and by (SMPL-X faces / stand-in faces), must be under 12 hours; otherwise the Kaggle run does not start. A CPU-measured tiny timing gives an upper bound, so a CPU result under 12 hours passes the gate early. Mitigations, in order, each a configuration key: `predict.n_samples: 16` instead of 32 (halves the cost); `measure.step_cm: 1.0` (halves the slices; it applies to ground truth and predictions alike, so generation re-runs); `predict.measure_mode: linearized`, a first-order expansion of the measurement map around each body's median sample (11 mesh evaluations per body and cell instead of 32, exact at the median). A mitigation is a configuration change recorded in the run record, never a change to the FR-014 rule.

## R14. Clean-room provenance (Principle I, FR-023)

Every algorithm above is taken from the cited public sources or written fresh for this plan. No file, weight, parameter, or output from `domattioli/Conv_AE-Human_Pose` or the UVA / NFL-Biocore project was opened, read, or listed while producing these artifacts. Each module docstring names its public source (R1 OpenCV, R6 convex-hull perimeter rule, R7 MVAE and 6D rotations, R8 split conformal). Pull requests carry the clean-room statement.
