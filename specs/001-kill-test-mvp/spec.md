<!-- provenance: author=domattioli model=claude-fable-5-1 effort=high date=2026-10-07 skill=speckit-specify repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
# Feature Specification: Kill-Test MVP for Calibrated Multi-View Body-Measurement Uncertainty

**Feature Branch**: `001-kill-test-mvp`  
**Created**: 2026-10-07  
**Status**: Draft  
**Input**: User description: "Kill-test MVP for calibrated multi-view body-measurement uncertainty. A clean-room Python package that: (1) generates synthetic silhouette renders of SMPL-X bodies seen by 1 to 4 cameras with randomized, known extrinsics; (2) trains a convolutional variational autoencoder whose per-view encodings fuse into one latent posterior (product of experts), so more views narrow the posterior; (3) maps latent samples to SMPL-X shape coefficients and then to anthropometric measurements in cm (height and circumferences such as chest, waist, hip, thigh); (4) calibrates the measurement intervals with split conformal prediction; (5) reports empirical coverage at the 90% level and median interval width in cm against number of views {1, 2, 4} and extrinsic noise {0, 2 deg, 5 deg}; (6) applies a kill criterion: the 4-view median interval width must be at least 30% narrower than 1-view at matched coverage. Real-image evaluation uses public data only: BodyM (front and side silhouettes with measurements) and SSP-3D, with SAM2 masks for real images."

Terms used throughout: a *view* is one camera's silhouette of one body. *Camera placement* is the camera's position and orientation (the extrinsics). *Placement noise* is a rotation error of the camera placement, in degrees. A *measurement* is one of five anthropometric quantities in cm. An *interval* is a calibrated prediction interval for one measurement. A *cell* is one experiment condition: a view count paired with a placement-noise level.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run the synthetic kill test and read the verdict (Priority: P1)

The researcher runs the experiment from one configuration file and one seed. The run generates a synthetic dataset of bodies seen by 1 to 4 cameras with known camera placement. It trains the multi-view uncertainty model. It calibrates the measurement intervals at the 90% level. It evaluates every cell of the grid: view count {1, 2, 4} by placement noise {0°, 2°, 5°}. It writes a results table and prints a PASS or KILL verdict.

**Why this priority**: The verdict is the product of this feature. Every other deliverable supports it.

**Independent Test**: Run the experiment with the tiny configuration on a CPU machine and with the full configuration on the training machine. Check that the results table has 9 cells, that every cell holds coverage, median width, and sample count for each of the five measurements, and that the verdict follows from the table by the stated rule.

**Acceptance Scenarios**:

1. **Given** a valid configuration and seed, **When** the researcher runs the experiment, **Then** a results table appears with one row per cell (9 cells) and per measurement (5 measurements), each row holding empirical coverage at 90% nominal, median interval width in cm, mean absolute error in cm, and the test sample count.
2. **Given** the results table, **When** the kill rule is applied, **Then** the verdict line states PASS when the 4-view median width is at most 70% of the 1-view median width at matched coverage, states KILL otherwise, and shows both widths and their ratio.
3. **Given** a cell whose empirical coverage falls outside the tolerance band, **When** the table is written, **Then** the cell is flagged, and the verdict line states that a width comparison involving that cell is invalid.
4. **Given** a completed run, **When** the researcher opens the output directory, **Then** it holds the per-sample predictions, the calibration quantiles, the results table in CSV and Markdown, the plots, and a run record with configuration hash, seed, code version, and hardware class.

---

### User Story 2 - Evaluate the calibrated model on real public images (Priority: P2)

The researcher points the package at local copies of BodyM and SSP-3D. For BodyM, the package uses the provided front and side silhouettes as a 2-view input and the tape measurements as ground truth. For SSP-3D, the package produces a person mask for each photograph with a promptable segmentation model (operator-named: SAM 2), uses the mask as a 1-view input, and derives ground-truth measurements from the dataset's body-shape annotations. The package reports coverage and median width per dataset and measurement next to the synthetic numbers.

**Why this priority**: Synthetic silhouettes are clean. Real masks carry segmentation errors and clothing. The real-image check shows whether the synthetic verdict transfers. It is P2 because the kill criterion is defined on synthetic data.

**Independent Test**: With the datasets at the configured path and a saved calibrated model, run the real-image evaluation alone. Check that a per-dataset table appears with coverage, median width, mean signed error, subjects evaluated, and subjects skipped.

**Acceptance Scenarios**:

1. **Given** BodyM at the configured path and a calibrated model, **When** the real-image evaluation runs, **Then** it reports, per measurement, empirical coverage at 90% nominal, median width in cm, and mean signed error in cm over the BodyM test subjects, using the front and side views.
2. **Given** SSP-3D at the configured path, **When** the evaluation runs, **Then** it reports the same metrics for 1-view input against ground-truth measurements derived from the dataset's body-shape annotations.
3. **Given** a photograph for which no usable person mask is produced, **When** the evaluation runs, **Then** the subject is counted in a skipped column and the run continues.
4. **Given** a dataset absent from the configured path, **When** the evaluation runs, **Then** it stops with a message that names the missing asset and the configuration key, and downloads nothing.

---

### User Story 3 - Validate a change on a CPU-only machine (Priority: P3)

A contributor changes any module: generation, model, calibration, evaluation, or reporting. The contributor runs the test suite on a machine with no GPU, no network, and none of the licensed assets. The suite exercises every stage on stand-in data and finishes within minutes.

**Why this priority**: The development container has no GPU; full training runs elsewhere. Without this loop no change can be checked before the expensive run. It is P3 because it produces no scientific result on its own.

**Independent Test**: Run the suite and the tiny end-to-end configuration on the reference machine (4 CPU cores, 15 GB RAM). Both pass within 10 minutes each with no GPU, no network, and no licensed asset present.

**Acceptance Scenarios**:

1. **Given** no GPU and no licensed asset, **When** the contributor runs the test suite, **Then** every stage runs on stand-in data and the suite passes within 10 minutes.
2. **Given** the tiny end-to-end configuration, **When** the contributor runs the experiment command, **Then** it completes on CPU within 10 minutes and writes a results table with the same columns as a full run.

---

### User Story 4 - Reproduce a reported result (Priority: P3)

A reviewer takes the configuration file, seed, and code version recorded with a results table. The reviewer re-runs the experiment on a machine of the same hardware class. The synthetic dataset manifest matches byte for byte. The reported numbers match within the stated tolerance.

**Why this priority**: The verdict is credible only when a second run produces the same numbers. It is P3 because it checks User Story 1 rather than adding a result.

**Independent Test**: Two runs with identical configuration and seed on the same hardware class; compare the manifests and the results tables.

**Acceptance Scenarios**:

1. **Given** the same configuration and seed, **When** the generation stage runs twice, **Then** the two dataset manifests (shape coefficients, camera placements, noise draws) are identical.
2. **Given** the same configuration, seed, code version, and hardware class, **When** the full experiment runs twice, **Then** every coverage value matches within 0.5 percentage points and every median width within 0.1 cm.

---

### Edge Cases

- Fewer views than requested are available (BodyM has 2; SSP-3D has 1). The model accepts any subset of 1 to 4 views. The report lists only the view counts the data supports.
- A silhouette is empty or the body is partly outside the frame. The sample is excluded from training and flagged. In evaluation it is counted as skipped.
- Placement noise 0° with 1 view. The fused posterior equals the single view's posterior exactly.
- Two cameras of one rig are nearly co-located. The generator enforces a minimum angular separation between cameras (default 20°).
- The calibration set is smaller than the documented minimum (default 200 bodies). Calibration refuses to run and names the minimum.
- A circumference interval has a lower bound below 0 cm. The bound is clipped at 0 cm and the clip count is reported.
- Empirical coverage lands exactly on a tolerance boundary. Boundaries are inclusive.
- The model produces a non-finite prediction. The run fails with an error that names the sample.
- A real photograph shows more than one person. The subject is skipped and counted.
- BodyM tape measurements and surface-derived circumferences follow different protocols. A systematic offset is expected. The report shows mean signed error per measurement next to coverage.
- SSP-3D body-shape annotations use an earlier generation of the body-model family. Ground-truth measurements come from the dataset's own meshes with the FR-004 definitions. The conversion is documented in the plan.
- Two views of one sample at the same noise level. Noise draws are independent per view.

## Requirements *(mandatory)*

### Functional Requirements

Synthetic data

- **FR-001**: The system MUST generate synthetic bodies by sampling the shape coefficients of a parametric body model (SMPL-X) over a documented range, in [NEEDS CLARIFICATION: pose protocol not specified: one fixed canonical standing pose (matches the BodyM protocol; smallest scope), small random jitter around that pose, or full pose randomization from a pose prior (SSP-3D becomes a fair test; largest scope)].
- **FR-002**: For each body, the system MUST place 1 to 4 virtual cameras at randomized placements (position and orientation) drawn from documented ranges with a minimum angular separation, and MUST record the true placement of every camera.
- **FR-003**: The system MUST render one binary silhouette per camera at a fixed resolution and MUST store, per sample, the silhouettes, the true camera placements, the shape coefficients, and the ground-truth measurements.
- **FR-004**: The system MUST compute ground-truth measurements from the body surface in centimeters: standing height and the circumferences of chest, waist, hip, and thigh. Each measurement MUST have one documented geometric definition that is applied identically to synthetic bodies and to real-data meshes.
- **FR-005**: The system MUST apply placement noise as a rotation of the camera placement that is given to the model, drawn independently per view, with angle equal to the condition's noise level (0°, 2°, or 5°). Silhouettes MUST stay rendered from the true placement.

Model

- **FR-006**: The system MUST encode each view on its own into a per-view estimate of the latent body-shape posterior (a mean and an uncertainty).
- **FR-007**: The system MUST fuse the per-view estimates of one sample into one latent posterior by product of experts. The fused uncertainty MUST NOT exceed the uncertainty of any single contributing view.
- **FR-008**: The system MUST map samples of the fused posterior to body-shape coefficients and then to the five measurements in cm, so that each measurement has a predictive distribution per sample.
- **FR-009**: One trained model MUST serve every view count from 1 to 4, so that a comparison across view counts isolates the effect of the number of views.

Calibration

- **FR-010**: The system MUST calibrate a prediction interval per measurement with split conformal prediction at 90% nominal coverage, on a calibration set disjoint from the training set and the test set, separately for each cell.
- **FR-011**: Calibration MUST refuse to run below the documented minimum calibration set size and MUST report the set sizes it used.

Evaluation and verdict

- **FR-012**: The system MUST evaluate a held-out synthetic test set for every cell of view count {1, 2, 4} by placement noise {0°, 2°, 5°} and MUST report, per cell and per measurement: empirical coverage at 90% nominal, median interval width in cm, mean absolute error in cm (secondary), and the number of test samples.
- **FR-013**: The system MUST flag every cell whose empirical coverage lies outside the tolerance band around 90% stated in the Assumptions.
- **FR-014**: The system MUST compute the kill verdict: PASS when the 4-view median interval width is at most 70% of the 1-view median interval width at matched coverage, KILL otherwise. [NEEDS CLARIFICATION: the rule leaves three choices open: (a) a per-measurement ratio required for all five measurements, or one pooled ratio; (b) judged at 0° placement noise only, or at every noise level; (c) "matched coverage" as both cells inside the tolerance band after per-cell calibration, or as re-calibration to equal empirical coverage before the widths are compared].
- **FR-015**: The system MUST write the results as a machine-readable table (CSV) and a human-readable table (Markdown) with the verdict line, generated by code from the saved per-sample outputs.
- **FR-016**: The system MUST plot coverage and median width against view count, one series per noise level, for each measurement.

Real images

- **FR-017**: The system MUST evaluate the calibrated model on the BodyM test subjects with the provided front and side silhouettes as a 2-view input and the dataset's tape measurements as ground truth for the five measurements.
- **FR-018**: The system MUST evaluate the calibrated model on SSP-3D as a 1-view input with ground-truth measurements derived from the dataset's body-shape annotations by the FR-004 definitions.
- **FR-019**: For a real photograph without a provided silhouette, the system MUST produce a person mask with a promptable segmentation model (operator-named: SAM 2), MUST record per subject whether a usable mask was produced, and MUST skip and count subjects without one.
- **FR-020**: The real-image evaluation MUST report, per dataset and measurement: empirical coverage at 90% nominal, median width in cm, mean signed error in cm, subjects evaluated, and subjects skipped. [NEEDS CLARIFICATION: role of the real-image results in the gate: report only (the verdict rests on synthetic data), a second pass bar on BodyM (for example coverage of at least 85% at 90% nominal), or deferral of real-image evaluation to a later feature].

Data and compliance

- **FR-021**: The system MUST read licensed assets (body-model files, BodyM, SSP-3D, segmentation-model weights) only from an operator-configured location outside the repository, MUST NOT download them, and MUST stop with a message that names the missing asset and the configuration key when one is absent.
- **FR-022**: The repository MUST hold no licensed asset, no derived render set, no per-subject table, and no trained weights. An automated check in the test suite MUST fail when such a file is under version control.
- **FR-023**: The implementation MUST be clean-room. No code, weights, or data MUST come from `domattioli/Conv_AE-Human_Pose` or from the UVA / NFL-Biocore project. Every change description MUST carry the clean-room statement.

Reproducibility and tests

- **FR-024**: Every run MUST be fully determined by one configuration file and one integer seed. The generation stage MUST produce identical sample manifests for identical inputs on any machine.
- **FR-025**: Every results table and run record MUST carry the configuration hash, the seed, the code version, and the hardware class.
- **FR-026**: Every module MUST have a smoke test that runs with no GPU, no network, and no licensed asset, on stand-in data. The full test suite MUST finish within 10 minutes on a machine with 4 CPU cores and 15 GB RAM.
- **FR-027**: A tiny end-to-end configuration MUST run the complete experiment on CPU within 10 minutes and MUST write the same table columns as a full run.
- **FR-028**: The system MUST select the compute device at run time and MUST fall back to CPU when no GPU is present.

### Key Entities *(include if feature involves data)*

- **Body Sample**: one synthetic subject: identifier, shape coefficients, pose, and the five ground-truth measurements in cm. Listed in the dataset manifest.
- **Camera Placement**: for one view: the true position and orientation, the fixed intrinsic parameters, the noisy placement given to the model, and the noise level.
- **Silhouette**: one binary image per (sample, view) at the fixed resolution, linked to its camera placement.
- **Per-View Posterior**: the latent mean and uncertainty estimated from one view.
- **Fused Posterior**: the product-of-experts combination of the per-view posteriors of one sample. Its uncertainty never exceeds that of any contributing view.
- **Measurement Interval**: per (sample, measurement): lower bound, upper bound, nominal level (90%), width in cm, and a covered flag (true value inside the interval).
- **Data Split**: training, calibration, and test sets, disjoint by body identifier, with recorded sizes.
- **Experiment Condition**: one cell: view count in {1, 2, 4} and placement noise in {0°, 2°, 5°}.
- **Result Cell**: per condition and measurement: empirical coverage, median width, mean absolute error, sample count, and the tolerance flag.
- **Kill Verdict**: PASS or KILL, with the 1-view median width, the 4-view median width, their ratio, the threshold (0.70), the coverage-band check, and the run record.
- **Real-Image Subject**: dataset name, subject identifier, available views, mask status, ground-truth measurements, and predicted intervals.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: One command produces the complete results table: 9 cells by 5 measurements, every entry holding coverage, median width, mean absolute error, and sample count, plus one verdict line. No entry is empty.
- **SC-002**: On the synthetic test set, every cell's empirical coverage at 90% nominal lies inside the tolerance band (87% to 93%) with at least 2,000 calibration bodies and 2,000 test bodies per cell.
- **SC-003**: The verdict is computed by code from the saved per-sample outputs. Two independent re-computations from the same outputs give the same verdict and the same ratio.
- **SC-004**: For 100% of test samples, the fused uncertainty with k views is at most the fused uncertainty with any subset of those views.
- **SC-005**: Two runs with the same configuration and seed give identical dataset manifests. On the same hardware class, every coverage value matches within 0.5 percentage points and every median width within 0.1 cm.
- **SC-006**: The test suite and the tiny end-to-end run each finish within 10 minutes on a 4-core, 15 GB machine with no GPU, no network, and no licensed asset.
- **SC-007**: The real-image evaluation reports all five measurements for 100% of BodyM test subjects (silhouettes are provided by the dataset) and for at least 90% of SSP-3D images (mask produced).
- **SC-008**: The automated repository check finds zero licensed assets, derived render sets, per-subject tables, or trained weights under version control on every change.
- **SC-009**: Every change description carries the clean-room statement, and review finds no material from the excluded sources.
- **SC-010**: The full-size experiment completes within 24 hours of wall time on the training machine.

## Assumptions

- Pose: unless the clarification on FR-001 decides otherwise, synthetic bodies stand in one canonical pose with arms held slightly away from the body, which matches the BodyM standing protocol. SSP-3D poses vary and are evaluated as a stress test.
- Measurement set: exactly five measurements: standing height and the circumferences of chest, waist, hip, and thigh. Each circumference is the perimeter of a horizontal slice of the body surface at a landmark height defined on the body model.
- Body model: one gender-neutral body model. Shape coefficients are sampled from a documented distribution (default: the first 10 coefficients, standard normal, clipped at plus or minus 3).
- Camera model: pinhole cameras with fixed intrinsic parameters. Distance and height are sampled within documented ranges; azimuth is uniform around the body; cameras of one rig are at least 20° apart.
- Noise model: placement noise is a rotation by the cell's angle about a uniformly random axis, applied per view to the placement given to the model. Renders use the true placement. Position error is out of scope.
- Training: one model trained with view counts drawn uniformly from 1 to 4 per sample and placement noise drawn uniformly from 0° to 5°, so that every evaluation cell lies inside the training distribution.
- Data split: training, calibration, and test sets are disjoint by body (default sizes 20,000; 2,000; 2,000). Calibration and test bodies are reused across cells with cell-specific view subsets and noise draws.
- Calibration score: the absolute error normalized by the model's predicted spread, so that calibrated widths follow the fused posterior and shrink when it shrinks.
- Tolerance band: 87% to 93% at 90% nominal (plus or minus 3 percentage points), given the default calibration and test set sizes.
- Real data: BodyM silhouettes are used as provided; the mask step applies to SSP-3D photographs. BodyM front and side views are treated as nominal 0° and 90° placements at a fixed default distance with no placement noise.
- Real-data ground truth: BodyM tape measurements as published; SSP-3D measurements derived from the dataset's body-shape annotations with the FR-004 definitions.
- Licensing: body-model files, BodyM, SSP-3D, and segmentation weights are treated as research-only, non-redistributable assets. The operator obtains them under their own agreements and places them at the configured path. Renders of licensed bodies stay out of the repository.
- Compute: development on a container with 4 CPU cores, 15 GB RAM, and no GPU; full training on a separate GPU machine. The full experiment budget is 24 hours of wall time.
- Scope: no README, no web surface, no packaging for distribution, and no further datasets or model families until the verdict is recorded (constitution Principle IV).
- Deliverable: an importable package with one command-line entry point. Run outputs go to an output directory outside version control. The results tables for the gate decision are committed as text under this feature's spec directory.
