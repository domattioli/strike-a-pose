# Feature Specification: Synthetic Pose Dataset Generation

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `001-dataset-generation`)
**Created**: 2026-10-17
**Status**: Approved (operator, 2026-10-17); amended 2026-10-20 (FR-002 pose source, operator decision)
**Input**: Roadmap slice 001 + operator answers 2026-10-17: "images in, keypoints
as labels; use defaults (SMPL, AMASS-sampled poses, headless pyrender, fixed
named camera ring, all labels saved, render at 256×192, CPU, data outside git,
non-commercial licenses OK)."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate a dataset from one command (Priority: P1)

The researcher runs a single CLI command with a config file and gets a dataset
directory of rendered body images, each paired with its pose, body shape,
camera, 2D/3D keypoints, and silhouette mask, plus a manifest describing it all.

**Why this priority**: Every later slice (loader, baselines, autoencoder)
consumes this output. The predecessor's data could not be regenerated.

**Independent Test**: Run the command on a small config; inspect that every
sample listed in the manifest has an image and all labels, and that a projected
2D keypoint overlay lands on the rendered body.

**Acceptance Scenarios**:

1. **Given** the body-model files and pose source are installed and a config
   requesting N poses × C cameras, **When** the generate command runs, **Then**
   the output contains N×C samples, each with image, mask, and labels, and a
   manifest listing all of them.
2. **Given** a sample, **When** its 3D keypoints are projected with its stored
   camera, **Then** they coincide with its stored 2D keypoints and fall inside
   its silhouette mask (joint-level tolerance).

---

### User Story 2 - Reproduce a dataset exactly (Priority: P1)

The researcher re-runs generation with the same config and seed on another day
or machine and gets the same dataset.

**Why this priority**: Constitution Principle III (reproducibility, non-negotiable).

**Independent Test**: Generate twice with the same config + seed; compare
manifests.

**Acceptance Scenarios**:

1. **Given** the same config, seed, code commit, and input data, **When**
   generation runs twice, **Then** the label data and manifest are identical
   (images identical or within a documented pixel tolerance if the renderer is
   not bit-exact across machines).
2. **Given** a dataset directory, **When** the researcher opens its manifest,
   **Then** it records config, seed, code commit, input data versions, and
   counts.

---

### User Story 3 - Tiny fixture dataset for tests (Priority: P2)

A very small dataset (a handful of samples) is committed to the repo so later
slices' unit tests and CI smoke runs work without the licensed body model or
pose source.

**Why this priority**: CI cannot download SMPL/AMASS (licensed, registration
required); slices 002–004 need data in CI.

**Independent Test**: In a clean checkout with no body-model files, load the
fixture manifest and read every sample.

**Acceptance Scenarios**:

1. **Given** a clean checkout, **When** tests read the fixture, **Then** all
   samples and labels load with the same schema as a full dataset.

---

### Edge Cases

- Body-model or pose-source files missing → fail fast with a message saying
  what to download and where to put it; write nothing.
- Output directory already exists → refuse unless an explicit overwrite flag is given.
- A pose renders with the body partly or fully outside a camera's view → still
  saved; flagged in labels (per-keypoint visibility), not silently dropped.
- Run interrupted → re-running resumes or restarts cleanly; a half-written
  dataset is never marked complete in its manifest.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a CLI command that generates a dataset from a
  config file and an explicit seed.
- **FR-002** *(amended 2026-10-20)*: System MUST pose the SMPL body model with a
  configurable number of poses (default 2,000) from a config-selected pose source:
  - `procedural` (**default**): each body joint's rotation is sampled uniformly,
    per SMPL joint-local axis, within a hardcoded, documented, anatomically
    plausible range table (degrees; versioned, e.g. `proc-v1`), seeded and
    deterministic. Root orientation zero; hand joints zero.
  - `amass` (optional): poses sampled from configured AMASS sub-collections
    after fixed frame-stride subsampling (root zeroed, hands zero).
  The manifest MUST record the pose source and, for `procedural`, the ranges
  table and its version. Body shape is the neutral SMPL model at mean shape.
- **FR-003**: System MUST render each pose from a fixed, named, configurable
  ring of cameras (default 8 at 45° steps, torso height, fixed distance, names
  per Clarification Q2), headless on CPU, as a plain grey body on a black
  background with simple lighting.
- **FR-004**: Each sample MUST include: RGB image (256×192 default, configurable),
  silhouette mask, 3D keypoints, 2D keypoints with per-joint visibility, pose
  parameters, body-shape parameters, camera intrinsics and extrinsics, and
  camera name.
- **FR-005**: System MUST write a manifest recording config, seed, code commit,
  input data identifiers/versions, sample list, and a completion flag.
- **FR-006**: Same config + seed + inputs MUST yield identical labels and
  manifest (images per US2 tolerance).
- **FR-007**: Generated datasets MUST be written outside git; default location
  `data/generated/<dataset-name>/`, which the repo MUST ignore.
- **FR-008**: System MUST include a script/doc stating how to obtain SMPL and
  AMASS (registration, license terms, expected paths).
- **FR-009**: A fixture dataset of ≤ 16 samples MUST be committed, generated by
  the same command from real SMPL renders, using the same schema.
- **FR-010**: Keypoint set MUST be a documented, fixed skeleton definition
  (joint names + order) shared with later slices: SMPL's 24 joints.

### Key Entities

- **Sample**: one (pose, camera) render — image, mask, and all labels; unique ID.
- **Camera**: name, intrinsics, extrinsics; part of a fixed ring defined in config.
- **Pose source record**: where a pose came from — AMASS sequence/frame, or the procedural ranges version and sampled per-joint angles.
- **Skeleton definition**: ordered joint names used for all keypoints.
- **Manifest**: dataset-level record of provenance, samples, and completeness.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: One command produces a complete dataset of N×C samples with zero
  samples missing any label.
- **SC-002**: Two runs with same config + seed produce identical manifests and labels.
- **SC-003**: For 100% of fixture samples, projected 3D keypoints match stored
  2D keypoints within 1 pixel.
- **SC-004**: A default-size dataset generates on a laptop CPU in under ~1 hour.
- **SC-005**: Fixture loads and validates in a clean checkout in under 5 seconds.

## Clarifications

### Session 2026-10-17

All 8 batched questions accepted at their proposed defaults (operator).

- Q1: Default dataset size? → A: 2,000 poses × 8 cameras = 16,000 images.
- Q2: Camera ring? → A: 8 cameras at 45° steps around the body, level with the
  torso, fixed distance; names `Front_0`, `FrontRight_45`, `Right_90`,
  `BackRight_135`, `Back_180`, `BackLeft_225`, `Left_270`, `FrontLeft_315`.
- Q3: AMASS subset? → A: one or two small sub-collections, poses subsampled
  (fixed frame stride) to avoid near-duplicate consecutive frames.
- Q4: Body shape? → A: neutral SMPL body, mean shape (betas = 0); random shapes later.
- Q5: Appearance? → A: plain grey body on black background, simple lighting; no
  textures/clothing.
- Q6: Keypoint skeleton? → A: SMPL's 24 joints.
- Q7: Default output location? → A: `data/generated/<dataset-name>/`, git-ignored.
- Q8: Fixture from real SMPL renders? → A: yes, commit ≤16 low-res derived
  renders generated by the same command.

### Session 2026-10-20

- Q9: Pose source for the MVP? → A (operator decision): skip AMASS for the MVP.
  Default to a procedural source: per-joint rotations sampled uniformly within
  hardcoded, ad hoc but anatomically realistic ranges (the predecessor did
  something similar; its ranges are unavailable, so new ranges are chosen and
  documented per joint, in degrees, around SMPL joint-local axes). Root
  orientation zero, hands/fingers zero, mean shape. AMASS remains an optional,
  config-selectable source (not deleted). The fixture is regenerated with the
  procedural source and needs only SMPL.

## Assumptions

- The operator registers for and downloads SMPL (and, only for the optional AMASS source, AMASS) manually; the tool
  never downloads licensed assets.
- Non-commercial license terms of SMPL/AMASS are acceptable alongside PolyForm NC.
- CPU-only generation is sufficient for MVP sizes; GPU/Kaggle out of scope.
- Real photos, clothing, textures, multiple people, and random backgrounds are
  out of scope for this slice.
