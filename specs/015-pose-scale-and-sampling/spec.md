# Feature Specification: Pose Scale, Latent-Aware Promotion and VAE Sampling

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `015-pose-scale-and-sampling`)
**Created**: 2026-10-22
**Status**: Approved at surrogate scope; operator decisions 2026-10-22 as approval basis
(clarification batch 1 answered by surrogate ruling 2026-10-22, below)
**Input**: Operator decisions 2026-10-22, relayed by the orchestrator: (1) next campaign =
(a) dataset `smpl_proc_v3` = `smpl_proc_v1` + 8,000 procedural poses (config-only like v2d:
v1 table, mean shape, ring cameras, split seed 0) to locate where the more-poses gain
saturates; (b) on v3: plain AE control R0 + R0b (seed 1), VAE β 0.01 (R2), mask-weighted MSE
α 4 (R1); (c) VAE β 0.1 on `smpl_proc_v1` plus a sampling evaluation of the N(0, I) prior with
a simple, defensible quality measure and a figure; (2) a new promotion rule (pixel win OR
latent win) replaces the pixel-only 012 rule for this and later campaigns, amended into the 012
documents and retro-applied to `docs/results/smpl-campaign/README.md`.

Settled context (checked at the branch head, 2026-10-22):

- **Datasets.** v1 = 2,000 poses × 8 ring cameras; split 0.8 / 0.1 / 0.1 by pose
  (`DEFAULT_FRACTIONS`). v3 at 8,000 poses: 64,000 images, 6,400 / 800 / 800 poses, i.e.
  51,200 train and 6,400 val / test images of 256 × 192 (49,152 px).
- **Trainer memory** (`train/trainer.py`): `load_split` holds train as uint8
  (51,200 × 49,152 B = 2.52 GB) and val as uint8 image + bool mask (0.63 GB); during scoring
  train is deleted and each scored split is converted whole to float32 (`to_float`, 1.26 GB per
  split; val + test ≈ 3.8 GB together). With `context_pca_k: null` the AE peak is ≈ 4 GB:
  fits 15 GB, no trainer change needed.
- **Baselines memory** (`eval/baselines.py::load_arrays`): train images are held as
  **float32** `[n, H*W]` even with `image_pca_ks: []`: 51,200 × 49,152 × 4 B = 10.1 GB, plus
  val / test, ≈ 12-13 GB peak: does **not** safely fit 15 GB (v2d ≈ 6 GB). Fix: FR-004.
- **Probe memory** (`eval/probe.py::extract`): per-batch encoding; caches latents
  (64 floats) and labels only, ≈ 0.1 GB at 64,000 images. Fits.
- **VAE** (`models/autoencoder.py`): `encode_dist → (mu, logvar)`, `decode(z)`; evaluation
  and probe use μ. `configs/ae_vae_beta0.1.yaml` exists (v1, run name
  `ae-vae-beta0.1-smpl_proc_v1`, guards set). No sampling code exists.
- **Campaign results** (`docs/results/smpl-campaign/README.md`): R0 val MSE 0.00218, R0b
  −0.6 % (seed gap); probe world MPJPE val R0 174.8, R0b 172.0 mm (probe gap 2.8 mm); R1
  +8.4 % MSE, 162.4 mm; R2 +0.2 %, 174.4 mm; v2d −18 %, 150.2 mm.
- **Timing.** v2d ≈ 2× v1 per epoch; v3 ≈ 4× v1, ≈ 330 s/epoch on 4 vCPU, so a guarded
  AE run (early stop near epoch 45-55) ≈ 4-5 h; four v3 AE runs + probes ≈ 18-22 h queued.
  Generation ≈ 20 min, ≈ 1.4 GB on disk.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The 8,000-pose dataset and its control (Priority: P1)

The researcher generates, validates and splits `smpl_proc_v3` from committed configs and runs
its baselines, R0-v3 and R0b-v3 with probes inside 15 GB, so the v1 → v2d → v3 curve (1,600 /
3,200 / 6,400 train poses) shows whether the pose-count gain saturates.

**Independent Test**: `configs/smpl_proc_v3/*` differ from their v2d counterparts only in names,
paths and `num_poses: 8000`; the baselines run on a v3-sized synthetic memory check stays under
the FR-004 bound; the campaign results table gains v3 rows.

**Acceptance Scenarios**:

1. **Given** the committed v3 configs, **When** generate / validate / split (seed 0) run,
   **Then** the split has 6,400 / 800 / 800 poses and its content hash is recorded.
2. **Given** v3, **When** baselines, R0-v3, R0b-v3 and their probes run, **Then** each finishes
   without OOM and the README reports MSE, IoU, region MSE, probe MPJPE and the v3 seed / probe
   gaps.

### User Story 2 - Variants at scale (Priority: P1)

R2 (VAE β 0.01) and R1 (mask α 4) are rerun on v3 and judged against R0-v3 / R0b-v3 with the
new rule, to see whether the v1 readings (R2 free, R1 trades pixels for pose) hold with 4× data.

**Independent Test**: each v3 variant config differs from its v1 counterpart only in names /
paths and from `configs/smpl_proc_v3/ae.yaml` only in the variant keys.

### User Story 3 - Latent-aware promotion rule (Priority: P1)

A variant that keeps pixels near the control but clearly improves the pose latent is promoted,
not dropped; the rule is in the 012 documents and the finished SMPL campaign is re-scored.

**Independent Test**: a pure function over (variant, R0, R0b) numbers returns WIN-pixel /
WIN-latent / DROPPED / REPORTED, unit-tested on the campaign numbers.

### User Story 4 - Sampling from the VAE prior (Priority: P2)

For R2b (β 0.1, v1) and the R2 runs, the researcher decodes N samples from N(0, I) and gets a
quality measure plus a figure, so "sampleable latent" is measured, not asserted.

**Independent Test**: on a fixture VAE checkpoint, the sampling command writes `samples.json`
and `samples.png` deterministically for a seed; on a plain AE checkpoint it refuses.

### Edge Cases

- Baselines or eval OOM on v3: FR-004 fixes baselines; the trainer path is measured (FR-005).
- A container restart mid-run: resume as in the 012 procedure (watchdog retry pass).
- VAE β 0.1 collapses (last train KL < 1 nat/image): the run is reported as collapsed; sampling
  still runs and is reported, but no promotion.
- A decoded sample with no foreground pixel above the IoU threshold: counts as invalid (0
  components).
- R0-v3 / R0b-v3 gap larger than on v1: the rule uses the v3 gaps for v3 variants.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** `configs/smpl_proc_v3/{generate,baselines,ae,probe}.yaml`: copies of the v2d
  files with `smpl_proc_v3` names / paths and `num_poses: 8000`; `context_pca_k: null`,
  `image_pca_ks: []`; 012 guards as v2d. Covered by the existing realism config tests.
- **FR-002** v3 run configs `ae_seed1.yaml` (R0b), `ae_vae.yaml` (R2, β 0.01), `ae_mask.yaml`
  (R1, α 4) and probe configs, each = the v3 `ae.yaml` + the v1 variant keys; run names
  `ae-seed1-smpl_proc_v3`, `ae-vae-smpl_proc_v3`, `ae-mask-smpl_proc_v3`.
- **FR-003** R2b: `configs/ae_vae_beta0.1.yaml` as committed (v1); its probe on μ.
- **FR-004** Baselines memory: when `image_pca_ks` is empty, train images are not materialized
  as float32 (mean image from uint8 / per-batch, as the 012 batched mean); peak for v3 ≤ 6 GB.
  Numbers bit-identical to the current path on the fixture (tested).
- **FR-005** Measured peak memory: every run command (train, baseline, probe, sample) records
  its process peak RSS (`peak_rss_mb`, from `resource.getrusage`) in `run_info.json` at the
  end. The queue runs v3 baselines and R0-v3 first and stops the v3 part (log `MEMORY STOP`)
  if either peak exceeds 12,288 MB; the peaks go into the README.
- **FR-006** Promotion rule (pure function, `eval/promotion.py` or the campaign script),
  with `seed_gap = |MSE(R0b) − MSE(R0)| / MSE(R0)`, `probe_gap = |MPJPE(R0b) − MPJPE(R0)|`
  (world-frame MLP, val), each on the variant's own dataset:
  - WIN (pixel): ΔMSE < −seed_gap and ΔMPJPE ≤ +probe_gap;
  - WIN (latent): ΔMSE ≤ +5 % and ΔMPJPE < −2 × probe_gap;
  - DROPPED: ΔMSE > +5 % and not a latent win; else REPORTED (not carried).
  VAE variants additionally need last train KL ≥ 1 nat/image to win (012 rule kept).
- **FR-007** Docs: 012 `spec.md` / `tasks.md` and the HANDOFF campaign procedure carry the new
  rule (old rule marked superseded 2026-10-22); `docs/results/smpl-campaign/README.md` gains a
  "015 verdict" column for the variants R1, R2, R3 (datasets v2a-v2d get no verdict: they are
  dataset changes judged against their own baselines, not variants). Under FR-006 the v1
  verdicts are: R1 DROPPED (+8.4 % MSE > +5 %; the README notes it misses the bound with a
  probe gain of 12.4 mm, 4.4× the 2.8 mm gap; Q1); R2 REPORTED (+0.2 %, 174.4 vs 174.8 mm:
  neither a pixel nor a latent win; it was a WINNER under the superseded 012 rule, whose VAE
  clause only needed "not worse"); R3 REPORTED (+1.5 %, probe 10 mm worse). The β 0.1 run
  (R2b) is an operator decision, not a consequence of R2's verdict.
- **FR-008** Sampling: `strike-a-pose-eval sample --checkpoint … --n 1024 --seed 0` on a VAE
  checkpoint decodes z ~ N(0, I) (fixed generator) and reports: (i) **valid fraction**: the
  share of samples whose silhouette (pixels > `iou_threshold`) has exactly one 8-connected
  component holding ≥ 95 % of the foreground, with foreground area inside the [1st, 99th]
  percentile of the train masks' areas; (ii) **mean NN-MSE**: each sample's minimum MSE to the
  train images (batched, uint8 train in memory); with two references: the same measures for μ
  reconstructions of 1,024 val images (upper bar) and for N(0, I) samples decoded by R0's
  plain AE decoder (lower bar). Figure: 4 × 8 grid of samples, each with its nearest train
  image. Writes `samples.json` + `samples.png` in a run dir. No new dependency
  (connected components by a small torch/numpy labelling, Q3).
- **FR-009** Campaign queue: the 012 queue / watchdog scripts run the 015 order (Q2);
  `scripts/run_status.py` lists the new runs.

### Key Entities

- **Promotion verdict**: {variant, dataset, ΔMSE %, ΔMPJPE mm, seed_gap, probe_gap, verdict}.
- **Sampling report**: {checkpoint, n, seed, valid fraction, area bounds, mean / median
  NN-MSE, reference values}.

## Success Criteria *(mandatory)*

- **SC-001** v3 baselines, four AE runs and four probes finish within 15 GB (peaks logged).
- **SC-002** The README states the v1 / v2d / v3 MSE and MPJPE curve and whether v3's gain over
  v2d is > 3× the v3 seed gap (not saturated) or not.
- **SC-003** Every run of both campaigns has a verdict under FR-006, unit-tested on the numbers.
- **SC-004** R2b and both R2 runs have a sampling report with both reference bars.

## Assumptions

- Mean shape, v1 table and ring cameras for v3 (operator); val / test are a new pose
  permutation (as v2d), so cross-dataset comparisons are by relative change to each dataset's
  own baselines and control.
- 4 vCPU, 15 GB RAM, no swap; Python 3.11; no new dependencies.

## Clarifications

### Batch 1 (surrogate ruling 2026-10-22)

Answers: Q1 → (a): rule as worded; R1 stays DROPPED; the README notes it misses the +5 %
bound with a probe gain of 12.4 mm (≈ 4.4× the gap). Q2 → (a). Q3 → (a). FR-004 and FR-005
accepted. Questions as asked:

1. **R1 on v1 under the new rule.** R1's val MSE is +8.4 % vs R0, above the +5 % bound of the
   latent win, so the rule as worded makes R1 DROPPED, not a latent winner (the operator's
   expected outcome). Options: (a) apply the rule as worded; R1 stays DROPPED and the README
   says the latent win just misses; (b) widen the latent-win bound to +10 %; (c) latent win
   with no MSE bound when IoU / region MSE are reported. Default: (a), a pre-registered rule
   should not be tuned to one result; the README flags R1 as the case to revisit.
2. **Queue order.** (a) R2b + sampling on v1 first (≈ 1.5 h, independent of v3), then v3
   generate / baselines / R0 / R0b / R2 / R1; (b) v3 first. Default (a): the cheap run gives
   the sampling evaluation a real checkpoint early and de-risks FR-008 before the 20 h queue.
3. **Sampling quality measure.** (a) FR-008 as written (valid fraction + NN-MSE, with AE-prior
   and μ-reconstruction bars); (b) add a probe-based pose plausibility (encode the samples, MLP
   joints, fraction within the train joint ranges); (c) only NN-MSE. Default (a): simple,
   label-free, defensible; (b) adds a circular dependency on the same encoder.
