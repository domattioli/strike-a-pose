# Feature Specification: Generative Pose VAE

**Feature Branch**: `021-generative-pose-vae` (slice `021-generative-pose-vae`)
**Created**: 2026-11-03
**Status**: Approved at operator scope 2026-11-03 (Fable scoping; operator: "take it and run").
Clarification batch 1 answered by surrogate ruling 2026-11-03 (below); the operator may overrule.
**Input**: Project goal line (operator 2026-11-03): "Encode, reconstruct and generate: one latent
≤ 128 numbers whose pose head/probe is ≤ 80 mm MPJPE, whose reconstructions are within +5 % MSE
of the 64-d plain-AE seed mean, and whose N(0,I) samples are valid bodies ≥ 59 % of the time."
Latent cap 128.

Settled context (read-only from `/home/user/strike-a-pose/runs`, 2026-11-03):

- L2 (128-d plain AE + pose head λ 0.03, `ae-l128-pose-l0.03-smpl_proc_v3-20261028T101221Z`)
  and its seed1 (`…-seed1-…-20261029T055258Z`): head 67.0 / 64.7 mm (seed1 66.8 / 65.0);
  image MSE val 0.00166906 / 0.00168247, test 0.0016472 / 0.00165959 (means val 0.001675765,
  test 0.001653395). A plain AE: no prior, so it cannot generate.
- R0-v3 64-d plain-AE seeds (R0, R0b, R0c) val MSE 0.00157751 / 0.0015873 / 0.0015731 (mean
  0.00157930), test 0.00155548 / 0.00156344 / 0.00155676 (mean 0.00155856).
- R2b-v3 (64-d VAE β 0.1, `ae-vae-beta0.1-smpl_proc_v3-20261024T065708Z`): val MSE 0.00160545,
  test 0.00158037; prior valid fraction 0.589844 (sample run
  `sample-vae-beta0.1-smpl_proc_v3-20261024T143458Z`, n 1024); probe
  `probe-smpl_proc_v3-20261024T134824Z`; never round-tripped.
- R2-v3 (64-d VAE β 0.01) round trip `roundtrip-vae-smpl_proc_v3-20261024T222459Z`: prior
  valid 0.324, plausible 0.457 vs shuffled-pose control 0.326 (margin 13.1 points).
- Code today: `train/config.py` refuses `pose_weight > 0` and `gradient_weight > 0` with
  `variational: true`; `train/trainer.py::train_epoch` puts the pose / gradient terms only in
  the non-VAE branch.

## Clarifications

### Session 2026-11-03 (batch 1, surrogate rulings; operator may overrule)

- Q1 Pixel bound of the verdict: the goal line says "+5 % of the 64-d plain-AE seed mean"
  (val ≤ 0.00165827), the operator's verdict rule says "V1 MSE ≤ L2 MSE + 5 %" (val ≤
  0.00175955). They differ (L2 itself is +6.1 % over the 64-d mean). **Ruling: the verdict
  uses the rule as fixed (L2 mean + 5 %)**; the goal-line comparison is reported as an
  informational column ("vs R0 64-d", PASS / FAIL), never gated. Flagged to the operator.
- Q2 Where the head reads: **μ** (deterministic encoder mean, as 018 / 019), before any
  sampling; never `z`. Pose evaluation already encodes through `model.encode` = μ, so train and
  eval read the same quantity. Head-on-z is the named follow-up if generation fails.
- Q3 Values: **β 0.1, λ 0.03, latent 128**, one seed each. β 0.1 is the only β with a measured
  prior valid fraction near the bar (0.590 at 64-d); λ 0.03 is L2's. No sweep fits the budget.
- Q4 Loss: one composition for every model:
  `loss = recon [+ γ·GDL] [+ β·KL/(H·W)] [+ λ·pose(head(μ))]`, terms added in this order,
  each only when on. With every existing combination this is the same float operations in the
  same order as today, so all shipped configs stay bitwise identical (tested against a frozen
  copy of the pre-021 loop).
- Q5 Gradient loss with the VAE: refusal **lifted** (and tested on fixtures), **not run**
  (deferred to 022).
- Q6 Report: a **new small module** `eval/generative.py` + `scripts/generative.py report`,
  not an extension of `fidelity.py` / `pose_track.py`: those judge against seed-sd of
  references; this verdict is fixed thresholds over three artifact kinds (train metrics, sample
  run, round-trip run). ~150 lines, pure functions tested on hand numbers.
- Q7 Generation axes: prior valid fraction from the sample run (`sets.vae_prior.valid_fraction`,
  n 1024); round-trip margin = 100·(`plausible_fraction` − `control_plausible_fraction`) of
  `sets.vae_prior.round_trip` in the round-trip run, in percentage points. Both are split-free;
  head and MSE are checked on val **and** test.
- Q8 Thresholds (operator, fixed before runs): head ≤ 80 mm; MSE ≤ L2 mean · 1.05; valid ≥
  0.56 (59 % − 2 binomial sd at n 1024: sd 0.0154); margin ≥ 13 points (R2-v3's measured 13.1).
  Inclusive bounds with a 1e-12 tolerance.
- Q9 V2 vs V1: "does the head help or hurt sampling" = two-proportion test on prior valid
  fraction: diff = V1 − V2, sd = √(p₁q₁/n₁ + p₂q₂/n₂); **HELPS** if diff > 2 sd, **HURTS** if
  diff < −2 sd, else **NO-DIFFERENCE**. Round-trip margins of both printed beside it, not gated.
- Q10 R2b-v3 retroactive round trip: context only (does β 0.1 at 64-d already clear 13
  points?); printed as an extra row, never enters the verdict.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train a VAE with a pose head (Priority: P1)

The researcher sets `variational: true, beta: 0.1, pose_weight: 0.03`; the run trains with
recon + β·KL/(H·W) + λ·pose(head(μ)), logs `train_kl` and `train_pose`, and scores
`pose_head.mpjpe_mm` as for L2.

**Independent Test**: fixture runs; every pre-021 combination bitwise equals a frozen copy of
the pre-021 epoch loop; the VAE + pose run repeats, resumes equal to uninterrupted, and its
head input is μ.

**Acceptance Scenarios**:

1. **Given** any shipped config, **When** it trains, **Then** losses and weights are bitwise
   today's (no new history column, no new RNG draw).
2. **Given** a fixture config with `variational, beta 0.1, pose_weight 0.03`, **Then**
   `train_loss == train_recon + 0.1·train_kl/(H·W) + 0.03·train_pose` per row (1e-6 rel),
   `metrics.json.splits.val.pose_head.mpjpe_mm` finite, probe and sample accept `best.pt`.

### User Story 2 - Goal verdict report (Priority: P1)

`scripts/generative.py report` reads V1 / V2 train runs, their sample and round-trip runs, the
L2 seeds and the R0 seeds, prints one Markdown table per split and a final verdict line.

### User Story 3 - Campaign queue (Priority: P2)

`docs/campaign-scripts/vae021_queue.sh` trains and probes V1 then V2, samples both, round-trips
V1, V2 and R2b-v3; idempotent, resumable, shared lock, 12 GB gate.

### Edge Cases

- V1 aborted (divergence guard) or missing any artifact → verdict `INCOMPLETE` naming it.
- V1 run without `pose_head` in `metrics.json` → pose axis `None` → `INCOMPLETE`.
- Resume of a pre-021 run dir: no new config key, nothing to fill.
- `gradient_weight > 0` with `mask_weight > 0` stays refused (unchanged).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** `train/config.py`: delete the two refusals `pose_weight … not with variational`
  and `gradient_weight … not with variational`; nothing else in validation changes.
- **FR-002** `train/trainer.py::train_epoch`: one loss composition (Q4); head reads μ; KL,
  pose, gradient accumulators as today. Docstring cites spec 021 FR-002.
- **FR-003** Defaults and every pre-021 combination bitwise unchanged (test against a frozen
  pre-021 loop in the test file).
- **FR-004** `eval/generative.py` + `scripts/generative.py` (data-model.md): constants
  `HEAD_MAX_MM 80.0`, `MSE_COST_REL 0.05`, `PRIOR_VALID_MIN 0.56`, `MARGIN_MIN_PTS 13.0`,
  `COMPARE_SD 2.0`; verdict `GOAL-MET` / `PARTIAL` / `INCOMPLETE`, failing axes in order
  pose, pixel, valid, margin; next step "beta 0.03" for pose / pixel, "head on z" for
  valid / margin.
- **FR-005** Configs `configs/smpl_proc_v3/ae_l128_vae_beta0.1_pose_l0.03.yaml` (V1),
  `ae_l128_vae_beta0.1.yaml` (V2) — L2's file with `variational: true, beta: 0.1` (V2:
  `pose_weight: 0.0`).
- **FR-006** Sample configs `sample_l128_vae_beta0.1_pose_l0.03.yaml`,
  `sample_l128_vae_beta0.1.yaml` (n 1024, reference L1 128-d plain AE) and round-trip configs
  `roundtrip_l128_vae_beta0.1_pose_l0.03.yaml`, `roundtrip_l128_vae_beta0.1.yaml` (placeholders
  overridden by the queue).
- **FR-007** Retroactive R2b-v3 round trip `configs/smpl_proc_v3/roundtrip_vae_beta0.1.yaml`
  (concrete paths, verified to exist 2026-11-03).
- **FR-008** Queue `docs/campaign-scripts/vae021_queue.sh` (lead, committed with the design).
- **FR-009** Docs: HANDOFF "## 021" section, roadmap bullet (lead); results README after the
  campaign.

### Key Entities

`GenEntry`, axis checks, verdict tuple, prior comparison (data-model.md).

## Success Criteria *(mandatory)*

- **SC-001** Full suite green; pre-021 combinations bitwise unchanged.
- **SC-002** Campaign ≈ 15 h (≤ 18 h), peak RSS < 12 GB.
- **SC-003** Verdict line for V1 on both splits under the Q8 thresholds, the V2-vs-V1 sampling
  comparison, and the R2b-v3 round-trip context row. The goal-line pixel column is reported.
