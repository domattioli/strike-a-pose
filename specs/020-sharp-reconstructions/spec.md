# Feature Specification: Sharper Reconstructions

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `020-sharp-reconstructions`)
**Created**: 2026-10-28
**Status**: Approved at surrogate scope; operator decision 2026-10-28 ("a parallel effort to
improve fidelity of the renderings") as approval basis, read as the autoencoder's
reconstructions (see research.md R1 for why not the dataset renderer). Clarification batch 1
answered by surrogate ruling 2026-10-28 (below); the operator may overrule.
**Input**: The plain conv AE on `smpl_proc_v3` (R0-v3) reaches val MSE 0.00158, IoU 0.905;
its reconstructions are blurry, and pose supervision (018 P2, λ 0.03) makes them blurrier
(IoU 0.870, MSE +36 %, region MSE hands 0.025 → 0.033, feet 0.022 → 0.032, head 0.013 →
0.017). Mask weighting (011 R1) made pixels worse (+14.5 %, IoU 0.856). Make the
reconstructions sharper without buying it with pose information, and measure sharpness.

Runs **after** slice 019 (latent 128 + pose head, designed in parallel) on the same machine;
020 changes only the reconstruction loss and the image metrics, never the latent or the head.

Settled context (cheap checks at `dd61515`, 320 val images = every 20th, R0-v3 and P2
`best.pt`; research.md R2–R3):

- The blur is low-pass: the reconstruction keeps 72 % (R0-v3) / 71 % (P2) of the true mean
  gradient magnitude; squared gradient error (both axes summed) 0.00232 (R0-v3) vs 0.00383
  for an all-black prediction, i.e. 40 % of the true gradient energy is recovered.
- Silhouette boundaries: boundary F1 at 1 px tolerance 0.929 (R0-v3) vs 0.861 (P2); at 2 px
  0.978 vs 0.951 (near ceiling). The 1-px metric separates the two models clearly.
- Figures (`recon_val.png` of R0-v3 and P2): soft contours, hands and feet shortened or
  missing, fingers and toes never present, thin raised arms faded, folded limbs smeared into
  a dark ghost; no checkerboard pattern from the transposed convolutions.

## Clarifications

### Session 2026-10-28 (batch 1, surrogate rulings; operator may overrule)

- Q1 What does "fidelity of the renderings" mean? (a) AE reconstructions; (b) the synthetic
  dataset renderer (`datagen/render.py`). **Ruling (a)**: the numbers the request cites (IoU
  0.905 → 0.870, MSE +36 %) are reconstruction metrics; the dataset images are the sharp
  ground truth (mesh render, R1); no renderer complaint is on record. Flagged to the operator.
- Q2 Loss candidate(s): (a) squared image-gradient difference (GDL) alongside MSE; (b) SSIM /
  MS-SSIM; (c) BCE on the sigmoid output; (d) a perceptual (VGG) loss. **Ruling (a) only**.
  Blur is the MSE's regression to the conditional mean; a gradient term directly penalises the
  missing high-frequency energy measured in R2, is 6 lines of torch, has the same units as MSE
  (squared intensity differences) so its weight reads as a ratio, and needs no window
  hyper-parameters (SSIM) or pretrained network / dependency (VGG: also not licence-checked).
  BCE on grey shaded targets has its minimum at the same mean as MSE (only the gradient scale
  differs), so it does not remove the averaging that causes blur. Rejected ones: research R4.
- Q3 Decoder change (nearest-upsample + conv instead of transposed conv; more decoder
  capacity)? **No (this slice)**. The figures show no checkerboard, so the transposed convs are
  not the visible failure; blur appears with an unchanged decoder when the latent is shared with
  pose (P2), pointing at the 64-number bottleneck + MSE, not decoder capacity. The budget holds
  two runs; a decoder run is the follow-up if the loss change fails (research R5).
- Q4 Skip connections (U-Net)? **No, by construction**: skips would let the image bypass the
  latent, making the reconstruction sharp for reasons the latent does not carry; the latent is
  the product (probe, sampling, pose). Only the latent → decoder path may produce pixels.
- Q5 Weight γ of the gradient term: total loss = `MSE + γ·GDL (+ λ·pose)`. Calibration (R3):
  GDL ≈ 1.8 × MSE at R0-v3 convergence. **γ = 0.5** puts the term at ≈ 0.9 × the MSE term,
  "balanced", the same reading as 018's λ calibration. One value; a second γ does not fit.
- Q6 Sharpness metric: **boundary F1 at 1 px tolerance** of the thresholded silhouette (pred >
  `iou_threshold` vs true mask; FR-002). It is not the trained quantity (the loss acts on
  intensity gradients, the metric on binarised contours), so it is not optimised directly;
  it captures shortened limbs (missing contour) and soft edges (contour displaced by the
  threshold), and is cheap (two max-pools). Rejected: gradient-magnitude error (that is what
  GDL trains; reported only in research as a diagnostic), edge-F1 on Canny maps (thresholds,
  scikit-image dependency).
- Q7 Variants (≤ 18 h): **F1** plain AE + γ 0.5 (`configs/smpl_proc_v3/ae_grad.yaml`);
  **F2** pose λ 0.03 + γ 0.5 (`ae_pose_l0.03_grad.yaml`). References (already trained, only
  rescored): R0-v3, R0b-v3, R0c-v3 (plain seeds), P2, P2-seed1 (pose λ 0.03 seeds).
- Q8 Verdict rule (pre-registered, per split, FR-005). Against its reference mean (F1: the
  three plain seeds; F2: the two P2 seeds), with sd from the three plain seeds (ddof 1):
  **SHARPER** iff Δ boundary F1 > +2 sd AND Δ whole-image MSE ≤ +5 % AND Δ probe MPJPE ≤
  +2 probe sd; **SHARPER-COSTLY** iff the F1 gain is real but MSE or probe fails;
  **NO-GAIN** otherwise. IoU and region MSE (hands, feet, head) are reported, not gated.
  The 015 promotion rule and the 018 pose track are unchanged and also run.
- Q9 Combination with `mask_weight` / `variational`: refused (`ConfigError`); neither is in
  the campaign and mask weighting already failed on pixels.

### Operator rulings 2026-10-28

- Q2 (operator ruling): if F2 is SHARPER-COSTLY (probe worse than 2 sd), it does not count as a
  win for the pose system. The verdict rule (Q8) stays as fixed; a decoder-upsampling run
  (research R5) is the next lever.
- Q1 (operator ruling 2026-10-28): "renderings" means the AE reconstructions (ruling a). The
  dataset renderer is out of scope.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train with a gradient term from a config (Priority: P1)

The researcher sets `gradient_weight: 0.5`; the run adds γ·GDL to the reconstruction loss,
logs the epoch mean of the GDL term as `train_grad`, and selects / early-stops on val pixel
MSE as before.

**Independent Test**: fixture runs — `gradient_weight: 0` is bitwise today's trainer; `> 0`
repeats exactly and resumes equal to uninterrupted.

**Acceptance Scenarios**:

1. **Given** any shipped config, **When** it trains, **Then** history (minus `epoch_s` and
   the new empty `train_grad`), weights and pre-existing metrics are bit for bit today's.
2. **Given** a fixture config with `gradient_weight: 0.5`, **When** it trains twice, **Then**
   `history.csv` (minus timings) and `metrics.json` are identical; `train_grad` is finite and
   `train_loss == train_recon + 0.5·train_grad` per row (1e-6 relative).

### User Story 2 - Sharpness metric everywhere images are scored (Priority: P1)

Every `score_images` result (train runs, baselines) gains `boundary_f1`.

### User Story 3 - Rescore and fidelity report (Priority: P1)

`scripts/fidelity.py rescore` scores an existing checkpoint (whole-image MSE, IoU, boundary F1,
region MSE) into `<run>/fidelity.json`; `scripts/fidelity.py report` prints, per split, each
variant against its reference with Δ, Δ/sd, probe MPJPE and the Q8 verdict.

### User Story 4 - Campaign queue (Priority: P2)

`docs/campaign-scripts/fidelity020_queue.sh` rescores the five references, trains and probes
F1 then F2, rescores both, idempotent and resumable, under the shared lock
`data/smpl_queue.lock` (so it never runs beside the 019 queue).

### Edge Cases

- `gradient_weight < 0` → `ConfigError`; `> 0` with `mask_weight > 0` or `variational: true`
  → `ConfigError`.
- Resume of a pre-020 run directory → `LEGACY_VALUES["gradient_weight"] = 0.0`; old history
  rows (no `train_grad`) → `_complete_row` fills `None`.
- Boundary F1 with both contours empty → 1.0; exactly one empty → 0.0.
- `fidelity.py report` with < 2 `--seed` → exit 2; a run without `fidelity.json` → exit 2
  naming it; a run without `probed.txt` → probe columns "n/a" and verdict "NO-PROBE".

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** Config key `gradient_weight: float = 0.0` (≥ 0, finite; `LEGACY_VALUES` 0.0;
  refused with `mask_weight > 0` or `variational`). `Objective` gains `gradient_weight`.
- **FR-002** `eval/metrics.py`: `gradient_loss(y, x)` (data-model.md; torch, differentiable)
  and `boundary_f1(pred, mask, threshold=IOU_THRESHOLD, tol_px=1)` per image; constants
  `BOUNDARY_TOL_PX = 1`.
- **FR-003** Training: loss = `mse + gradient_weight·gradient_loss(y, x) (+ pose term)`;
  computed only when `gradient_weight > 0` (no extra op, no RNG draw otherwise); history column
  `train_grad` appended after `train_pose` (empty when off); `train_recon` stays the MSE term;
  `eval_loss` / selection / early stop / divergence guard unchanged (pixel MSE, total train
  loss as today).
- **FR-004** `score_images` gains `boundary_f1` (mean per image) at the same threshold as IoU;
  units entry `boundary_f1` in `baselines.UNITS` and the trainer's `IMAGE_UNITS`;
  `METRICS_VERSION` unchanged (additive key).
- **FR-005** `eval/fidelity.py` + `scripts/fidelity.py` (`rescore`, `report`): rescore writes
  `fidelity.json` (schema data-model.md) without retraining; report implements Q8 with
  `REAL_GAIN_SD = 2.0`, `MSE_COST_REL = 0.05`, `--split {val,test,both}`.
- **FR-006** Configs `configs/smpl_proc_v3/ae_grad.yaml`, `ae_pose_l0.03_grad.yaml`; queue
  `docs/campaign-scripts/fidelity020_queue.sh` (lead, committed with this design).
- **FR-007** Docs: HANDOFF names the 020 queue and report; results go to
  `docs/results/020-sharp-reconstructions/README.md` (lead, after the campaign).

### Key Entities

`Objective.gradient_weight`, `gradient_loss`, `boundary_f1`, `fidelity.json`, fidelity-report
row (data-model.md).

## Success Criteria *(mandatory)*

- **SC-001** Default configs bitwise unchanged; full suite green.
- **SC-002** Campaign ≈ 14.5 h (≤ 18 h), peak RSS < 12 GB, two probes, seven rescores.
- **SC-003** Fidelity report for F1 and F2 on val and test with the Q8 verdict, plus the
  015 promotion and 018 pose-track tables; F2's probe reported so a sharpness gain is never
  bought silently with pose. No other success threshold is pre-registered.
