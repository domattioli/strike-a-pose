# Feature Specification: Training Variants (regularization, mask-weighted MSE, VAE)

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `011-training-variants`)
**Created**: 2026-10-21
**Status**: Implemented 2026-10-21 (verification: [tasks.md](tasks.md) T013; runs: its
experiment procedure, needs SMPL). Approved at surrogate scope (surrogate rulings 2026-10-21;
operator sign-off pending). Approval basis: the operator's #16 choice quoted below; the surrogate ruled (a) on
Q1–Q3 and accepted every lead default.
**Input**: Operator answer to roadmap open question #16 ("Training improvements / VAE (blur,
lost extremities, mild overfitting in 004)?"), options (ii)–(iv), chosen 2026-10-20 from
`docs/HANDOFF.md` ("Operator decisions requested", #16) and relayed by the orchestrator:
"(ii) regularization (weight decay, small pixel-shift augmentation, bottleneck dropout),
(iii) loss shaping (mask-weighted MSE first; a silhouette term deferred), (iv) a VAE variant
(KL-regularized latent, small beta, enabling sampling/interpolation later), each as a
config-selectable option of the existing trainer/model whose defaults reproduce today's run
bit for bit, with the plain AE as control." Option (i) landed as slice 009; (v) (capacity
sweep) is not chosen.
Orchestrator brief (Fable surrogate scope): spec and batched clarifications first; "done" is
code + tests + docs + configs + a documented experiment procedure, not numbers (no SMPL
model, dataset or checkpoint in this container). Input: a read-only architect memo
(2026-10-20, session scratchpad), verified below; its `trainer.py` / `config.py` line
numbers predate 008/009, its content holds except where noted.

Settled context (checked at `19479e2` with read-only experiments in the locked environment,
torch 2.14.1, Python 3.13):

- **Training loop.** `train_epoch` shuffles with its own generator, scales each uint8 batch
  `/255`, steps `F.mse_loss(model(x), x)` with `torch.optim.Adam(params, lr)` (built in
  `_train_and_evaluate`) and a per-step cosine `LambdaLR`, and returns `(mean loss, last
  lr)`; train is held as uint8 without masks. `eval_loss` (val, eval mode, per-image MSE)
  drives `best.pt` and the 008 selection; the scored model is `predict_rows` behind the 009
  seam `score_images(…, labels=…)` (also the figure). Every RNG is saved per epoch, so
  `--resume` equals an uninterrupted run on the same machine and thread count.
- **Config.** `TrainConfig` is a strict flat YAML mapping; a resume compares
  `with_legacy_values(ckpt["config"])` with the re-parsed `config.yaml`, so every new key
  needs its pre-key value in `LEGACY_VALUES` (008) or older run dirs stop resuming.
- **Model.** `ConvAutoencoder(image_size, latent_dim, channels)`: `encoder` =
  `Sequential(conv blocks…, Flatten, Linear(3072, 64))`, `decoder` ends in a 1-channel
  sigmoid (shaded grey 0.15–0.8 on black); 3,126,897 parameters. `load_checkpoint` rebuilds
  it from `ckpt["model_hparams"]`; the probe (006) loads any `checkpoint.pt` / `best.pt`
  that way and uses `encode` only; its random-encoder control is built from
  `(image_size, latent_dim, channels)` under the probe seed.
- **Measured.** `AdamW(weight_decay=0)` gives bitwise the same parameters as `Adam` (5 steps,
  CPU); a saved `Adam` state loads into `AdamW` (`decoupled_weight_decay` forced true,
  `weight_decay` 0). `encoder[-1](encoder[:-1](x))` equals `encoder(x)` bitwise. A
  `Linear(3072, 64)` log-variance head adds 196,672 parameters (3,323,569 in all).
  `F.dropout(p=0, training=True)` draws no RNG. The general weighted formula at α = 0 equals
  `F.mse_loss` bitwise in forward *and* backward at the full batch shape (64 × 1 × 256 × 192,
  channels-last); the memo's "backward kernels differ" did not reproduce, but the explicit
  α = 0 branch is kept as a guarantee (FR-003).
- **Mask without masks.** The 001 renderer writes body pixels as
  `round(255·(0.15 + 0.65·shade)) ≥ 38` and sets the mask at exactly those pixels on a zero
  background, so `x > 0` is the silhouette (memo: 16 / 16 fixture samples; body ≈ 7.5 % of
  the pixels); 010's options change poses / shapes / cameras, not the renderer.
- **Budget.** 4 vCPU; `configs/ae.yaml` ≈ 85 min per run, probe ≈ 8 min; pytest ≈ 36 s;
  the fixture smoke train (10 epochs × 2 steps) ≈ 3–4 s (constitution IV: < ~2 min).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Today's run is the control, unchanged (Priority: P1)

The operator trains with `configs/ae.yaml` (or any pre-011 config) and gets exactly the
numbers the pre-011 code gives; a run directory started before 011 still resumes.

**Why this priority**: operator constraint ("defaults reproduce today's run bit for bit");
the re-baseline run R0 is the control of every variant.

**Independent Test**: regression test of `train_epoch` against an inline copy of today's
loop; same-machine before/after fixture comparison; the legacy resume test.

**Acceptance Scenarios**:

1. **Given** a config without any 011 key, **When** it trains, **Then** weights, optimizer
   state tensors, `history.csv` losses / lr and every pre-011 `metrics.json` value equal the
   pre-011 code's; the only differences are the additive keys and columns of FR-009.
2. **Given** a run dir whose `config.yaml`, `checkpoint.pt` and `best.pt` predate 011,
   **When** `--resume` runs, **Then** it continues with the legacy values (FR-002) and
   finishes like an uninterrupted pre-011 run.

### User Story 2 - Mask-weighted MSE (Priority: P1)

The operator sets `mask_weight: 4` so errors on body pixels weigh 5× those on background,
targeting the blur and the lost extremities.

**Why this priority**: operator decision (iii), "mask-weighted MSE first"; R1 of the
experiment procedure.

**Independent Test**: `recon_loss` unit values (2 × 2 case, α = 0 identity); fixture runs
repeat exactly.

**Acceptance Scenarios**:

1. **Given** `mask_weight: α > 0`, **When** a batch is trained, **Then** the loss is FR-003's
   weighted mean with `m = (x > 0)`, and `train_loss` in `history.csv` is that objective.
2. **Given** the same run, **When** it is evaluated, **Then** `val_loss`, the selection and
   every `metrics.json` image metric are computed exactly as for the plain AE.

### User Story 3 - VAE variant (Priority: P1)

The operator sets `variational: true, beta: 0.1` and gets a KL-regularized latent whose
checkpoint the probe reads and a later slice can sample or interpolate.

**Why this priority**: operator decision (iv); R2 of the experiment procedure.

**Independent Test**: shapes, parameter count, KL values; a fixture VAE run repeats,
resumes exactly and is probed.

**Acceptance Scenarios**:

1. **Given** `variational: true`, **When** training, **Then** `z = μ + exp(½·logvar)·ε`, the
   loss is `recon + β·KL/(H·W)` (FR-003) and `history.csv` records `train_recon` and
   `train_kl`.
2. **Given** a VAE checkpoint, **When** it is evaluated or probed, **Then** `encode` returns
   μ and `model(x)` = `decode(μ)` (deterministic): `val_loss`, the figure, the metrics and the
   probe latents need no sampling.

### User Story 4 - Regularization knobs (Priority: P2)

The operator sets any of `weight_decay`, `shift_px`, `dropout` to fight the mild overfitting
(004: val minimum at epoch 34 of 60).

**Why this priority**: operator decision (ii); R3 of the experiment procedure (one combined
run, Q1).

**Independent Test**: off = identity without RNG draws; shift moves image and mask together
with zero borders; dropout acts in training only; AdamW decay at 0 is a no-op.

**Acceptance Scenarios**:

1. **Given** `shift_px: 4`, **When** a batch is trained, **Then** each image moves by its own
   integer offset in `[-4, 4]²` with zero fill; val and test images are never shifted.
2. **Given** `dropout: 0.1` / `weight_decay: 0.05`, **When** training, **Then** latent
   coordinates are dropped (rest scaled by 1/0.9) before decoding, eval unchanged / AdamW
   applies decoupled decay to every parameter.

### User Story 5 - Experiment configs and procedure (Priority: P2)

A later session with SMPL access runs R1–R3 against R0 by following a documented procedure.

**Why this priority**: "plain AE as control" (operator); the brief's definition of done.

**Independent Test / Acceptance**: the committed configs load and differ from
`configs/ae.yaml` only in `run_name` and their knobs; the procedure lists commands, order,
cost and the report table.

### Edge Cases

- `variational: true` with `beta: 0`, `beta > 0` without `variational`, or `variational`
  as `0` / `1` / a string: config error (exit 2), nothing created (a VAE with β = 0 only
  learns to switch its noise off).
- A batch without body pixels: `w ≡ 1`, so the weighted loss equals the plain MSE.
- A NaN objective (e.g. an overflowing `exp(logvar)`): recorded as is; a NaN val loss never
  pins `best.pt` (008); empty val (the fixture) falls back to the final weights, as today.
- Shift near the frame edge: zero fill clips silhouettes that reach the border (004 research
  R0 shows bodies touching x = 0); accepted for small `shift_px`, stated with the results.
- A VAE checkpoint read by pre-011 code fails (`variational` is an unknown argument):
  forward-incompatible by design, documented in the 004 contract.
- `overfit_steps` (constitution IV smoke helper) stays plain Adam + MSE.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (config surface): `TrainConfig` MUST gain six keys, validated in
  `config_from_dict` / `_check` like the existing ones (numbers normalized to `float`):

  | Key | Type, range | Default | Lives in |
  |---|---|---|---|
  | `mask_weight` | finite float ≥ 0 (α) | `0.0` | objective (`train_epoch`) |
  | `variational` | YAML bool | `false` | model (`ConvAutoencoder`, `model_hparams`) |
  | `beta` | finite float ≥ 0; > 0 iff `variational` | `0.0` | objective (`train_epoch`) |
  | `weight_decay` | finite float ≥ 0 | `0.0` | optimizer (`_train_and_evaluate`) |
  | `shift_px` | int in `[0, 32]` | `0` | train batches (`train_epoch`) |
  | `dropout` | finite float in `[0, 1)` | `0.0` | train forward (`train_epoch`) |

  No CLI flag is added. `train_epoch` takes the four batch-level knobs as one keyword
  argument (a small frozen objective record built from the config; `None` = defaults).
- **FR-002** (legacy values): `LEGACY_VALUES` MUST gain the six keys with their defaults
  (pre-011 runs trained exactly that way), so pre-011 run dirs resume (US1-2).
- **FR-003** (objective): per training batch, with `x` the (shifted) batch in `[0, 1]`,
  `y` the reconstruction, `d = y − x`, `m = 1[x > 0]`, `P = H·W`, `B` the batch size:
  - `recon = F.mse_loss(y, x)` when α = 0 (today's call, an explicit branch);
    otherwise `w = 1 + α·m`, `recon = Σ w·d² / Σ w` (sums over the whole batch);
  - variational: `KL = (1/B)·Σ_b ½·Σ_j (μ_bj² + exp(lv_bj) − 1 − lv_bj)` (nats per image),
    `loss = recon + β·KL / P`; otherwise `loss = recon`.

  `β·KL/P` keeps per-pixel units: the loss is `(SSE + β·KL)/P`, so β = 2σ² of a Gaussian
  decoder (β = 1 ↔ σ² = ½; the 004 AE's val MSE ≈ 0.0022 ↔ β ≈ 0.004). `recon_loss` and
  `gaussian_kl` MUST be pure functions (unit-testable).
- **FR-004** (shift): with `shift_px = k > 0`, each training batch MUST draw
  `torch.randint(-k, k + 1, (B, 2))` (global torch RNG) and move sample b's content by
  `(dy_b, dx_b)` (`out[i, j] = in[i − dy, j − dx]`, zero outside), on the uint8 batch before
  `to_float` (values copied, never interpolated); `m` follows from the shifted image. k = 0:
  no draw, batch untouched. Eval paths never shift.
- **FR-005** (bottleneck dropout): with `dropout = p > 0`, `train_epoch` MUST apply
  `F.dropout(z, p, training=True)` to the latent after sampling (FR-007) and before
  `decode`; p = 0: not called. Not a model attribute (no hparam, no state key).
- **FR-006** (optimizer): `_train_and_evaluate` MUST build
  `torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)` (explicit:
  AdamW's own default is 0.01), one param group (every parameter, BatchNorm and biases
  included), so pre-011 `Adam` states load on resume. Coupled L2 is not offered.
- **FR-007** (VAE model): `ConvAutoencoder(…, variational: bool = False)`. With
  `variational`, a `logvar = nn.Linear(flat, latent_dim)` head on the flattened bottleneck
  (the input of `encoder[-1]`) MUST be created after `encoder` and `decoder`, so a plain AE's
  parameters, and a VAE's shared ones, initialize exactly as today at the same seed. API:
  `encode(x)` → μ (unchanged code); `encode_dist(x)` → `(μ, logvar)`, `(encode(x), None)`
  for the plain AE; `decode` unchanged; `forward(x) = decode(encode(x))` (deterministic in
  both modes); `hparams()` adds `"variational"`. Sampling `z = μ + exp(½·lv)·randn_like(μ)`
  (global torch RNG) happens in `train_epoch` only. Draw order per batch: shift, ε, dropout,
  each only when on.
- **FR-008** (checkpoints, probe): `checkpoint.pt` / `best.pt` keep their structure;
  `model_hparams` gains `variational`, a VAE's `model` gains `logvar.weight` /
  `logvar.bias`. `load_checkpoint` rebuilds from `model_hparams` (missing key → plain AE).
  The probe's encoding path stays unchanged (μ latents); its random-encoder control stays a
  plain encoder, whose weights equal an untrained VAE's at the same seed (`logvar` is built
  last, FR-007); probe `metrics.json` gains an additive `encoder.variational`
  (`probe_version` 1).
- **FR-009** (records): `HISTORY_FIELDS` MUST append `train_recon, train_kl` (header
  `epoch,train_loss,val_loss,lr,epoch_s,train_recon,train_kl`): `train_loss` = epoch mean of
  the objective (as today), `train_recon` = epoch mean of `recon` (= `train_loss` unless
  variational), `train_kl` = epoch mean KL in nats per image (empty unless variational).
  Checkpointed rows from before 011 are rewritten with `train_recon = train_loss` and an
  empty `train_kl`. `metrics.json` gains `model.variational` (through `hparams()`) and
  `training.objective = {mask_weight, beta, weight_decay, shift_px, dropout}`;
  `train_version` 2 and `metrics_version` 1 stay (additive). The epoch log line appends
  `kl <train_kl>` for a VAE only. `run_info.json` unchanged (`model_parameters` counts the
  head).
- **FR-010** (evaluation unchanged): `eval_loss` (the val loss behind `best.pt` and the 008
  selection), `predict_rows`, `score_images` (009 regions), the baselines and the figure MUST
  stay as they are: eval mode, plain per-image MSE of `decode(μ)`, no shift, dropout,
  sampling or weighting, so every variant's metrics compare with R0's.
- **FR-011** (defaults): with every 011 key at its default, training MUST be bitwise the
  pre-011 computation (same ops, same RNG draws), US1-1.
- **FR-012** (tests, written first; pytest, fixture, CPU):
  (a) config: defaults; each rejection (negative, non-finite, `dropout` ≥ 1, `shift_px` > 32
  or non-int, `variational` non-bool, the `beta` / `variational` coupling); `LEGACY_VALUES`
  equals the defaults for the six keys; (b) regression: 2 epochs of `train_epoch` with
  defaults (AdamW) vs an inline copy of today's loop (Adam, `F.mse_loss`) on the fixture train
  images, batch 3 (partial last batch): `torch.equal` on parameters, optimizer state tensors,
  losses and `torch.get_rng_state()`; (c) model: VAE shapes, `encode` = μ, 3,323,569
  parameters, state keys = today's + `logvar.*`, shared params of seeded AE and VAE equal,
  `forward` deterministic in train mode, `encode_dist` of the plain AE, a checkpoint without
  `variational` loads as plain; (d) objective: α = 0 `torch.equal` to `F.mse_loss` (value and
  gradient), a hand-computed 2 × 2 weighted value, body error costs (1 + α)× background; KL
  ≥ 0 on random input, = 0 at μ = 0, lv = 0, a hand value; (e) shift: k = 0 identity without
  RNG draws, per-sample offsets, zero borders, exact values, `x > 0` moves with the image,
  `eval_loss` independent of `shift_px`; (f) dropout: p = 0 no draw; p > 0 changes the
  training output only; (g) fixture runs (2–3 epochs, small channels as the 008 tests):
  repeatability for each committed variant's knobs; one all-knobs-on run stopped after epoch 1
  and resumed equals the uninterrupted run (history, weights, `metrics.json` bytes); history
  columns and `metrics.json` keys present; a VAE's `train_kl` finite and > 0; (h) legacy: a
  run dir with the six keys stripped from `config.yaml` / `checkpoint.pt` / `best.pt`,
  `variational` from `model_hparams` and the new columns from the checkpointed rows resumes
  and finishes; (i) probe: a VAE fixture checkpoint probes end to end, latents
  `[n, latent_dim]`, `encoder.variational` true; (j) data: `(image > 0) == mask` on all 16
  fixture samples; (k) the committed variant configs load and differ from `configs/ae.yaml`
  only in `run_name` and their knobs. Existing tests change only where the `train_epoch`
  return shape changes.
- **FR-013** (experiment configs): `configs/ae_mask.yaml` (`mask_weight: 4.0`),
  `configs/ae_vae.yaml` (`variational: true`, `beta: 0.1`) and `configs/ae_reg.yaml`
  (`weight_decay: 0.05`, `shift_px: 4`, `dropout: 0.1`; one combined run, Q1), each a copy of
  `configs/ae.yaml` with `run_name` `ae-{mask,vae,reg}-smpl_proc_v1`; `configs/ae.yaml` gains
  the six keys at their default values, commented (its resolved config is unchanged). The
  fixture, 010 and probe configs are unchanged.
- **FR-014** (documentation, same slice): see *Documentation scope*.
- **FR-015** (constraints): no new dependency (`pyproject.toml`, `uv.lock` untouched);
  Python 3.11-compatible; `uv run --locked` only; the CI smoke step stays the plain fixture
  config.

### Documentation scope (FR-014)

- `specs/004-autoencoder-mvp/spec.md`: dated note "amended 2026-10-21 by slice 011" on the
  model / training FRs (optional variants; defaults unchanged); original text kept.
- `specs/004-autoencoder-mvp/contracts/train-run.md`: model API (`variational`,
  `encode_dist`, `logvar` head), the six config rows, the objective formulas, `history.csv`
  header, `checkpoint.pt` `model_hparams`, `metrics.json` `model.variational` and
  `training.objective`, pre-011 code cannot load VAE checkpoints; `train_version` stays 2.
- `specs/006-latent-probe/contracts/probe-run.md`: one note (a VAE checkpoint is probed
  through μ; additive `encoder.variational`).
- `README.md`: status table (011 row); Usage: the six keys, the three configs, a pointer to
  the experiment procedure; Results unchanged.
- `specs/roadmap.md`: #16 (ii)–(iv) chosen 2026-10-20 and implemented by 011, (v) still open;
  post-MVP table row; status line; changelog entry 2026-10-21, appended at the end.
- `docs/HANDOFF.md`: state (011 implemented, approval status); next work: the experiment
  procedure after the re-baseline (item 4); #16 updated in "Operator decisions requested".
- Not edited: `docs/results/` (its `history.csv` keeps the 5 columns of its run;
  `plot_loss_curve.py` reads columns by name), 001–010
  `plan.md` / `tasks.md`, `.github/workflows/ci.yml`.

## Success Criteria *(mandatory)*

- **SC-001**: ruff, ruff format, pytest and the CI smoke commands green locally and, after the
  push, in CI (3.11 and 3.13).
- **SC-002**: FR-011 holds: the regression test passes, and a same-machine fixture comparison
  of `19479e2` with the slice head gives equal weights, optimizer state tensors, history
  losses / lr, and `metrics.json` / probe `metrics.json` byte-equal after removing
  `model.variational`, `training.objective` and `encoder.variational`.
- **SC-003**: every knob, each committed variant and the all-knobs-on run pass the FR-012
  repeatability and resume tests; the legacy run dir resumes.
- **SC-004**: pytest grows by ≤ 30 s on 4 vCPU; the CI smoke train is unchanged.
- **SC-005**: `docs/results/`, `splits/`, the fixture and the 010 configs are byte-unchanged;
  the documentation scope is done.

## Assumptions

- `x > 0` ⟺ silhouette holds for every dataset of the 001 renderer (pinned on the fixture,
  FR-012 (j)); a dataset with a non-black background would need the train masks.
- Bitwise claims hold per machine and thread count, as the 004 resume guarantee.
- No KL warm-up or free bits: if `train_kl` collapses early, a linear warm-up by global step
  (resumable without new state) is a follow-up, not part of 011.
- Out of scope: the silhouette term (operator: deferred; options in Clarifications 10),
  sampling / interpolation CLIs (a later slice), the capacity sweep (v), new metrics.

## Experiment procedure (follow-up E; needs SMPL; not part of "done")

*(Amended 2026-10-21 by slice [012](../012-run-guards/spec.md), operator decisions
2026-10-21: the first VAE run R2 uses β = 0.01 (`configs/ae_vae.yaml`), a collapse retry
β = 0.001 and a healthy follow-up β = 0.1; the runs follow the 012 campaign procedure
(`docs/HANDOFF.md`), with per-variant promotion metrics and a seed-1 control R0b. The text
below is the procedure as specified by 011.)*

Prerequisites: F1 and the re-baseline run R0 (HANDOFF "Next work" item 4: `configs/ae.yaml`,
`best_val`, plus its probe) on `smpl_proc_v1`, same machine type and locked environment.
R0 is reused as the control, never rerun.

1. R1 mask-weighted MSE: `strike-a-pose-train --config configs/ae_mask.yaml` (detached,
   resumable), then the probe on its `metrics.json` `selection.checkpoint`.
2. R2 VAE: `configs/ae_vae.yaml` (β = 0.1, Q3), then the probe; a second β (0.01 or 1) only
   if R2's reconstructions blur badly or its `train_kl` collapses.
3. R3 regularization: `configs/ae_reg.yaml`, then the probe; single-knob runs (copies of
   `ae_reg.yaml` with the other two knobs at 0) only if R3's best val loss is below R0's (Q1).
4. Report per run next to R0 (README Results, `docs/results/011-<run>/`): best-val epoch and
   loss; val / test `image_mse`, `psnr_db`, `silhouette_iou`; region MSE (hands / feet /
   head, 009); probe ridge / MLP MPJPE and yaw accuracy; R2's final `train_kl`; the figure.
   `train_loss` is each variant's own objective and is not compared. One seed per variant:
   stated as such; no success bar (constitution: honest baselines).
5. 010 datasets (Q2): the 010 chain's AE runs stay plain (`configs/smpl_proc_v*/ae.yaml`);
   each winning variant (best val loss below R0's) runs once on the last 010 dataset, next to
   that dataset's plain run, from a copy of that dataset's `ae.yaml` plus the variant's keys.

Cost: ≈ 85 min per training run + ≈ 8 min per probe; R1 ≈ 100 min (the mask-weighted loss
measured +15 % per step; the VAE and the regularization knobs within the ±10 % timing noise;
research R8); R1–R3 ≈ 5 h sequential on 4 vCPU.

## Clarifications

### Session 2026-10-21 (autonomous run, Fable surrogate scope)

Defaults chosen by the lead within the surrogate's scope are marked *(lead default)*; the
three questions sent to the surrogate in one batch carry *(surrogate ruling 2026-10-21)*. The
surrogate accepted every lead default, naming in particular: six keys with `LEGACY_VALUES` =
defaults; dropout in `train_epoch`; `variational` in the model with the `logvar` head created
last; AdamW always with an explicit `weight_decay`; `train_recon` / `train_kl` always in
`history.csv` (the 7-column format goes into the 004 contract, while
`docs/results/004-ae-smpl_proc_v1/history.csv` keeps the 5 columns of its run); selection by
the plain eval-mode val MSE; `model.variational`, `training.objective` and the probe's
`encoder.variational`; the configs `ae_mask.yaml`, `ae_vae.yaml`, `ae_reg.yaml`. Every answer
is provisional and overturnable by the operator.

1. Q: Slice number and scope? → A: `011-training-variants`: options (ii)–(iv) in one slice;
   numbers wait for SMPL (follow-up E). *(lead default)*
2. Q: Where does each knob live? → A: FR-001's table; only `variational` (it changes the
   parameters) is in the model. Dropout is not a model attribute (memo: an `nn.Dropout` in
   the model), so eval paths cannot see it and `hparams()` stays architectural.
   *(lead default)*
3. Q: Weighted-loss normalization? → A: per batch, `Σ w·d² / Σ w` (memo): equal weight per
   pixel class whatever the body size, plain-MSE scale at α = 0; explicit α = 0 branch.
   Considered: per-image normalization (upweights small bodies), unnormalized `mean(w·d²)`.
   *(lead default)*
4. Q: KL scaling? → A: `β·KL/(H·W)` with KL in nats per image (memo), i.e. the usual
   `SSE + β·KL` up to the constant 1/P. *(lead default)*
5. Q: Which loss drives `best.pt` for a variant? → A: the plain eval-mode val MSE of
   `decode(μ)` (FR-010): comparable across variants and with R0. Considered: each variant's
   own objective on val. *(lead default)*
6. Q: Optimizer? → A: AdamW always, `weight_decay` explicit, one param group (FR-006);
   bitwise equal to Adam at 0 (measured); coupled L2 through Adam is divided by √v and not
   offered; BatchNorm / bias decay accepted. *(lead default)*
7. Q: `history.csv`? → A: two columns appended in every run (FR-009): one header for all
   runs; readers by name and the first five columns are unaffected. *(lead default)*
8. Q: Where does `logvar` attach, and what does the probe read? → A: beside `encoder[-1]`,
   created last (FR-007); `encode` = μ, so the probe code is untouched apart from the
   additive `encoder.variational`. *(lead default)*
9. Q: Warm-up, free bits, logvar clamp? → A: none in 011 (Assumptions). *(lead default)*
10. Q: Silhouette term? → A: deferred (operator); options for later: a 2-channel mask head
    (new last layer, state dict, figure) or a soft-threshold BCE on `σ((y − 0.075)/T)`.
    *(lead default)*
11. Q: Mask source for α? → A: `m = (x > 0)` from the (shifted) batch; the train split stays
    mask-free in memory (≈ 0.6 GB saved at 12,800 images). *(lead default)*
12. Q: First parameter values? → A: α = 4 (body ≈ 7.5 % of pixels → ≈ 29 % of the weight);
    β = 0.1 (Q3); `weight_decay` 0.05 (cumulative decay `exp(−wd·Σlr)` ≈ exp(−0.3) over
    12,000 cosine steps; 1e-4 would do nothing); `shift_px` 4; `dropout` 0.1 (the decoder's
    first BatchNorm sees a train / eval variance gap; kept ≤ 0.1).
    *(lead default)*
13. Q: Regularization runs: one combined run or one run per knob? → A: one combined run R3
    first (`ae_reg.yaml`); single-knob runs only if it beats R0 on best val loss (batch Q1,
    option (a)). *(surrogate ruling 2026-10-21)*
14. Q: Variant runs on the 010 datasets? → A: R1–R3 on `smpl_proc_v1` against R0; the 010
    chain stays plain; each winning variant runs once on the last 010 dataset next to that
    dataset's plain run (batch Q2, option (a)). *(surrogate ruling 2026-10-21)*
15. Q: VAE runs: one β or a bracket? → A: β = 0.1 first (batch Q3, option (a)).
    *(surrogate ruling 2026-10-21)*
