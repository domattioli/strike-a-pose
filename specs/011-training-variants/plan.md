# Implementation Plan: 011 Training Variants

**Branch**: `claude/charming-wozniak-5ouldf` | **Date**: 2026-10-21 | **Spec**: [spec.md](spec.md)
**Input**: spec approved at surrogate scope (rulings 2026-10-21: Q1–Q3 option (a), every lead
default accepted). Research: [research.md](research.md); records: [data-model.md](data-model.md).

## Summary

Six train-config keys select three training variants of the 004 autoencoder: regularization
(`weight_decay` through AdamW, `shift_px` integer shifts of the train batches, `dropout` on the
latent), mask-weighted MSE (`mask_weight` α, `w = 1 + α·[x > 0]`) and a VAE (`variational`
with a `logvar` head, `beta` on `KL/(H·W)`). Every key defaults to today's computation, bit
for bit, and carries that value in `LEGACY_VALUES`, so pre-011 run dirs resume. Evaluation
(val loss, selection, metrics, figure, probe) stays the plain deterministic reconstruction
`decode(μ)`. Three configs (`configs/ae_{mask,vae,reg}.yaml`) and a documented experiment
procedure (R1 → R2 → R3 against the re-baseline R0) complete the slice; no run happens here.

## Technical Context

Python ≥ 3.11 (CI 3.11 and 3.13); torch 2.14.1 CPU (locked PyPI wheel), PyYAML, NumPy; no new
dependency. Tests: pytest on the committed 16-sample fixture (256 × 192, train 1 pose / val 0
/ test 1); ruff. Budget: pytest ≈ 36 s today, at most +30 s; the CI smoke train and probe stay
the plain fixture configs. Determinism: same machine and thread count (004 resume guarantee).

## Constitution Check

| Gate | Status | How |
|---|---|---|
| I Library-first | PASS | Logic in `strike_a_pose.train` (new `objective.py`) and `models.autoencoder`; no notebook. |
| II CLI + config | PASS | Six versioned config keys; no new CLI flag; variant configs under `configs/`. |
| III Reproducibility | PASS | Every new draw (shift offsets, VAE ε, dropout masks) comes from the global torch RNG, already saved per epoch; defaults draw nothing new; resume equality tested under all knobs. |
| IV Test-first | PASS | Each stage writes its tests first (red run noted in tasks.md): regression against an inline copy of today's loop, identities at 0, legacy resume, VAE probe. |
| V Honest evaluation | PASS | Variants are scored exactly like the control (plain MSE / PSNR / IoU / regions of `decode(μ)`, selection by plain val MSE); R0 reused as control; one seed per variant stated. |
| VI Simplicity | PASS | One small module, one optional model head; no warm-up, no silhouette term, no sampling CLI, no per-parameter-group decay. |

## Design

**Modules.**
- `train/objective.py` (new): `Objective` (frozen: `mask_weight`, `beta`, `shift_px`,
  `dropout`; `from_config`), `shift_batch(u8, k)`, `recon_loss(y, x, mask_weight)`,
  `gaussian_kl(mu, logvar)`, `reparameterize(mu, logvar)`.
- `train/config.py`: six fields (appended after `figure_samples`), validation, `MAX_SHIFT_PX`,
  `LEGACY_VALUES` entries.
- `train/trainer.py`: `make_optimizer(params, cfg)` (AdamW, explicit `weight_decay`);
  `train_epoch(…, objective=None)` returns `{"train_loss", "train_recon", "train_kl", "lr"}`;
  `HISTORY_FIELDS` + `train_recon`, `train_kl`; legacy rows normalized on resume;
  `training.objective` in `metrics.json`; `kl` in the epoch log line of a VAE.
- `models/autoencoder.py`: `variational` argument, `logvar` head built last, `encode_dist`,
  `hparams()["variational"]`; `load_checkpoint` unchanged (rebuilds from `model_hparams`).
- `eval/probe.py`: additive `encoder.variational`; encoding path untouched.

**`train_epoch`, per batch** (draws only when the knob is on, in this order):

```python
u8 = images[idx]
if obj.shift_px:
    u8 = shift_batch(u8, obj.shift_px)  # torch.randint
x = to_float(u8).contiguous(memory_format=CL)
mu, logvar = model.encode_dist(x)  # (encoder(x), None) for the AE
z = mu if logvar is None else reparameterize(mu, logvar)  # torch.randn_like
if obj.dropout:
    z = F.dropout(z, obj.dropout, training=True)  # dropout mask
recon = recon_loss(model.decode(z), x, obj.mask_weight)  # α == 0: F.mse_loss(y, x)
kl = None if logvar is None else gaussian_kl(mu, logvar)
loss = recon if kl is None else recon + obj.beta * kl / (H * W)
```

With the defaults this is `F.mse_loss(decoder(encoder(x)), x)` with the same draws as today
(research R1–R2), and `AdamW(weight_decay=0)` steps like `Adam` (R1).

**Where the next slice hooks in.** Sampling / interpolation: `model.decode(z)` with
`z ~ N(0, I)` or between two `encode(x)` (= μ) of a `variational` checkpoint loaded with
`load_checkpoint`; a silhouette term would join `recon_loss`.

## Stages (each green and committed: ruff check, ruff format --check, full pytest; trainer
stages also the CI fixture train + probe smoke commands)

1. `plan(011)`: rulings folded into the spec; plan, research, data model, tasks; 004 / 006
   contract amendments.
2. `feat(011): config + optimizer + regularization`: six keys + validation + legacy values;
   AdamW; `objective.py` with `Objective` and `shift_batch`; dropout; history columns;
   `training.objective`.
3. `feat(011): mask-weighted MSE`: `recon_loss` with the α = 0 branch; fixture mask pin.
4. `feat(011): VAE variant`: model head, `encode_dist`, `gaussian_kl`, `reparameterize`,
   KL in the objective and history, `model.variational`, probe `encoder.variational`.
5. `feat(011): experiment configs` + `docs(011)`: three configs (+ `configs/ae.yaml` keys at
   their defaults), README, roadmap, HANDOFF, 004 spec notes, tasks verification record.

## Risks

- AdamW = Adam at 0 relies on torch skipping the decay step at `weight_decay == 0`; pinned by
  the regression test, so a torch upgrade that changes it fails loudly.
- Time box (hard stop 10:40 UTC): stages 4–5 may not finish; tasks.md then lists the
  remainder with exact instructions and HANDOFF notes it.
- VAE collapse or NaN at β = 0.1 (exp of a large `logvar`): visible in `train_kl`; 008 never
  selects a NaN val loss; warm-up stays a follow-up (spec Assumptions).
