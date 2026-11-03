# Data Model: 011 Training Variants

## Train config (`TrainConfig`, appended fields)

| Key | Type, range | Default = `LEGACY_VALUES` | Used by |
|---|---|---|---|
| `mask_weight` | finite float ≥ 0 | `0.0` | `recon_loss` |
| `variational` | YAML bool | `false` | `ConvAutoencoder(variational=…)` |
| `beta` | finite float ≥ 0; > 0 iff `variational` | `0.0` | objective (KL weight) |
| `weight_decay` | finite float ≥ 0 | `0.0` | `make_optimizer` (AdamW) |
| `shift_px` | int in `[0, MAX_SHIFT_PX = 32]` | `0` | `shift_batch` |
| `dropout` | finite float in `[0, 1)` | `0.0` | `train_epoch` (latent) |

`Objective` (frozen dataclass, `train/objective.py`): `mask_weight`, `beta`, `shift_px`,
`dropout`; `Objective()` = the plain AE; `Objective.from_config(cfg)`.

## Model

`ConvAutoencoder(image_size, latent_dim, channels, variational=False)`;
`hparams() = {image_size, latent_dim, channels, variational}`; a VAE's state dict adds
`logvar.weight [latent_dim, flat]`, `logvar.bias [latent_dim]`. `encode(x)` → μ;
`encode_dist(x)` → `(μ, logvar | None)`; `forward(x) = decode(encode(x))`.

## Records

- `history.csv`: `epoch,train_loss,val_loss,lr,epoch_s,train_recon,train_kl`; `train_kl`
  empty unless variational. Checkpointed rows carry the same keys; pre-011 rows are completed
  on resume (`train_recon = train_loss`, `train_kl = None`).
- `checkpoint.pt` / `best.pt`: structure unchanged; `model_hparams.variational`; `config`
  with the six keys.
- `metrics.json` (`train_version` 2, `metrics_version` 1; additive): `model.variational`;
  `training.objective = {"mask_weight", "beta", "weight_decay", "shift_px", "dropout"}`.
- Probe `metrics.json` (`probe_version` 1; additive): `encoder.variational`.
- `run_info.json`: unchanged (`model_parameters` counts the head).
