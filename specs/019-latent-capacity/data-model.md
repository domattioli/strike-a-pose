# Data Model: 019 Latent Capacity

No new entities. Unchanged records, with the values this slice adds:

- `TrainConfig.latent_dim` (int ≥ 1): 128 in the three FR-001 configs.
- Checkpoint `hparams.latent_dim` = 128; `pose_head_hparams.latent_dim` = 128 when λ > 0.
- `metrics.json`, probe `metrics.json` (`model.latent_dim` = 128): unchanged schema.

Variant naming in reports: L1 `ae-l128`, L2 `ae-l128-pose-l0.03`, L3 `ae-l128-pose-l0.01`;
018 rows keep P1 / P2.
