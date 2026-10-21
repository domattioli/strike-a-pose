# Feature Specification: Autoencoder MVP

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `004-autoencoder-mvp`)
**Created**: 2026-10-20
**Status**: Implemented 2026-10-20 (approved: autonomous run, operator delegation 2026-10-20; Operator sign-off 2026-10-21 ("approve all")); full run recorded in [tasks.md](tasks.md) T015, results in `docs/results/004-ae-smpl_proc_v1/`
**Input**: Roadmap slice 004 + lead brief 2026-10-20: "plain convolutional
autoencoder, latent 64, deterministic (VAE out of scope); config-driven training
CLI that seeds Python/NumPy/torch and writes a run dir via `strike_a_pose.runlog`
(resolved config, commit, env, seed, per-epoch train/val loss, checkpoint never
committed); eval reusing the 003 metrics on val (test once at the end) with a recon
figure in the 003 format plus an AE column (PCA k=64 for context); CI smoke train on
the committed fixture (< ~60 s, loss decreases on a fixed batch) replacing the 005
placeholder; one real run on the full dataset within a ~3 h CPU budget, results
committed under `docs/results/004-ae-smpl_proc_v1/`."

Settled context (operator): MVP bar = the image autoencoder *works* — trains from
config without error, loss decreases, held-out reconstructions are recognizable;
003 baselines (PCA k=64 val MSE 0.00459, IoU 0.644) are reported alongside for
context, **not** a bar to beat (roadmap Q3). Input: 1×256×192 grey-on-black in
[0, 1] (002 sample contract). Predecessor (`Conv_AE-Human_Pose`): conv AE, latent
64, input 256×192 (its VAE attempt was unfinished). Data: `smpl_proc_v1` (16,000
samples) with frozen pose-level split `splits/smpl_proc_v1.json` (12,800 / 1,600 /
1,600). Compute: one 4-vCPU, 15 GB RAM machine, no GPU.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train the autoencoder from a config (Priority: P1)

A researcher runs `strike-a-pose-train --config configs/ae.yaml`; the model trains
on the train split, logs train and val loss per epoch, saves a checkpoint, and
writes a self-describing run directory.

**Why this priority**: This is the MVP (roadmap goal); constitution II + III.

**Independent Test**: Run the CLI on the committed fixture with a tiny config into a
temp dir; check the run directory contents and that the train loss decreases.

**Acceptance Scenarios**:

1. **Given** a dataset, its split file and a train config, **When** the CLI runs,
   **Then** a new run directory holds the resolved config, run info (git commit +
   dirty flag, argv, seed, environment, dataset manifest hash, split hash, wall
   time), a per-epoch loss history (train + val), a checkpoint and `metrics.json`.
5. **Given** a run interrupted after epoch N (process killed), **When**
   `strike-a-pose-train --resume RUN_DIR` runs, **Then** training continues at
   epoch N+1 and the final history, weights and `metrics.json` equal an
   uninterrupted run's.
2. **Given** a fixed batch, **When** the model takes a few optimizer steps on it,
   **Then** the loss decreases (overfit smoke).
3. **Given** the same config, data, code and machine, **When** training runs twice,
   **Then** the loss histories are identical (seeded, deterministic CPU training).
4. **Given** an invalid config or a split file that does not fit the dataset,
   **When** the CLI runs, **Then** it exits non-zero with a clear message and no run
   directory.

### User Story 2 - Evaluate the AE next to the baselines (Priority: P1)

After training, the final model is scored with the 003 image metrics on val and
test; the same run also scores the mean image and PCA k=64 (fit on train) so the
AE and the baselines sit side by side in one `metrics.json`, and a reconstruction
figure in the 003 format (true | AE | mean image | PCA k=64) is written.
*(Amended 2026-10-20 by slice [008](../008-best-val-selection/spec.md): the scored model
is the selected weights, by default those of the epoch with the lowest val loss
(`best.pt`), the final weights when val is empty or with `select_weights: final`;
`metrics.json` names them in `selection`.)*

**Why this priority**: Constitution V; the MVP "recognizable" check is visual.

**Independent Test**: Fixture run: `metrics.json` has `ae`, `mean_image` and
`pca_k<K>` entries per evaluated split; figure size = header + rows × columns.

**Acceptance Scenarios**:

1. **Given** a trained model, **When** evaluation runs, **Then** every eval split
   reports `image_mse`, `psnr_db`, `silhouette_iou` for `ae`, `mean_image`, `pca_k64`
   using the 003 metric functions.
2. **Given** an empty eval split (the fixture's val), **When** evaluation runs,
   **Then** it is reported with `samples: 0` and no metrics, and the val loss column
   is empty, without failing.

### User Story 3 - Full-dataset MVP run recorded (Priority: P2)

One training run on `smpl_proc_v1` within the CPU budget; loss curve, val/test
metrics, figure and wall time recorded in docs, citing run dir, config and commit.

**Independent Test**: Cited numbers equal the committed `metrics.json` copy; train
and val loss decrease; the figure shows recognizable held-out bodies.

### Edge Cases

- Image height/width not divisible by `2^len(channels)` → config/model error before
  training (no run dir).
- Last train batch smaller than `batch_size` → used (no samples dropped).
- `context_pca_k` larger than the train data's rank (fixture) → PCA keeps the
  numerical rank; `k_effective` recorded (003 behaviour).
- Empty val split → `val_loss` empty in the history, `samples: 0` in metrics.
  *(Amended 2026-10-20 by slice [008](../008-best-val-selection/spec.md): and the final
  weights are scored, `selection.selected_by` = `"final (empty val)"`.)*
- Run directory already exists → exit 4.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Package `strike_a_pose.models` MUST provide `ConvAutoencoder`: a
  deterministic convolutional encoder (`[B, 1, H, W]` → latent `[B, latent_dim]`)
  and decoder (latent → `[B, 1, H, W]` in `[0, 1]`), configurable latent size
  (default 64) and channel widths.
- **FR-002**: Package `strike_a_pose.train` MUST provide a YAML train config
  (strict keys/types), a training loop (Adam, MSE loss, seeded shuffling, optional
  cosine learning-rate decay) and a console script `strike-a-pose-train --config
  PATH [--seed INT] [--run-root DIR]` whose overrides are written into the resolved
  config.
- **FR-003**: Training MUST fit on the train split only; the val loss (MSE, model in
  eval mode) is computed after every epoch and never used to change the run.
  *(Amended 2026-10-20 by slice [008](../008-best-val-selection/spec.md): val still
  never changes training (epochs, lr, optimizer); it selects which epoch's weights are
  scored (`select_weights`, default `best_val`).)*
- **FR-004**: Each run MUST write `config.yaml`, `run_info.json` (003 fields +
  model parameter count, epochs, wall time), `history.csv` (per epoch: train loss,
  val loss, learning rate, epoch seconds), `checkpoint.pt` (model + optimizer state,
  epoch, resolved config; written at the end of every epoch), `metrics.json`
  (deterministic for the same machine) and `recon_<figure_split>.png`.
  *(Amended 2026-10-20 by slice [008](../008-best-val-selection/spec.md): also
  `best.pt`, the weights of the epoch with the lowest finite val loss, written after
  that epoch's `checkpoint.pt` whenever val has samples; it is the scored file by
  default, and `metrics.json` `selection.checkpoint` names the scored file.)*
- **FR-005**: Seeds for Python, NumPy and torch MUST be set from the config seed and
  logged (constitution III).
- **FR-006**: After training, the final model MUST be scored on the configured eval
  splits (default val, test) with the 003 functions `image_mse`, `psnr`,
  `silhouette_iou` (threshold 0.075); the mean image and PCA with `context_pca_k`
  components (default 64), fit on train, MUST be scored in the same run.
  *(Amended 2026-10-20 by slice [008](../008-best-val-selection/spec.md): the selected
  weights are scored in place of the final model: `best_val` (default) or `final`. Only
  they are scored, test once, so their gain over the final weights is known on val
  only.)*
  *(Amended 2026-10-21 by slice [009](../009-region-metrics/spec.md): every scored image
  model (`ae`, `mean_image`, `pca_k<K>`) is also scored with the 003 `masked_mse` inside
  the 003 `REGIONS` (`hands`, `feet`, `head`, from the true 2D keypoints, which the
  scored splits now keep in memory), in the same `score_images` call as its headline
  metrics; `metrics.json` only gains keys, `train_version` stays 2.)*
- **FR-007**: The figure MUST use the 003 grid layout (header labels; one row per
  evenly spaced sample) with columns true, AE, mean image, PCA k.
- **FR-008**: CI MUST run a CPU smoke train on the committed fixture (replacing the
  005 placeholder step); pytest MUST include an overfit test on a fixed batch and a
  fixture end-to-end CLI test; per-leg CI wall time stays < 5 min.
- **FR-010**: `strike-a-pose-train --resume RUN_DIR` MUST continue an interrupted
  run from its last end-of-epoch checkpoint (model, optimizer, lr schedule, all RNG
  states, loss history) so that the finished run equals an uninterrupted one on the
  same machine/thread count; it refuses finished runs, runs without a checkpoint,
  and changed config/dataset/split files.
- **FR-009**: Checkpoints and run directories MUST NOT be committed (`runs/` is
  git-ignored); only small summaries go under `docs/results/`.

### Key Entities

- **Train config** (YAML) and **run directory**: `contracts/train-run.md`.
- **ConvAutoencoder**: `contracts/train-run.md` (model section).

## Success Criteria *(mandatory)*

- **SC-001**: Fixture smoke train (CLI) finishes in < 60 s on a CI runner and its
  train loss decreases from the first to the last epoch.
- **SC-002**: Full-dataset run finishes in ≤ ~3 h on the 4-vCPU machine; train and
  val loss decrease; val/test metrics + figure recorded next to the 003 baselines.
- **SC-003**: Held-out (val) reconstructions in the figure are recognizable as the
  true poses (lead's visual check, recorded).
- **SC-004**: `uv sync --locked`, ruff, pytest green locally and on CI (py3.11 +
  py3.13).

## Assumptions

- 002 loader and 003 metrics/PCA are reused unchanged (behaviour-preserving
  refactor of the 003 figure/scoring helpers into a shared module is allowed).
- Matching hardware/thread count is needed for bit-identical training; across
  machines low-order differences are possible (recorded in research).

## Clarifications

### Session 2026-10-20 (autonomous run, defaults chosen by lead)

1. Q: Package / CLI names? → A: `strike_a_pose.models` (`ConvAutoencoder`) and
   `strike_a_pose.train` (`config`, `trainer`, `cli`); console script
   `strike-a-pose-train`. Shared 003 figure/scoring helpers move to
   `strike_a_pose.eval.report` (003 behaviour unchanged).
2. Q: Architecture? → A: predecessor shape, sized for 4 CPUs: 6 stride-2 3×3 conv
   blocks (BatchNorm + LeakyReLU 0.2) with channels `[16, 32, 64, 128, 256, 256]`
   (256×192 → 4×3), flatten → Linear → latent 64; decoder mirrors it (Linear →
   4×3 → 6 ConvTranspose 4×4 stride-2 blocks) with a sigmoid output. ≈ 3.1 M
   parameters. Full 256×192 input (no resize): measured ≈ 170 img/s training on
   this machine (≈ 1.3 min / epoch, research R2).
3. Q: Loss / optimizer? → A: per-pixel MSE (= the 003 `image_mse`, predecessor's
   loss); Adam, lr 1e-3, batch 64, cosine decay to 0 over the run, no weight decay,
   no augmentation.
4. Q: Epochs / budget? → A: 60 epochs (≈ 80 min of training + load/eval), chosen a
   priori from the throughput benchmark to sit well inside the ~3 h budget; not
   tuned on val or test.
5. Q: Which weights are evaluated? → A: the final epoch's (no early stopping or
   best-val selection; val is logged only).
   *(Amended 2026-10-20 by slice [008](../008-best-val-selection/spec.md), operator
   choice B for roadmap question #13: runs score the weights of the epoch with the
   lowest val loss by default (`select_weights: best_val`), the final weights when val
   is empty or with `select_weights: final`; 60 epochs and the cosine schedule
   unchanged, no early stopping. The published 004 run, scored on its final weights,
   stays as reported.)*
6. Q: Baselines in the AE run? → A: mean image + PCA `context_pca_k = 64` refit on
   train in the same run with the 003 code (exact PCA, deterministic), so the AE
   `metrics.json` holds AE and baselines side by side; they must equal the 003
   numbers (consistency check). Pose baselines are not repeated (the AE outputs
   images only).
7. Q: Figure? → A: 003 grid via the shared helper: columns true | ae | mean_image |
   pca k=64; split `val`, 6 evenly spaced samples (the same samples as 003's figure).
8. Q: Checkpoint / resume? → A: `checkpoint.pt` overwritten atomically after each
   epoch (model, optimizer, epoch, config). *Superseded by Session 2026-10-20b:
   resume is implemented.*
9. Q: Test split? → A: scored once, at the end of the real run; no decision is made
   from test numbers. A pilot (if any) evaluates val only.
10. Q: CI smoke? → A: `configs/ae_fixture.yaml` (fixture train = 8 images; batch 4,
    10 epochs = 20 steps, `context_pca_k 4`, figure on `test`), run by the CLI in CI
    and by a pytest end-to-end test; separate pytest overfit test on a fixed batch
    of 4 fixture images (loss after 15 steps < ½ initial).
11. Q: VAE? → A: out of scope for the MVP (deterministic AE only); later candidate
    in the roadmap.
12. Q: Determinism? → A: seeds set and logged; shuffling uses a seeded
    `torch.Generator`; on one machine with a fixed thread count training repeats
    bit-for-bit (tested on the fixture). Torch's default intra-op threads (= CPU
    count) are used and logged.
13. Q: Exit codes? → A: as 003: 0 OK; 1 dataset/split invalid; 2 config error;
    4 run dir exists.
14. Q: PyTorch CPU wheel index (002 R1 open question)? → A: re-tested 2026-10-20:
    `download.pytorch.org` still returns 403 at CONNECT from this session → unchanged
    (PyPI torch), stays an operator question.

### Session 2026-10-20b (operator requirement, lead resuming after a container restart)

15. Q: Restart resilience? → A (operator): the container can restart and kill
    background processes, so training MUST be resumable: a checkpoint at the end of
    each epoch (model + optimizer + epoch + RNG states + history) and
    `--resume RUN_DIR` continuing deterministically; tested (interrupted + resumed
    == uninterrupted on the fixture, bit-exact on CPU); the full run is launched
    detached (`setsid nohup … > log`) with the resume command recorded in tasks.md;
    wall-clock budget ≤ ~3 h total. Implementation: checkpoint adds `scheduler`,
    `rng` (shuffle generator, torch, NumPy, Python), `history`, `elapsed_s`,
    `train_wall_s`; resume rewrites `history.csv` from the checkpoint (drops rows
    written after it), restores state and continues; if all epochs were done but
    evaluation was not, only evaluation runs. `run_info.json` lists `sessions`
    (start epoch, argv, git, start time). `--seed`/`--run-root` are rejected with
    `--resume` (the run dir's `config.yaml` is authoritative). Exit 2 for a run that
    cannot be resumed (finished, no checkpoint, config mismatch); exit 1 if the
    dataset manifest or split file hash changed. Supersedes Q8's "resume not
    implemented".
