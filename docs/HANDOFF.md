# Handoff — strike-a-pose (as of 2026-10-21, after slice 014)

Branch: `claude/charming-wozniak-5ouldf` (all work here; `main` untouched; no PRs —
operator: "one-shot, no PRs"). CI green on `ubuntu-24.04`, Python 3.11 + 3.13.

## State

- MVP done and signed off: slices 001–005, plus post-MVP 006 (latent probe).
  The operator's "approve all" (labelled 2026-10-21) covers slices 002–006, including their
  autonomous clarifications, and the surrogate rulings of 2026-10-20 only; 007–012 await
  operator sign-off (list below); 014 follows operator decisions directly.
- Results: `docs/results/{003-baselines,004-ae,006-probe}-smpl_proc_v1/`; summary in
  README "Results". AE test MSE 0.00219 / IoU 0.881 (PCA-64: 0.00452 / 0.646).
  Probe: camera yaw 96 % (MLP), 3D joints 172 mm MPJPE (mean pose 253 mm).
- Slice 007 `split-content-hash` (roadmap #14) implemented 2026-10-20; approved at
  surrogate scope, operator sign-off pending. Split files are `split_version` 2: they
  pin the manifest content (manifest without its generating `code`), keep the code as
  `manifest_code`, and warn (`SplitCommitWarning`) when the same content was generated
  by other code; v1 files stay readable with the old strict check. The 004 resume
  check and the 006 latent cache still key on the raw manifest hash.
- Slice 008 `best-val-selection` (roadmap #13, operator choice B, 2026-10-20)
  implemented 2026-10-20; approved at surrogate scope, operator sign-off pending.
  Training runs score the weights of the epoch with the lowest val loss (`best.pt`)
  by default (`select_weights: best_val`), the final weights when val is empty (the
  fixture, CI) or with `select_weights: final`; training is unchanged (60 epochs,
  cosine). `metrics.json` names the scored weights in `selection` (`train_version` 2);
  their gain over the final weights is known on val only. `checkpoint.pt` is written
  before `best.pt`, and `--resume` repairs or refuses an inconsistent `best.pt`; run
  dirs started before a config key existed resume with its legacy value
  (`train.config.LEGACY_VALUES`). The published 004 / 006 numbers stay final-epoch
  weights until the re-baseline run. `configs/probe.yaml` names a placeholder run dir.
- Slice 009 `region-metrics` (roadmap #16, operator choice (i), 2026-10-20) implemented
  2026-10-21; approved at surrogate scope, operator sign-off pending. Every image model
  (003 mean image and each PCA k; 004 `ae`, `mean_image`, `pca_k<K>`) also reports its MSE
  in three regions around the true 2D keypoints: `hands` (21-px squares on joints 22 / 23),
  `feet` (17 px on 7 / 8 / 10 / 11), `head` (29 px on 15); `metrics.json` gains `regions`
  per image entry, a top-level definitions block and `units.regions` (`metrics_version` 1,
  `train_version` 2 unchanged; no config key, training or figure change). The scoring
  seam is `score_images(…, labels=(kp2d, visible, image_size))`; the next slice's
  training variants keep their predictors behind it (`predict_rows`) and get regions
  without further change (009 plan, "Where the next slice hooks in"). The published
  numbers predate 009.
- Slice 010 `dataset-realism` (roadmap #15, operator decision 2026-10-20) implemented
  2026-10-21; approved at surrogate scope, operator sign-off pending. Opt-in generator
  options, each left out of the manifest `config` while off, so `smpl_proc_v1` regenerates
  byte for byte (fixture pins; same-machine before/after synthetic comparison identical):
  `ranges_version: armsdown-v1` (renamed by 014; arms lowered ∓65° after sparsification,
  shoulder z `[-20, 20)`, collar z `[-10, 10)`: the collar narrowing departs from the
  predecessor's table; every other draw equals v1's for the same seed), `body_shape`
  (per-pose betas N(0, 1) clipped at ±2, own stream) and `camera_jitter` (per-sample camera
  moved ±0.3 m in height and 0–0.5 m farther, aimed at the ring's look-at point and tilted
  0–5° down, kept only if every mesh vertex projects 2 px inside the image, else the ring
  camera; labels record `camera.jitter`). 003 accepts `image_pca_ks: []`. Its cumulative
  config chain is replaced by 014's independent datasets (below). No new dataset, split or run
  exists: steps v2a–v2d of the campaign procedure below.
- Slice 011 `training-variants` (roadmap #16, operator options (ii)–(iv), 2026-10-20)
  implemented 2026-10-21; approved at surrogate scope (rulings Q1–Q3 (a)), operator
  sign-off pending. Six train config keys, each off by default (a default run equals the
  pre-011 code's bit for bit: regression test and a same-machine before/after fixture
  comparison) and in `LEGACY_VALUES` (older run dirs resume): `weight_decay` (AdamW),
  `shift_px` (per-sample integer shifts of train images), `dropout` (latent, training
  only), `mask_weight` (α on the rendered silhouette `x > 0`), `variational` + `beta` (VAE:
  `logvar` head, `beta·KL/(H·W)`, `encode` = μ). Val loss, selection, metrics, figure and
  probe stay the plain deterministic reconstruction; `history.csv` gains `train_recon`,
  `train_kl` (`docs/results/004-ae-smpl_proc_v1/history.csv` keeps its 5 columns);
  `metrics.json` gains `model.variational`, `training.objective`; the probe
  `encoder.variational`. Configs `configs/ae_{mask,vae,reg}.yaml`; no run yet (campaign procedure below).
- Slice 012 `run-guards` (operator decision 2026-10-21: the ≈ 20 h campaign with checks)
  implemented 2026-10-21; approved at surrogate scope, operator sign-off pending. Trainer
  guards, off by default: `early_stop {patience, min_epochs, min_delta}` (stop on a val
  plateau; schedule and 008 selection unchanged; `training.stopped_early`) and
  `abort_on_divergence` + `divergence_ratio` (`aborted.json`, exit 5, not resumable); a run-dir
  lock (`RUN_DIR/lock`, a live owner refuses a second start); the batched mean image (the
  4,000-pose OOM); `env.cpu_model` and `dataset.checksums_sha256` in `run_info.json`;
  `scripts/run_status.py`; campaign configs (guards in `configs/ae{,_mask,_vae,_reg}.yaml`,
  `ae_vae.yaml` β 0.01, new `ae_seed1`, `ae_vae_beta0.1`, `ae_vae_beta0.001`). Slices 007–012
  are verified on the fixture and the synthetic body only: no full-dataset run yet.
  *Fix 2026-10-21 (`fix(012)`, found in live use): the lock also treats a trainer launched as
  `python -m strike_a_pose.train.cli` as alive (before, `run_status.py` said "looks dead" and a
  `--resume` would have replaced a live lock).*
- Slice 014 `realism-ablations` (operator decisions 2026-10-21, amending 010) implemented
  2026-10-21. Independent single-change datasets, each compared with `smpl_proc_v1` (R0):
  `configs/smpl_proc_v2a/` (+ `ranges_version: armsdown-v1`), `v2b` (+ `body_shape`), `v2c`
  (+ `camera_jitter`), `v2d` (+ `num_poses: 4000`, `image_pca_ks: []`, `context_pca_k: null`),
  each with `generate`, `baselines`, `ae` (012 guards, as `configs/ae.yaml`) and `probe`;
  `configs/smpl_proc_v{2,3,4,5}/` removed. The arms-down table is `armsdown-v1` (no alias;
  never generated under its old name); v2a's confound (narrowed shoulder / collar z: less pose
  variety; more limb–torso overlap) is documented in 010 FR-003. `scripts/intersection_check.py`
  (`datagen/intersection.py`, NumPy): share of poses with at least 0.4 % of the arm vertices
  (11 of 2,559 on SMPL) inside the torso (per-segment elliptic cylinders, Kabsch-fitted rest
  frames, 1 cm margin, spine3 half-width capped at the rest shoulder half-span − 5 cm; spec 017
  recalibration of 014). Hard gate: v2a's split is not frozen above 5 %; the visual check stays.
  Real SMPL: v1 1.75 %, v2d 2.23 %, v3 2.24 %, armsdown-v1 23 % (FAIL).
- Slice 013 `capsule-body` (operator decision 2026-10-21: SMPL still unavailable) implemented
  2026-10-21; approved at surrogate scope (rulings Q1 mirror the campaign with v2a–v2d, Q2 (a),
  Q3 (b)), operator sign-off pending. `datagen/capsule.py` + `strike-a-pose-data
  make-capsule-body`: an interim license-free SMPL-format body (28 rigid capsules on SMPL's rest
  skeleton, recovered at build time from the committed fixture labels; 5,952 vertices, 18.9 MB,
  SHA-256 `bc1f3eb7…` pinned). Twin configs `configs/capsule_proc_{v1,v2a,v2b,v2c,v2d}/`
  (test-enforced copies of the SMPL configs). `capsule_proc_v1` is generated, its split
  `splits/capsule_proc_v1.json` is frozen (same pose assignment as `smpl_proc_v1`) and its
  baselines reproduce the published pose numbers (SC-005). Results are infrastructure tests:
  `docs/results/013-capsule/`.

## Pending operator sign-off

Each slice is approved at surrogate scope; key surrogate rulings in one line:

- 007 `split-content-hash`: split files pin the manifest content (`split_version` 2), warn on a
  commit mismatch; F1 rewrites `splits/smpl_proc_v1.json` with SMPL.
- 008 `best-val-selection`: batch Q1 (a): only the selected weights are scored (test once); the
  gain over the final weights is known on val only.
- 009 `region-metrics`: Q1–Q3 (a): hands / feet / head boxes of fixed size on the true 2D
  keypoints, all pixels, MSE only.
- 010 `dataset-realism`: Q1–Q4: cumulative chain (now replaced by the operator's independent
  v2a–v2d, 2026-10-21), aim-and-tilt jitter, standard split for 4,000 poses, no image PCA there.
- 011 `training-variants`: Q1–Q3 (a): one combined regularization run, variants on v1 against
  R0, β 0.1 first (now 0.01 by operator decision 2026-10-21).
- 012 `run-guards`: Q1 (a) with per-variant promotion metrics (**operator confirmation
  requested**), Q2 (a) plus the pixel diagnostic, Q3 (a) unattended check-ins advance the
  campaign and commit results.
- 014 `realism-ablations`: the operator's 2026-10-21 decisions; lead defaults: the intersection
  test (per-segment inscribed elliptic cylinders, 1 cm margin, ≥ 1 vertex = intersecting pose),
  a script rather than a CLI subcommand, and a gate failure skipping v2a (no unattended table
  change).
- 013 `capsule-body`: Q1 mirror the campaign on capsule twins, Q2 (a) results only under
  `docs/results/013-capsule/`, Q3 (b) skeleton derived at build time from the fixture
  (**operator confirmation of the license reading requested**: 72 SMPL-derived joint
  coordinates, already public in the fixture labels; no SMPL array used).

Open operator decisions: the promotion rule (above); the winners' dataset: which of v2a–v2d
(or v1 only) the winning variants run on (default if no answer: v2a).

## Next work

1. **The campaign** (needs `assets/smpl/SMPL_NEUTRAL.npz`; `domattioli/smpl-assets` exists
   but was still empty at the last check): "Campaign procedure" below, starting with F1
   ("F1 commands"). The v2a–v2d configs are committed (slice 014). Until then, the interim
   capsule campaign ("Interim capsule campaign" below; slice 013) runs without SMPL.
2. **#12 approved, on hold**: switch torch to the CPU wheel index
   (`specs/002-data-pipeline/research.md` R1) only after the campaign's runs are done:
   changing `uv.lock` mid-campaign breaks "same locked environment". `download.pytorch.org`
   was still refused from the agent session on 2026-10-20 (check with
   `curl -sS -o /dev/null -w "%{http_code}" https://download.pytorch.org/whl/cpu/`).
3. Rules for later slices: a new train config key adds its pre-key value to
   `train.config.LEGACY_VALUES` (008); new image predictors are scored through `score_images`
   with the split's labels (009); not started: #16 option (v) (capacity sweep, not chosen).

## Campaign procedure

Operator decision 2026-10-21: the campaign below may use ≈ 20 h of CPU, with guards that end a
stalled or diverging run (slice 012) and check-ins that apply these rules. *(Amended 2026-10-21
by slice 014: the realism steps v2a–v2d are independent single-change datasets with committed
configs, v2a has an automatic intersection gate, W waits for the operator.)* Prerequisite:
`assets/smpl/SMPL_NEUTRAL.npz` in place (never committed). Run everything from the repository
root, detached and resumable, and commit each finished run's records when it ends: nothing may
depend on the container surviving.

### Steps (in order; compute time on 4 vCPU)

| Step | What | Configs | Time |
|---|---|---|---|
| F1 | regenerate `smpl_proc_v1`; SMPL-file, manifest and pixel diagnostics; split v2; the 003 baselines (also R0's) | "F1 commands" | 15 min |
| R0 | re-baseline AE (the control) + probe | `configs/ae.yaml`, `configs/probe.yaml` | 95 min (≈ 70 if it stops near epoch 44) |
| R0b | R0 with seed 1 + probe: the noise bar | `configs/ae_seed1.yaml`, `configs/probe.yaml` | 95 min |
| v2a | `smpl_proc_v2a` = v1 + arms-down table `armsdown-v1`: generate, validate, intersection check (≤ 5 % of poses), visual check, split, baselines, AE, probe | `configs/smpl_proc_v2a/` | 100 min |
| v2b | `smpl_proc_v2b` = v1 + random body shape: generate, validate, split, baselines, AE, probe | `configs/smpl_proc_v2b/` | 100 min |
| v2c | `smpl_proc_v2c` = v1 + camera ring jitter: the same steps + the jitter fallback count | `configs/smpl_proc_v2c/` | 100 min |
| v2d | `smpl_proc_v2d` = v1 + 4,000 poses: the same steps (baselines without image PCA, AE with `context_pca_k: null`) | `configs/smpl_proc_v2d/` | 200 min |
| R1 | mask-weighted MSE (α 4) + probe | `configs/ae_mask.yaml` | 110 min |
| R2 | VAE β 0.01 + probe, then once R2r (β 0.001) or R2b (β 0.1) | `configs/ae_vae.yaml`, `configs/ae_vae_beta0.001.yaml`, `configs/ae_vae_beta0.1.yaml` | 95 min each |
| R3 | weight decay + shifts + latent dropout + probe | `configs/ae_reg.yaml` | 95 min |
| W | winning variants on a realism dataset | pending operator decision (default v2a) | ≈ 3 h each |

F1 → R3 ≈ 18 h, less when runs stop early. v2a–v2d are independent single-change datasets
(operator decision 2026-10-21; 010 FR-010 as amended by slice 014): each is `smpl_proc_v1` plus
one change, is compared with R0 (and the R0b noise bar) only, never with another v2x, and runs
the training and evaluation configs of R0 (guards included) with its own names and paths. v2a
also changes pose variety (shoulder z ±20°, collar z ±10° instead of ±50° / ±30°; 010 FR-003):
its effect is arms down plus less variety plus more limb–torso overlap. v2d needs the batched
mean image of slice 012 (landed) and runs without PCA (`image_pca_ks: []`,
`context_pca_k: null`): peak ≈ 3 GB for its AE run and ≈ 6 GB for its baselines (≈ 16.7 /
15.1 GB before, an OOM on 15 GB; 012 research R4).

### Commands

```bash
uv sync --locked && mkdir -p runs
launch() { n=$1; shift; setsid nohup "$@" >> "runs/$n.log" 2>&1 < /dev/null & }   # detached
newest() { ls -td runs/"$1"-2*/ | head -1; }   # newest run dir of a run name
scored() { uv run --locked python -c 'import json, sys; print(json.load(open(sys.argv[1] + "/metrics.json"))["selection"]["checkpoint"])' "$1"; }
keep() {  # keep RUN_DIR STEP KIND: the run's records, never weights
  d=docs/results/campaign/$2/$3; mkdir -p "$d"
  for f in config.yaml run_info.json metrics.json history.csv aborted.json; do
    if [ -f "$1/$f" ]; then cp "$1/$f" "$d/"; fi
  done
  cp "$1"/*.png "$d/" 2>/dev/null || true
}

# training step, R0 shown (R0b: ae_seed1.yaml / ae-seed1-smpl_proc_v1; R1: ae_mask.yaml /
# ae-mask-smpl_proc_v1; R2: ae_vae.yaml / ae-vae-smpl_proc_v1; R3: ae_reg.yaml / ae-reg-…)
launch r0 uv run --locked strike-a-pose-train --config configs/ae.yaml
RUN=$(newest ae-smpl_proc_v1)
uv run --locked python scripts/run_status.py "$RUN"            # every check-in
launch r0 uv run --locked strike-a-pose-train --resume "$RUN"    # only when it "looks dead"
launch r0-probe uv run --locked strike-a-pose-eval probe --config configs/probe.yaml \
    --checkpoint "$RUN/$(scored "$RUN")"                      # once metrics.json exists
keep "$RUN" r0 ae && keep "$(newest probe-smpl_proc_v1)" r0 probe
git add docs/results/campaign/r0 docs/HANDOFF.md   # commit "docs(campaign): R0 results"

# dataset step (DS = smpl_proc_v2a, smpl_proc_v2b, smpl_proc_v2c, smpl_proc_v2d)
launch "$DS" bash -c "uv run --locked strike-a-pose-data generate --config configs/$DS/generate.yaml --seed 0 && uv run --locked strike-a-pose-data validate data/generated/$DS"
# v2a only: the intersection gate (spec 017 rule; exit 1 = FAIL: more than 5 % of the poses have
# >= 0.4 % of the arm vertices inside the torso; real SMPL armsdown-v1 ~ 23 %; then no split, see the step rule), then the visual check (010
# tasks.md, Follow-up R); v2c: the jitter fallback count (same place); then the split
if [ "$DS" = smpl_proc_v2a ]; then
  uv run --locked python scripts/intersection_check.py --config configs/$DS/generate.yaml \
      --seed 0 > runs/intersection-$DS.txt; echo "exit $? (0 PASS, 1 FAIL, 2 INVALID)"
  cat runs/intersection-$DS.txt
fi
uv run --locked strike-a-pose-data split data/generated/$DS --seed 0 --out splits/$DS.json
launch "$DS" bash -c "uv run --locked strike-a-pose-eval baseline --config configs/$DS/baselines.yaml && uv run --locked strike-a-pose-train --config configs/$DS/ae.yaml"
# then its probe, keep (STEP v2a … v2d; KIND baselines, ae, probe), commit with splits/$DS.json
# (v2a: also runs/intersection-smpl_proc_v2a.txt as docs/results/campaign/v2a/intersection.txt)
```

### Decision rules

At every check-in, for the current step (the campaign log names it and its run dir):

1. **Dead**: `run_status.py` says `looks dead` (no live lock owner, no `metrics.json` or
   `aborted.json`, nothing written for more than max(2 × median epoch, 10 min)) → resume it
   detached. If the previous check-in already resumed it and no history row was added since →
   stop that step and report (no blind retries).
2. **Stalled** (live lock owner, nothing written for 3× that threshold) → report; do not kill or
   resume.
3. **Aborted** (exit 5, `aborted.json`) → keep and commit its records; never resume it or rerun
   the same config; the step rule says what follows.
4. **Stopped early or finished** → run its probe, keep and commit the records, apply the step
   rule; an early stop is a normal finish (log its stop and best epochs).
5. A non-training command (generate, validate, split, baselines, probe) died without its output
   → rerun it once from scratch (`generate --overwrite`; a new run dir); died again → stop that
   step and report.
6. Every result commit carries its campaign-log line; README and roadmap prose wait for an
   attended wrap-up. After a stop outcome nothing new starts; the next message reports it first.

Step rules:

- **F1**: as "F1 commands" says: reproduced (manifest check 1, or check 2 with a one-line note,
  plus the pixel diagnostic) → note it; otherwise continue with the caveat (operator decision
  2026-10-21) and report it prominently. Commit `splits/smpl_proc_v1.json` and keep the
  baselines run as `f1 baselines` (R0's baselines).
- **R0**: accepted if the epoch-1 val loss is within 0.010030 ± 2 % and `final_val_loss` within
  0.0022078 ± 3 % (the published curve stays within −1.8 … +0.1 % of it from epoch 30 on, so an
  early stop stays in the band); otherwise stop the campaign and investigate (environment,
  dataset, code). Aborted → stop the campaign (no control).
- **R0b**: the noise bar = |R0 − R0b| on the whole-image val MSE, the mean val region MSE and
  the probe's val world-frame MLP MPJPE, stated next to every later comparison. Aborted →
  recorded; comparisons then say "no noise bar".
- **v2a … v2d** (independent; each compared with R0): each starts after the previous step
  finished. v2a: `intersection_check.py` FAIL (more than 5 % of the 2,000 poses intersect) or
  INVALID (exit 2) → do not split, delete `data/generated/smpl_proc_v2a`, keep the report, stop
  the v2a step and continue with v2b (amending the table is code: a new version string such as
  `armsdown-v2`, in an attended session); PASS → the visual check, recorded with the rate in
  001 research R9; a visual check that shows frequent intersections overrides a PASS (same
  outcome as FAIL). For every v2x: its AE run's best val loss (`selection.val_loss`) above
  2 × R0's, or an aborted AE run → stop that dataset and check it (intersection count, jitter
  fallback rate, the visual check); the other datasets and the variants still run. v2d starts
  only if the most recent 2,000-pose v2x AE run's `run_info.json` `train_wall_s` ≤ 7,650 s
  (1.5 × 85 min).
- **R1, R3**: the promotion rule below; aborted → dropped.
- **R2**: at the first check-in with ≥ 10 epochs, an epoch-10 `train_kl` below 1 nat/image →
  stop the run (`kill` the lock's pid), record "abandoned: KL collapse", keep its records,
  start R2r (`configs/ae_vae_beta0.001.yaml`); R2 aborted → R2r. R2r collapsed or aborted → the
  VAE is dropped. R2 healthy at its end (last `train_kl` ≥ 1, whole-image val MSE ≤ 1.05 × R0's)
  → R2b (`configs/ae_vae_beta0.1.yaml`) once. The VAE candidate is the largest β among the VAE
  runs that meets the promotion rule.
- **Promotion (spec 015 FR-006, operator decision 2026-10-22; supersedes the rule below for
  this and later campaigns)**: on val, against the variant's own dataset's R0 and R0b, with
  seed_gap = |MSE(R0b) − MSE(R0)| / MSE(R0) (whole-image MSE) and probe_gap = |MPJPE(R0b) −
  MPJPE(R0)| (probe world-frame MLP): WIN (pixel) if ΔMSE < −seed_gap and ΔMPJPE ≤ +probe_gap;
  WIN (latent) if ΔMSE ≤ +5 % and ΔMPJPE < −2 × probe_gap; DROPPED if ΔMSE > +5 % and not a
  latent win; otherwise reported, not carried. A VAE also needs its last `train_kl` ≥ 1
  nat/image. Implemented in `strike_a_pose.eval.promotion`; `scripts/promotion.py` prints the
  verdicts from the run dirs.
- **Promotion, superseded 2026-10-22 by spec 015 FR-006** (surrogate ruling 2026-10-21, operator confirmation requested): every number on
  val; every "beats" by more than the R0 vs R0b gap on that metric. R1 wins if its mean val
  region MSE beats R0's while its whole-image val MSE ≤ 1.05 × R0's. A VAE run wins if its
  whole-image val MSE ≤ 1.05 × R0's, its last `train_kl` ≥ 1 nat/image and its probe's val
  world-frame MLP MPJPE ≤ R0's + the gap. R3 wins if its whole-image val MSE beats R0's.
  Dropped: whole-image val MSE > 1.05 × R0's (R1, R3), or collapsed / aborted after the retry
  (VAE). Otherwise reported, not carried (one seed cannot resolve 0–5 %).
- **W**: pending operator decision: which of v2a–v2d (or v1 only) the winning variants run
  on; default if no answer: v2a. Ask when R3 ends; with no answer by the next check-in, run
  the winners on `smpl_proc_v2a` (each winner's config with v2a's names and paths, plus a plain
  AE control `configs/smpl_proc_v2a/ae.yaml` if v2a's own AE run is not already committed).

Where the numbers are: whole-image val MSE `metrics.json` `splits.val.image.ae.image_mse`;
region MSE the mean of `splits.val.image.ae.regions.{hands,feet,head}.image_mse`; best val
loss `selection.val_loss`; `train_kl` in `history.csv`; probe MPJPE
`splits.val.targets.joints_world.mlp.mpjpe_mm` of the probe's `metrics.json`. The status
script prints the first three and `train_kl`.

Pending operator decisions: the promotion rule (surrogate ruling); the winners' dataset (default
v2a, see W).

### Check-in

The orchestrator arms it with `send_later` (≈ 50 min); every check-in re-arms it with this text,
verbatim:

```text
Campaign check-in (strike-a-pose, slice 012). In /home/user/strike-a-pose on branch
claude/charming-wozniak-5ouldf, follow docs/HANDOFF.md "Campaign procedure" exactly: read the
"Campaign log" (current step, run dir); run `uv run --locked python scripts/run_status.py
<run dir>` (non-training steps: their log under runs/ and their output); apply the decision
rules; act: resume a dead run once, stop a collapsed VAE, keep and commit each finished run's
records with its log line (explicit paths, never git add -A), start the next step only when
its rule passes, start nothing after a stop outcome. Re-arm this check-in with send_later in
about 50 minutes, verbatim, unless the campaign stopped or finished. Reply in at most 5 lines:
any failure, abort or stop first, then step, state, action, next check-in.
```

## Interim capsule campaign

Spec 013; runs now, without SMPL. Procedure = the campaign procedure above with the capsule
configs, F1 skipped:

- Body: `uv run --locked strike-a-pose-data make-capsule-body` (writes
  `assets/synthetic/capsule_v1.npz`, checks the pinned SHA-256 `bc1f3eb7…`; "unchanged" if
  present).
- Substitute `configs/default.yaml` → `configs/capsule_proc_v1/generate.yaml`,
  `configs/<run>.yaml` → `configs/capsule_proc_v1/<run>.yaml`, `configs/smpl_proc_v2{a,b,c,d}/`
  → `configs/capsule_proc_v2{a,b,c,d}/`, `splits/smpl_proc_*` → `splits/capsule_proc_*`; the
  capsule R0 (`ae-capsule_proc_v1`) is the control of capsule runs only; v2a's visual check is
  done on the capsule images.
- Order: v1 R0 → R0b → R1 → R2 → R3 → v2a → v2b → v2c → v2d; decision rules, guards, check-ins
  and `scripts/run_status.py` as above.
- Results: `docs/results/013-capsule/<run>/` and its README table (infrastructure tests, never
  README headline numbers); campaign-log rows marked `capsule`.
- SMPL arrival preempts: the session installing SMPL stops the running capsule step (its run
  dir stays resumable), logs it and starts F1; capsule steps resume only if the operator asks.
  Capsule outcomes may lead to code fixes (a slice, tests first), never to changes of the SMPL
  campaign's configs, thresholds or order.

## Campaign log

One row per step, updated by the check-in that changes it (committed with the step's
results); the last row is the current step.

| Step | Run dir(s) | Started → ended (UTC) | Outcome | Rule → next |
|---|---|---|---|---|
| F1 | — | not started (needs SMPL) | — | — |

## F1 commands

Verbatim from `specs/007-split-content-hash/spec.md` ("Follow-up F1", amended 2026-10-21 by
slice 012):

```bash
uv sync --locked            # the locked environment (uv.lock unchanged since 406b68f)

# the SMPL file the 2026-10-20 datasets were generated from (fixture manifest:
# inputs.smpl_model.sha256 = f408141b…); a mismatch is reported, generation continues
uv run --locked python - <<'EOF'
import hashlib, json
want = json.load(open("tests/fixtures/smpl_fixture/manifest.json"))["inputs"]["smpl_model"]["sha256"]
got = hashlib.sha256(open("assets/smpl/SMPL_NEUTRAL.npz", "rb").read()).hexdigest()
print("SMPL file matches" if got == want else f"SMPL FILE DIFFERS: {got[:12]}… is not {want[:12]}…")
EOF

uv run --locked strike-a-pose-data generate --config configs/default.yaml --seed 0  # ≈ 5 min, 351 MB

# manifest diagnostics (config, inputs, sample list, labels): same content as the dataset behind
# the published 003/004/006 runs? check 1 as generated; check 2 with the recorded SMPL file hash
uv run --locked python - <<'EOF'
import hashlib
from strike_a_pose.datagen.io import dumps, load_manifest
m = load_manifest("data/generated/smpl_proc_v1")
recorded = "f408141b4cd5b6ca6466e4e218aeb88a11c60f598114951b1d6d1f8078c1da16"
for check, smpl in ((1, m["inputs"]["smpl_model"]["sha256"]), (2, recorded)):
    m["inputs"]["smpl_model"]["sha256"] = smpl
    for dirty in (False, True):
        m["code"] = {"commit": "406b68f990c403dd34747dfc41914407d9ec2ea9", "dirty": dirty}
        print(check, dirty, hashlib.sha256(dumps(m)).hexdigest())
EOF
# 9000c915c0b9e5db495fc188821d97027c18b80a1b5859ebbee921ed745c9588 on a check-1 line: identical
# manifest content; only on a check-2 line: identical except the SMPL file's bytes (a
# re-converted file with the same arrays; note it in one line); on neither: labels differ

uv run --locked strike-a-pose-data split data/generated/smpl_proc_v1 --seed 0 \
    --out splits/smpl_proc_v1.json --overwrite        # default fractions 0.8 0.1 0.1

# the same seed, fractions and algorithm give the same pose lists (this proves the split
# procedure, not the dataset's identity)
uv run --locked python - <<'EOF'
import json, subprocess
old = json.loads(subprocess.check_output(["git", "show", "HEAD:splits/smpl_proc_v1.json"]))
new = json.loads(open("splits/smpl_proc_v1.json").read())
keys = ("algorithm", "unit", "dataset_name", "seed", "fractions", "poses", "counts")
assert [k for k in keys if old[k] != new[k]] == [], "pose assignment changed: do not commit"
assert (old["split_version"], new["split_version"]) == (1, 2)
print("pose assignment unchanged; split_version 1 -> 2")
EOF

# pixel diagnostic: the 003 baselines (deterministic, ≈ 3 min; also R0's baselines) against
# every number of the published run (relative difference <= 1e-5; strings and newer keys skipped)
uv run --locked strike-a-pose-eval baseline --config configs/baselines.yaml
uv run --locked python - <<'EOF'
import json, pathlib
old = json.load(open("docs/results/003-baselines-smpl_proc_v1/metrics.json"))
run = sorted(pathlib.Path("runs").glob("baselines-smpl_proc_v1-2*"))[-1]
new = json.load(open(run / "metrics.json"))
bad = []
def walk(a, b, path):
    if isinstance(a, dict):
        for k in a:
            walk(a[k], b.get(k) if isinstance(b, dict) else None, f"{path}.{k}")
    elif isinstance(a, int | float) and not isinstance(a, bool):
        if not isinstance(b, int | float) or abs(b - a) > 1e-5 * abs(a):
            bad.append((path, a, b))
walk(old, new, "")
print(run, "pixels reproduced" if not bad else f"{len(bad)} NUMBERS DIFFER, e.g. {bad[:3]}")
EOF
```

Reproduced = manifest check 1 (or check 2, with a one-line note) and "pixels reproduced": commit
only `splits/smpl_proc_v1.json` (and the baselines run's records, campaign procedure); update
README (drop the "still v1" note), `docs/HANDOFF.md` and the roadmap changelog (record the
outcome); tick F1 in 007 `tasks.md`. Otherwise (operator decision 2026-10-21): keep the
rewritten split, run the campaign, and record in README "Results", the roadmap changelog and
`docs/HANDOFF.md` that the published 003 / 004 / 006 numbers come from the `406b68f` dataset
(raw manifest `9000c915…`), which no longer reproduces, naming the part that differs: the SMPL
file (step 1), the labels or config (neither manifest check), the pixels (the baselines);
report it first and prominently in the next operator message.

## Operator decisions requested (2026-10-20 closing message)

Put to the operator in one batch (answers noted per item). Sequencing that
applies to every option needing numbers: SMPL access → F1 (regenerate
`smpl_proc_v1`, rewrite its split as v2) → retrain (≈ 85 min CPU per AE run on
4 vCPU, 2–3 h wall-clock with container restarts; probe ≈ 8 min, baselines
≈ 3 min). No checkpoint or dataset exists in a fresh container. Generation of
the #15 datasets waits for F1 and a same-code v1 re-baseline (slice 010 keeps
its new config fields out of v1 manifests, so F1's provenance check is
unaffected).

- **SMPL access**: `domattioli/smpl-assets` was not reachable from this session
  (GitHub access refused; not in the session's repo list). Options: grant the
  Claude GitHub App access to that repo and start the next session with it
  attached, or supply `SMPL_NEUTRAL.npz` another way (sha256 must equal the
  fixture manifest's `inputs.smpl_model.sha256`, `f408141b…`).
- **#13 epochs / early stopping**: *answered 2026-10-20, option B, implemented in
  slice 008.* Options as put (then: 60 epochs, cosine lr, final weights
  evaluated; `best.pt` saved but informational; 004 val min 0.00217 at epoch 34,
  final 0.00221, i.e. 1.8 % above the minimum). Options: A keep; B report
  `best.pt` (best val loss) as the model, 60 epochs unchanged (small code: config
  flag, fallback to final weights when val is empty as in the fixture,
  `configs/probe.yaml` repointed, 004 FR-006/Q5 amended; gain on the 004 curve
  1.8 % val MSE ≈ 0.08 dB, one seed, measured on the selecting split); C early
  stopping on val with patience P (P = 10 on the 004 curve stops after epoch 44,
  ≈ 25 % less compute, same epoch-34 weights as B; changes 004 FR-003 "val never
  changes the run" and the resume logic); D fewer epochs (e.g. 40; a 40-epoch
  cosine run is not epoch 40 of the 60-epoch run, result unknown). Lead
  recommendation: B, landed before the one re-baseline run that F1 forces anyway.
- **#15 dataset realism**: *answered 2026-10-20 (all four, in order (a)–(d), each
  its own dataset version and experiment), implemented in slice 010; runs: Next
  work, campaign procedure.* Options as put (then: predecessor range table, arms near the SMPL
  rest pose, mean shape, 8-camera level ring at 3.5 m; none of the options is
  config-selectable yet, each needs code + tests + 001 FR-002/FR-003 and layout
  contract amendments + operator sign-off, and its own dataset version
  `smpl_proc_v2…`). Options: (a) arms-down range table (reopens the 001 answer
  "keep arms-out limitation"; most limb–torso overlap); (b) random betas per pose
  (changes bone lengths, so MPJPE is not comparable with v1); (c) camera ring
  jitter — look-at the body, pitch ≤ ~5°, distance ≥ 3.5 m, height small, with an
  in-frame check (a 15° pitch or +0.3 m height pushes feet past the 256-px edge);
  (d) more poses (`num_poses`, config only; train has 1,600 distinct poses;
  2× doubles generation and epoch time and needs `context_pca_k: null`, PCA
  memory); AMASS mocap poses (config-selectable today, arms-down for free, needs
  the AMASS download + license). Lead recommendation: one change per dataset
  version, (a) first.
- **#16 training improvements / VAE** (today: plain MSE, Adam + cosine,
  BatchNorm, no dropout / weight decay / augmentation, sigmoid grey output).
  Options: (i) per-region MSE metrics (hands / feet / head box; no face joint in
  SMPL-24) — metrics only, lands before the re-baseline run; (ii) regularization
  (weight decay, small shifts, bottleneck dropout: 3 runs); (iii) loss shaping
  (mask-weighted MSE is small; a silhouette BCE term needs a mask head or soft
  threshold); (iv) VAE (model, loss, config, probe and checkpoint loading
  changes; reconstruction usually slightly worse at equal latent size; enables
  sampling / interpolation); (v) capacity sweep (latent 32 / 128; wider channels
  ≈ 2.5× slower). Every option after a #15 dataset change needs a matching plain
  AE control run on that dataset. Lead recommendation: (i) first, then (iii) or
  (iv) by goal (detail vs generation). Answered: (i) (2026-10-20, slice 009) and
  (ii)–(iv) (2026-10-20: each config-selectable, defaults = today's run, plain AE as
  control; slice 011, runs in the campaign procedure); (v) not chosen.

## Session pause (2026-10-20 23:30 UTC)

- Pushed and CI green at `5382075` (slices 007 and 008 landed). Slice 009
  `region-metrics`: spec approved at surrogate scope
  (`specs/009-region-metrics/spec.md`, "Notes for the plan"); plan/tasks/implementation
  not started — next step is phase 2 with a fresh lead. *(Phase 2 landed 2026-10-21; see
  State.)*
- `domattioli/smpl-assets` now exists and is attachable but is still empty
  (operator just created it); F1 and every retrain wait for `SMPL_NEUTRAL.npz`
  (expected SHA-256 `f408141b…`, the fixture manifest's `inputs.smpl_model.sha256`).
- Design memos for 010 (dataset realism) and 011 (training variants) were produced
  by read-only architect passes; copies live in the session scratchpad only
  (`010-design-memo.md`, `011-training-variants-memo.md`); re-derive from the
  roadmap #15/#16 decisions if the scratchpad is gone.
- Operator instruction: resume at 2026-10-21 08:04 UTC for a 3-hour window
  (scheduled), wrap-up reminder at 10:49 UTC.

## End of the 2026-10-21 autonomous window (20:30 UTC)

- Capsule campaign on `capsule_proc_v1` finished R0, R0b, R1, R2, R3 (+ probes); summaries and
  the campaign verdict are in `docs/results/013-capsule/README.md`. Not run: the β 0.1 VAE
  follow-up (`configs/capsule_proc_v1/ae_vae_beta0.1.yaml`, indicated by R2's healthy KL) and the
  independent realism datasets `capsule_proc_v2a`–`v2d` (generate with
  `configs/capsule_proc_v2x/generate.yaml`, split with seed 0, then baselines / AE / probe as for
  v1; v2a needs `scripts/intersection_check.py` first). Extend `data/capsule_queue.sh` with these
  steps (it is git-ignored under `data/`; it does not exist in a fresh container) or run them by
  hand; keep a tracked keep-alive task running while they train.
- Operator decisions still open: confirm the variant promotion rule (R1's probe gain suggests
  adding a probe-based criterion); the winners' dataset; sign-off on 007–014; the capsule
  license reading (skeleton derived from the committed fixture labels).
- SMPL: `domattioli/smpl-assets` was still empty at 20:18 UTC; F1 and the SMPL campaign wait
  for `SMPL_NEUTRAL.npz`.

## F1 outcome (2026-10-22)

`SMPL_NEUTRAL.npz` arrived via `domattioli/smpl-assets` (re-converted by the operator from the
original `.pkl`; SHA-256 `f09c8ece…`, 300 shape components of which the generator uses 10; not
the recorded `f408141b…` file). `smpl_proc_v1` regenerated at `3ba4259`+: manifest checks 1 and 2
both miss `9000c915…` because label values differ by ≤ 5e-6 (6th-decimal rounding of the
re-converted arrays; fixture regeneration: joints ≤ 5e-6, pixels ≤ 1 grey level). Pixel
diagnostic: the 003 baselines rerun (`runs/baselines-smpl_proc_v1-20261022T011445Z`, copy in
`docs/results/007-f1-baselines-smpl_proc_v1/`) reproduces **every** published 003 number to
≤ 1e-5 relative. Verdict: reproduced in substance, not byte for byte (caveat: the published
003/004/006 runs used the `406b68f` dataset, raw manifest `9000c915…`, which differs from the
regenerated one at the 1e-6 level). `splits/smpl_proc_v1.json` rewritten as `split_version` 2
with unchanged pose lists.

## SMPL campaign (started 2026-10-22 01:23 UTC, operator: run everything autonomously)

Queue `data/smpl_queue.sh` (git-ignored): v1 baselines → R0 → R0b → v2a gate (intersection
check, 5 %) → v2a / v2b / v2c / v2d (generate, split, baselines, AE, probe each) → R1 mask →
R2 VAE β 0.01 → R3 reg; every step idempotent (skips done, resumes unfinished). Progress
checks, three layers: (1) trainer guards (early stop, divergence abort → `aborted.json`);
(2) `data/watchdog.sh` as a tracked background task, re-armed every ≤ 110 min: keeps the
container alive, relaunches the queue if it died, and kills a trainer whose history has not
advanced for > max(3 × median epoch, 15 min) so the queue resumes it from its checkpoint;
(3) an hourly heartbeat (scheduled task) that compares epoch counts and queue steps with the
previous hour (`data/heartbeat.log`), handles FAILED / ABORTED / SKIPPED / STALL lines, records
finished runs under `docs/results/` and `docs/results/smpl-campaign/README.md`, and pushes.
Committed copies: `docs/campaign-scripts/` (copy back to `data/` after a fresh clone).
Incident 2026-10-22 01:23–02:00: the first watchdog never recognised the running queue
(process-name check), relaunched it every 5 min, and the parallel queues started R0b, R1–R3
and v2b–v2d at once (lock clashes, exit 2); it also killed R0 during the data loading after a
resume (stall clock read only `history.csv`). Fixed: the queue holds an exclusive `flock` (a
second instance exits), the watchdog reads that lock, the stall clock starts at the later of
the last epoch and the lock file, and the queue loops (≤ 5 passes) until every run has
`metrics.json` or `aborted.json`. Partial runs and datasets from the parallel queues were
deleted; R0 (3 clean epochs) resumed. v2a intersection gate: diagnostic only — on real SMPL
the 014 check flags 79 % of v1-table poses (and 100 % of `armsdown-v1`), so it is
miscalibrated (arm vertices near the shoulder fall inside the fitted torso); v2a runs anyway;
recalibrating the check is an open item.

## SMPL campaign finished (2026-10-22 16:49 UTC)

All queued runs done; summary in `docs/results/smpl-campaign/README.md` and README "Results".
Open operator decisions: (1) promotion metric: pixels only (current) or add the probe (R1
mask-MSE is dropped on pixels but its latent is 12 mm better, twice reproduced); (2) next
steps candidates: more poses (8,000 / AMASS), the VAE β 0.1 follow-up, R1 + VAE on v2d, a
combined realistic dataset; (3) recalibrate the 014 intersection check; (4) sign-off on
007-014 and the capsule license reading.

## 015 campaign (done 2026-10-24; results docs/results/smpl-campaign-015/)

QUEUE_DONE 2026-10-24 02:46 UTC. R0-v3 0.00158 / 131 mm; R0b-v3 noise 0.6 % / 3.3 mm; R2-v3 (VAE β 0.01) REPORTED (+0.7 %, 132 mm, prior 32 % valid); R1-v3 (mask) DROPPED (+14.5 %, 124.5 mm). Open: operator call on whether the mask-weighted pose/pixel trade should be accepted; more poses still helps.

Original plan:

Slice 015 `pose-scale-and-sampling` (spec, plan, tasks under `specs/015-pose-scale-and-sampling/`):
`smpl_proc_v3` (8,000 poses) configs, baselines with uint8 train images (no image PCA; v3 peak
≈ 2.7 GB instead of ≈ 10 GB), `peak_rss_mb` in every `run_info.json`, the 015 promotion rule
(`strike_a_pose.eval.promotion`, `scripts/promotion.py`; procedure text above), and
`strike-a-pose-eval sample` (VAE prior samples: valid-silhouette fraction, NN-MSE to train,
plain-AE and μ-reconstruction reference bars, `samples.png`). Queue
`docs/campaign-scripts/smpl015_queue.sh` (order: R2b + probe → sampling R2b, R2 → v3 generate /
split / `load_split` timing / baselines [memory gate 12 GB] → R0-v3 [gate] → R0b-v3 → R2-v3 →
R1-v3 → sampling R2-v3; ≈ 22–25 h), same lock `data/smpl_queue.lock`, log
`data/smpl015_queue.log`; terminal lines `QUEUE_DONE`, `QUEUE_INCOMPLETE`, `MEMORY_STOP`
(peak > 12 GB, trainer exit 137, or a run still unfinished after one resume). Launch (watchdog
as a harness-tracked background task, re-armed every ≤ 110 min):

    cd /home/user/strike-a-pose && cp docs/campaign-scripts/smpl015_queue.sh docs/campaign-scripts/watchdog.sh data/ && data/watchdog.sh 6600 >> data/watchdog015.log 2>&1

After the runs: verdicts with `scripts/promotion.py` (v3 variants against R0-v3 / R0b-v3),
the v1 → v2d → v3 curve and the SC-002 statement (v3's gain over v2d > 3× the v3 seed gap, or
saturated) in `docs/results/smpl-campaign-015/README.md`.

Slice 018 `pose-supervised-ae` (spec, plan, tasks under `specs/018-pose-supervised-ae/`): optional
MLP pose head on the latent (`pose_weight` > 0; loss = recon + `pose_weight` · standardized
root-relative `joints_world` MSE; head stored under its own checkpoint keys, so probe / sample /
round-trip are unchanged). Queue `docs/campaign-scripts/pose018_queue.sh` (P1 λ 0.003, P2 λ 0.03,
P3 λ 0.03 + mask 4, each probed; ≈ 17 h), same lock `data/smpl_queue.lock`, terminal lines
`QUEUE_DONE`, `QUEUE_INCOMPLETE`, `MEMORY_STOP`. Launch (watchdog as a harness-tracked background
task, re-armed every ≤ 110 min):

    cd /home/user/strike-a-pose && cp docs/campaign-scripts/pose018_queue.sh docs/campaign-scripts/watchdog.sh data/ && LOG=data/pose018_queue.log QUEUE=data/pose018_queue.sh data/watchdog.sh 6600 >> data/watchdog018.log 2>&1

After `QUEUE_DONE` (probe dirs: `runs/<run>/probed.txt`), two reports (`--split both`):

    uv run --locked python scripts/promotion.py --r0 <R0-v3> --r0-probe <probe> --r0b <R0b-v3> --r0b-probe <probe> --variant <NAME> <RUN> <PROBE> ... --split both
    uv run --locked python scripts/pose_track.py --seed <R0-v3> <probe> --seed <R0b-v3> <probe> --seed <R0c-v3> <probe> --variant <NAME> <RUN> <PROBE> ... --split both

Operator rulings 2026-10-24: (1) if the pose head reaches <= 80 mm but the probe does not, the
head's val / test MPJPE answers grill point 8 as the pose-system result; the probe stays the
latent-quality measure; report both. (2) The best pose variant gets a second seed (about +5.5 h)
only if its probe gain is within 3 sd of the next variant or its head / probe MPJPE is within 3 sd
of 80 mm (`pose_track.py` prints this verdict per split); otherwise not. Results go to
`docs/results/pose-campaign-018/README.md` (lead).

## Process rules

- Spec-kit per slice: specify → clarify → plan → tasks → analyze → implement;
  `spec.md` sign-off before plan; questions batched.
- Opus leads own design; sonnet implements fully specified tasks only.
- Constitution: `.specify/memory/constitution.md` (config-driven runs, seeds +
  run_info logged, test-first core logic, honest baselines, MVP discipline).
- Commit trailers per the session's attribution rules; push with
  `git push -u origin claude/charming-wozniak-5ouldf`.

## Environment gotchas

- **Licensed assets are not in git.** SMPL slim model lives in private repo
  `domattioli/smpl-assets` (`SMPL_NEUTRAL.npz`, 16 MB). Fresh container: attach that
  repo, clone it, copy to `assets/smpl/SMPL_NEUTRAL.npz`. Never commit it.
- **Generated data/runs are not in git** (`data/`, `runs/` ignored). Regenerate:
  `strike-a-pose-data generate --config configs/default.yaml --seed 0` (~5 min,
  351 MB). The committed `splits/smpl_proc_v1.json` is still `split_version` 1 and only
  matches a dataset generated at commit `406b68f`: after regenerating, do F1 above
  (v2 split files accept a regeneration at any commit, with a `SplitCommitWarning`). AE
  checkpoint must be retrained (`strike-a-pose-train --config configs/ae.yaml`,
  ~84 min CPU) before re-running the probe.
- **Container is reclaimed when the session idles**, killing background jobs. Run
  long jobs detached (`setsid nohup … &`), resumable (`strike-a-pose-train --resume
  RUN_DIR`), and keep the campaign check-in armed (`send_later`, text in "Campaign
  procedure"); `scripts/run_status.py` says when a run looks dead, and the run-dir lock
  refuses a resume while its process is alive.
- `.venv` is ~5.5 GB (PyPI torch with CUDA wheels) until #12 lands.

## Capsule campaign: live state (2026-10-21, autonomous window)

- Running detached: `data/capsule_queue.sh` (idempotent, git-ignored under `data/`; re-run it to
  continue after a container restart: `cd /home/user/strike-a-pose && setsid nohup
  data/capsule_queue.sh >> data/capsule_queue.log 2>&1 < /dev/null &`). Order: R0 (resume) →
  probe → R0b → R1 mask → R2 VAE β 0.01 → R3 reg, probe after each; no new run starts after
  18:40 UTC 2026-10-21 (override with `LAST_START_UTC`).
- R0 = `runs/ae-capsule_proc_v1-20261021T111109Z` (started at e131cc7 from the clean worktree
  `/home/user/sap-run` with absolute-path configs in `data/configs-wip/`; killed by a container
  restart after epoch 16 and resumed 13:24 UTC; the stale lock of the old boot was replaced as
  designed).
- Keep-alive: the container is reclaimed whenever the session has no running tracked task, even
  with a detached job (R0 lost minutes 13:29–14:22 that way). While a campaign runs, keep a
  harness-tracked background watcher (≤ 2 h, re-armed on completion) alive; the hourly heartbeat
  alone is not enough.
- Capsule baselines `runs/baselines-capsule_proc_v1-20261021T110723Z`: pose baselines equal the
  published SMPL ones to ≤ 1e-4 (013 SC-005).
