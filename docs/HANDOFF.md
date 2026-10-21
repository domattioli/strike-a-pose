# Handoff — strike-a-pose (as of 2026-10-21, after slice 011)

Branch: `claude/charming-wozniak-5ouldf` (all work here; `main` untouched; no PRs —
operator: "one-shot, no PRs"). CI green on `ubuntu-24.04`, Python 3.11 + 3.13.

## State

- MVP done and signed off: slices 001–005, plus post-MVP 006 (latent probe).
  Operator "approve all" 2026-10-21 covers every autonomous clarification in
  `specs/00*/spec.md` and the surrogate rulings 2026-10-20 (see `specs/roadmap.md`).
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
  `ranges_version: predecessor-v2-armsdown` (arms lowered ∓65° after sparsification,
  shoulder z `[-20, 20)`, collar z `[-10, 10)`: the collar narrowing departs from the
  predecessor's table; every other draw equals v1's for the same seed), `body_shape`
  (per-pose betas N(0, 1) clipped at ±2, own stream) and `camera_jitter` (per-sample camera
  moved ±0.3 m in height and 0–0.5 m farther, aimed at the ring's look-at point and tilted
  0–5° down, kept only if every mesh vertex projects 2 px inside the image, else the ring
  camera; labels record `camera.jitter`). 003 accepts `image_pca_ks: []`. Configs for the
  chain `configs/smpl_proc_v2` (arms-down) → `v3` (+ shape) → `v4` (+ jitter) → `v5`
  (+ 4,000 poses). No new dataset, split or run exists: run procedure below (item 5).
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
  `encoder.variational`. Configs `configs/ae_{mask,vae,reg}.yaml`; no run yet (item 6).

## Next work (from roadmap open questions)

1. **F1 (slice 007 follow-up; needs SMPL):** rewrite `splits/smpl_proc_v1.json` as
   `split_version` 2. It is still v1 and only fits `smpl_proc_v1` generated at
   `406b68f`. With `assets/smpl/SMPL_NEUTRAL.npz` in place (never committed), run the
   commands under "F1 commands" below.

   Then commit only `splits/smpl_proc_v1.json`; update README (drop the "still v1"
   note), this file and the roadmap changelog (record the provenance-check outcome);
   tick F1 in `specs/007-split-content-hash/tasks.md`. If the provenance check fails
   although the SMPL file and the environment match, still rewrite the split (after the
   pose-list check) and record a reproducibility caveat in the roadmap changelog, the
   README "Results" section and here: the published 003 / 004 / 006 numbers come from
   the `406b68f` content (raw manifest `9000c915…`), which the regenerated dataset does
   not reproduce; the operator rules on it.
2. **#12 approved → do when possible:** switch torch to the CPU wheel index
   (`specs/002-data-pipeline/research.md` R1). Re-tested from the agent session
   2026-10-20: `download.pytorch.org` still refused (HTTP 403 at the proxy CONNECT,
   3 attempts); torch stays on PyPI; the switch waits for a session whose egress policy
   allows that host. Check with
   `curl -sS -o /dev/null -w "%{http_code}" https://download.pytorch.org/whl/cpu/`.
3. **#13 answered** (operator choice B, 2026-10-20) and implemented in slice 008;
   **#16 option (i)** (per-region image metrics) implemented in slice 009 (2026-10-21).
   Any new train config key must add its pre-key value to `train.config.LEGACY_VALUES`
   (008 plan); new image predictors are scored through `score_images` with the split's
   labels (009 plan); **#15** (dataset realism) implemented as generator options in slice
   010 (2026-10-21), its datasets and runs are item 5; **#16 options (ii)–(iv)** (training
   variants) implemented in slice 011 (2026-10-21), its runs are item 6. Not started: the
   SMPL-asset access problem (`domattioli/smpl-assets` exists but is still empty, which
   blocks F1 and every run) and #16 option (v) (capacity sweep, not chosen).
4. **Re-baseline run** (after SMPL access → F1; 009 has landed): `strike-a-pose-train
   --config configs/ae.yaml` (as committed: `select_weights: best_val`), launched
   detached and resumable (see "Environment gotchas"); then set `checkpoint` in
   `configs/probe.yaml` to that run's `best.pt` (its `metrics.json`
   `selection.checkpoint`) and run the probe. README, roadmap and `docs/results/` then
   report best-val weights, compared with the published final-weight 004 numbers with
   the val-only caveat. Its `metrics.json` brings the first per-region numbers (`ae`,
   `mean_image`, `pca_k64` on val and test); re-running the 003 baselines
   (`configs/baselines.yaml`, ≈ 3 min) adds them for every PCA k (optional for the
   re-baseline alone, required as the reference row of item 5).
5. **Dataset realism runs (#15, slice 010; needs SMPL):** generation waits for F1 and a
   same-code `smpl_proc_v1` re-baseline (item 4 plus the 003 baselines). Then, in order,
   `smpl_proc_v2` → `v3` → `v4` → `v5`, each with `configs/<dataset>/{generate,baselines,
   ae,probe}.yaml`: generate (seed 0), validate, split (`--seed 0`, default fractions;
   commit `splits/<dataset>.json`), baselines, AE (detached, resumable), probe; before the
   `smpl_proc_v2` split, the arms-down visual check with real SMPL (record it in 001
   research R9). Commands, the jitter acceptance one-liner and what to commit: 010
   `tasks.md`, "Follow-up R". Time on 4 vCPU: ≈ 100 min CPU per 2,000-pose dataset
   (generate ≈ 5 min / 351 MB, baselines ≈ 3 min, AE ≈ 85 min, probe ≈ 8 min);
   `smpl_proc_v5` ≈ 200 min (700 MB; baselines without image PCA, AE without the PCA
   column: image PCA on 25,600 images needs ≈ 25–30 GB, a ≥ 32 GB machine); ≈ 8.5 h CPU in
   all, about twice that in calendar time with container restarts. Comparisons:
   `smpl_proc_v2` vs `smpl_proc_v1` is controlled (same draws except four arm angles);
   `v3`+ are compared with their predecessor; 009 region numbers across datasets carry
   the caveat of 010 clarification 21.
6. **Training-variant runs (#16, slice 011; needs SMPL):** after F1 and the re-baseline
   run R0 (item 4: `configs/ae.yaml` + its probe on `smpl_proc_v1`, reused as the control,
   never rerun), same machine type and locked environment, each detached and resumable as
   item 4, each followed by the probe on its `metrics.json` `selection.checkpoint`:
   R1 `configs/ae_mask.yaml` (mask-weighted MSE, α = 4) → R2 `configs/ae_vae.yaml` (VAE,
   β = 0.1; a second β only if R2 blurs badly or `train_kl` collapses) → R3
   `configs/ae_reg.yaml` (weight decay 0.05 + shifts ≤ 4 px + latent dropout 0.1 in one
   run; single-knob runs only if R3's best val loss is below R0's). Cost ≈ 85 min per run
   (R1 ≈ 100 min: the mask-weighted loss measured +15 % per step) + ≈ 8 min per probe,
   ≈ 5 h CPU for R1–R3. Report each next to R0 (README Results,
   `docs/results/011-<run>/`): best-val epoch / loss, val / test `image_mse`, `psnr_db`,
   `silhouette_iou`, region MSEs, probe MPJPE and yaw accuracy, R2's `train_kl`; one seed
   each, no success bar (011 spec, "Experiment procedure"). With the 010 chain (item 5,
   surrogate ruling Q2): its AE runs stay plain; each winning variant runs once on the
   last 010 dataset, next to that dataset's plain run, from a copy of its `ae.yaml` plus
   the variant's keys. Item 5 and item 6 both start after R0; their order is free.

## F1 commands

Verbatim from `specs/007-split-content-hash/spec.md` ("Follow-up F1"):

```bash
uv sync --locked            # the locked environment (uv.lock unchanged since 406b68f)

# the SMPL file must be the one the 2026-10-20 datasets were generated from
# (recorded in the committed fixture manifest: inputs.smpl_model.sha256 = f408141b…)
uv run --locked python - <<'EOF'
import hashlib, json
want = json.load(open("tests/fixtures/smpl_fixture/manifest.json"))["inputs"]["smpl_model"]["sha256"]
got = hashlib.sha256(open("assets/smpl/SMPL_NEUTRAL.npz", "rb").read()).hexdigest()
assert got == want, f"SMPL file {got[:12]}… is not the recorded {want[:12]}…: stop here"
print("SMPL file matches", want[:12])
EOF

uv run --locked strike-a-pose-data generate --config configs/default.yaml --seed 0  # ≈ 5 min, 351 MB

# provenance check: same content as the dataset behind the published 003/004/006 runs?
uv run --locked python - <<'EOF'
import hashlib
from strike_a_pose.datagen.io import dumps, load_manifest
m = load_manifest("data/generated/smpl_proc_v1")
for dirty in (False, True):
    m["code"] = {"commit": "406b68f990c403dd34747dfc41914407d9ec2ea9", "dirty": dirty}
    print(dirty, hashlib.sha256(dumps(m)).hexdigest())
EOF
# identical content iff one line prints 9000c915c0b9e5db495fc188821d97027c18b80a1b5859ebbee921ed745c9588

uv run --locked strike-a-pose-data split data/generated/smpl_proc_v1 --seed 0 \
    --out splits/smpl_proc_v1.json --overwrite        # default fractions 0.8 0.1 0.1

# the pose assignment must not change (same seed, fractions, algorithm and pose set)
uv run --locked python - <<'EOF'
import json, subprocess
old = json.loads(subprocess.check_output(["git", "show", "HEAD:splits/smpl_proc_v1.json"]))
new = json.loads(open("splits/smpl_proc_v1.json").read())
keys = ("algorithm", "unit", "dataset_name", "seed", "fractions", "poses", "counts")
assert [k for k in keys if old[k] != new[k]] == [], "pose assignment changed: do not commit"
assert (old["split_version"], new["split_version"]) == (1, 2)
print("pose assignment unchanged; split_version 1 -> 2")
EOF
```

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
  work item 5.* Options as put (then: predecessor range table, arms near the SMPL
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
  control; slice 011, runs in "Next work" item 6); (v) not chosen.

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
  RUN_DIR`), and keep a scheduled heartbeat check-in that resumes them.
- `.venv` is ~5.5 GB (PyPI torch with CUDA wheels) until #12 lands.
