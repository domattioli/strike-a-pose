# Tasks: Run Guards and Campaign Procedure

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md). Tests first within each stage (constitution IV); red runs noted.
Before every commit: ruff check, ruff format --check, full pytest (count, wall time); for trainer
changes also the fixture train + probe smoke commands. Commits name explicit paths.

## Stage 1 — plan (`plan(012)`)

- [x] T001 Spec: rulings Q1–Q3, the red-team additions and the operator's decisions of
  2026-10-21 folded in (`spec(012)`).
- [x] T002 plan.md, research.md, data-model.md, tasks.md (the campaign procedure below is the
  text HANDOFF carries); analyze pass (below).

## Stage 2 — run guards (`feat(012): run guards`)

- [x] T003 Tests (red first), `tests/test_train_guards.py`: config defaults, partial
  `early_stop`, every rejection, `LEGACY_VALUES`, YAML round trip; `early_stop_state` and
  `divergence_reason` units; fixture runs with scripted val / train losses: stop epoch, record,
  log line, history prefix and lr, selection / `splits` / `best.pt` equal to the full run's,
  resume across both kill points, empty val; abort paths (a)–(c) through the CLI, `aborted.json`,
  resume refusal, kill before `aborted.json`; lock (live helper process refused, stale lock
  replaced, released after finish / abort / error, foreign pid stale); legacy resume; default run
  unchanged. Red run: collection error (no `EarlyStop`). 43 tests.
- [x] T004 `train/config.py`: `EarlyStop`, three keys, validation, `LEGACY_VALUES`.
- [x] T005 `train/guards.py`: rules, `abort_record`, `RunAborted`, lock helpers.
- [x] T006 `train/trainer.py` (loop, resume refusals, lock, records), `train/cli.py` (exit 5).
- [x] T007 Checks; fixture train + probe smoke. ruff check and ruff format --check clean;
  pytest 417 passed in 45.2 s (374 before); validate, fixture train and probe exit 0 (8.6 s);
  the fixture train's `metrics.json` and figure byte-identical to `e98134e`'s, history identical
  except `epoch_s`, weights equal, `config.yaml` + the three keys.

## Stage 3 — status script, configs, memory, provenance (`feat(012): campaign status script + configs`)

- [ ] T008 Tests (red first): `tests/test_run_status.py` (finished, stopped early, aborted,
  killed mid-run with `--now`, live lock, VAE `train_kl`, not a run dir → 2, nothing written);
  `tests/test_run_records.py` (`image_mean` = one-shot float64 mean bit for bit on uint8 and
  float rows with odd batch sizes; `env.cpu_model`; `dataset.checksums_sha256` in the 003 / 004
  / 006 fixture `run_info.json`); configs (guards in the four campaign configs, not in the
  fixture config; each new config differs from its parent only in its knob and run name);
  FR-015 pins in `tests/test_realism.py`.
- [ ] T009 `eval/baselines.py` `image_mean`, used by `fit_baselines` and the trainer (float32
  train matrix only for the PCA column).
- [ ] T010 `runlog.py` `cpu_model`, `checksums_sha256`; 003 / 004 / 006 `run_info.json`.
- [ ] T011 `scripts/run_status.py`.
- [ ] T012 Configs (FR-012); `tests/test_train_variants.py` (β 0.01), `tests/test_realism.py`
  (guard keys exempt until the 010 amendment).
- [ ] T013 Checks; smoke; SC-002 same-machine comparison against the `e98134e` fixture outputs.

## Stage 4 — docs (`docs(012)`)

- [ ] T014 Contracts: 004 `train-run.md`, 003 `eval-run.md`, 006 `probe-run.md`, 001
  `dataset-layout.md`.
- [ ] T015 Dated notes: 004, 008, 011 specs; 007 spec + tasks F1 block; 010 tasks note.
- [ ] T016 README; T017 roadmap; T018 HANDOFF (state, sign-off list, next work, campaign
  procedure + log, F1 block).
- [ ] T019 Verification record: checks, SC-002, SC-004 (procedure and F1 copies identical,
  `bash -n`), the F1 manifest and pixel methods tried on the fixture.

## Campaign procedure

Operator decision 2026-10-21: the campaign below may use ≈ 20 h of CPU, with guards that end a
stalled or diverging run (slice 012) and check-ins that apply these rules. Prerequisite:
`assets/smpl/SMPL_NEUTRAL.npz` in place (never committed). Run everything from the repository
root, detached and resumable, and commit each finished run's records when it ends: nothing may
depend on the container surviving.

### Steps (in order; compute time on 4 vCPU)

| Step | What | Configs | Time |
|---|---|---|---|
| F1 | regenerate `smpl_proc_v1`; SMPL-file, manifest and pixel diagnostics; split v2; the 003 baselines (also R0's) | "F1 commands" | 15 min |
| R0 | re-baseline AE (the control) + probe | `configs/ae.yaml`, `configs/probe.yaml` | 95 min (≈ 70 if it stops near epoch 44) |
| R0b | R0 with seed 1 + probe: the noise bar | `configs/ae_seed1.yaml`, `configs/probe.yaml` | 95 min |
| v2a | arms-down dataset (`armsdown-v1`): generate, checks, split, baselines, AE, probe | 010 amendment | 100 min |
| v2b | body-shape dataset, same steps | 010 amendment | 100 min |
| v2c | camera-jitter dataset, same steps | 010 amendment | 100 min |
| v2d | 4,000-pose dataset, same steps (baselines without image PCA, AE with `context_pca_k: null`) | 010 amendment | 200 min |
| R1 | mask-weighted MSE (α 4) + probe | `configs/ae_mask.yaml` | 110 min |
| R2 | VAE β 0.01 + probe, then once R2r (β 0.001) or R2b (β 0.1) | `configs/ae_vae.yaml`, `configs/ae_vae_beta0.001.yaml`, `configs/ae_vae_beta0.1.yaml` | 95 min each |
| R3 | weight decay + shifts + latent dropout + probe | `configs/ae_reg.yaml` | 95 min |
| W | winning variants on a realism dataset | operator decision pending | ≈ 3 h each |

F1 → R3 ≈ 18 h, less when runs stop early. v2a–v2d are independent single-change datasets, each
compared with v1 (R0); their order, the arms-down table and their configs follow the operator's
2026-10-21 decision and the 010 amendment (pending, separate lead), whose AE configs carry this
slice's guard keys. If those configs are not committed when R0b ends, run R1 → R3 first. v2d
needs the batched mean image of slice 012: peak ≈ 3 GB for its AE run and ≈ 6 GB for its
baselines (≈ 16.7 / 15.1 GB before, an OOM on 15 GB; 012 research R4).

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

# dataset step (DS = the v2a … v2d dataset name of the 010 amendment)
launch "$DS" bash -c "uv run --locked strike-a-pose-data generate --config configs/$DS/generate.yaml --seed 0 && uv run --locked strike-a-pose-data validate data/generated/$DS"
# v2a: the automatic intersection count (threshold per the 010 amendment) plus the visual
# check; v2c: the jitter fallback count (010 tasks.md, Follow-up R); then
uv run --locked strike-a-pose-data split data/generated/$DS --seed 0 --out splits/$DS.json
launch "$DS" bash -c "uv run --locked strike-a-pose-eval baseline --config configs/$DS/baselines.yaml && uv run --locked strike-a-pose-train --config configs/$DS/ae.yaml"
# then its probe, keep (STEP v2a … v2d; KIND baselines, ae, probe), commit with splits/$DS.json
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
- **v2a … v2d**: each starts after the previous step finished. Its AE run's best val loss
  (`selection.val_loss`) above 2 × R0's, or an aborted AE run → stop the dataset series and
  check that dataset (intersection count, jitter fallback rate, the visual check); the variants
  still run. v2d starts only if the v2c AE run's `run_info.json` `train_wall_s` ≤ 7,650 s
  (1.5 × 85 min).
- **R1, R3**: the promotion rule below; aborted → dropped.
- **R2**: at the first check-in with ≥ 10 epochs, an epoch-10 `train_kl` below 1 nat/image →
  stop the run (`kill` the lock's pid), record "abandoned: KL collapse", keep its records,
  start R2r (`configs/ae_vae_beta0.001.yaml`); R2 aborted → R2r. R2r collapsed or aborted → the
  VAE is dropped. R2 healthy at its end (last `train_kl` ≥ 1, whole-image val MSE ≤ 1.05 × R0's)
  → R2b (`configs/ae_vae_beta0.1.yaml`) once. The VAE candidate is the largest β among the VAE
  runs that meets the promotion rule.
- **Promotion** (surrogate ruling 2026-10-21, operator confirmation requested): every number on
  val; every "beats" by more than the R0 vs R0b gap on that metric. R1 wins if its mean val
  region MSE beats R0's while its whole-image val MSE ≤ 1.05 × R0's. A VAE run wins if its
  whole-image val MSE ≤ 1.05 × R0's, its last `train_kl` ≥ 1 nat/image and its probe's val
  world-frame MLP MPJPE ≤ R0's + the gap. R3 wins if its whole-image val MSE beats R0's.
  Dropped: whole-image val MSE > 1.05 × R0's (R1, R3), or collapsed / aborted after the retry
  (VAE). Otherwise reported, not carried (one seed cannot resolve 0–5 %).
- **W**: stop and ask the operator which dataset (the cumulative v5 no longer exists).

Where the numbers are: whole-image val MSE `metrics.json` `splits.val.image.ae.image_mse`;
region MSE the mean of `splits.val.image.ae.regions.{hands,feet,head}.image_mse`; best val
loss `selection.val_loss`; `train_kl` in `history.csv`; probe MPJPE
`splits.val.targets.joints_world.mlp.mpjpe_mm` of the probe's `metrics.json`. The status
script prints the first three and `train_kl`.

Pending operator decisions: the promotion rule (surrogate ruling); the winners' dataset; the 010
amendment (v2a–v2d configs, `armsdown-v1`, the intersection threshold).

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

## Analyze pass (2026-10-21, before stage 2)

- Coverage: FR-001 → T003–T004 · FR-002 / FR-003 → T003, T005–T006 · FR-004 / FR-005 /
  FR-006 → T003, T005–T006 · FR-007 → T003, T005–T006, T011 · FR-008 / FR-009 → T003, T013 ·
  FR-010 → T008–T009, T013 · FR-011 → T008, T010, T014 · FR-012 → T008, T012 · FR-013 → T008,
  T011 · FR-014 → T002, T018, T019 · FR-015 → T008, T014 · FR-016 → T015, T018, T019 · FR-017 →
  T003, T008 · FR-018 → T014–T018 · FR-019 → every commit's file list.
- FR-017 (m) (the F1 methods on the fixture) is a scripted check in T019, as 007 did for its
  provenance check: the F1 snippets are documentation, not package code.
- Consistency: "anchor" (early stop) vs "best epoch" (008 selection) are distinct terms and
  coincide at `min_delta: 0`; "looks dead" / "stalled" are the status script's states and the
  procedure's rule names; dataset names v2a–v2d are the operator's, configs follow the 010
  amendment.
- Conflict found: the 010 series test requires `configs/smpl_proc_v*/ae.yaml` to equal
  `configs/ae.yaml` except names and paths, but this slice may not edit those configs (operator
  decision): the test exempts the three guard keys until the 010 amendment adds them (T012).
- Conflict found: the 011 config test pins `ae_vae.yaml` at β 0.1; the operator moved the first
  run to 0.01 → test expectation updated with the config (T012).
- Risk: the status script imports the package (torch import ≈ 1–2 s); acceptable for a check
  every 50 min (SC-003: < 5 s).
