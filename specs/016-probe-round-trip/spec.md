# Feature Specification: Probe Round-Trip Plausibility of VAE Samples

**Feature Branch**: `claude/charming-wozniak-5ouldf` (slice `016-probe-round-trip`)
**Created**: 2026-10-24
**Status**: Approved at surrogate scope; operator decision 2026-10-24 (grill point 5) as approval
basis (clarification batch 1 answered by surrogate ruling 2026-10-24, below)
**Input**: Operator decision 2026-10-24 (grill point 5): the 015 sampling "valid" criterion (one
body-sized connected silhouette) is too weak. Add a probe round-trip plausibility metric: encode
each sample with the VAE encoder (μ), predict pose / camera with the trained probe, re-render the
predicted body with the repo's NumPy SMPL rasterizer, score silhouette IoU between re-render and
sample; report a plausible fraction at an a-priori threshold calibrated on real test images, the
IoU distribution and the existing valid fraction, in `samples.json` and a figure. CPU-cheap.

Settled context (checked at the branch head, 2026-10-24):

- **What the probe predicts** (`eval/probe.py`, `probe_config.TARGETS`): ridge and MLP probes for
  four targets: `joints_world`, `joints_cam` (root-relative 24 × 3), `yaw` (8-way ring-camera
  class) and **`smpl_pose`** (the full 72-d axis-angle SMPL pose, `a["pose"]`). So SMPL pose
  parameters *are* predicted: no nearest-pose or new-target workaround is needed. v3 VAE probe
  (`runs/probe-smpl_proc_v3-20261023T190738Z`): MLP `smpl_pose` val 22.55° mean geodesic angle
  (random encoder 25.7°), `yaw` val accuracy 0.979 (random 0.744).
- **Probe weights are not saved.** A probe run dir holds `config.yaml`, `metrics.json`
  (incl. the selected MLP weight decay per target), `run_info.json` and figures. The latents +
  labels of every split are in the probe cache `runs/probe-cache/<ckpt12>-<manifest12>-<split12>.pt`
  (v3 VAE: `33b816f9ea8f-05e13c862ec9-1b3cae265288.pt`, 181 MB). `fit_mlp` is seeded
  (`torch.manual_seed(seed)` + own generator) per call, so the probe's selected MLP is re-fit
  exactly from the cache with the run's config and selected weight decay (research R1).
- **Render path** (`datagen/generate.py`): `SMPLModel.forward(pose[72], betas)` → verts;
  `render(verts, faces, cam)` → `(rgb, mask)`; v1 / v3 use mean shape (betas 0), ring cameras
  without jitter (manifest `cameras[k]`, `config.camera_jitter` / `body_shape` absent), root
  orientation and hands zero. Generation runs ≈ 19 ms / image on 4 vCPU (64,000 in ≈ 20 min).
- **Sampling** (`eval/sample.py`, 015 FR-008): sets `vae_prior` (z ~ N(0, I), seeded),
  `ae_prior` (same z, plain-AE decoder), `vae_recon_val` (μ reconstructions of the first
  `n_recon` images of `recon_split`). Silhouette = pixel > `iou_threshold` (0.075). Existing v3
  run `runs/sample-vae-smpl_proc_v3-20261024T024345Z`: valid 0.324 (prior), 0.932 (recon),
  0.000 (AE prior: all 1,024 empty).
- **Asset**: `assets/smpl/SMPL_NEUTRAL.npz` (SHA-256 `f09c8ece…`, = the v1 / v3 manifests'
  `inputs.smpl_model.sha256`); `/assets/` and `/runs/` are git-ignored; never committed.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Round-trip plausibility of prior samples (Priority: P1)

The researcher runs `strike-a-pose-eval round-trip --config …` on an existing sample run and its
VAE's probe run and gets, per set (`vae_prior`, `ae_prior`, `vae_recon_val`, `real_test`), the
round-trip IoU distribution and the fraction of images whose IoU reaches a threshold calibrated
on real val images, next to the 015 valid fraction, so "the sample looks like *a* body in *some*
pose" is measured, not only "one blob of body size".

**Independent Test**: on the fixture (random VAE, fixture probe run, synthetic body model) the
command writes a run dir with `samples.json` (source sets + `round_trip` blocks),
`round_trip_per_image.json`, `round_trip.png`, `round_trip_iou.png`; two runs give byte-identical
JSON; `real_test.plausible_fraction ≥ 1 − p/100`.

**Acceptance Scenarios**:

1. **Given** the v3 sample run and probe run 190738Z, **When** round-trip runs, **Then** it
   finishes in ≤ 15 min CPU and ≤ 3 GB RSS, and `samples.json` reports τ, the four distributions,
   plausible fractions, shuffled-pose control and the refit check.
2. **Given** a probe run of a different checkpoint than the sample run's VAE, **Then** it refuses
   (`ConfigError`) before creating anything.
3. **Given** a sample run whose re-decoded sets do not reproduce its recorded
   `foreground_px_mean` / `valid_fraction`, **Then** it refuses (`ConfigError`).

### User Story 2 - Applied to the existing runs (Priority: P1)

The three existing sample runs (R2 v1, R2b v1, R2 v3) get a round-trip run each; the results
README gains the table.

### Edge Cases

- Empty sample silhouette (all of R0-v3's AE-prior samples): IoU 0, counted implausible.
- Empty union (empty silhouette and empty re-render): IoU 0.
- Dataset with `camera_jitter` or `body_shape` in its manifest config: refused (the re-render
  would need per-image camera / shape the probe does not predict).
- SMPL file whose SHA-256 differs from the manifest's: run proceeds, `matches_manifest: false`
  recorded and printed as a WARNING (the fixture tests use a synthetic body).
- Probe cache missing: re-extracted with the probe's own config (logged; slow on v3, ≈ 15 min).
- Refit not reproducing the probe's recorded scores (thread nondeterminism): recorded
  (`refit.matches: false`) and printed as a WARNING, not fatal.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** New command `strike-a-pose-eval round-trip --config C [--sample-run D]
  [--probe-run D] [--smpl-model P] [--run-root R]`, YAML config `RoundTripConfig`
  (data-model.md). The 015 `sample` command is unchanged; past run dirs are never modified.
- **FR-002** Inputs and checks (before creating anything): the sample run's `config.yaml`
  (→ `SampleConfig`) and `samples.json`; the probe run's `config.yaml` (→ `ProbeConfig`) and
  `metrics.json`; `metrics.encoder.checkpoint_sha256` == SHA-256 of the sample run's VAE
  checkpoint and probe / sample `dataset.manifest_sha256` equal, else `ConfigError`; the probe
  run's targets include `smpl_pose` and `yaw`; dataset renderable (edge case); SMPL file exists
  (`MissingAssetError` → `ConfigError`).
- **FR-003** Rebuild the sets exactly as `run_sample` does (same seed, decoders, `n_recon`,
  `recon_split`) plus **`real_test`** = the first `n_recon` images of the `test` split (operator ruling, Clarifications batch 2); verify each rebuilt 015 set reproduces the source
  `foreground_px_mean` (after `round_sig`) and `valid_fraction` (with the source area bounds),
  else `ConfigError("sample run not reproduced: …")`. No NN-MSE is recomputed.
- **FR-004** Probe refit: load latents with `probe.load_or_extract` (probe config, its encoders,
  same hashes → cache hit), standardize trained latents with `Standardizer.fit(train)`, re-fit the
  `smpl_pose` MLP (on `Standardizer.fit(pose)`-standardized targets) and the `yaw` MLP with the
  probe config's `mlp_hidden / epochs / batch_size / lr / seed` and the weight decay in
  `metrics.selection.mlp.<target>.value`. Refit check: score the refit MLPs on the probe split
  equal to `recon_split` and compare to `metrics.splits.<split>.targets.{smpl_pose.mlp.angle_deg,
  yaw.mlp.accuracy}`; `matches` iff |Δangle| ≤ 0.05° and |Δacc| ≤ 0.005 (null if that split
  has no samples).
- **FR-005** Round trip per image: μ = `vae.encode(image)`; z = probe standardizer(μ);
  pose = MLP prediction (72-d, used as predicted: dims constant in train come back as their
  train mean); camera = manifest `cameras[argmax yaw]`; betas = zeros(`num_betas`);
  re-render mask = `render(SMPLModel.forward(pose, betas)[0], faces, cam)[1] > 0`; sample
  silhouette = image > `iou_threshold` of the sample run; IoU = |A ∧ B| / |A ∨ B| (0 if union
  empty).
- **FR-006** Threshold (pre-registered): τ = `np.percentile(IoU(real_test), p, method="lower")`
  with p = `calibration_percentile` = 5.0: a sample is **plausible** iff IoU ≥ τ, i.e. it
  round-trips at least as well as the worst 5 % of real images. τ depends on real images only,
  is computed before any sample set is scored, and is reported.
- **FR-007** Shuffled-pose control (chance floor): for each image, re-render a random train pose
  (seeded, `np.random.default_rng([seed, 16, set_ordinal])`, one row per train pose) with the
  same predicted camera; report the control IoU median and control plausible fraction.
- **FR-008** Outputs in a new run dir `<run_root>/<run_name>-<UTC>`: `config.yaml`;
  `samples.json` = source `samples.json` with `round_trip_version`, `source_sample_run`,
  `round_trip_criteria` and per set `round_trip` blocks added (and the new set `real_test` whose
  015 fields are `valid_fraction` / fail counts / `foreground_px_mean`, `nn_mse_*` null);
  `round_trip_per_image.json`; `round_trip.png` (8 rows: VAE prior sample | its re-render | real
  val image | its re-render); `round_trip_iou.png` (IoU histogram per set, τ line);
  `run_info.json` with git, env, input hashes, `wall_s`, `peak_rss_mb`.
- **FR-009** Campaign: `docs/campaign-scripts/roundtrip016.sh` runs the three round trips after
  the running queue ends; results table in `docs/results/smpl-campaign-015/README.md`
  (new section "016 round-trip plausibility").

### Key Entities

- **Round-trip block** (per set): n, IoU mean and quantiles, plausible fraction, plausible-and-valid
  fraction, empty-silhouette count, predicted-yaw histogram, control median / plausible fraction.
- **Round-trip criteria**: τ and its rule, probe run + refit check, SMPL file + manifest match.

## Success Criteria *(mandatory)*

- **SC-001** Each v1 / v3 round-trip run ≤ 15 min wall on 4 vCPU, ≤ 3 GB peak RSS.
- **SC-002** The README states, per run, τ, the plausible fraction of `vae_prior` vs `ae_prior`,
  `vae_recon_val`, `real_test` and the control, and whether `vae_prior` exceeds the control by
  more than the control's own spread.
- **SC-003** Unit and fixture tests cover IoU, τ, camera parsing, re-render identity,
  summaries, refusals, determinism; suite green.

## Assumptions

- Mean shape, ring cameras, zero root orientation hold for every target dataset (v1, v3).
- The probe's encoder is the sampled VAE's encoder (circularity accepted; research R4).
- 4 vCPU, 15 GB RAM, no new dependency.

## Clarifications

### Batch 1 (surrogate ruling 2026-10-24)

Answers: Q1 → (a). Q2 → (a). Q3 → (a). Q4 → (a). Q5 → (a). Questions as asked:

1. **How to re-render.** The probe predicts SMPL pose (72-d) and yaw class, but saves no weights.
   (a) Re-fit the selected `smpl_pose` and `yaw` MLPs from the probe cache with the probe run's
   config and selected weight decay (seeded, ≈ 2-4 min), checked against the recorded scores;
   (b) change the probe to save weights and re-run all probes; (c) nearest train pose by
   predicted joints (the joints probe). Default (a): uses the probe that was evaluated, costs
   minutes, no re-run; (b) is heavier and changes 006 outputs; (c) is the fallback only for a
   probe without a pose target, and would quantize poses to the train set (a sample in a novel
   pose would be scored on its nearest train pose).
2. **Threshold.** (a) τ = 5th percentile (`method="lower"`) of real-val round-trip IoU;
   (b) 1st percentile; (c) fixed IoU 0.5. Default (a): relative to what the probe+renderer
   achieves on real images (the ceiling), so probe error (22.5° mean) is not charged to the VAE;
   p5 tolerates the probe's tail failures on real images; p1 rests on ~10 images of 1,024;
   a fixed value ignores the probe's quality.
3. **Where it lives.** (a) New `round-trip` command reading an existing sample run, writing a new
   run dir with an augmented `samples.json`; (b) extend `sample` and re-run it (NN-MSE recomputed,
   45-75 min on v3). Default (a): cheap, old runs immutable, reproduction checked (FR-003).
4. **Camera / shape.** (a) Predicted yaw class → ring camera, mean shape, refuse jittered /
   shaped datasets; (b) also probe a camera / shape target. Default (a): v1 / v3 have exactly
   that generative model.
5. **Chance control.** (a) Add the shuffled-pose control (random train pose, same predicted
   camera); (b) none. Default (a): an IoU between two body silhouettes at the same camera is
   high by chance (shared torso / legs), so plausibility needs a floor to be read against.

### Batch 2 (operator ruling 2026-10-24)

1. **Calibration set.** The ceiling that calibrates τ is the round-trip IoU of real **test**
   images (set `real_test`: the first `n_recon` images of the `test` split), not val: the probe
   used val to select its weight decay. `vae_recon_val` is unchanged (its inputs are
   `recon_split`). `calibration.split` is `"test"`.
2. **Promotion.** The plausible fraction stays report-only; it is not part of any promotion rule.
