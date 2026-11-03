# Implementation Plan: Probe Round-Trip Plausibility of VAE Samples

**Branch**: `claude/charming-wozniak-5ouldf` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

## Summary

A new `strike-a-pose-eval round-trip` command that, for an existing 015 sample run and the 006
probe run of the same VAE, rebuilds the sample sets (checked against the recorded numbers), adds
real test images, re-fits the probe's selected `smpl_pose` and `yaw` MLPs from the probe cache,
encodes every image with the VAE (μ), predicts pose + ring camera, re-renders the mean-shape SMPL
body with the generator's NumPy rasterizer, scores silhouette IoU, calibrates τ = p5 of real-val
IoU, adds a shuffled-pose control, and writes an augmented `samples.json`, per-image JSON and two
figures in a new run dir. Then a script applies it to the three existing sample runs and the
results README gets a table. Sampling, probe, generator and trainer code are unchanged.

## Technical Context

Python ≥ 3.11, PyTorch (CPU), NumPy, PyYAML, Pillow; pytest; ruff (line 100). No new dependency
(histogram figure drawn with Pillow; no matplotlib). Commands: `uv run --locked …`; never
`uv sync`. Machine: 4 vCPU, 15 GB, a training queue running (`data/grill_queue.sh`): the
implementer runs tests only; the real round-trip runs go through FR-009's script after the queue.

## Constitution Check

- **I Library-first**: logic in `eval/roundtrip.py`; CLI thin. PASS.
- **II CLI-driven, config-defined**: `configs/roundtrip_*.yaml` + `RoundTripConfig`. PASS.
- **III Reproducibility**: seeded control stream; sets rebuilt from the sample run's own config
  and verified; probe refit verified against recorded scores; run dir with config, git, env,
  input hashes, peak RSS; JSON byte-deterministic (tested). PASS.
- **IV Test-first**: every task lists its tests, run red first. PASS.
- **V Honest evaluation**: τ pre-registered and calibrated on real images; ceiling (real test),
  floor (AE prior), chance control (shuffled pose); the 015 valid fraction reported alongside;
  circularity discussed (research R4). PASS.
- **VI Simplicity**: one module, one config module, one command; reuses probe / sample /
  datagen functions; no change to existing outputs. PASS.

## Design

| Batch | Content | Files |
|---|---|---|
| B1 | config + pure functions (IoU, τ, camera, renderability, re-render, summary, histogram figure) | `eval/roundtrip_config.py`, `eval/roundtrip.py`, `configs/roundtrip_fixture.yaml`, `tests/test_roundtrip.py` |
| B2 | probe refit, set rebuild + check, `run_round_trip`, CLI | `eval/roundtrip.py`, `eval/cli.py`, `tests/test_roundtrip.py` |
| B3 | run configs, campaign script, README (runs after the queue) | `configs/roundtrip_vae*.yaml`, `docs/campaign-scripts/roundtrip016.sh`, `docs/results/smpl-campaign-015/README.md`, `specs/roadmap.md` changelog |

B2 depends on B1; B3 on B2.

### Flow of `run_round_trip(cfg)`

1. Read sample run (`SampleConfig` via `sample_config.config_from_dict(yaml)`), `samples.json`;
   probe run (`probe_config.config_from_dict(yaml)`), `metrics.json`. Load VAE (refuse non-VAE),
   reference AE if any. Check hashes (FR-002), targets, renderability, SMPL file. Load manifest
   via the train `PoseImageDataset(...).manifest`.
2. Rebuild `vae_prior`, `ae_prior`, `vae_recon_val`, `real_test`; reproduction check (FR-003).
3. Probe: `load_or_extract` → refit (FR-004) → refit check.
4. For each set: predict (μ → z → pose, yaw), re-render, IoU, valid flags; control renders.
5. τ from `real_test`; summaries; figures; write run dir (only now: refusals create nothing).

## Risks

- Refit drift across thread counts: recorded, not fatal (FR-004).
- Probe cache evicted: re-extraction ≈ 15 min on v3 (51k + 12.8k images); acceptable once.
- Render cost of degenerate predicted poses (self-overlap): same rasterizer as generation;
  bounded by face count.

## Analysis (2026-10-24, lead)

Coverage: FR-001 → T011, T012, T022; FR-002 → T021 (`run_round_trip` checks), T020 (refusals);
FR-003 → T021 `rebuild_sets` / `check_reproduced`, T020 (b); FR-004 → T021 `refit_probe` /
`refit_check`, T020 `test_refit_matches_probe`; FR-005 → T013 `rerender_masks`,
`silhouette_iou`, T021 `predict`; FR-006 → T013 `calibrate_threshold`, T010; FR-007 → T021
control, T013 `summarize`; FR-008 → T021, T020 run / determinism tests; FR-009 → T030-T032.
SC-001 is measured on the real runs (T031, recorded `wall_s` / `peak_rss_mb`); SC-002 → T032;
SC-003 → T010, T020.

Consistency findings fixed in the documents: (1) `real_test` (operator ruling) is the first `n_recon` images of the `test` split; the split is
recorded in `calibration.split`. (2) The refit check uses the probe split equal to `recon_split`, so on the
fixture it is `test` (the fixture has no val). (3) The control draws train *poses* (rows with
`camera_index == 0`), not images, so it is independent of the camera distribution.
(4) Calibration moved to real test images (operator ruling), so it no longer shares the split
the probe used to select weight decay (research R3).
No constitution conflict; no new dependency; no change to existing outputs.
