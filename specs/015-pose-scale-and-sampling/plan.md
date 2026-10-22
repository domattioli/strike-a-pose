# Implementation Plan: Pose Scale, Latent-Aware Promotion and VAE Sampling

**Branch**: `claude/charming-wozniak-5ouldf` | **Date**: 2026-10-22 | **Spec**: [spec.md](spec.md)

## Summary

Configs for an 8,000-pose dataset `smpl_proc_v3` and its four AE runs; a memory fix in the
baselines (train images stay uint8 when no image PCA is fit); a peak-RSS record in every
`run_info.json`; a pure promotion-rule module; a new `strike-a-pose-eval sample` command that
decodes N(0, I) samples of a VAE checkpoint and scores them (valid-silhouette fraction,
nearest-train-image MSE, two reference bars, a figure); the 015 campaign queue; docs. The
generator, trainer objective and probe are unchanged.

## Technical Context

Python ≥ 3.11, PyTorch (CPU), NumPy, PyYAML, Pillow; pytest; ruff (line 100). No new
dependency (connected components: a small union-find / iterative flood fill in NumPy, research
R3). Commands run as `uv run --locked …`; never `uv sync`. Machine: 4 vCPU, 15 GB RAM, no swap.

## Constitution Check

- **I Library-first**: promotion in `eval/promotion.py`, sampling in `eval/sample.py`; CLI and
  script are thin. PASS.
- **II CLI-driven, config-defined runs**: v3 datasets / runs from committed YAML;
  `sample` takes a YAML config (`eval/sample_config.py`) like `probe`. PASS.
- **III Reproducibility**: sampling uses a seeded `torch.Generator`; run dir with `config.yaml`,
  `run_info.json` (git, env, hashes of checkpoint / manifest / split, `peak_rss_mb`). The
  baselines fix is bit-identical (research R1). PASS.
- **IV Test-first**: every task below lists its tests, written and run red first. PASS.
- **V Honest evaluation**: sampling reports two reference bars (plain-AE decoder on the same
  prior samples; μ reconstructions of val) so the VAE number is read against a floor and a
  ceiling; the promotion rule is pre-registered (spec FR-006) and applied as worded (Q1). PASS.
- **VI Simplicity**: one new command, one pure module, configs. No trainer change. PASS.

## Design

### Batches (independent; each ends green and is committed separately)

| Batch | Content | Files |
|---|---|---|
| B1 | v3 configs + tests | `configs/smpl_proc_v3/*`, `tests/test_realism.py` |
| B2 | baselines uint8 train + `peak_rss_mb` | `eval/baselines.py`, `runlog.py`, `train/trainer.py`, `eval/probe.py`, tests |
| B3 | promotion rule | `eval/promotion.py`, `scripts/promotion.py`, `tests/test_promotion.py` |
| B4 | sampling command | `eval/sample_config.py`, `eval/sample.py`, `eval/cli.py`, `configs/sample*.yaml`, `tests/test_sample.py` |
| B5 | queue + docs (after B1-B4) | `docs/campaign-scripts/smpl015_queue.sh`, watchdog, 012 docs, HANDOFF, README |

B1-B4 touch disjoint files except B2/B4 both read `runlog.peak_rss_mb` (B4 calls it if
present: B4's implementer imports it from `runlog`; if B2 is not merged yet, B4 adds the same
function verbatim, which B2 then keeps — the text is fixed in tasks.md T020).

### Sampling (spec FR-008; research R2-R4)

`run_sample(cfg)`: load the VAE checkpoint (refuse a plain AE: `ConfigError`), the reference
plain-AE checkpoint (refuse a VAE or a latent-size / image-size mismatch), train split images
as uint8 and train mask areas; draw `z ~ N(0, I)` `[n, d]` from `torch.Generator().manual_seed
(seed)`; decode with both decoders; μ-reconstruct the first `n_recon` val images (in split
order); score the three sets with the same function; figure; write the run dir.

### Promotion (spec FR-006)

Pure function on numbers; a script reads the run dirs' `metrics.json` (+ the probe's) and prints
a Markdown table that B5 pastes into the README.

## Risks

- NN search cost: 1,024 × 51,200 × 49,152 MACs ≈ 2.6 TFLOP for v3, ≈ 10-20 min on 4 vCPU; v1
  ≈ 4 min. Accepted (research R4).
- Resume: `ru_maxrss` is the current process's peak; a resumed run reports the resumed
  process only (documented in `run_info.json` key name and README).
