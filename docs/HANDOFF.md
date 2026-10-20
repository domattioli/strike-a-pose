# Handoff — strike-a-pose (as of 2026-10-21)

Branch: `claude/charming-wozniak-5ouldf` (all work here; `main` untouched; no PRs —
operator: "one-shot, no PRs"). CI green on `ubuntu-24.04`, Python 3.11 + 3.13.

## State

- MVP done and signed off: slices 001–005, plus post-MVP 006 (latent probe).
  Operator "approve all" 2026-10-21 covers every autonomous clarification in
  `specs/00*/spec.md` and the surrogate rulings 2026-10-20 (see `specs/roadmap.md`).
- Results: `docs/results/{003-baselines,004-ae,006-probe}-smpl_proc_v1/`; summary in
  README "Results". AE test MSE 0.00219 / IoU 0.881 (PCA-64: 0.00452 / 0.646).
  Probe: camera yaw 96 % (MLP), 3D joints 172 mm MPJPE (mean pose 253 mm).

## Next work (from roadmap open questions)

1. **#14 approved → implement:** split identity hashes the manifest without
   `code.commit` / `code.dirty`; commit stored informationally; loader warns on
   commit mismatch, still refuses on content mismatch. Update 002 contract + tests.
2. **#12 approved → do when possible:** switch torch to the CPU wheel index
   (`specs/002-data-pipeline/research.md` R1). Needs `download.pytorch.org`
   reachable (`curl -sS -o /dev/null -w "%{http_code}" https://download.pytorch.org/whl/cpu/`);
   it was blocked (403) in the old session despite the env allow-list change.
3. **Need an operator choice (ask in one batch):** #13 early stopping / fewer
   epochs default; #15 dataset realism (arms-down, random betas, camera
   variation); #16 training improvements / VAE. Candidate: per-region
   (hands/face/feet) image MSE.

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
  351 MB). Until #14 lands, the committed `splits/smpl_proc_v1.json` only matches a
  dataset generated at commit `406b68f`; regenerate the split if needed. AE
  checkpoint must be retrained (`strike-a-pose-train --config configs/ae.yaml`,
  ~84 min CPU) before re-running the probe.
- **Container is reclaimed when the session idles**, killing background jobs. Run
  long jobs detached (`setsid nohup … &`), resumable (`strike-a-pose-train --resume
  RUN_DIR`), and keep a scheduled heartbeat check-in that resumes them.
- `.venv` is ~5.5 GB (PyPI torch with CUDA wheels) until #12 lands.
