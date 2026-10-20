# Handoff — strike-a-pose (as of 2026-10-20, third session that day, after slice 007)

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
3. **Need an operator choice:** #13 early stopping / fewer epochs default; #15 dataset
   realism (arms-down, random betas, camera variation); #16 training improvements /
   VAE. Candidate: per-region (hands/face/feet) image MSE. These, and the SMPL-asset
   access problem (GitHub access to `domattioli/smpl-assets` was refused in this
   session, which blocks F1), were put to the operator in this session's closing
   message; nothing on them was started.

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
