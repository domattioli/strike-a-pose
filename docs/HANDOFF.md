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

## Operator decisions requested (2026-10-20 closing message)

Put to the operator in one batch; nothing below was started. Sequencing that
applies to every option needing numbers: SMPL access → F1 (regenerate
`smpl_proc_v1`, rewrite its split as v2) → retrain (≈ 85 min CPU per AE run on
4 vCPU, 2–3 h wall-clock with container restarts; probe ≈ 8 min, baselines
≈ 3 min). No checkpoint or dataset exists in a fresh container. Any generator
change (#15) must land *after* F1, because new config fields enter
`manifest.config` and F1's provenance check could then never match `9000c915…`.

- **SMPL access**: `domattioli/smpl-assets` was not reachable from this session
  (GitHub access refused; not in the session's repo list). Options: grant the
  Claude GitHub App access to that repo and start the next session with it
  attached, or supply `SMPL_NEUTRAL.npz` another way (sha256 must equal the
  fixture manifest's `inputs.smpl_model.sha256`, `f408141b…`).
- **#13 epochs / early stopping** (today: 60 epochs, cosine lr, final weights
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
- **#15 dataset realism** (today: predecessor range table, arms near the SMPL
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
  (iv) by goal (detail vs generation).

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
