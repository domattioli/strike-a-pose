# Research: 002 Data Pipeline

## R1 — How PyTorch is installed (CPU, CI speed)

- **Preferred**: PyTorch CPU wheel index via uv:
  `[[tool.uv.index]] name = "pytorch-cpu", url = "https://download.pytorch.org/whl/cpu", explicit = true`
  + `[tool.uv.sources] torch = { index = "pytorch-cpu" }`. Linux CPU wheel ≈ 200 MB,
  no CUDA runtime wheels.
- **Blocker**: `download.pytorch.org` is denied (HTTP 403 at CONNECT) by this
  agent session's egress policy. `uv lock` must read that index (and `uv sync`
  download from it), so a lock that uses it can be neither produced nor verified
  here; hand-writing lock entries with hashes from elsewhere would be unverified
  and is rejected. Fetching it through CI to work around the session policy is
  also rejected.
- **Decision**: `torch>=2.5` from PyPI, universal lock (resolved 2.14.1). On
  Linux x86_64 this pulls the CUDA runtime wheels (`nvidia-*`, `cuda-toolkit`,
  `triton`), a multi-GB download; torch still runs on CPU (no GPU needed).
  Dropping those wheels with `override-dependencies` is not viable: `import torch`
  preloads the CUDA libraries when `libtorch_global_deps.so` cannot resolve them
  and raises if they are absent.
- **Consequence (measured)**: CI run 39357737620 (127ad51): `uv sync --locked`
  took 20 s (py3.11) / 26 s (py3.13), whole leg ≈ 37 s — well inside the 005
  SC-003 budget (< 5 min). Local venv ≈ 5.5 GB (torch 1.2 GB, nvidia 3.2 GB,
  triton 0.9 GB). Switching to the CPU index is a two-stanza `pyproject.toml` change plus
  `uv lock` on a machine that can reach `download.pytorch.org` — listed as an
  open operator question, not done silently.

## R2 — Image representation

Fixture check: RGB renders have R = G = B exactly; max grey 204; mask is
exactly `{0, 255}` and equals `image > 0`. One channel loses nothing.
`[0, 1]` scaling (`/ 255`) is invertible on uint8 and matches a sigmoid decoder
output; per-dataset mean/std standardization would need statistics computed on
the train split (more machinery, Principle VI) and is left to 004 if wanted.

## R3 — Split unit and leakage

Each pose is rendered from 8 cameras; the 8 samples share `pose`,
`joints_3d_world`, and are rigid transforms of each other in camera space. A
per-sample split would place most test poses in training from other angles;
held-out image reconstruction and MPJPE would then measure memorized poses.
Pose-level split removes that leak. Viewpoint is not held out (all splits share
the fixed ring); held-out-camera evaluation is a later, separate question.

## R4 — Split algorithm and file determinism

Seeded `numpy.random.default_rng(seed).permutation` over sorted pose indices
(PCG64 stream is stable across NumPy versions per NumPy's policy for
`default_rng`), round-based counts with a minimum of 1 for non-zero fractions
(tiny datasets such as the 2-pose fixture still get a held-out pose), train
must be non-empty. Serialized with the 001 `dumps` (sorted keys, 6-decimal
floats) so equal inputs give equal bytes. The file excludes timestamps and the
code commit; `algorithm` names the procedure instead. The manifest SHA-256 ties
the split to one exact dataset (the 001 manifest is byte-reproducible).

## R5 — Validation cost

Full 001 validation reads and hashes every label JSON (16k files for the
default dataset) — seconds to tens of seconds. Default construction checks the
manifest and file existence only; each item re-checks its own labels (keys,
shapes, id consistency, SC-003 for 24 joints — microseconds). The split CLI,
which runs once per dataset, always runs the full validator.

## R6 — Pose index and camera from the manifest

Manifest sample entries carry only `id` and paths. `sample_id =
f"{pose_index:05d}_{camera_name}"` (001 data model), so the pose index is the
integer before the first `_` and the camera name is the rest (camera names may
contain `_`). The loader checks the parsed values against the camera ring on
construction and against each label file on read.
