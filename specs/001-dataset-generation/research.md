# Research: 001 Dataset Generation

Decisions made by the lead (opus) at plan time, 2026-10-17.

## R1. Renderer: in-package NumPy z-buffer rasterizer (not pyrender)

- **Decision**: Render with a small, pure-NumPy triangle rasterizer
  (`strike_a_pose.render`): z-buffer, flat Lambert shading, light from the
  camera, grey body on black, silhouette mask from coverage.
- **Rationale**: The spec's appearance (Q5: plain grey, simple light, no
  textures) needs nothing pyrender adds. pyrender needs an OpenGL backend
  (EGL/OSMesa) for headless CPU use, which is fragile in CI/containers and not
  bit-exact across drivers. A NumPy rasterizer is deterministic (float64,
  deterministic tie-break), CPU-only, testable on synthetic triangles, and
  adds zero dependencies (Principle VI). It makes US2's "documented pixel
  tolerance" fallback unnecessary on a given platform.
- **Alternatives rejected**: pyrender (GL backend, non-bit-exact, extra deps:
  pyrender, trimesh, PyOpenGL); Blender (huge, out-of-process); PyTorch3D
  (heavy build, GPU-centric).
- **Deviation note**: roadmap Q5 / spec input listed "headless pyrender" as a
  default. FR-003 only requires "headless on CPU"; this plan satisfies FR-003
  with a different renderer. Recorded in the roadmap changelog.
- **Performance**: faces are rasterized in vectorized chunks (bbox-padded
  candidate pixels per face, sorted by bbox area). Target ≤ 0.2 s/image at
  256×192 → 16,000 images in < 1 h (SC-004). Measured on the operator's first
  real run; if missed, the follow-up is a perf task, not a renderer swap.

## R2. SMPL forward: in-package NumPy LBS (not smplx/torch)

- **Decision**: Implement SMPL linear blend skinning in NumPy
  (`strike_a_pose.smpl`), loading the model from `.npz` (keys `v_template`,
  `shapedirs`, `posedirs`, `J_regressor`, `weights`, `kintree_table`, `f`) or
  from the official `.pkl` via `pickle` (`encoding="latin1"`; scipy-sparse
  `J_regressor` densified). Chumpy-pickled files fail fast with conversion
  instructions.
- **Rationale**: smplx pulls torch only for a ~60-line forward pass; this slice
  is data generation, not training. NumPy keeps generation deterministic and
  test-time light, and lets tests drive the real LBS code with a synthetic
  SMPL-format model. Torch arrives with slice 004.
- **Alternatives rejected**: smplx + torch (heavy dep, CPU float nondeterminism
  risks, nothing gained).

## R3. Pose canonicalisation

- AMASS `poses` (SMPL+H, 156 values) → SMPL 72: body joints 0–21 from
  `poses[:66]`, SMPL hand joints 22–23 set to zero.
- Global orientation (joint 0) set to zero and translation dropped: the body
  is upright, faces +Z, pelvis at the world origin, so the fixed camera ring
  sees comparable views. The original root orientation is recorded in the
  pose-source record for later use.

## R4. Sampling

- Candidate frames: for each configured sub-collection (config order), every
  `*.npz` with a `poses` array (sorted by relative POSIX path; files named
  `shape.npz` skipped), frames `0, stride, 2·stride, …`.
- Selection: `numpy.random.default_rng(seed).choice(len(candidates), N,
  replace=False)`, kept in draw order. `N > len(candidates)` → error.
- Python `random` and NumPy global seeds are also set and logged (Principle
  III). PyTorch is not imported in this slice, so there is no torch seed.

## R5. Coordinates and cameras

- World: SMPL canonical, metres, +Y up, body faces +Z, pelvis at origin.
- Camera i (i = 0..count-1), yaw θ = 360°·i/count measured toward the
  subject's right: centre `C = (-d·sinθ, h, d·cosθ)`, looking at `(0, h, 0)`.
- Extrinsics: OpenCV convention (x right, y down, z forward), world→camera
  `X_c = R·X_w + t`, `t = -R·C`.
- Intrinsics: `K = [[f,0,W/2],[0,f,H/2],[0,0,1]]`; pixel (row r, col c) covers
  continuous coords `[c, c+1) × [r, r+1)`; its centre is `(c+0.5, r+0.5)`.
- Defaults: count 8, d = 3.5 m, h = 0.0 m (pelvis/torso level), f = 320 px,
  H×W = 256×192 (images are portrait: height 256, width 192).
- Names for count = 8 (Q2): `Front_0, FrontRight_45, Right_90, BackRight_135,
  Back_180, BackLeft_225, Left_270, FrontLeft_315`. Other counts:
  `Cam<i>_<deg>`.

## R6. Visibility

Per-joint visibility = joint in front of the camera (z_c > 0) and its 2D
keypoint inside the image `[0,W)×[0,H)`. Self-occlusion is not modelled
(documented; adequate for MVP labels).

## R7. Reproducibility / identity

- Labels are JSON with floats rounded to 6 decimals; JSON written with sorted
  keys, 2-space indent, trailing newline.
- `manifest.json` contains no timestamps or host info; it is the identity
  artifact (SC-002). Per-label SHA-256 is in the manifest.
- Image/mask SHA-256 go to `checksums.json` (not part of identity, so a
  renderer difference across machines can't break the manifest).
- Timestamps, Python/NumPy/Pillow versions, platform → `run_info.json`.
- Code commit from `git rev-parse HEAD` (+ dirty flag); `"unknown"` outside git.

## R8. Lockfile

`uv.lock` generated with `uv lock` and committed (constitution: pinned deps +
lockfile). Runtime deps: numpy, pillow, pyyaml. Dev: pytest, ruff.

## R9. Procedural pose source (amendment 2026-10-20)

- Decision: default pose source samples, per joint and per local axis (x, y, z),
  an angle uniformly from a hardcoded range table (`procedural.RANGES_DEG`,
  version `proc-v1`), composes intrinsic XYZ Euler (`R = Rx @ Ry @ Rz`) and
  converts to axis-angle. `numpy.random.default_rng(seed)`, one `uniform`
  draw of shape (24, 3) per pose, in pose order → deterministic.
- SMPL rest frame: +Y up, faces +Z, +X = subject's left; rest is a T-pose.
- Ranges (degrees, x / y / z; right side mirrors y and z of the left):

| Joint | x | y | z |
|---|---|---|---|
| pelvis (root) | 0 | 0 | 0 |
| L_hip / R_hip | -90..20 (− = thigh forward) | -25..25 | -10..40 / -40..10 (abduction) |
| L_knee / R_knee | 0..120 (flexion) | 0 | 0 |
| L_ankle / R_ankle | -20..30 | -10..10 | -10..10 |
| L_foot / R_foot | -10..10 | 0 | 0 |
| spine1 | -10..30 (+ = forward bend) | -15..15 | -15..15 |
| spine2, spine3 | -5..15 | -10..10 | -10..10 |
| neck | -20..30 | -30..30 | -15..15 |
| head | -20..20 | -30..30 | -15..15 |
| L_collar / R_collar | 0 | -10..10 | -10..15 / -15..10 |
| L_shoulder / R_shoulder | -30..30 (twist) | -80..20 / -20..80 (forward) | -80..30 / -30..80 (down from T-pose) |
| L_elbow / R_elbow | 0 | -130..0 / 0..130 (flexion) | 0 |
| L_wrist / R_wrist | -20..20 | -10..10 | -30..30 |
| L_hand / R_hand | 0 | 0 | 0 |

- Ranges are ad hoc (predecessor's were unavailable). Checked by rendering a
  12-pose sheet on real SMPL: knees/elbows bend the right way, no obvious
  limb-through-torso. Joint limits are independent per axis, so mild
  self-intersection (e.g. hand into torso at extreme shoulder+elbow) is
  possible and accepted for the MVP.
- Any change to the table bumps `RANGES_VERSION`.
- Alternative kept: AMASS (R3/R4), selected with `pose_source: amass`.
