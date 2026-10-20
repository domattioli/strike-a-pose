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

Source: predecessor body-model-visualizer `generate_random_slider_vals` +
`refine_random_sliders` (operator-provided 2026-10-20). Faithful port; this
replaces an interim self-chosen table that was never released.

- Per non-root joint and axis: integer degrees uniform in `[-A, B)` (upper
  exclusive), `numpy.random.default_rng(seed).integers`.
- Sparsification: of the 69 values, draw `k` uniform integer in `[0, 69)` until
  `k >= 5`; keep `k` randomly chosen entries (without replacement), zero the rest.
- Euler `'xyz'` degrees → rotation as scipy `Rotation.from_euler('xyz')`
  (lower-case = extrinsic, `R = Rz @ Ry @ Rx`) → rotvec, implemented in NumPy;
  scipy is a dev/test-only dependency used to verify equivalence.
- Root orientation zero; mean shape (predecessor randomized betas — not adopted).
- Version string `predecessor-v1-nohands` (2026-10-20: hands 22/23 zeroed by operator; predecessor ranges kept below for reference); manifest records it plus the full table. Fixture seed 3 (seed 0 gave two near-T-poses).

| SMPL joint | x | y | z |
|---|---|---|---|
| 1 L_hip / 2 R_hip | [-130,80) | [-20,20) | [-10,10) |
| 3 spine1 | [-50,100) | [-75,75) | [-15,15) |
| 4 L_knee / 5 R_knee | [0,130) | [-20,20) | 0 |
| 6 spine2 | [-20,20) | [-15,15) | [-20,20) |
| 7 L_ankle / 8 R_ankle | [-30,70) | [-30,30) | [-25,30) |
| 9 spine3 | [-35,25) | [-50,50) | [-45,45) |
| 10 L_foot / 11 R_foot | [-40,40) | 0 | [-20,20) |
| 12 neck | [-45,35) | [-50,50) | [-20,20) |
| 13 L_collar / 14 R_collar | [-10,10) | [-10,10) | [-30,30) |
| 15 head | [-30,30) | [-30,30) | [-30,30) |
| 16 L_shoulder / 17 R_shoulder | [-20,20) | [-40,20) | [-50,50) |
| 18 L_elbow | [-60,110) | [-145,0) | [-35,5) |
| 19 R_elbow | [-60,110) | [0,145) | [-35,5) |
| 20 L_wrist / 21 R_wrist | [-50,50) | [-35,35) | [-50,50) |
| 22 L_hand / 23 R_hand | [-25,25) | [-15,15) | [-90,45) |

- Visual check (12 poses, real SMPL): knees/elbows bend plausibly. Observations,
  kept faithful: shoulder z is ±50° around the SMPL T-pose, so arms are never
  lowered to the sides; left/right shoulder and wrist ranges are not mirrored;
  some strong spine bends; hand joints 22/23 are non-zero (predecessor table).
- Predecessor cameras (option only, not adopted; we keep our ring): look-at
  (0,1,0), up +Y, eyes at y=2: Front (0,2,3), FL (4/3,2,8/3), FR (-4/3,2,8/3),
  Back (0,2,-3), BL (4/3,2,-8/3), BR (-4/3,2,-8/3), SideL (3,2,0), SideR (-3,2,0);
  vertical FOV 60°; feet translated to y=0; unlit inverted silhouette.
