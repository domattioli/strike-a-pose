# Research: 010 Dataset Realism Options

Measurements by the lead, 2026-10-21, read-only experiments in the locked environment
(numpy 2.5.4) on the committed fixture (`tests/fixtures/smpl_fixture`, real SMPL, seed 3).

## R1 — Arms-down table (`predecessor-v2-armsdown`)

*(Amended 2026-10-21 by slice 014: the table is named `armsdown-v1` — it is not the
predecessor's: besides the 65° offset it narrows shoulder z to `[-20, 20)` and collar z to
`[-10, 10)`, so a v2a effect mixes arms down, less pose variety and more limb–torso overlap. The
joint-level proxy below is complemented by a mesh-level count, `scripts/intersection_check.py`
(014 spec FR-004), which gates v2a's split at 5 % of poses.)*

- **Rest skeleton** recovered from the fixture labels by undoing the kinematic chain
  (`j_rest[i] = j_rest[p] + G_pᵀ (J[i] − J[p])`); both fixture poses agree to 2.7 µm. T-pose,
  joint-local axes world-aligned. Pelvis (−0.002, −0.223, 0.028); shoulders (0.172, 0.226),
  (−0.175, 0.225); elbows x ±0.43; wrists ±0.68; hand joints ±0.77; hips (±0.07, −0.31);
  upper arm 0.261 m, forearm 0.249 m, wrist → hand 0.086 m.
- **Axis and sign.** `R = Rz·Ry·Rx` (z outermost): adding δ to a shoulder's z angle rotates
  the whole sampled shoulder rotation about the parent's z axis (frontal-plane normal).
  Left shoulder (16) lowers with negative z, right (17) with positive z.
- **Forward kinematics** (left wrist (x, y), m): lowering 45° (0.53, −0.14); 55° (0.46, −0.19);
  65° (0.38, −0.24), hand (0.41, −0.32) at hip-joint height; 75° (0.30, −0.27); 85°
  (0.21, −0.28) beside the thigh; 90° (0.17, −0.28); 95° (0.12, −0.28); 110° (0.00, −0.25)
  through the torso. With 65° and elbow y −90° the wrist is at z = +0.21 m (forearm forward).
  The elbow z range `[-35, 5)` is not mirrored: at 65°, left elbow z −35° moves the wrist in to
  x = 0.24 m, the right one out and up to (−0.50, −0.13).
- **Candidate tables** (2,000 poses, seed 0; abduction of the upper arm in the spine3 frame,
  0° = down, 90° = T-pose, negative = toward the midline; proxy = an elbow, wrist or hand
  joint with |x| < 0.14 m, |z| < 0.12 m, −0.42 < y < 0.20 m in the spine3 frame):

  | Table | Abduction 1st / 5th / 50th / 95th pct (L) | Proxy L / R | Proxy (|x| < 0.18) L / R |
  |---|---|---|---|
  | v1 | 21 / 42 / 87 / 129° | 0.5 / 0.1 % | 0.8 / 0.2 % |
  | ∓70°, z ±20, collar ±30 | −25 / −11 / 17 / 42° | 9.6 / 5.9 % | 18.4 / 12.2 % |
  | ∓65°, z ±20, collar ±30 | −20 / −6 / 22 / 47° | 6.4 / 3.1 % | 12.4 / 8.1 % |
  | ∓60°, z ±20, collar ±30 | −15 / −1 / 27 / 52° | 4.2 / 1.9 % | 8.2 / 5.4 % |
  | ∓70°, z ±20, collar ±10 | −8 / −1 / 17 / 33° | 4.3 / 1.8 % | 11.6 / 5.9 % |
  | **∓65°, z ±20, collar ±10** | −3 / 4 / 22 / 38° | **1.8 / 1.0 %** | 6.1 / 3.5 % |

  Decision: the last row (lead default, surrogate-accepted). Collar z shares the shoulder's
  axis, so its ±30° adds directly to the elevation; ±10° keeps shrugs without crossing the arm
  into the torso. The collar narrowing departs from the predecessor's table (flagged).
- **Draw structure.** Narrowing non-zero ranges (shoulder z, collar z) leaves every other
  drawn angle and the sparsification pattern identical (2,000 poses, seeds 0 / 3 / 7): as
  measured, the generator consumes the same random words per non-zero range whatever its
  width. Zero-width ranges draw nothing, so a table must never turn a zero range non-zero (or
  the reverse) if this property is to hold. `smpl_proc_v2` therefore has `smpl_proc_v1`'s
  poses except the four changed angles (the controlled comparison of the Q1 ruling).
- Sparsification keeps each value with probability ≈ 0.52: about half of the arms sit exactly
  at ∓65°, as half sit exactly in the T-pose today.

## R2 — Body shape

- `forward` uses `shapedirs[..., :nb] @ betas[:nb]`, `nb = min(len(betas), components)`:
  betas beyond the model's components would be ignored but recorded, hence the refusal.
- N(0, 1) per component clipped at ±2 (the common SMPL choice); P(|z| > 2) ≈ 4.6 % per
  component is clipped. Stream `default_rng([seed, 1, pose_index])` (NumPy `SeedSequence`
  entropy list), independent of the pose stream `default_rng(seed)`: poses are byte-identical
  with and without `body_shape`. Shared by the 8 views (the split unit is the pose).
- Betas move bone lengths and the root (`j_rest = J_regressor @ v_shaped`): MPJPE and pose PCA
  are not comparable with mean-shape datasets (`pose` is).

## R3 — Camera jitter geometry (surrogate ruling Q2 (a))

- Ring camera i: yaw θ = 360°·i/count, centre `(−d sin θ, h, d cos θ)`, look-at `(0, h, 0)`
  (0.22 m above the pelvis), `look_at` builds x = z × up (no roll).
- Candidate: centre `(−(d+δd) sin θ, h+δh, (d+δd) cos θ)`; aim angle `a = atan2(δh, d+δd)` (the
  depression toward the look-at point); target `(0, h + δh − (d+δd)·tan(a + δp), 0)`, i.e. the
  camera aims at the look-at point, then tilts down by δp. Same `look_at` call as the ring:
  zero deltas give the ring's `R, t` bit for bit (`atan2(0, d) = 0`, `tan(0) = 0`).
- Effects (91.4 px/m; T-pose soles at y ≈ −1.20 m and crown ≈ +0.52 m estimated): δh ±0.3 m →
  elevation ±4.9°, soles move +2.5 / −3.9 px, crown ≤ 1 px; δd +0.5 m → scale 0.875; δp 1° →
  5.6 px upward shift. The memo's level translation instead moves the body 9 px per 0.1 m.
- Fixture margins at the ring cameras (16 silhouettes): top ≥ 44 px, bottom ≥ 8 px, left ≥ 16,
  right ≥ 19 (bottom 20–23 px for pose 0, 8–13 px for pose 1). An upward tilt would eat the
  small bottom margins; hence δp `[0, 5]°` down only.
- In-frame test: all mesh vertices with camera depth > 1e-6 m and `margin ≤ u < W − margin`,
  `margin ≤ v < H − margin`. The rasterizer fills pixel (r, c) when its centre (c + 0.5,
  r + 0.5) is inside a triangle, so the silhouette lies inside the vertices' bounding box: a
  vertex margin of m px leaves at least m empty rows and columns at every border. Joints alone
  would miss soles, crown and fingertips by 5–15 px.
- Stream `default_rng([seed, 2, pose_index, ring_index])`; per attempt δh, δd, δp in this
  order; up to `max_attempts` candidates; fallback = ring camera with zero deltas.
- Cost: projecting 6,890 vertices per candidate is well under 1 ms vs ≈ 18 ms of rendering per
  sample (290 s / 16,000).

## R4 — Serialization and identity

- `manifest.config` is `to_dict()`; the split's content hash and F1's provenance check cover
  it, and `inputs`, `cameras` and every label's SHA-256. Fields added after layout v1 are left
  out while at their default; enabled blocks serialize resolved (floats, lists).
- Pins: fixture `config` dict; `sample_procedural_poses(2, 3)` = fixture `euler_deg` (exact)
  and `pose` (6 decimals); `inputs.pose_source` and `cameras` rebuilt for the fixture config;
  explicit defaults ≡ absent; a same-machine before / after synthetic generation.

## R5 — Image PCA memory (FR-014)

- 003 fits image PCA on every train image via the Gram matrix: 12,800 images ≈ 10 GB peak
  (ran on 15 GB); 25,600 images: data 5.0 GB + centred copy 5.0 GB + float64 Gram 5.2 GB +
  eigenvectors 5.2 GB + `eigh` working memory ≈ 25–30 GB → a ≥ 32 GB machine. With
  `image_pca_ks: []` the 4,000-pose run holds the float32 train images (5.0 GB) for the mean
  image only; the AE run uses `context_pca_k: null`.

## R6 — Run time estimates (4 vCPU)

Generation 290 s per 16,000 samples (+ < 1 % for jitter); 003 ≈ 3 min; AE 60 epochs ≈ 85 min
CPU per 12,800 train images (≈ 170 min for 25,600); probe ≈ 8 min (≈ 16 min). Per 2,000-pose
dataset ≈ 100 min CPU, `smpl_proc_v5` ≈ 200 min; ≈ 8.5 h for the series after the v1
re-baseline; calendar about twice that (container restarts).
