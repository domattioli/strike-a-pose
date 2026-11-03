# Research: 006 Latent Probe

## R1 — Encoding cost on 4 vCPUs

Benchmark (2026-10-20, torch 2.14.1 CPU, 4 threads, 004 final checkpoint, batch
256 of random 256×192 images, `no_grad`, eval mode): encoder only, NCHW 486 img/s,
channels_last 747 img/s (max |Δ latent| 1.4e-5 between layouts). → channels_last
(as the 004 trainer): ≈ 21 s per encoder for all 16,000 images; loading the
images + labels through the 002 loader costs ≈ 60 s (004 T014: 46 s for train).
With the random-encoder control the full extraction is ≈ 2 min; the probes
themselves are small (64-d inputs). Well inside the 30-min cap, so no pilot is
needed for budget reasons.

## R2 — Linear probe: closed-form ridge

Ridge with intercept on standardized latents `Z` (train mean / std):
`W = (ZcᵀZc + λ·n·I)⁻¹ ZcᵀYc`, `b = ȳ`, solved in float64 with
`torch.linalg.solve` (64×64 system; deterministic). The `λ·n` scaling makes λ
comparable across split sizes (λ = 1 ↔ penalty equal to the per-sample data
term for unit-variance features). λ > 0 keeps the system non-singular even for the
fixture (8 samples, 64 features). For yaw, ridge on one-hot targets + arg-max
("least-squares classifier") keeps the linear probe closed-form; logistic
regression would need an iterative solver for little interpretive gain.

## R3 — Why yaw is a classification target

The 001 ring has 8 cameras at yaw 0°, 45°, …, 315° and nothing else (fixed
distance 3.5 m, height 0, focal 320 px). A regression target would be a circular
variable on 8 points: direct angle regression breaks at the 0°/360° seam, and a
(sin, cos) regression must be snapped back to the ring anyway. 8-way
classification is exact for this label set; its confusion matrix shows the
structured errors that matter for a single-view image model — front/back flips
(180°, near-identical silhouettes, only shading differs) and adjacent-view
confusions (45°).

## R4 — Joint frames

Root orientation is zeroed in `smpl_proc_v1`, so world-frame root-relative joints
are identical for the 8 views of a pose; the camera frame differs per view
(rotation about the vertical axis by the yaw). The 003 baselines are world frame,
so `joints_world` is the headline (a probe must recover the pose *and* undo the
view; a 180° view error mirrors front/back). `joints_cam` asks only for what is
visible. MPJPE is invariant to a common rotation, so mean-pose and PCA baselines
differ between the frames only because averaging / compressing across views
differs (the camera-frame mean pose blurs 8 orientations).

## R5 — SMPL pose error

Axis-angle vectors are not a Euclidean space; the error reported is the geodesic
angle `arccos((tr(R̂ᵀR) − 1)/2)` between predicted and true joint rotations
(Rodrigues), averaged over joints 1–21 (root and the two hand joints are zero in
the data). Probes still regress the raw 72-vector (simple, deterministic).
