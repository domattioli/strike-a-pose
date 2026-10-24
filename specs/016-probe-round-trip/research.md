# Research: slice 016

## R1 What the probe predicts, and recovering it without saved weights

`probe_config.TARGETS = (joints_world, joints_cam, yaw, smpl_pose)`. `target_values(a,
"smpl_pose")` is `a["pose"]` [n, 72] (axis-angle, SMPL joint order, root and hands zero in the
procedural data); scored by geodesic angle over joints 1-21. `yaw` is the 8-way ring-camera
class. So the round trip can render the predicted SMPL pose directly; the joints probes are not
needed (a joints-only probe would have forced a nearest-train-pose lookup, which cannot score
novel poses: rejected, Q1 (c)).

`run_probe` keeps no weights. It fits, per target, ridge for each λ and one MLP per weight decay,
and selects on val (`metrics.selection.mlp.<target>.value`). `fit_mlp` begins with
`torch.manual_seed(seed)` and draws batches from its own `torch.Generator().manual_seed(seed)`,
so calling it again with the same inputs (standardized train latents `Standardizer.fit(train
latents)`, targets standardized by `Standardizer.fit(y)` for regression, class ids for yaw), the
same config and the selected weight decay reproduces the selected MLP. Inputs come from the
probe cache (`load_or_extract` with the probe run's own `ProbeConfig` and `_encoders(model,
pcfg)` → identical cache meta → hit). CPU float32 matmul on the same machine and thread count
is deterministic in practice; across thread counts tiny differences can occur, hence the refit
check (FR-004) rather than an equality requirement. Cost (v3): 51,200 × 64 latents, MLP
64-256-256-72, 100 epochs × 200 steps ≈ 20k steps ≈ 1-2 min per MLP on 4 vCPU; two MLPs.

## R2 Re-render

Exactly the generator's path: `SMPLModel.load(path)`, `forward(pose, zeros(num_betas))`,
`render(verts, faces, cam)`, mask = `mask > 0`. Camera from the manifest's `cameras[k]` dict
(`K, R, t, width, height, yaw_deg, name`) → `datagen.cameras.Camera` with `np.asarray(…,
dtype=np.float64)`; for v1 / v3 this equals `camera_ring(…)[k]` bit for bit (tested on the
fixture manifest). Root orientation is part of the 72-d pose (predicted ≈ 0). Cost ≈ 15-20 ms
per render (generation ≈ 19 ms / image incl. PNG); 4 sets × 1,024 × 2 (round trip + control)
≈ 8,200 renders ≈ 2.5-3 min.

## R3 Threshold calibration

The probe is imperfect (22.5° mean joint angle on v3 val), so even real images do not round-trip
to IoU 1. The ceiling is the round-trip IoU distribution of real val images; a sample is
plausible if it does as well as a typical real image's lower tail. τ = 5th percentile with
`method="lower"` (an order statistic, so `fraction(IoU ≥ τ) ≥ 0.95` on real val exactly, ties
included). The rule is fixed here before any number is seen; only its value is data-derived,
from real images only. Real val images were also used by the probe to pick the weight decay
(3 values): a negligible optimistic bias on the ceiling, noted in the README; test is reserved.

## R4 Circularity (the 015 Q3 objection)

015 rejected a probe-based measure as circular (same encoder). The round trip is not a pure
latent statistic: the IoU is computed in pixel space between the sample and an *independent
renderer's* body; a blob can only score high if some SMPL body at a ring camera covers it. The
remaining circularity (the encoder may map an off-distribution sample to a "typical" pose whose
render overlaps a typical blob) is measured by the shuffled-pose control (R5) and by the AE-prior
and real-val bars.

## R5 Shuffled-pose control

Two body silhouettes at the same camera overlap a lot (torso, legs). The control re-renders a
random train pose (one row per train pose: rows with `camera_index == 0`) with the same predicted
camera; its IoU distribution is the chance level. Seeded per set, so deterministic.

## R6 Memory

v3: probe cache ≈ 0.18 GB (all splits, both encoders), 3 × 1,024 decoded images float32
(0.2 GB each), VAE and AE models (< 0.1 GB), SMPL model; < 2 GB. No train images loaded (no
NN-MSE); the area bounds come from the source `samples.json`.
