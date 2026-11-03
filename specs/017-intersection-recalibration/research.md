# Research: 017 Intersection Recalibration

All measurements: real `assets/smpl/SMPL_NEUTRAL.npz` (`f09c8ece…`), betas 0, table
`predecessor-v1-nohands`, seed 0 (v1, v2d, v3 share the table and seed; they differ in pose count),
CPU, each run < 1 min and < 1 GB, during the training queue.

## R1 Where the 014 hits come from

Rest torso segments (centre, shrunk half-extents x, y, z in m): pelvis 0 (0.177, 0.116, 0.135),
spine1 3 (0.145, 0.062, 0.123), spine2 6 (0.150, 0.048, 0.118), **spine3 9 (0.195, 0.120,
0.119)**. Rest joints: J16 x = +0.172, J17 x = −0.175 → shoulder half-span 0.1735 < 0.195.
500 v1 poses: 80.6 % flagged; segment with hits: 9 in 403, 0 in 2, 6 in 1; hit vertices'
dominant joints: 16 in 329 poses, 17 in 281, 20 in 12, 21 in 3, 22 in 2, 18 in 1; poses with any
18–23 hit 3.2 %. Inside-count percentiles of flagged poses 10/50/90: 2 / 7 / 24.
Decision: diagnosis (a). Renders (front and side, hit vertices red) of poses 4, 5, 15 (counts
9, 6, 4): clean bodies, red dots at the armpit. Poses 1407, 1915 (717, 711): forearm genuinely
inside chest / back.

## R2 Candidate fixes (2,000 v1 poses; controls as in spec FR-004)

| variant | flag | ≥10 | rest | down60 | down80 | fwd90 | hips | L-chest | L-belly | crossed |
|---|---|---|---|---|---|---|---|---|---|---|
| 014 | 80.3 % | 30.3 % | 0 | 57 | 126 | 97 | 66 | 742 | 262 | 1738 |
| 18–23 only | 3.0 % | 2.6 % | 0 | 0 | 0 | 0 | 0 | 696 | 156 | 1663 |
| drop < 10 cm of shoulder | 7.5 % | 4.3 % | 0 | 24 | 70 | 10 | 23 | 619 | 224 | 1535 |
| **spine3 cap, 5 cm** | 2.1 % | 1.8 % | 0 | 0 | 2 | 0 | 0 | 641 | 188 | 1521 |
| cap + drop | 2.1 % | 1.8 % | 0 | 0 | 2 | 0 | 0 | 571 | 188 | 1409 |

Clearance sweep (cap only): 0 cm 18.5 % (down80 83, fwd90 60), 3 cm 3.1 %, 5 cm 2.1 %,
8 cm 1.8 %. Decision: cap at 5 cm (plateau onset, ≈ upper-arm radius); arms down 80° (2) is
below the per-pose minimum (R3).

## R3 Per-pose minimum

Residual flagged poses at 1–7 vertices (714, 1575, 125, 1470, 1928, 155, 1553) render as upper
arm pressed against the side or hand touching the back: contact, not penetration. From ≈ 10
vertices on, forearm / hand clearly sunk in. Decision: `ceil(0.004 × arm_vertices)` (11 on
SMPL), scale-free so the 64-vertex synthetic body keeps threshold 1.

## R4 Final rates (cap 5 cm, minimum 11)

| set | poses | 014 | 017 any | 017 ≥ 11 |
|---|---|---|---|---|
| v1 | 2,000 | 80.30 % | 2.15 % | 35 = 1.75 % |
| v2d | 4,000 | 81.05 % | 2.70 % | 89 = 2.23 % |
| v3 | 8,000 | 80.70 % | 2.72 % | 179 = 2.24 % |
| armsdown-v1 | 2,000 | 100 % | 29.9 % | 460 = 23.0 % |

Pose table: no change (< 5 %). Rates use mean shape; `body_shape` configs (v2c) are not re-surveyed
here (the script does it per pose with its betas).
