# Research: 013 Capsule Body

Read-only measurements by the lead, 2026-10-21 (locked environment, numpy 2.5.4, Python
3.13); prototype in the session scratchpad, then the committed module.

## R1 — Rest skeleton from the fixture

Undoing the chain on each fixture pose (`j[0] = J[0]`, `j[i] = j[p] + G_pᵀ (J[i] − J[p])`)
gives the same 24 joints for both poses to 2.7 µm; the mean rounded to 1 µm is within 1.6 µm
of each. Labels of the 8 views of a pose are identical (joints and pose). Landmarks: pelvis
(−0.0018, −0.2233, 0.0282), ankles y −1.09, foot joints y −1.14, head joint y 0.353, neck
0.288, shoulders x ±0.17. Mirror differences ≤ 7 mm (SMPL's own; kept).

## R2 — Skinning

`SMPLModel.forward`: joint i's rotation moves the vertices weighted to i, i.e. the bones from
i to its children; a capsule p → c is weighted to p. Caps centred on a shared joint stay
concentric under any rotation (tested: posed cap centres = posed joints), so the union never
opens at a joint.

## R3 — Render cost and resolution

The rasterizer's chunk arrays scale with (faces in chunk) × (largest face box): capsules
without cylinder rings (2,096 vertices) render at 370 ms per sample; 12 × 3 rings and rings
≤ 4 cm: 25 ms; 16 × 4 and ≤ 3 cm (chosen): 5,952 vertices, 11,744 faces, longest edge
0.073 m, 19 ms (SMPL ≈ 18 ms), 30 ms per sample through `generate_dataset` with PNG / JSON
writes (≈ 8 min per 2,000-pose dataset).

## R4 — Fidelity on the fixture (16 views, β = 0)

Joints equal the labels to 1.1 µm; all keypoints inside the capsule silhouette; IoU with the
real-SMPL masks 0.73–0.82 (mean 0.77); body area 7.45 % vs 7.46 % (per view 0.97–1.06×);
margins top / bottom / left / right ≥ 44 / 2 / 14 / 17 px (SMPL 44 / 8 / 16 / 19). Ring
cameras, 2,000 procedural poses: no joint outside the image at β = 0; samples with a vertex
outside 0.9 % (v1 table) / 0.7 % (arms-down), 2.6 % / 2.2 % with `body_shape`.

## R5 — Determinism

`np.savez` uncompressed: zip entries dated 1980-01-01, keys in call order; the stdlib zip
bytes are identical on Python 3.11 / 3.12 / 3.13. `savez_compressed` depends on the zlib
build: avoided. Floats rounded to 1e-6 with −0.0 → +0.0 absorb last-bit libm / SIMD
differences; `posedirs` float32 zeros (exact) keep the file at 18.9 MB (float64: 34 MB).
Pinned digest `bc1f3eb7275ce0fabf6fdab3fb80e36dc2bcd29eb96a6590b009dac5d0b4116a` (numpy
2.5.4); the numpy 2.4.6 leg is verified by CI (not available offline here).

## R6 — Same poses, split and joints as `smpl_proc_v1`

`make_split` depends only on pose indices and the seed: a 2,000-pose dataset with seed 0 gives
`splits/smpl_proc_v1.json`'s pose lists exactly (checked). Same generator seed and table →
same poses and cameras; β = 0 joints within µm of SMPL's → SC-005.
