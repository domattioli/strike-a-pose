# Research: 004 Autoencoder MVP

## R0 — Suspected 003 figure bug (true bodies look smaller / shifted)

Report (lead brief): in `recon_val.png` of run
`runs/baselines-smpl_proc_v1-20261020T022918Z` the "true" column's bodies look
smaller / shifted against the mean-image and PCA columns.

Checks (2026-10-20, scratch script, full val split, 1,600 images):

- Figure assembly: every "true" tile cut back out of the PNG is byte-equal to
  `tensor_to_image` of the loaded image, and its foreground bounding box equals the
  sample's mask bounding box (e.g. `00002_Front_0`: y 80–188, x 42–165 in all three).
  Tiles are 192 wide × 256 high at the documented offsets; the raw PNGs are
  192×256 (W×H) RGB with R = G = B, loaded as `[1, 256, 192]` — no transpose.
- Same data to PCA and figure: PCA k=128 (refit on train, as in the run) vs true,
  per image: best integer cross-correlation shift (±12 px search) is (0, 0) for
  97.9 % of images and ±1 px for the rest; mean |centroid difference| 1.1 px (y) /
  0.8 px (x), mean signed difference 0.06 / −0.01 px; median bounding-box
  difference (y0, y1, x0, x1) = (−2, +4, −5, +5) px at threshold 0.075, i.e. a
  slight blur halo, median height ratio 1.045.
- Mean image (threshold 0.075) spans y 79–237, x 54–137, inside the union of true
  silhouettes (y 36–252, x 0–189).

**Finding: no bug — the columns share one image space.** The impression comes from
(1) the blurred, brighter mean/PCA columns (soft halo, arms smeared across poses)
versus the crisp, shaded true renders with thin dark limb edges; (2) poses whose
extent differs from the mean body (e.g. row 1, a tucked leg: true y 80–188, while
low-k PCA adds faint mean-like legs down to y 226); (3) viewer downscaling of the
1376×1580 grid. An upscaled crop of row 6 shows true and PCA k=128 bodies at the
same position and height. 003 numbers stand; no re-run needed.

Regression guard added anyway (004 refactors the grid into a shared helper used by
both figures): `tests/test_eval_report.py` checks that each tile cut out of the
grid at its documented offset equals the column input, for non-square tiles.

## R1 — PyTorch CPU wheel index (002 R1 follow-up)

`curl https://download.pytorch.org/whl/cpu/` → `CONNECT tunnel failed, response
403` (2026-10-20). Still blocked by the session egress policy, so the lock cannot
be produced or verified here → torch stays on PyPI (002 R1 decision); open operator
question unchanged.

## R2 — Architecture and throughput on 4 vCPUs

Benchmark (torch 2.14.1 CPU, 4 threads, random data, Adam step incl. backward;
img/s at batch 32 / 64, NCHW vs channels_last):

| Encoder channels | Params | b32 NCHW | b32 CL | b64 NCHW | b64 CL |
|---|---|---|---|---|---|
| 16-32-64-128-256-256 | 3.13 M | 152 | 161 | 150 | **177** |
| 32-64-128-128-256-256 | 3.52 M | 61 | 80 | 55 | 72 |
| 16-32-64-96-128-128 | 1.14 M | 175 | 221 | 141 | 176 |

Choice: `[16, 32, 64, 128, 256, 256]` with channels_last, batch 64: ≈ 1.2–1.4 min
per 12,800-image epoch at full 256×192 resolution, so no input resize is needed and
60 epochs ≈ 80 min. Same channel progression as the predecessor (16 → 512 over 6
stride-2 convs) capped at 256 (the 512-wide last block and its 6,144 → 64 linear
are the predecessor's largest cost for little benefit at a 4×3 map). Wider
first layers (32+) cost 2.5× for an unknown gain — not justified for an MVP bar of
"works". Decoder: ConvTranspose 4×4 stride 2 padding 1 (exact 2× upsampling,
kernel divisible by stride → no checkerboard from uneven overlap); sigmoid output
matches images in [0, 1] (002 R2).

## R3 — Data in memory

Train split as uint8 `[12800, 1, 256, 192]` = 629 MB; val/test 79 MB each. Loaded
once through the 002 `PoseImageDataset` (labels checked per item), then batches are
sliced from memory and scaled `/255` (bit-identical to the loader's float32 values).
Avoids re-decoding PNGs every epoch (~100 s per pass). The context PCA needs train
as float32 (2.5 GB) + Gram (12.8k², float32 + float64) — built after training, so
peak RSS stays ≈ the 003 run's (≈ 8 GB) and inside 15 GB.

## R4 — Determinism

`seed_everything(seed)` before model init; shuffling via `torch.Generator`
seeded from the config seed; no dropout; BatchNorm is deterministic. CPU kernels
used here (conv, conv-transpose, BN, Adam) are deterministic for a fixed thread
count, so the same machine reproduces the loss history exactly (fixture test).
Different CPU / thread counts can change float reduction order → tiny differences;
`run_info.json` records `cpu_count` and `torch_threads`.

## R5 — Model selection and test usage

Final-epoch weights are evaluated (no early stopping) — no val-driven choice in the
run, so val stays a monitor and test is scored exactly once by the real run. A
pilot on the full data (if run) uses `eval_splits: [val]`.

## R6 — Side-by-side baselines

Re-fitting the mean image and PCA k=64 inside the AE run (same code as 003, exact
PCA) costs ≈ 1–2 min and puts AE and baselines into one `metrics.json`; equality
with the 003 run's numbers is a free consistency check of data loading. The 003
pose baselines are not repeated (no pose output).
