# Research: Run Guards and Campaign Procedure

## R1 Early-stop semantics

Decision: Keras / Lightning semantics: an epoch improves iff its val loss is below the anchor
(the val loss of the last improving epoch) by more than `min_delta`; stop after an epoch with a
finite val loss when `epoch ≥ min_epochs` and `epoch − anchor_epoch ≥ patience`. With
`min_delta: 0` the anchor is the strict finite minimum, i.e. 008's best epoch, so `best.pt`, the
selection and the stop agree. Alternative: compare against the running minimum; then a steady
decline by less than `min_delta` per epoch never resets the counter and stops a run that is
still improving. Simulation on the published 004 curve (`patience 10, min_epochs 20,
min_delta 0`): stop after epoch 44, best epoch 34 (the same weights a 60-epoch run selects), 16
of 60 epochs saved. The lr schedule stays the 60-epoch cosine, so a stopped run is a prefix of
the full run and its selection equals the full run's whenever the full run's best epoch is at or
before the stop.

## R2 Divergence thresholds

The 004 run's epoch-1 train loss is 0.0352 and every later one ≤ 0.0036; a 10× threshold
(0.352) is reached only by a collapse (e.g. a saturated output: an all-white image scores ≈ 0.9
on these mostly black renders). Three grace epochs leave the first lr-warm epochs alone; three
consecutive non-finite val losses tolerate an isolated NaN (never selected, 008). A VAE's
objective adds `β·KL/(H·W)` (≈ 1e-4 at β 0.1, KL 50 nats): far below the threshold.

## R3 Lock liveness

The container can be reclaimed and restarted; PIDs restart small and get reused. A lock is
alive only if (1) its `boot_id` equals `/proc/sys/kernel/random/boot_id` when both are known,
and (2) `/proc/<pid>/cmdline` exists and contains `strike-a-pose-train` (a zombie's command
line is empty: not alive). Without `/proc` (not Linux) the check falls back to `os.kill(pid, 0)`.
The lock is written to a temp file and moved in place, then re-read: of two launches in the
same instant the later writer wins and the other refuses; the remaining window (check, then
replace) is milliseconds wide, acceptable for "never a second resume while a run is alive"
(the 004 run's two resumes were 19 s apart). `kill -9` and reclaims leave a stale lock, which the
next start replaces. No `fcntl` lock: it dies with the process, which is fine, but it does not
survive into the status script's view of a run dir and adds platform code for no gain here.

## R4 Mean-image memory and exactness

`x.sum(dim=0, dtype=torch.float64)` on a float32 matrix allocates a float64 copy: measured on
a 4,000 × 49,152 matrix (750 MB float32), the extra peak over the matrix itself was ≈ 750 MB in
that process, the batched version 0 MB. Pixel values are `k/255` in float32: every non-zero one
is ≥ 2⁻⁸, so its 24-bit significand puts it on the grid of multiples of 2⁻³¹; sums of n ≤ 2²²
such values stay on that grid below 2²² and fit float64's 53 bits exactly, in any order. Hence
the batched float64 accumulation is bit-identical to the one-shot sum (tested).

Peak memory, 4,000-pose dataset (25,600 train images of 49,152 px; val and test 3,200 each),
dominant arrays only:

| Run | before 012 | with 012 |
|---|---|---|
| AE (`context_pca_k: null`), evaluation | uint8 train 1.3 GB + float32 train 5.0 GB + float64 copy 10.1 GB + val 0.3 GB ≈ 16.7 GB (OOM on 15 GB) | uint8 train 1.3 GB + val / test 0.6 GB + model and batches ≈ 1 GB ≈ 3 GB |
| 003 baselines (`image_pca_ks: []`) | float32 train 5.0 GB + float64 copy 10.1 GB ≈ 15.1 GB (OOM) | float32 train 5.0 GB + an eval split 0.8 GB ≈ 6 GB |

The 2,000-pose datasets halve these numbers; their image PCA (Gram matrix) fits as before.

## R5 F1 diagnostics

What each part proves: the SMPL-file hash compares bytes, not arrays (a re-conversion can
change bytes only); manifest check 1 compares the whole manifest content (config, inputs incl.
the SMPL hash, sample list, every label file's SHA-256) with the `406b68f` dataset; check 2
repeats it with the recorded SMPL hash substituted, so a match there means everything but the
model file's bytes is identical; the manifest does not cover pixels, so the 003 baselines are
rerun (deterministic: exact PCA, float64 means; 3 min) and every number of the published
`metrics.json` (122 numbers, 6 significant digits) must match within a relative 1e-5 (exact for
zeros; the published rounding resolution is ≈ 1e-6 relative), skipping strings (hashes differ
by design: split file v2, raw manifest) and keys added since (009 regions). The baselines need
the split file, and the committed v1 split pins the `406b68f` raw manifest, so the split is
rewritten first; the pose-list check then proves only that the seed, fractions and algorithm
give the same pose lists. Failure policy (operator decision 2026-10-21): continue with a
caveat naming the part that differs.

## R6 Dead-run detection

A resumed process loads its data (≈ 1 min for 16,000 samples, ≈ 2–3 min for 32,000) before it
rewrites `history.csv`; the final evaluation (≈ 1–3 min, longer with PCA) writes no row. A
"2 × median epoch" threshold (≈ 2.6 min on v1) alone would call such a live run dead. The script
therefore uses the lock (a live owner is never dead), the newest mtime of every file the trainer
writes, and a 10-min floor.

## R7 Promotion metrics and noise

Ruled by the orchestrator (operator confirmation requested): each variant has its own metric,
fixed before the runs (whole-image val MSE is what R1 and R2 are designed to give up). One seed
per run: the R0 vs R0b gap (operator decision) is the noise bar; a "beat" must exceed it. Every
promotion metric is a val number (test is reported, never used to decide).

## R8 Time estimates (4 vCPU, compute)

From 004 / 010 / 011: AE 85 min (78 s / epoch × 60 + loading and evaluation), mask-weighted +15 %,
probe 8 min, baselines 3 min, generation 5 min per 2,000 poses; the 4,000-pose dataset doubles
generation, epochs and probe. With `patience 10, min_epochs 20` an AE run that plateaus like
004 stops ≈ 25 % earlier. F1 → R3 ≈ 18 h, plus the winners step.
