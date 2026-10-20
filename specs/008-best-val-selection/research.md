# Research: 008 Best-Val Weight Selection

## R1 — Selection rule

The operator chose "lowest val loss". The rule is a plain argmin over the per-epoch val
losses at full precision (the floats kept in the checkpointed history; `history.csv`
rounds to 8 significant digits), earliest epoch on ties, finite values only.

- No tolerance, smoothing or "prefer later / earlier within x %": none was asked for, and
  any threshold would be a new tuned knob. On the 004 curve the minimum is 0.0021685 at
  epoch 34; epoch 39 is 0.12 % higher (0.0021712) and the final epoch 1.8 % higher. Which
  of 34 / 39 wins is within noise either way; the spec records the near-tie.
- Ties → earliest: the rule `best.pt` already used (strict `<`).
- Non-finite: today `vl < best["val_loss"]` with a NaN best is always false, so a NaN at
  epoch 1 would pin the best to epoch 1 for the whole run. Requiring `math.isfinite` in
  the one predicate used by the epoch loop, the resume recomputation and the selection
  removes that. Infinity cannot occur (sigmoid output and images in [0, 1] bound the MSE
  by 1) but is excluded by the same test.
- The per-epoch val loss is the mean per-image MSE in eval mode, the same quantity as the
  003 `image_mse` the evaluation reports: the published 004 run has `final_val_loss` =
  val `ae` `image_mse` = 0.00220785. So `selection.val_loss` is the selected weights' val
  MSE, and its gain over `training.final_val_loss` is measured on the split that selected
  it (the caveat the README and the 004 contract state).

## R2 — Write order and resume safety

Per epoch `e`: (1) train, (2) val loss `v_e`, (3) write `checkpoint.pt` (atomic: temp
file + `os.replace`), (4) if `v_e` is a new finite minimum, write `best.pt` (atomic).
Invariant after a completed iteration: `checkpoint.pt` is at `e` and `best.pt` holds the
minimum over epochs ≤ `e`. Kill points:

| Killed | State on disk | Resume |
|---|---|---|
| in (1)–(2) of `e+1`, or in (3) before the replace | checkpoint `e`, `best.pt` = min ≤ `e` | consistent; epoch `e+1` re-runs |
| between (3) and (4) of a new-best epoch `e` | checkpoint `e` (history has `v_e` as the minimum), `best.pt` = previous best `k < e` | best epoch = checkpoint epoch → `best.pt` rewritten from the checkpoint's weights, which are epoch `e`'s (same `state_dict`, BatchNorm buffers included) |
| in (4) before the replace | as the row above (a stray `best.pt.tmp` is harmless) | as above |
| after the last epoch, before or in evaluation | checkpoint `E`, `best.pt` = min ≤ `E` (or the row above with `e = E`) | evaluation only, after the same repair |

Every state is made consistent from what is on disk, without re-running an epoch, so
it does not depend on the machine. The old order (`best.pt` first) was safe only on
the same machine: a kill between the writes left `best.pt` at `e` and the checkpoint at
`e−1`, with the previous best `k`'s weights already overwritten; resume re-runs `e`,
which on the same machine reproduces the file bit for bit, but on other hardware (a
reclaimed container can come back on another CPU) `v_e` may land just above `v_k`, and
`best.pt` would then hold weights the history does not select, with `k`'s weights gone.
That was harmless while `best.pt` was informational; it is not once `best.pt` is the
reported model.

A `best_val` resume whose `best.pt` does not hold the history's best epoch while that
epoch is older than the checkpoint (a deleted or replaced file) refuses with exit 2
before training: the weights are gone, and silently scoring the final weights would
misreport. A `final` run leaves the informational file alone. At evaluation,
`load_selected_weights` checks epoch, val loss and config again, which only a hand edit
during the run can fail.

Equality with an uninterrupted run (same machine and thread count): re-run epochs are
bit-identical (004 FR-010), the selection is recomputed from the checkpointed
full-precision history, and the repaired `best.pt` holds the same tensors, so
`metrics.json` bytes and `best.pt` weights match (tests at the three kill points).

## R3 — Config keys added after a run started

`resume_training` compares the checkpoint's stored config with the run dir's
`config.yaml` parsed by `TrainConfig`; a new key with a default makes them differ for
every run dir created before the key existed (exit 2, "checkpoint does not match
config.yaml"). The fix reads both sides through `with_legacy_values`, a table that maps
each such key to the behaviour a run without it was started under:
`{"select_weights": "final"}`. It applies to run directories (`--resume`) only; a new
run from a config file without the key gets the default (`best_val`), as the operator
decided for future runs. The resumed legacy run records `selection.select_weights:
"final"` (`selected_by: "config"`), and its pre-008 `best.pt` (config without the key)
compares equal after the same fill. Rejected: refusing such run dirs (cheaper, loses an
in-flight run); resuming them under `best_val` (changes the rule mid-run, and a resumed
run would no longer equal the uninterrupted one). No such run dir exists today; the
table is the hook for later keys (009).

## R4 — Tests: scripted val losses, small model

On this machine the fixture run on a split with a val pose (train 1 / val 1 pose, 8
images each) has its val minimum at the last epoch in every setting tried (6 and 10
epochs at cosine lr 1e-3; 10 epochs at constant lr 3e-3): the fixture is far from
overfitting, so best-val and final weights coincide and no test could tell them apart;
on other CPUs the curve could differ. The selection tests therefore replace the
per-epoch val loss with a scripted curve: a callable installed with `monkeypatch` over
`trainer.eval_loss` (looked up by name at call time) that returns `script[epoch − 1]`,
or None for an empty split. It counts calls, so it continues correctly across a
`stop_after_epoch` session and a resume in the same process. Training itself is real,
so the weights of the scripted best epoch differ from the final ones. A small model
(channels `[4]*6`, latent 8) runs a 4–6 epoch fixture run in ≈ 0.4 s instead of ≈ 2 s;
the rule does not depend on the model size. To compare scores exactly, the tests
rebuild a checkpoint's model with `load_checkpoint`, move it to the channels-last layout
the trainer uses, and score it with the trainer's own `score_images` / `predict_rows`.

## R5 — Scoring the selected weights

The selected weights are read back from `best.pt` in every case (uninterrupted, resumed,
evaluation-only), so one code path produces the bytes that the resume tests compare.
`load_state_dict` copies into the existing channels-last parameters, so when the best
epoch is the last one, the scored tensors equal the in-memory final weights bit for bit
and the metrics and figure equal a `final` run's (tested). Reading a 12.5 MB file costs
< 1 s per full run.

## R6 — Versions and readers

`train_version` 1 → 2 (one constant for `metrics.json`, `checkpoint.pt`, `best.pt`): in v1
`splits.*.image.ae` always meant the final weights; in v2 it means the weights the
`selection` block names. Nothing in the code reads `train_version` (only one test
asserts it). The 006 probe needs nothing new: `load_checkpoint` reads only
`model_hparams` and `model`, the probe records `epoch` from the file (present in
`best.pt`) and `train_run_git` from the `run_info.json` next to it (checked in this
session with a probe run on a fixture `best.pt`).
