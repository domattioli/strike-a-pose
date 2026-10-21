# Tasks: 011 Training Variants

Input: [spec.md](spec.md) (approved at surrogate scope, rulings 2026-10-21), [plan.md](plan.md),
[research.md](research.md), [data-model.md](data-model.md). Tests first in every stage (red run
noted), then code; each stage ends green (ruff check, ruff format --check, full pytest; stages
touching the trainer also the CI fixture train + probe smoke commands) and is committed with
explicit paths. Commands: `uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked pytest -q`; smoke: `uv run --locked strike-a-pose-train --config
configs/ae_fixture.yaml --run-root <tmp>/runs`, then `uv run --locked strike-a-pose-eval probe
--config configs/probe_fixture.yaml --checkpoint <run>/checkpoint.pt --run-root <tmp>/probe`.

## Stage 1 — plan (`plan(011)`)

- [x] T001 Spec: rulings folded in (Q1–Q3 option (a)), status approved at surrogate scope.
- [x] T002 plan.md, research.md, data-model.md, tasks.md; analyze pass (below).
- [x] T003 Contract amendments: `specs/004-autoencoder-mvp/contracts/train-run.md` (model API,
  six config rows, objective, `history.csv` header, `model_hparams`, `metrics.json` keys),
  `specs/006-latent-probe/contracts/probe-run.md` (VAE checkpoints, `encoder.variational`).

## Stage 2 — config + optimizer + regularization (`feat(011): config + optimizer + regularization`)

- [x] T004 Tests (red first), `tests/test_train_variants.py`: config defaults / rejections /
  `beta`–`variational` coupling / `LEGACY_VALUES` = defaults; regression: 2 epochs of
  `train_epoch` (defaults, `make_optimizer`) vs an inline copy of the pre-011 loop (Adam,
  `F.mse_loss`) on the fixture train images, batch 3: equal losses, parameters, optimizer
  state tensors, `torch.get_rng_state()`; `shift_batch` semantics (k = 0 no draw; offsets,
  zero fill, exact values); dropout p = 0 no draw; fixture runs (small channels): regularized
  run repeats exactly, all-knobs resume equals uninterrupted, legacy run dir resumes;
  `history.csv` columns, `training.objective`.
- [x] T005 `train/config.py`: six fields, validation, `MAX_SHIFT_PX`, `LEGACY_VALUES`.
- [x] T006 `train/objective.py`: `Objective`, `shift_batch`; `train/trainer.py`:
  `make_optimizer` (AdamW), `train_epoch(…, objective=None)` → dict, dropout on z, history
  columns + legacy row completion, `training.objective`; adapt
  `tests/test_train.py::test_train_epoch_partial_last_batch` to the dict return.
  Stage record: tests written before the code (not run red: the module cannot import before
  T005–T006). Shipped without `mask_weight` / `variational` / `beta` (stages 3–4 add each
  key with its code, so no key is ever accepted and ignored). Checks: ruff and format clean;
  pytest 354 passed in 37.95 s (343 before); CI smoke train + probe OK.

## Stage 3 — mask-weighted MSE (`feat(011): mask-weighted MSE`)

- [ ] T007 Tests (red first): `recon_loss` α = 0 `torch.equal` to `F.mse_loss` (value and
  gradient); 2 × 2 hand value; body error costs (1 + α)× background; all-background batch =
  MSE; fixture `(image > 0) == mask` (16 / 16); mask run repeats; `train_loss` = weighted
  objective while `metrics.json` image metrics stay plain.
- [ ] T008 `recon_loss` in `train/objective.py`, used by `train_epoch`.

## Stage 4 — VAE variant (`feat(011): VAE variant`)

- [ ] T009 Tests (red first): model (`variational` shapes, `encode` = μ, 3,323,569
  parameters, state keys = plain + `logvar.*`, shared init equals the plain AE's at the same
  seed, train-mode `forward` deterministic, `encode_dist` of the plain AE, checkpoint without
  `variational` loads as plain); `gaussian_kl` (≥ 0, 0 at μ = 0 / lv = 0, hand value);
  fixture VAE run: `train_kl` finite > 0, repeatable, resume (all knobs incl. VAE) equals
  uninterrupted, `model.variational`; probe on a VAE fixture checkpoint end to end
  (`encoder.variational` true, latents `[n, latent_dim]`).
- [ ] T010 `models/autoencoder.py` (`variational`, `logvar` built last, `encode_dist`,
  `hparams`); `train/objective.py` (`gaussian_kl`, `reparameterize`); trainer (model build,
  KL in the objective and history, `kl` in the log line); `eval/probe.py`
  (`encoder.variational`).

## Stage 5 — configs and docs (`feat(011): experiment configs`, `docs(011)`)

- [ ] T011 `configs/ae_mask.yaml`, `configs/ae_vae.yaml`, `configs/ae_reg.yaml` (copies of
  `configs/ae.yaml`, `run_name` `ae-{mask,vae,reg}-smpl_proc_v1`); `configs/ae.yaml` gains the
  six keys at their defaults; test: each loads and differs from `ae.yaml` only in `run_name`
  and its keys.
- [ ] T012 Docs: 004 `spec.md` dated notes; README (status row, Usage: variants and configs,
  Results: numbers come with the runs); roadmap (#16 (ii)–(iv) implemented, 011 row, status
  line, changelog 2026-10-21); HANDOFF (state; experiment procedure R0 → R1 → R2 → R3 with
  costs; the 010 interplay per Q2; #16 entry).
- [ ] T013 Verification record (below): SC-001…SC-005, pytest count and wall time, the
  same-machine before/after fixture comparison (SC-002) against `19479e2`.

## Analyze pass (2026-10-21, before stage 2)

- Coverage: FR-001/002 → T004–T005; FR-003 → T007–T010; FR-004/005/006 → T004, T006;
  FR-007/008 → T009–T010; FR-009 → T004, T006, T010; FR-010/011 → T004 (regression), T007
  (plain metrics); FR-012 → T004, T007, T009, T011; FR-013 → T011; FR-014 → T003, T012;
  FR-015 → every stage (no `pyproject.toml` / `uv.lock` change). SC-001…005 → T013.
- Consistency: spec, plan, data model and contracts agree on key names, defaults, formulas,
  the 7-column history header and the additive `metrics.json` keys. No open ambiguity.
- Constitution: no violation (plan, Constitution Check).

## Verification record

(filled at the end of stage 5)
