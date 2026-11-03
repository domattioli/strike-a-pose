# Data Model: Generative Pose VAE (slice 021)

## Config

No new key. `_check` loses exactly two lines pairs:
```
if cfg.pose_weight > 0 and cfg.variational:
    raise ConfigError("pose_weight: not with variational: true")
...
if cfg.gradient_weight > 0 and cfg.variational:
    raise ConfigError("gradient_weight: not with variational: true")
```

## `train_epoch` loss (trainer.py, replaces the `if logvar is None: … else: …` block)

```
recon = recon_loss(y, x, obj.mask_weight)
loss = recon
if obj.gradient_weight:  # spec 020: + gamma * gradient term
    g = gradient_loss(y, x)
    loss = loss + obj.gradient_weight * g
    grad_total += float(g.detach()) * len(idx)
if logvar is not None:  # spec 011: KL in per-pixel units, (SSE + beta·KL) / (H·W) per image
    kl = gaussian_kl(mu, logvar)
    loss = loss + obj.beta * kl / (x.shape[2] * x.shape[3])
    kl_total = (kl_total or 0.0) + float(kl.detach()) * len(idx)
if pose is not None:  # spec 018 / 021 FR-002: the head reads mu (deterministic), never z
    pl = pose_loss(pose.head(mu), pose.targets[idx], pose.active)
    loss = loss + obj.pose_weight * pl
    pose_total += float(pl.detach()) * len(idx)
```
Bitwise note: old AE+grad was `recon + γ·g`, new is `recon + γ·g` (same); old VAE was
`recon + β·kl/(H·W)`, new is `recon + β·kl/(H·W)` (same). History columns unchanged.

## `eval/generative.py`

```python
GENERATIVE_VERSION = 1
HEAD_MAX_MM = 80.0
MSE_COST_REL = 0.05
PRIOR_VALID_MIN = 0.56
MARGIN_MIN_PTS = 13.0
COMPARE_SD = 2.0
EPS = 1e-12
AXES = ("pose", "pixel", "valid", "margin")
NEXT_STEP = {"pose": "beta 0.03", "pixel": "beta 0.03", "valid": "head on z", "margin": "head on z"}

@dataclass(frozen=True)
class GenEntry:
    name: str
    head_mm: dict[str, float | None]      # split -> metrics.json splits[s].pose_head.mpjpe_mm
    mse: dict[str, float]                 # split -> splits[s].image.ae.image_mse
    prior_valid: float | None             # sample run samples.json sets.vae_prior.valid_fraction
    prior_n: int | None                   # sets.vae_prior.n
    rt_plausible: float | None            # round-trip samples.json sets.vae_prior.round_trip.plausible_fraction
    rt_control: float | None              # … .round_trip.control_plausible_fraction

    @property
    def margin_pts(self) -> float | None  # 100 * (rt_plausible - rt_control), None if either None
```

- `load_gen_entry(name, train_run, sample_run=None, roundtrip_run=None, splits=("val","test"))
  -> GenEntry`: reads `<train_run>/metrics.json`; `head_mm[s]` None when `pose_head` absent;
  `<sample_run>/samples.json`, `<roundtrip_run>/samples.json`; missing files → `FileNotFoundError`
  naming the path; `sample_run`/`roundtrip_run` None → fields None.
- `mean_mse(runs: list[str|Path], splits) -> dict[str, float]`: arithmetic mean of
  `image_mse` per split; `ValueError("need >= 1 run")` on empty.
- `axis_checks(e, l2_mse, splits) -> dict[str, bool | None]` keys `AXES`:
  pose = all(head ≤ 80 + EPS) over splits (None if any head None); pixel = all(mse ≤
  l2_mse[s]·1.05 + EPS); valid = prior_valid ≥ 0.56 − EPS (None if None); margin =
  margin_pts ≥ 13 − EPS (None if None).
- `goal_verdict(checks) -> tuple[str, list[str], str]`: any None → `("INCOMPLETE", [axes that
  are None], "")`; all True → `("GOAL-MET", [], "")`; else `("PARTIAL", failing in AXES order,
  "; ".join(unique NEXT_STEP values, "head on z" before "beta 0.03" per operator ruling 2026-11-03))`.
- `compare_prior(v1, v2) -> tuple[str, float, float]`: `(label, diff, sd)` per spec Q9; any
  prior None → `("INCOMPLETE", nan, nan)`.
- `goal_line_pixel(e, r0_mse, splits) -> bool`: all mse ≤ r0_mse[s]·1.05 + EPS (informational).
- `format_markdown(entries, l2_mse, r0_mse, split) -> str`: header
  `| model | head mm | MSE | MSE / L2 | pixel (≤ L2 +5 %) | vs R0 64-d +5 % | prior valid | RT margin pts |`
  one row per entry (missing → "n/a"; ratios `.4f`, mm `.1f`, MSE `.6g`, valid `.3f`, margin `.1f`).

## `scripts/generative.py`

`report --v1 TRAIN SAMPLE RT --v2 TRAIN SAMPLE RT --l2 RUN [--l2 RUN] --r0 RUN [--r0 RUN …]
[--extra NAME TRAIN SAMPLE RT] [--split {val,test,both}] [--json PATH]`. Prints per split
`## <split>` + table, then `V2 vs V1 prior: <label> (diff +0.xxx, sd 0.xxx)`, then last line
`VERDICT <GOAL-MET|PARTIAL|INCOMPLETE> failing=<comma list or -> next=<text or ->` (verdict
over the requested splits; `both` = val and test). `--json` writes `{"generative_version": 1,
"verdict", "failing", "next", "checks", "compare": {...}, "entries": {name: asdict}}`. Exit 0
on a verdict (even INCOMPLETE); exit 2 when a path is missing (stderr names it). Never prints
`QUEUE_DONE`.
