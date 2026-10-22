"""Variant promotion rule (spec 015 FR-006; supersedes the pixel-only 012 rule).

A variant is compared with its dataset's control R0 and the seed-1 control R0b on val:
``seed_gap = |MSE(R0b) - MSE(R0)| / MSE(R0)`` (whole-image MSE, relative) and
``probe_gap = |MPJPE(R0b) - MPJPE(R0)|`` (probe world-frame MLP MPJPE, mm).

- WIN-pixel: ΔMSE < -seed_gap and ΔMPJPE <= +probe_gap;
- WIN-latent: ΔMSE <= +5 % and ΔMPJPE < -2 x probe_gap;
- DROPPED: ΔMSE > +5 % and not a latent win; otherwise REPORTED (not carried).

A VAE also needs its last train KL >= 1 nat/image to win (012 collapse check kept).
"""

from dataclasses import dataclass

MSE_DROP_REL = 0.05
LATENT_GAP_FACTOR = 2.0
KL_MIN_NATS = 1.0
EPS = 1e-12  # float tolerance on the +5 % bound only


@dataclass(frozen=True)
class Measures:
    mse: float  # whole-image val MSE
    mpjpe_mm: float  # probe val world-frame MLP MPJPE
    train_kl: float | None = None  # last train KL, nats/image (VAE)


@dataclass(frozen=True)
class Verdict:
    label: str  # WIN-pixel | WIN-latent | DROPPED | REPORTED
    d_mse_rel: float
    d_mpjpe_mm: float
    seed_gap_rel: float
    probe_gap_mm: float
    reason: str


def verdict(variant: Measures, r0: Measures, r0b: Measures, variational: bool = False) -> Verdict:
    seed_gap = abs(r0b.mse - r0.mse) / r0.mse
    probe_gap = abs(r0b.mpjpe_mm - r0.mpjpe_mm)
    d_mse = (variant.mse - r0.mse) / r0.mse
    d_mp = variant.mpjpe_mm - r0.mpjpe_mm
    pixel = d_mse < -seed_gap and d_mp <= probe_gap
    latent = d_mse <= MSE_DROP_REL + EPS and d_mp < -LATENT_GAP_FACTOR * probe_gap
    nums = (
        f"{d_mse * 100:+.1f} % MSE (gap {seed_gap * 100:.1f} %); "
        f"{d_mp:+.1f} mm probe (gap {probe_gap:.1f})"
    )
    note = ""
    if variational:
        if variant.train_kl is None:
            raise ValueError("a VAE variant needs train_kl")
        if variant.train_kl < KL_MIN_NATS:
            pixel = latent = False
            note = "; KL < 1 nat (collapse)"
    if pixel:
        label = "WIN-pixel"
    elif latent:
        label = "WIN-latent"
    elif d_mse > MSE_DROP_REL + EPS:
        label, note = "DROPPED", note + "; MSE > +5 %"
    else:
        label = "REPORTED"
    return Verdict(label, d_mse, d_mp, seed_gap, probe_gap, nums + note)
