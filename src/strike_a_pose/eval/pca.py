"""Exact PCA via the smaller Gram / covariance eigenproblem (research R1)."""

from dataclasses import dataclass

import torch

REL_TOL = 1e-6  # eigenvalues <= REL_TOL * largest count as zero (numerical rank)


@dataclass(frozen=True)
class PCA:
    mean: torch.Tensor  # [d]
    components: torch.Tensor  # [k_eff, d], orthonormal rows, descending variance
    variances: torch.Tensor  # [k_eff], float64, per-component variance (n - 1 denominator)
    total_variance: float  # sum of all per-dimension variances
    n_samples: int
    numerical_rank: int  # eigenvalues > rel_tol * largest (of the centred data)

    @property
    def n_components(self) -> int:
        """Kept components: min(requested k, numerical rank)."""
        return int(self.components.shape[0])

    def k_effective(self, k: int) -> int:
        return min(int(k), self.n_components)

    def explained_variance_ratio(self, k: int) -> float:
        if self.total_variance <= 0:
            return 1.0
        return float(self.variances[: self.k_effective(k)].sum() / self.total_variance)

    def reconstruct(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Project ``[n, d]`` rows onto the first ``min(k, k_eff)`` components and back."""
        v = self.components if k is None else self.components[: self.k_effective(k)]
        xc = x.to(self.mean.dtype) - self.mean
        return self.mean + (xc @ v.T) @ v


def fit_pca(x: torch.Tensor, k: int, rel_tol: float = REL_TOL) -> PCA:
    """Fit PCA on rows of ``x [n, d]`` keeping up to ``k`` components.

    ``n <= d``: eigendecomposition of the Gram matrix ``Xc Xcᵀ``; else of ``Xcᵀ Xc``.
    Products in ``x``'s dtype (Gram) or float64 (covariance); eigh in float64.
    """
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"expected [n, d] with n >= 1, got {tuple(x.shape)}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n, d = x.shape
    dtype = x.dtype if x.is_floating_point() else torch.float32
    mean = (x.sum(dim=0, dtype=torch.float64) / n).to(dtype)
    xc = x.to(dtype) - mean
    denom = max(n - 1, 1)
    if n <= d:
        gram = (xc @ xc.T).to(torch.float64)
        evals, evecs = torch.linalg.eigh(gram)
        total = float(gram.trace()) / denom
    else:
        cov = xc.to(torch.float64).T @ xc.to(torch.float64)
        evals, evecs = torch.linalg.eigh(cov)
        total = float(cov.trace()) / denom
    evals, evecs = evals.flip(0), evecs.flip(1)  # descending
    top = float(evals[0]) if evals.numel() else 0.0
    keep = int((evals > rel_tol * top).sum()) if top > 0 else 0
    kk = min(k, keep)
    lam = evals[:kk]
    if n <= d:
        u = evecs[:, :kk]
        comps = (xc.T @ u.to(dtype)) / lam.sqrt().to(dtype)  # [d, kk]
        comps = comps.T.contiguous()
    else:
        comps = evecs[:, :kk].T.to(dtype).contiguous()
    return PCA(
        mean=mean,
        components=comps,
        variances=lam / denom,
        total_variance=total,
        n_samples=n,
        numerical_rank=keep,
    )
