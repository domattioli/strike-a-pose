"""Exact PCA (spec 003 FR-005, research R1)."""

import pytest
import torch

from strike_a_pose.eval.pca import fit_pca


def _low_rank(n: int, d: int, r: int, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scales = torch.tensor([2.0**-i for i in range(r)], dtype=torch.float64)
    z = torch.randn(n, r, generator=g, dtype=torch.float64) * scales
    basis = torch.linalg.qr(torch.randn(d, r, generator=g, dtype=torch.float64))[0].T
    return (z @ basis + 0.3).to(dtype)


@pytest.mark.parametrize(("n", "d"), [(20, 50), (60, 12)], ids=["gram", "covariance"])
def test_exact_at_rank_and_variance_accounting(n, d):
    r = 5
    x = _low_rank(n, d, r)
    p = fit_pca(x, k=10)
    assert p.numerical_rank == r and p.n_components == r and p.k_effective(10) == r
    v = p.components
    torch.testing.assert_close(v @ v.T, torch.eye(r, dtype=v.dtype), atol=1e-10, rtol=0)
    torch.testing.assert_close(p.reconstruct(x), x, atol=1e-10, rtol=0)
    assert p.explained_variance_ratio(r) == pytest.approx(1.0)
    # k < r: squared reconstruction error = (n - 1) * sum of dropped variances
    for k in (1, 3):
        sse = float(((p.reconstruct(x, k) - x) ** 2).sum())
        assert sse == pytest.approx(float(p.variances[k:].sum()) * (n - 1), rel=1e-9)
    ratios = [p.explained_variance_ratio(k) for k in range(1, r + 1)]
    assert ratios == sorted(ratios) and 0 < ratios[0] < 1
    total = float(((x - x.mean(0)) ** 2).sum()) / (n - 1)
    assert p.total_variance == pytest.approx(total)


def test_gram_and_covariance_paths_agree():
    x = _low_rank(30, 30, 4, seed=3)  # n == d -> Gram path
    a = fit_pca(x, 4)
    b = fit_pca(torch.cat([x, x]), 4)  # n > d -> covariance path, same directions
    torch.testing.assert_close((a.components @ b.components.T).abs(), torch.eye(4, dtype=x.dtype))
    torch.testing.assert_close(a.mean, b.mean)


def test_float32_gram_path_drops_noise_components():
    x = _low_rank(8, 400, 3, seed=1, dtype=torch.float32)
    p = fit_pca(x, 7)
    assert p.numerical_rank == 3 and p.n_components == 3
    assert p.components.dtype == torch.float32
    torch.testing.assert_close(p.reconstruct(x), x, atol=1e-5, rtol=0)


def test_degenerate_single_sample_and_constant_rows():
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    p = fit_pca(x, 2)
    assert p.n_components == 0 and p.numerical_rank == 0
    torch.testing.assert_close(p.reconstruct(torch.zeros(2, 3, dtype=x.dtype)), x.expand(2, 3))
    assert p.explained_variance_ratio(2) == 1.0
    q = fit_pca(x.expand(5, 3).clone(), 2)  # identical rows -> mean only
    assert q.n_components == 0


def test_bad_args():
    with pytest.raises(ValueError):
        fit_pca(torch.zeros(3), 1)
    with pytest.raises(ValueError):
        fit_pca(torch.zeros(3, 2), 0)
