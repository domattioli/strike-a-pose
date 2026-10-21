"""Per-region image metrics (spec 009): region masks, masked MSE, scoring seam, placement."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from strike_a_pose.data import PoseImageDataset
from strike_a_pose.datagen.skeleton import SMPL24_JOINT_NAMES
from strike_a_pose.eval import REGIONS, masked_mse, region_mask
from strike_a_pose.eval.report import region_definitions, score_images
from strike_a_pose.runlog import dumps_exact

FIXTURE = Path(__file__).parents[1] / "tests" / "fixtures" / "smpl_fixture"
BACKENDS = [np.asarray, lambda x: torch.as_tensor(np.asarray(x))]
HEADLINE = ("image_mse", "psnr_db", "silhouette_iou")


def _np(y):
    return y.numpy() if isinstance(y, torch.Tensor) else y


@pytest.fixture(params=BACKENDS, ids=["numpy", "torch"])
def as_(request):
    return request.param


def _square(size: tuple[int, int], r: int, c: int, h: int) -> np.ndarray:
    """Expected mask: the (2h + 1)² square centred on pixel (r, c), clipped to the image."""
    m = np.zeros(size, dtype=bool)
    m[max(r - h, 0) : max(r + h + 1, 0), max(c - h, 0) : max(c + h + 1, 0)] = True
    return m


def _mask(as_, uv, vis, size, h, joints=None):
    """``region_mask`` of one sample whose joints are ``uv`` (list of (u, v))."""
    kp = as_(np.array([uv], dtype=np.float64))
    v = as_(np.array([vis], dtype=bool))
    m = region_mask(kp, v, size, range(len(uv)) if joints is None else joints, h)
    assert type(m) is type(kp) and tuple(m.shape) == (1, *size) and _np(m).dtype == bool
    return _np(m)[0]


# ------------------------------------------------------------------ regions (FR-001)


def test_regions_table_and_definitions():
    names = {r: [SMPL24_JOINT_NAMES[j] for j in joints] for r, (joints, _) in REGIONS.items()}
    assert names == {
        "hands": ["left_hand", "right_hand"],
        "feet": ["left_ankle", "right_ankle", "left_foot", "right_foot"],
        "head": ["head"],
    }
    assert {r: h for r, (_, h) in REGIONS.items()} == {"hands": 10, "feet": 8, "head": 14}
    assert region_definitions() == {
        r: {"joints": names[r], "half_size_px": h} for r, (_, h) in REGIONS.items()
    }


# ------------------------------------------------------------------ region mask (FR-002)


def test_interior_square_fractional_and_integer(as_):
    """(a) the pixel holding (u, v) is row ⌊v⌋, column ⌊u⌋; u = 10.0 → column 10."""
    size = (20, 16)
    m = _mask(as_, [(7.6, 12.3)], [True], size, 2)
    np.testing.assert_array_equal(m, _square(size, 12, 7, 2))
    assert m.sum() == 25
    m = _mask(as_, [(10.0, 5.0)], [True], size, 1)
    np.testing.assert_array_equal(m, _square(size, 5, 10, 1))
    assert m[4:7, 9:12].all() and m.sum() == 9
    m = _mask(as_, [(3.99, 0.0)], [True], size, 0)  # h = 0: the joint's pixel only
    assert m.sum() == 1 and m[0, 3]


@pytest.mark.parametrize(
    ("uv", "rc", "count"),
    [
        ((0.3, 10.5), (10, 0), 5 * 3),  # left border: columns 0..2
        ((15.2, 10.5), (10, 15), 5 * 3),  # right border (W = 16)
        ((np.nextafter(16.0, 0.0), 10.5), (10, 15), 5 * 3),  # visible, u just below W
        ((7.5, 0.0), (0, 7), 3 * 5),  # top
        ((7.5, 19.9), (19, 7), 3 * 5),  # bottom (H = 20)
        ((0.0, 0.0), (0, 0), 3 * 3),  # corners
        ((15.5, 0.5), (0, 15), 3 * 3),
        ((0.5, 19.5), (19, 0), 3 * 3),
        ((15.9, 19.9), (19, 15), 3 * 3),
        ((1.5, 1.5), (1, 1), 4 * 4),  # one pixel in from the corner
    ],
)
def test_clipping_at_borders_and_corners(as_, uv, rc, count):
    """(b) squares are clipped to the image: count = product of the clipped extents."""
    size = (20, 16)
    m = _mask(as_, [uv], [True], size, 2)
    np.testing.assert_array_equal(m, _square(size, *rc, 2))
    assert m.sum() == count


def test_union_of_squares(as_):
    """(c) overlapping squares count each pixel once; two joints on one pixel = one square."""
    size = (20, 16)
    m = _mask(as_, [(5.5, 5.5), (7.5, 6.5)], [True, True], size, 2)
    np.testing.assert_array_equal(m, _square(size, 5, 5, 2) | _square(size, 6, 7, 2))
    assert m.sum() == 25 + 25 - 4 * 3
    m = _mask(as_, [(5.2, 5.7), (5.9, 5.1)], [True, True], size, 2)
    np.testing.assert_array_equal(m, _square(size, 5, 5, 2))
    m = _mask(as_, [(1.0, 1.0), (14.0, 18.0), (8.0, 8.0)], [True, True, True], size, 1, [0, 1])
    assert m.sum() == 18 and not m[7:10, 7:10].any()  # only the listed joints count


def test_visibility(as_):
    """(d) invisible joints add nothing, even just outside the image with a square that
    would reach in; a region whose joints are all invisible is empty."""
    size = (20, 16)
    m = _mask(as_, [(5.5, 5.5), (12.5, 15.5)], [True, False], size, 2)
    np.testing.assert_array_equal(m, _square(size, 5, 5, 2))
    m = _mask(as_, [(-0.5, 10.5), (16.0, 3.0)], [False, False], size, 2)  # outside, reach in
    assert not m.any()
    m = _mask(as_, [(5.5, 5.5), (12.5, 15.5)], [False, False], size, 2)
    assert not m.any()
    m = _mask(as_, [(float("nan"), float("inf")), (5.5, 5.5)], [False, True], size, 1)
    np.testing.assert_array_equal(m, _square(size, 5, 5, 1))  # unused coordinates ignored


def test_region_mask_rejects_bad_input(as_):
    kp, vis = as_(np.zeros((2, 3, 2))), as_(np.ones((2, 3), dtype=bool))
    with pytest.raises(ValueError):
        region_mask(as_(np.zeros((2, 3, 3))), vis, (8, 8), [0], 1)
    with pytest.raises(ValueError):
        region_mask(kp, as_(np.ones((2, 4), dtype=bool)), (8, 8), [0], 1)
    with pytest.raises(ValueError):
        region_mask(kp, vis, (8, 8), [0], -1)


# ------------------------------------------------------------------ masked MSE (FR-003)


def test_masked_mse_hand_computed(as_):
    """(e) error only outside the mask → 0; only inside → its mean; empty mask → NaN."""
    target = np.zeros((4, 2, 3))
    pred = target.copy()
    mask = np.zeros((4, 2, 3), dtype=bool)
    pred[0, 1, 2] = 0.5  # sample 0: error outside its mask
    mask[0, 0, :2] = True
    pred[1, 0, 0], pred[1, 0, 1] = 0.5, 0.25  # sample 1: errors inside, plus one zero pixel
    mask[1, 0, :3] = True
    pred[2] = 1.0  # sample 2: empty mask
    pred[3] = 0.5  # sample 3: whole image, uniform error
    mask[3] = True
    m = masked_mse(as_(pred), as_(target), as_(mask))
    assert type(m) is type(as_(pred)) and tuple(m.shape) == (4,)
    m = _np(m)
    np.testing.assert_array_equal(m[[0, 1, 3]], [0.0, (0.25 + 0.0625) / 3, 0.25])
    assert np.isnan(m[2])


def test_masked_mse_shapes_and_dtype(as_):
    x = np.ones((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        masked_mse(as_(x), as_(x), as_(np.ones((2, 12), dtype=bool)))
    with pytest.raises(ValueError):
        masked_mse(as_(x), as_(x[:, :, :3]), as_(np.ones((2, 3, 4), dtype=bool)))
    full = as_(np.ones((2, 3, 4), dtype=bool))
    assert _np(masked_mse(as_(x), as_(x * 0), full)).dtype == np.float32
    assert _np(masked_mse(as_(x.astype(np.float64)), as_(x * 0), full)).dtype == np.float64


# ------------------------------------------------------------------ NumPy = torch (FR-009 f)


def test_numpy_torch_and_float32_float64_agree():
    """(f) identical masks and values from NumPy and torch, float32 and float64 keypoints."""
    rng = np.random.default_rng(0)
    size = (40, 30)
    kp = rng.uniform(-15, 50, size=(16, 24, 2)).astype(np.float32).astype(np.float64)
    vis = rng.random((16, 24)) < 0.7  # float64 keypoints holding float32 values, as in 003
    for joints, h in REGIONS.values():
        ref = region_mask(kp, vis, size, joints, h)
        assert isinstance(ref, np.ndarray) and ref.any() and not ref.all()
        for k in (kp.astype(np.float32), torch.from_numpy(kp), torch.from_numpy(kp).float()):
            m = region_mask(
                k, torch.from_numpy(vis) if torch.is_tensor(k) else vis, size, joints, h
            )
            np.testing.assert_array_equal(_np(m), ref)
    for dt in (np.float32, np.float64):
        p = rng.random((16, 40, 30)).astype(dt)
        t = rng.random((16, 40, 30)).astype(dt)
        mask = rng.random((16, 40, 30)) < 0.3
        mask[0] = False  # one empty sample
        a = masked_mse(p, t, mask)
        b = masked_mse(torch.from_numpy(p), torch.from_numpy(t), torch.from_numpy(mask))
        assert a.dtype == dt and b.dtype == torch.from_numpy(p).dtype
        np.testing.assert_array_equal(a, b.numpy())
        assert np.isnan(a[0]) and not np.isnan(a[1:]).any()


# ------------------------------------------------------------------ placement (FR-009 g)


def _body_fill(kp: torch.Tensor, vis, body, size) -> list[float]:
    fills = []
    for joints, h in REGIONS.values():
        m = region_mask(kp, vis, size, joints, h)
        n = m.sum(dim=(1, 2))
        fills += ((m & body).sum(dim=(1, 2)) / n.clamp_min(1)).tolist()  # empty → 0
    return fills


def test_regions_sit_on_the_body_in_the_fixture():
    """(g) every region of every fixture sample holds ≥ 10 % true-body pixels (minimum 23 %);
    with u and v swapped, 33 of the 48 sample-region pairs fall below 10 %."""
    ds = PoseImageDataset(FIXTURE)
    items = [ds[i] for i in range(len(ds))]
    kp = torch.stack([s["keypoints_2d"] for s in items])
    vis = torch.stack([s["visibility"] for s in items])
    body = torch.stack([s["mask"][0] for s in items])
    fills = _body_fill(kp, vis, body, ds.image_size)
    assert len(fills) == 48 and min(fills) >= 0.10
    swapped = _body_fill(kp.flip(-1), vis, body, ds.image_size)
    assert sum(f < 0.10 for f in swapped) == 33


# ------------------------------------------------------------------ scoring seam (FR-004)

SIZE = (64, 48)
HEAD_RC = (15, 24)  # head square rows 1..29, columns 10..38: away from hands and feet


def _labels(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Hands at the bottom corners, feet at the bottom centre, head at the top, all visible."""
    kp = np.zeros((n, 24, 2))
    kp[:, 15] = (HEAD_RC[1] + 0.5, HEAD_RC[0] + 0.5)
    kp[:, 22], kp[:, 23] = (8.5, 50.5), (40.5, 50.5)
    kp[:, [7, 8, 10, 11]] = [(20.5, 55.5), (28.5, 55.5), (20.5, 60.5), (28.5, 60.5)]
    return kp, np.ones((n, 24), dtype=bool)


class Counting:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, x):
        self.calls += 1
        return self.fn(x)


def test_score_images_regions_hand_computed():
    """(h) zero targets, error 0.5 inside the head square and 0.25 elsewhere: head 0.25,
    hands / feet 0.0625; one sample's hand joints invisible → left out of hands only."""
    n = 4
    head = torch.from_numpy(_square(SIZE, *HEAD_RC, REGIONS["head"][1])).reshape(-1)
    err = torch.where(head, 0.5, 0.25).to(torch.float32)
    x = torch.zeros(n, SIZE[0] * SIZE[1])
    masks = torch.zeros(n, SIZE[0] * SIZE[1], dtype=torch.bool)
    kp, vis = _labels(n)
    vis[1, [22, 23]] = False
    pred = Counting(lambda b: b + err)
    labels = (torch.from_numpy(kp), torch.from_numpy(vis), SIZE)
    r = score_images(pred, x, masks, batch_size=3, labels=labels)
    assert pred.calls == 2  # once per batch: rows 0..2, 3
    assert r["regions"] == {
        "head": {"image_mse": 0.25, "samples": 4},
        "hands": {"image_mse": 0.0625, "samples": 3},
        "feet": {"image_mse": 0.0625, "samples": 4},
    }
    plain = score_images(pred, x, masks, batch_size=3)
    assert tuple(plain) == HEADLINE  # without labels: today's result
    assert {k: r[k] for k in HEADLINE} == plain  # headline bit-identical
    assert r["image_mse"] == pytest.approx((841 * 0.25 + (3072 - 841) * 0.0625) / 3072)

    vis[:, 15] = False  # every sample's head region empty
    r = score_images(pred, x, masks, 3, labels=(torch.from_numpy(kp), torch.from_numpy(vis), SIZE))
    assert math.isnan(r["regions"]["head"]["image_mse"]) and r["regions"]["head"]["samples"] == 0
    assert '"head": {\n      "image_mse": null,\n      "samples": 0' in dumps_exact(r)


def test_score_images_regions_match_per_sample_values():
    """(h) region values = mean of the per-sample masked MSE over the non-empty samples;
    the same result for batch sizes 1, 3 and n."""
    g = torch.Generator().manual_seed(0)
    n, (h, w) = 7, SIZE
    x = torch.rand(n, h * w, generator=g)
    masks = x > 0.5
    kp = torch.rand(n, 24, 2, generator=g) * torch.tensor([w + 10.0, h + 10.0]) - 5.0
    vis = torch.rand(n, 24, generator=g) < 0.7
    vis[2, [22, 23]] = False
    vis[4, [7, 8, 10, 11, 15, 22, 23]] = True
    pred = lambda b: (b * 0.8 + 0.1).clamp(0, 1)  # noqa: E731
    results = [score_images(pred, x, masks, bs, 0.4, labels=(kp, vis, SIZE)) for bs in (1, 3, n)]
    assert results[0] == results[1] == results[2]
    p = pred(x)
    for name, (joints, hs) in REGIONS.items():
        v = masked_mse(p, x, region_mask(kp, vis, SIZE, joints, hs).reshape(n, -1))
        ok = ~torch.isnan(v)
        want = {"image_mse": float(v[ok].double().mean()), "samples": int(ok.sum())}
        assert results[0]["regions"][name] == want, name
    assert results[0]["regions"]["hands"]["samples"] < n


def test_score_images_rejects_mismatched_labels():
    x = torch.zeros(4, SIZE[0] * SIZE[1])
    m = x > 0
    kp, vis = (torch.from_numpy(a) for a in _labels(4))
    with pytest.raises(ValueError):
        score_images(lambda b: b, x, m, 2, labels=(kp[:3], vis[:3], SIZE))
    with pytest.raises(ValueError):
        score_images(lambda b: b, x, m, 2, labels=(kp, vis, (SIZE[0], SIZE[1] + 1)))
