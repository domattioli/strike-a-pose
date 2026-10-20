"""Render the 004 loss curve (train/val MSE vs epoch, log y) from history.csv.

PIL only (no plotting dependency). Usage, from the repo root:

    uv run python docs/results/004-ae-smpl_proc_v1/plot_loss_curve.py \\
        docs/results/004-ae-smpl_proc_v1/history.csv docs/results/004-ae-smpl_proc_v1/loss_curve.png
"""

import csv
import math
import sys

from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
ep = [int(r["epoch"]) for r in rows]
tr = [float(r["train_loss"]) for r in rows]
va = [float(r["val_loss"]) for r in rows]

W, H = 900, 520
L, R, T, B = 90, 150, 60, 70
bg, ink, ink2, grid = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e6e5e0"
c_tr, c_va = "#2a78d6", "#eb6834"
img = Image.new("RGB", (W, H), bg)
d = ImageDraw.Draw(img)


def font(s):
    try:
        return ImageFont.load_default(size=s)
    except TypeError:
        return ImageFont.load_default()


f, fs, ft = font(15), font(13), font(18)

ymin, ymax = 1e-3, 4e-2
lo, hi = math.log10(ymin), math.log10(ymax)


def x(e):
    return L + (e - 1) / (len(ep) - 1) * (W - L - R)


def y(v):
    return T + (hi - math.log10(v)) / (hi - lo) * (H - T - B)


for v in [1e-3, 2e-3, 5e-3, 1e-2, 2e-2]:
    d.line([(L, y(v)), (W - R, y(v))], fill=grid, width=1)
    d.text((L - 8, y(v)), f"{v:g}", fill=ink2, font=fs, anchor="rm")
for e in [1, 10, 20, 30, 40, 50, 60]:
    d.text((x(e), H - B + 10), str(e), fill=ink2, font=fs, anchor="mt")
d.line([(L, H - B), (W - R, H - B)], fill=ink2, width=1)
d.text((L + (W - L - R) / 2, H - B + 34), "epoch", fill=ink2, font=f, anchor="mt")
d.text((L, 20), "ConvAutoencoder on smpl_proc_v1: per-pixel MSE (log scale)", fill=ink, font=ft)

for vals, col in [(tr, c_tr), (va, c_va)]:
    d.line([(x(e), y(v)) for e, v in zip(ep, vals, strict=True)], fill=col, width=2, joint="curve")
i = va.index(min(va))
cx, cy = x(ep[i]), y(va[i])
d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=c_va, outline=bg, width=2)
d.text((cx, cy - 12), f"val min {va[i]:.5f} @ epoch {ep[i]}", fill=ink, font=fs, anchor="mb")
d.text((W - R + 10, y(tr[-1])), f"train {tr[-1]:.5f}", fill=ink, font=f, anchor="lm")
d.text((W - R + 10, y(va[-1])), f"val {va[-1]:.5f}", fill=ink, font=f, anchor="lm")
# legend
lx, ly = W - R - 170, T + 10
for k, (lab, col) in enumerate([("train (12,800 imgs)", c_tr), ("val (1,600 imgs)", c_va)]):
    yy = ly + k * 22
    d.line([(lx, yy), (lx + 24, yy)], fill=col, width=3)
    d.text((lx + 32, yy), lab, fill=ink, font=fs, anchor="lm")
img.save(out, optimize=True)
print(out)
