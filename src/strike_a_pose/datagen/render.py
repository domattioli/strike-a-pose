"""Deterministic NumPy z-buffer rasterizer (research R1).

Grey flat-shaded body on black; light at the camera. Pixel (r, c) is sampled at
its centre (c + 0.5, r + 0.5); nearest depth wins, ties go to the lowest face index.
"""

import numpy as np

from .cameras import Camera, project

AMBIENT = 0.15
DIFFUSE = 0.65
_NEAR = 1e-6
_CHUNK = 4096


def render(
    verts_world: np.ndarray, faces: np.ndarray, cam: Camera
) -> tuple[np.ndarray, np.ndarray]:
    h, w = cam.height, cam.width
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return rgb, mask
    xc, uv = project(cam, verts_world)
    tri_uv = uv[faces]  # (F,3,2)
    tri_z = xc[faces][:, :, 2]  # (F,3)
    ok = np.all(tri_z > _NEAR, axis=1)
    e1 = tri_uv[:, 1] - tri_uv[:, 0]
    e2 = tri_uv[:, 2] - tri_uv[:, 0]
    area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    ok &= np.abs(area) > 1e-12
    cmin = np.maximum(np.ceil(tri_uv[:, :, 0].min(1) - 0.5), 0)
    cmax = np.minimum(np.floor(tri_uv[:, :, 0].max(1) - 0.5), w - 1)
    rmin = np.maximum(np.ceil(tri_uv[:, :, 1].min(1) - 0.5), 0)
    rmax = np.minimum(np.floor(tri_uv[:, :, 1].max(1) - 0.5), h - 1)
    ok &= (cmax >= cmin) & (rmax >= rmin)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return rgb, mask
    bw = (cmax - cmin + 1)[idx].astype(np.int64)
    bh = (rmax - rmin + 1)[idx].astype(np.int64)
    idx = idx[np.argsort(bw * bh, kind="stable")]

    pix_all, dep_all, face_all = [], [], []
    for s in range(0, len(idx), _CHUNK):
        f = idx[s : s + _CHUNK]
        cw = int((cmax[f] - cmin[f]).max()) + 1
        ch = int((rmax[f] - rmin[f]).max()) + 1
        oy, ox = np.meshgrid(np.arange(ch), np.arange(cw), indexing="ij")
        cols = cmin[f][:, None, None] + ox  # (n,ch,cw)
        rows = rmin[f][:, None, None] + oy
        inb = (cols <= cmax[f][:, None, None]) & (rows <= rmax[f][:, None, None])
        px, py = cols + 0.5, rows + 0.5
        t = tri_uv[f]
        a = area[f][:, None, None]

        def edge(p, q, px=px, py=py):
            return (q[:, 0, None, None] - p[:, 0, None, None]) * (py - p[:, 1, None, None]) - (
                q[:, 1, None, None] - p[:, 1, None, None]
            ) * (px - p[:, 0, None, None])

        b0 = edge(t[:, 1], t[:, 2]) / a
        b1 = edge(t[:, 2], t[:, 0]) / a
        b2 = edge(t[:, 0], t[:, 1]) / a
        inside = inb & (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        z = tri_z[f]
        invz = b0 / z[:, 0, None, None] + b1 / z[:, 1, None, None] + b2 / z[:, 2, None, None]
        sel = np.nonzero(inside)
        pix_all.append((rows[sel] * w + cols[sel]).astype(np.int64))
        dep_all.append(1.0 / invz[sel])
        face_all.append(f[sel[0]])
    pix = np.concatenate(pix_all)
    if len(pix) == 0:
        return rgb, mask
    dep = np.concatenate(dep_all)
    fid = np.concatenate(face_all)
    order = np.lexsort((fid, dep, pix))
    pix, fid = pix[order], fid[order]
    first = np.unique(pix, return_index=True)[1]
    pix, fid = pix[first], fid[first]

    tri_c = xc[faces[fid]]  # (P,3,3) camera frame
    n = np.cross(tri_c[:, 1] - tri_c[:, 0], tri_c[:, 2] - tri_c[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
    cen = tri_c.mean(1)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    shade = np.abs(np.sum(n * cen, axis=1))
    val = np.round(255.0 * (AMBIENT + DIFFUSE * shade)).astype(np.uint8)
    flat_rgb = rgb.reshape(-1, 3)
    flat_rgb[pix] = val[:, None]
    mask.reshape(-1)[pix] = 255
    return rgb, mask
