"""把用户导入的 3D 模型 / 轮廓图变成二维固体掩膜。

3D 网格沿指定轴做正投影得到剪影（与 web/model-import.js 的定义一致）：
长度单位都是网格间距 h=1，size 是剪影最长边的一半，剪影中心放在 (cx, cy)。
栅格化用超采样 + 三角形并集，图片先按 alpha（有透明通道时）或亮度二值化。
"""

import os
import re

import numpy as np


def _load_stl(path):
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) >= 84:
        n = int(np.frombuffer(buf, dtype="<u4", count=1, offset=80)[0])
        if n > 0 and 84 + n * 50 == len(buf):
            rec = np.frombuffer(buf, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
            flat = rec[:, 12:48].copy().view("<f4").reshape(n, 9)
            return flat.astype(np.float64).reshape(n, 3, 3)
    txt = buf.decode("utf-8", "replace")
    vals = re.findall(r"vertex\s+(-?[\d.eE+\-]+)\s+(-?[\d.eE+\-]+)\s+(-?[\d.eE+\-]+)", txt)
    if not vals:
        raise ValueError("STL 里没有三角形：%s" % path)
    return np.array(vals, dtype=np.float64).reshape(-1, 3, 3)


def _load_obj(path):
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "v" and len(t) >= 4:
                verts.append((float(t[1]), float(t[2]), float(t[3])))
            elif t[0] == "f" and len(t) >= 4:
                idx = []
                for tok in t[1:]:
                    p = tok.split("/")[0]
                    if not p:
                        continue
                    n = int(p)
                    idx.append(n - 1 if n > 0 else len(verts) + n)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    if not faces:
        raise ValueError("OBJ 里没有面：%s" % path)
    v = np.asarray(verts, dtype=np.float64)
    return v[np.asarray(faces, dtype=np.int64)]


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        return _load_stl(path)
    if ext == ".obj":
        return _load_obj(path)
    raise ValueError("只支持 .stl / .obj，收到 %s" % ext)


def _project(tris, axis):
    if axis == "z":
        return tris[:, :, :2]
    if axis == "y":
        return tris[:, :, [0, 2]]
    if axis == "x":
        return tris[:, :, [1, 2]]
    raise ValueError("轴必须是 x / y / z，收到 %r" % axis)


def _rasterize(v2, nx, ny, size, cx, cy, ss=4):
    """v2: (n,3,2) 投影后的三角形；返回 (ny, nx) 布尔掩膜。"""
    pts = v2.reshape(-1, 2)
    lo, hi = pts.min(0), pts.max(0)
    ext = float((hi - lo).max())
    if not np.isfinite(ext) or ext <= 1e-12:
        raise ValueError("模型在投影平面上退化成一条线，换个 --axis")
    scale = 2.0 * size / ext
    mid = 0.5 * (lo + hi)
    p = (v2 - mid) * scale + np.array([cx, cy], dtype=np.float64)
    xs = p[:, :, 0] * ss
    ys = p[:, :, 1] * ss
    W, H = nx * ss, ny * ss
    acc = np.zeros((H, W), dtype=bool)
    px = np.array([0.5, 0.5])
    for k in range(v2.shape[0]):
        x, y = xs[k], ys[k]
        x0 = max(0, int(np.floor(x.min() - 0.5)))
        x1 = min(W - 1, int(np.ceil(x.max() + 0.5)))
        y0 = max(0, int(np.floor(y.min() - 0.5)))
        y1 = min(H - 1, int(np.ceil(y.max() + 0.5)))
        if x1 < x0 or y1 < y0:
            continue
        den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + px[0], np.arange(y0, y1 + 1) + px[1])
        a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
        b = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
        inside = (a >= 0) & (b >= 0) & (a + b <= 1)
        acc[y0:y1 + 1, x0:x1 + 1] |= inside
    return acc.reshape(ny, ss, nx, ss).mean(axis=(1, 3)) >= 0.5


def mesh_mask(path, nx, ny, size, cx, cy, axis="z", ss=4):
    tris = load_mesh(path)
    return _rasterize(_project(tris, axis), nx, ny, size, cx, cy, ss)


def image_mask(path, nx, ny, size, cx, cy, invert=False, ss=4):
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)
    if np.any(a[:, :, 3] < 250):
        cov = a[:, :, 3].astype(np.float64)
    else:
        lum = a[:, :, 0] * 0.299 + a[:, :, 1] * 0.587 + a[:, :, 2] * 0.114
        cov = 255.0 - lum
    if invert:
        cov = 255.0 - cov
    ys, xs = np.where(cov > 127)
    if xs.size == 0:
        raise ValueError("图片里没有轮廓像素，试试 --invert")
    cov = cov[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = cov.shape
    scale = 2.0 * size / max(w, h)
    tw = max(1, int(round(w * scale * ss)))
    th = max(1, int(round(h * scale * ss)))
    sm = Image.fromarray(cov.astype(np.uint8), mode="L").resize((tw, th), Image.LANCZOS)
    m = (np.asarray(sm) > 127)[::-1]        # 数组行 0 是图片顶，网格 j 从下往上
    full = np.zeros((ny * ss, nx * ss), dtype=bool)
    x0 = int(round(cx * ss - tw / 2.0))
    y0 = int(round(cy * ss - th / 2.0))
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(nx * ss, x0 + tw), min(ny * ss, y0 + th)
    if dx1 > dx0 and dy1 > dy0:
        sx0, sy0 = dx0 - x0, dy0 - y0
        full[dy0:dy1, dx0:dx1] = m[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    return full.reshape(ny, ss, nx, ss).mean(axis=(1, 3)) >= 0.5


def encode_mask(mask):
    """按位打包再 base64；位序与 JS 端 (byte >> (k & 7)) & 1 对齐。"""
    import base64

    bits = np.packbits(mask.astype(np.uint8).ravel(), bitorder="little")
    return base64.b64encode(bits.tobytes()).decode("ascii")