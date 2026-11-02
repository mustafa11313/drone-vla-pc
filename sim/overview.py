"""
overview.py  -  look at the whole simulated cage from outside (for checking the scene, not for
the drone).

    python3 -m sim.overview --out overview.jpg

Draws a few views from free camera positions around the cage, plus a floor plan. The camera
can stand anywhere inside the lab box (the walls with the lab photos).
"""
import argparse, math

import numpy as np
from PIL import Image, ImageDraw

from sim.render import (ALU, BASE, BASE_HEIGHT, CAGE_HALF, CAGE_HEIGHT, LAB_CEIL, LAB_HALF,
                        LAB_HEIGHT, NET, POST, PAD_SIZE, Scene, Textures, _floor_colour,
                        _net_strand)


def _look(eye, target, w, h, hfov):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    tx = math.tan(math.radians(hfov) / 2)
    ty = tx * h / w
    px = (np.arange(w) + 0.5) / w * 2 - 1
    py = 1 - (np.arange(h) + 0.5) / h * 2
    PX, PY = np.meshgrid(px, py)
    d = f + PX[..., None] * tx * r + PY[..., None] * ty * u
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _lab_wall(tex, p, axis, sign):
    """Lab photo on the wall x = sign*LAB_HALF (axis 0) or y = sign*LAB_HALF (axis 1)."""
    along = p[:, 1] * sign if axis == 0 else -p[:, 0] * sign
    side = (0 if sign > 0 else 2) if axis == 0 else (3 if sign > 0 else 1)
    n = len(tex.lab)
    out = np.empty((len(p), 3), np.float32)
    out[:] = (212, 214, 216)
    if not n:
        return out
    half = (along > 0).astype(int)
    idx = (side * 2 + half) % n
    fu = np.mod(along + LAB_HALF, LAB_HALF) / LAB_HALF
    fv = 1 - np.clip(p[:, 2] / LAB_HEIGHT, 0, 1)
    for k in range(n):
        m = idx == k
        if m.any():
            im = tex.lab[k]
            hh, ww = im.shape[:2]
            out[m] = im[(fv[m] * (hh - 1)).astype(int), (fu[m] * (ww - 1)).astype(int)]
    return out


def render_free(eye, target, scene, w=640, h=400, hfov=90):
    tex = Textures.get()
    eye = np.asarray(eye, float)
    d = _look(eye, target, w, h, hfov).reshape(-1, 3)
    col = np.zeros((len(d), 3), np.float32)
    todo = np.ones(len(d), bool)
    planes = []                       # (kind, axis, value)
    for s in (-1, 1):
        planes += [("cage", 0, s * CAGE_HALF), ("cage", 1, s * CAGE_HALF), ("lab", 0, s * LAB_HALF), ("lab", 1, s * LAB_HALF)]
    planes += [("cage", 2, CAGE_HEIGHT), ("floor", 2, 0.0), ("ceil", 2, LAB_HEIGHT)]
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.stack([(v - eye[a]) / d[:, a] for _, a, v in planes], 1)
    ts = np.where(np.isfinite(ts) & (ts > 1e-4), ts, np.inf)
    order = np.argsort(ts, 1)
    for k in range(len(planes)):
        if not todo.any():
            break
        idx = order[:, k]
        t = ts[np.arange(len(d)), idx]
        for j, (kind, a, v) in enumerate(planes):
            m = todo & (idx == j) & np.isfinite(t)
            if not m.any():
                continue
            p = eye + d[m] * t[m][:, None]
            if kind == "floor":
                ok = (np.abs(p[:, 0]) <= LAB_HALF) & (np.abs(p[:, 1]) <= LAB_HALF)
                c = _floor_colour(scene, p[:, 0], p[:, 1], tex)
            elif kind == "ceil":
                ok = np.ones(len(p), bool)
                c = np.broadcast_to(LAB_CEIL, (len(p), 3)).astype(np.float32)
            elif kind == "lab":
                ok = (np.abs(p[:, 1 - a]) <= LAB_HALF) & (p[:, 2] >= 0) & (p[:, 2] <= LAB_HEIGHT)
                c = _lab_wall(tex, p, a, np.sign(v))
            else:                                   # cage face: net, base strip, posts
                if a == 2:
                    inside = (np.abs(p[:, 0]) <= CAGE_HALF) & (np.abs(p[:, 1]) <= CAGE_HALF)
                    aa, bb = p[:, 0], p[:, 1]
                    base = post = np.zeros(len(p), bool)
                else:
                    inside = (np.abs(p[:, 1 - a]) <= CAGE_HALF) & (p[:, 2] >= 0) & (p[:, 2] <= CAGE_HEIGHT)
                    aa, bb = p[:, 1 - a], p[:, 2]
                    base = p[:, 2] < BASE_HEIGHT
                    post = np.abs(aa) > CAGE_HALF - POST
                strand = _net_strand(scene, aa, bb)
                ok = inside & (strand | base | post)
                c = np.where(base[:, None], BASE, np.where(post[:, None], ALU, NET)).astype(np.float32)
            sel = np.flatnonzero(m)[ok]
            col[sel] = c[ok]
            todo[sel] = False
    return np.clip(col.reshape(h, w, 3), 0, 255).astype(np.uint8)


def floor_plan(scene, size=520):
    """Top view of the floor: mats, dark strip, pad, cage outline, which lab photo is where."""
    tex = Textures.get()
    m = LAB_HALF
    xs = np.linspace(-m, m, size)
    X, Y = np.meshgrid(xs, xs[::-1])
    c = _floor_colour(scene, X.ravel(), Y.ravel(), tex).reshape(size, size, 3)
    img = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    s = size / (2 * m)
    a, b = (m - CAGE_HALF) * s, (m + CAGE_HALF) * s
    d.rectangle([a, a, b, b], outline=(20, 20, 20), width=3)
    d.text((size / 2 - 20, 4), "north wall", fill=(0, 0, 0))
    d.text((size / 2 - 20, size - 14), "south wall", fill=(0, 0, 0))
    d.text((4, size / 2), "west", fill=(0, 0, 0))
    d.text((size - 30, size / 2), "east", fill=(0, 0, 0))
    d.text((a + 4, a + 4), "net cage 2 x 2 m", fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description="views of the simulated cage from outside")
    ap.add_argument("--out", default="overview.jpg")
    ap.add_argument("--pad", nargs=2, type=float, default=[0.2, 0.1])
    args = ap.parse_args()
    sc = Scene(pad_x=args.pad[0], pad_y=args.pad[1], seed=1)
    lh = LAB_HALF - 0.05
    views = [((lh, -lh, 3.4), (0, 0, 0.6), "from the south-east corner, high"),
             ((-lh, lh, 3.4), (0, 0, 0.6), "from the north-west corner, high"),
             ((0.0, -lh, 1.6), (0, 0.5, 1.0), "standing at the south side"),
             ((0.0, 0.0, 3.5), (0.0, 0.01, 0.0), "from above the cage")]
    w, h, pad = 520, 340, 24
    sheet = Image.new("RGB", (3 * w, 2 * (h + pad)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, (eye, tgt, title) in enumerate(views):
        x, y = (k % 3) * w, (k // 3) * (h + pad)
        dr.text((x + 8, y + 6), title, fill="black")
        sheet.paste(Image.fromarray(render_free(eye, tgt, sc, w, h, 95)), (x, y + pad))
    x, y = 4 % 3 * w, (h + pad)
    dr.text((x + 8, y + 6), "floor plan (top view, to scale)", fill="black")
    sheet.paste(floor_plan(sc, size=h).resize((h, h)), (x + (w - h) // 2, y + pad))
    sheet.save(args.out, quality=90)
    print(args.out)


if __name__ == "__main__":
    main()
