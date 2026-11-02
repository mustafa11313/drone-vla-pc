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

from sim.render import (CAGE_HALF, CAGE_HEIGHT, LAB_BOUNDS, LAB_CEIL, LAB_FLOOR, LAB_HEIGHT,
                        NET, POST, POST_COL, Scene, Textures, _floor_colour, _net_strand,
                        lab_wall_colour)

DISPLAY_GAIN = 1.5     # the scene is stored darker than the photos (render.LAB_DIV); show it like a phone would


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


def render_free(eye, target, scene, w=640, h=400, hfov=90, ss=2):
    """A free camera, rendered at ss x the size and scaled down (smooth net)."""
    if ss > 1:
        big = render_free(eye, target, scene, w * ss, h * ss, hfov, ss=1)
        return np.asarray(Image.fromarray(big).resize((w, h), Image.BOX))
    tex = Textures.get()
    eye = np.asarray(eye, float)
    d = _look(eye, target, w, h, hfov).reshape(-1, 3)
    col = np.zeros((len(d), 3), np.float32)
    todo = np.ones(len(d), bool)
    planes = []                       # (kind, axis, value)
    for s in (-1, 1):
        planes += [("cage", 0, s * CAGE_HALF), ("cage", 1, s * CAGE_HALF)]
    planes += [("lab", 0, LAB_BOUNDS["west"]), ("lab", 0, LAB_BOUNDS["east"]),
               ("lab", 1, LAB_BOUNDS["south"]), ("lab", 1, LAB_BOUNDS["north"])]
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
                ok = _in_lab(p)
                c = _floor_colour(scene, p[:, 0], p[:, 1], tex)
            elif kind == "ceil":
                ok = np.ones(len(p), bool)
                c = np.broadcast_to(LAB_CEIL, (len(p), 3)).astype(np.float32)
            elif kind == "lab":
                ok = _in_lab(p, axis=1 - a) & (p[:, 2] >= 0) & (p[:, 2] <= LAB_HEIGHT)
                side = ("east" if v > 0 else "west") if a == 0 else ("north" if v > 0 else "south")
                c = lab_wall_colour(tex, p, side)
            else:                                   # cage face: net and corner posts
                if a == 2:
                    inside = (np.abs(p[:, 0]) <= CAGE_HALF) & (np.abs(p[:, 1]) <= CAGE_HALF)
                    aa, bb = p[:, 0], p[:, 1]
                    post = np.zeros(len(p), bool)
                else:
                    inside = (np.abs(p[:, 1 - a]) <= CAGE_HALF) & (p[:, 2] >= 0) & (p[:, 2] <= CAGE_HEIGHT)
                    aa, bb = p[:, 1 - a], p[:, 2]
                    post = np.abs(aa) > CAGE_HALF - POST
                strand = _net_strand(scene, aa, bb)
                ok = inside & (strand | post)
                c = np.where(post[:, None], POST_COL, NET).astype(np.float32)
            sel = np.flatnonzero(m)[ok]
            col[sel] = c[ok]
            todo[sel] = False
    return np.clip(col.reshape(h, w, 3) * DISPLAY_GAIN, 0, 255).astype(np.uint8)


def _in_lab(p, axis=None):
    b = LAB_BOUNDS
    okx = (p[:, 0] >= b["west"] - 1e-6) & (p[:, 0] <= b["east"] + 1e-6)
    oky = (p[:, 1] >= b["south"] - 1e-6) & (p[:, 1] <= b["north"] + 1e-6)
    return okx if axis == 0 else oky if axis == 1 else okx & oky


def floor_plan(scene, px_per_m=90):
    """Top view to scale: floor, pad, cage, and which photos stand on which wall."""
    tex = Textures.get()
    b = LAB_BOUNDS
    W, H = int((b["east"] - b["west"]) * px_per_m), int((b["north"] - b["south"]) * px_per_m)
    X, Y = np.meshgrid(np.linspace(b["west"], b["east"], W), np.linspace(b["north"], b["south"], H))
    c = _floor_colour(scene, X.ravel(), Y.ravel(), tex).reshape(H, W, 3) * DISPLAY_GAIN
    img = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    to = lambda x, y: ((x - b["west"]) * px_per_m, (b["north"] - y) * px_per_m)
    d.rectangle([*to(-CAGE_HALF, CAGE_HALF), *to(CAGE_HALF, -CAGE_HALF)], outline=(230, 230, 230), width=2)
    d.text((W / 2 - 60, 4), "north: yellow robot cell", fill=(0, 0, 0))
    d.text((W / 2 - 50, H - 14), "south: the desk", fill=(0, 0, 0))
    d.text((4, H / 2), "west", fill=(0, 0, 0))
    d.text((W - 30, H / 2), "east", fill=(0, 0, 0))
    x0, y0 = to(-CAGE_HALF, CAGE_HALF)
    d.text((x0 + 4, y0 + 4), "net cage 2 x 2 m", fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description="views of the simulated cage from outside")
    ap.add_argument("--out", default="overview.jpg")
    ap.add_argument("--pad", nargs=2, type=float, default=[0.2, 0.1])
    args = ap.parse_args()
    sc = Scene(pad_x=args.pad[0], pad_y=args.pad[1], seed=1)
    b = LAB_BOUNDS
    views = [((b["east"] - 0.1, b["south"] + 0.1, 3.4), (0, 0, 0.6), "from the south-east corner, high"),
             ((b["west"] + 0.1, b["north"] - 0.1, 3.4), (0, 0, 0.6), "from the north-west corner, high"),
             ((0.0, b["south"] + 0.1, 1.6), (0, 0.5, 1.0), "standing at the desk (south), looking north"),
             ((0.0, b["north"] - 0.1, 1.6), (0, -0.5, 1.0), "from the robot cell (north), looking south"),
             ((0.0, 0.0, 3.5), (0.0, 0.01, 0.0), "from above the cage")]
    w, h, pad = 520, 340, 24
    sheet = Image.new("RGB", (3 * w, 2 * (h + pad)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, (eye, tgt, title) in enumerate(views):
        x, y = (k % 3) * w, (k // 3) * (h + pad)
        dr.text((x + 8, y + 6), title, fill="black")
        sheet.paste(Image.fromarray(render_free(eye, tgt, sc, w, h, 95)), (x, y + pad))
    x, y = 2 * w, (h + pad)
    dr.text((x + 8, y + 6), "floor plan (top view, to scale)", fill="black")
    fp = floor_plan(sc)
    fp.thumbnail((w - 10, h))
    sheet.paste(fp, (x + (w - fp.width) // 2, y + pad))
    sheet.save(args.out, quality=90)
    print(args.out)


if __name__ == "__main__":
    main()
