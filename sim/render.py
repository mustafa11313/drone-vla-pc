"""
render.py  -  the drone camera's view of the flight cage, drawn with numpy only (no GPU).

Realistic version: real textures cut from your own pictures by sim/make_textures.py

    sim/textures/pad.png     the landing pad, from a drone frame
    sim/textures/mat.png     the foam mats seen from above, from drone frames
    sim/textures/lab_*.jpg   the lab, from the walk-around video, as walls of the hall

plus the things a real camera adds: soft overhead light, vignetting, slight blur, sensor
noise, JPEG. When a texture file is missing, plain colours are used instead.

The scene:
    floor        beige foam mats in the middle, dark grey mats along the net
    landing pad  PAD_SIZE square with rounded corners
    cage         2 x 2 m, 3 m high: a black, slightly irregular net on the sides and top,
                 a white base strip under the net, aluminium posts in the corners
    lab          the hall around the cage, its walls covered with the lab video frames

Camera like the Mini 3 in the recordings: 16:9, about 70 deg horizontal field of view,
gimbal 50 deg down, stabilised (the drone's own tilt does not move the picture). Each frame
is rendered at 2 x the size and scaled down, which smooths the edges like a real lens.

The later Gaussian-Splatting copy of the cage can replace this module: anything with the
same render(x, y, z, heading, scene, camera) -> RGB array will do.
"""
import glob
import io
import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

CAGE_HALF = 1.0          # m, the cage is 2 x 2 m
CAGE_HEIGHT = 3.0        # m
LAB_HALF = 1.4           # m, the lab walls (photos) seen through the net
LAB_HEIGHT = 3.6
PAD_SIZE = 0.5           # m, side of the pad (landing_score.py uses 50 cm too)
PAD_CORNER = 0.04        # m, corner radius
BORDER = 0.35            # m, dark mats along the net
MAT_M_PER_PX = 0.002     # mat.png: 2 mm per pixel (make_textures.ipm)
NET_CELL = 0.09          # m, net mesh size
NET_STRAND = 0.0035      # m, strand half-thickness
BASE_HEIGHT = 0.10       # m, white base strip under the net
POST = 0.05              # m, corner posts

TEX_DIR = os.path.join(os.path.dirname(__file__), "textures")

PAD_RED = np.array([226, 48, 86])
WHITE = np.array([238, 232, 230])
MAT = np.array([196, 180, 168])
NET = np.array([30, 29, 29])
BASE = np.array([226, 224, 218])
ALU = np.array([175, 178, 182])
LAB_FLOOR = np.array([196, 196, 192])
LAB_CEIL = np.array([222, 224, 226])
LAB_WALL = np.array([212, 214, 216])


def _load(name):
    p = os.path.join(TEX_DIR, name)
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) if os.path.exists(p) else None


class Textures:
    _cache = None

    def __init__(self):
        self.pad = _load("pad.png")
        mat = _load("mat.png")
        self.mat = mat
        if mat is not None:
            lum = mat.mean(-1, keepdims=True)
            # dark mats: same foam pattern, grey and darker (as in the frames along the net)
            self.mat_dark = (lum - lum.mean()) * 0.8 + np.array([96, 94, 93], np.float32)
        else:
            self.mat_dark = None
        self.lab = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
                    for p in sorted(glob.glob(os.path.join(TEX_DIR, "lab_*.jpg")))]

    @classmethod
    def get(cls):
        if cls._cache is None:
            cls._cache = cls()
        return cls._cache


@dataclass
class Camera:
    width: int = 320
    height: int = 180
    hfov_deg: float = 62.0          # estimated from the pad's shape in a real frame (see README)
    gimbal_deg: float = -50.0       # negative = looking down
    supersample: int = 2
    blur: float = 0.6               # px at output size
    noise: float = 2.5              # sensor noise, grey levels
    realistic: bool = True          # False = flat colours, no camera effects (fast, old look)


@dataclass
class Scene:
    pad_x: float = 0.0
    pad_y: float = 0.0
    pad_yaw: float = 0.0            # deg
    show_pad: bool = True
    brightness: float = 1.0
    seed: int = 0
    tile_shade: np.ndarray = field(default=None, repr=False)
    net_phase: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.tile_shade = rng.uniform(-6, 6, size=(64, 64))
        self.net_phase = rng.uniform(0, 2 * math.pi, size=8)
        self.rng = rng


def _rays(cam, heading_deg, w, h):
    hd = math.radians(heading_deg)
    g = math.radians(cam.gimbal_deg)
    fh = np.array([math.sin(hd), math.cos(hd), 0.0])
    right = np.array([math.cos(hd), -math.sin(hd), 0.0])
    fwd = math.cos(g) * fh + np.array([0, 0, math.sin(g)])
    up = -math.sin(g) * fh + np.array([0, 0, math.cos(g)])
    tx = math.tan(math.radians(cam.hfov_deg) / 2)
    ty = tx * h / w
    px = (np.arange(w) + 0.5) / w * 2 - 1
    py = 1 - (np.arange(h) + 0.5) / h * 2
    PX, PY = np.meshgrid(px, py)
    d = fwd[None, None] + PX[..., None] * tx * right[None, None] + PY[..., None] * ty * up[None, None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _box_exit(o, d, half, top):
    """Distance along each ray to where it leaves the box |x|,|y| <= half, 0 <= z <= top, and
    which face: 0 floor, 1 side x, 2 side y, 3 top."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[..., 0] > 0, (half - o[0]) / d[..., 0], (-half - o[0]) / d[..., 0])
        ty = np.where(d[..., 1] > 0, (half - o[1]) / d[..., 1], (-half - o[1]) / d[..., 1])
        tz = np.where(d[..., 2] > 0, (top - o[2]) / d[..., 2], (0 - o[2]) / d[..., 2])
    tx = np.where(np.isfinite(tx) & (tx > 0), tx, np.inf)
    ty = np.where(np.isfinite(ty) & (ty > 0), ty, np.inf)
    tz = np.where(np.isfinite(tz) & (tz > 0), tz, np.inf)
    t = np.minimum(np.minimum(tx, ty), tz)
    face = np.where(t == tz, np.where(d[..., 2] > 0, 3, 0), np.where(t == tx, 1, 2))
    return t, face


def _sample(tex, u, v):
    """Nearest texel; u, v in pixels, mirrored outside the texture (no visible seams)."""
    h, w = tex.shape[:2]
    u = np.mod(u, 2 * w)
    v = np.mod(v, 2 * h)
    u = np.where(u >= w, 2 * w - 1 - u, u).astype(int).clip(0, w - 1)
    v = np.where(v >= h, 2 * h - 1 - v, v).astype(int).clip(0, h - 1)
    return tex[v, u]


def _floor_colour(scene, x, y, tex):
    border = (np.abs(x) > CAGE_HALF - BORDER) | (np.abs(y) > CAGE_HALF - BORDER)
    if tex.mat is not None:
        u, v = x / MAT_M_PER_PX, -y / MAT_M_PER_PX
        col = _sample(tex.mat, u, v).copy()
        col[border] = _sample(tex.mat_dark, u[border] + 137, v[border] + 59)
    else:
        col = np.broadcast_to(MAT, x.shape + (3,)).astype(float).copy()
        col[border] = (98, 96, 95)
    outside = (np.abs(x) > CAGE_HALF) | (np.abs(y) > CAGE_HALF)
    col[outside] = LAB_FLOOR
    if scene.show_pad:
        a = math.radians(scene.pad_yaw)
        dx, dy = x - scene.pad_x, y - scene.pad_y
        u = dx * math.cos(a) - dy * math.sin(a)          # pad's right
        v = dx * math.sin(a) + dy * math.cos(a)          # pad's far edge
        hs = PAD_SIZE / 2
        qu, qv = np.maximum(np.abs(u) - (hs - PAD_CORNER), 0), np.maximum(np.abs(v) - (hs - PAD_CORNER), 0)
        on = (np.abs(u) <= hs) & (np.abs(v) <= hs) & (np.hypot(qu, qv) <= PAD_CORNER)
        if tex.pad is not None:
            n = tex.pad.shape[0]
            pu = ((u[on] + hs) / PAD_SIZE * (n - 1))
            pv = ((hs - v[on]) / PAD_SIZE * (n - 1))
            col[on] = tex.pad[pv.astype(int).clip(0, n - 1), pu.astype(int).clip(0, n - 1)]
        else:
            col[on] = PAD_RED
    return col


def _net_strand(scene, a, b):
    """Slightly irregular diamond net on a wall with surface coordinates a, b (metres)."""
    ph = scene.net_phase
    a2 = a + 0.02 * np.sin(2.1 * b + ph[0]) + 0.01 * np.sin(5.3 * a + 1.7 * b + ph[1])
    b2 = b + 0.025 * np.sin(1.7 * a + ph[2]) + 0.012 * np.sin(4.1 * b + 2.3 * a + ph[5]) \
        + 0.04 * np.sin(0.9 * a + ph[3]) * np.clip(b - 0.3, 0, 1)          # the net sags a little
    p, q = (a2 + b2) / NET_CELL, (a2 - b2) / NET_CELL
    dp, dq = np.abs(p - np.round(p)) * NET_CELL, np.abs(q - np.round(q)) * NET_CELL
    w = NET_STRAND * (1 + 0.3 * np.sin(7 * a + 3 * b + ph[4]))
    return (dp < w) | (dq < w)


def _lab_colour(o, d, tex):
    t, face = _box_exit(o, d, LAB_HALF, LAB_HEIGHT)
    p = o + d * t[..., None]
    col = np.empty(t.shape + (3,), np.float32)
    col[:] = LAB_WALL
    col[face == 0] = LAB_FLOOR
    col[face == 3] = LAB_CEIL
    lights = (face == 3) & (np.mod(p[..., 0], 2.5) < 0.2) & (np.mod(p[..., 1], 3.0) < 1.6)
    col[lights] = (255, 255, 252)
    if tex.lab:
        n = len(tex.lab)
        wall = (face == 1) | (face == 2)
        along = np.where(face == 1, p[..., 1] * np.sign(p[..., 0]), -p[..., 0] * np.sign(p[..., 1]))
        side = np.where(face == 1, np.where(p[..., 0] > 0, 0, 2), np.where(p[..., 1] > 0, 3, 1))
        # two pictures per wall, each LAB_HALF wide, floor to LAB_HEIGHT
        half = (along > 0).astype(int)
        idx = (side * 2 + half) % n
        fu = np.mod(along + LAB_HALF, LAB_HALF) / LAB_HALF
        fv = 1 - np.clip(p[..., 2] / LAB_HEIGHT, 0, 1)
        for k in range(n):
            m = wall & (idx == k)
            if m.any():
                im = tex.lab[k]
                h, w = im.shape[:2]
                col[m] = im[(fv[m] * (h - 1)).astype(int), (fu[m] * (w - 1)).astype(int)]
    return col


def render(x, y, z, heading, scene=None, cam=None):
    """RGB uint8 array (H, W, 3) seen from the drone at (x, y, z) m with this heading (deg)."""
    scene = scene or Scene()
    cam = cam or Camera()
    tex = Textures.get() if cam.realistic else Textures.__new__(Textures)
    if not cam.realistic:
        tex.pad = tex.mat = tex.mat_dark = None
        tex.lab = []
    ss = cam.supersample if cam.realistic else 1
    W, H = cam.width * ss, cam.height * ss
    o = np.array([x, y, max(z, 0.05)])
    d = _rays(cam, heading, W, H)
    t, face = _box_exit(o, d, CAGE_HALF, CAGE_HEIGHT)
    p = o + d * t[..., None]
    col = np.empty(t.shape + (3,), np.float32)
    floor = face == 0
    col[floor] = _floor_colour(scene, p[..., 0][floor], p[..., 1][floor], tex)
    wall = ~floor
    a = np.where(face == 1, p[..., 1], p[..., 0])
    b = np.where(face == 3, p[..., 1], p[..., 2])
    side = (face == 1) | (face == 2)
    strand = wall & _net_strand(scene, a, b)
    base = side & (p[..., 2] < BASE_HEIGHT)
    post = side & (np.abs(a) > CAGE_HALF - POST)
    behind = wall & ~strand & ~base & ~post
    if behind.any():
        col[behind] = _lab_colour(o, d[behind], tex) * 0.9      # seen through the net
    col[strand] = NET
    col[base] = BASE
    col[post] = ALU * (0.85 + 0.15 * np.abs(d[post][:, 0]))[:, None]
    # light: overhead lamps make the floor near the centre a bit brighter, far things dimmer
    r = np.hypot(p[..., 0], p[..., 1])
    light = 1.04 - 0.06 * np.clip(r - 0.5, 0, 2) - 0.012 * np.clip(t, 0, 6)
    col *= light[..., None] * scene.brightness
    if not cam.realistic:
        return np.clip(col, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))
    if ss > 1:
        img = img.resize((cam.width, cam.height), Image.BOX)
    if cam.blur:
        img = img.filter(ImageFilter.GaussianBlur(cam.blur))
    out = np.asarray(img, np.float32)
    # vignetting and sensor noise
    yy, xx = np.mgrid[0:cam.height, 0:cam.width]
    rr = np.hypot((xx - cam.width / 2) / (cam.width / 2), (yy - cam.height / 2) / (cam.height / 2))
    out *= (1 - 0.12 * rr ** 2)[..., None]
    if cam.noise:
        out += scene.rng.normal(0, cam.noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_jpeg(rgb, quality=80):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    return buf.getvalue()
