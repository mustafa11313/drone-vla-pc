"""
render.py  -  the drone camera's view of the flight cage, drawn with numpy only (no GPU).

The scene is built from simple shapes so the whole Claude + VLA loop can be tested before the
lab video exists:

    floor        grey-beige foam mats (60 cm tiles), a darker strip along the net
    landing pad  50 x 50 cm, pink-red with a white ring and an H (like the real one)
    cage         2 x 2 m, 3 m high, black diamond net on the four sides and the top
    lab          a large light-grey hall behind the net, with a few coloured machines

Later the realistic 3D copy of the cage (Gaussian Splatting from the video) replaces this
module: anything with the same render(pose) -> image interface will do.

Camera: like the Mini 3 in the recordings: 16:9, about 74 deg horizontal field of view,
gimbal 50 deg down, stabilised (the drone's own tilt does not move the picture).
"""
import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

CAGE_HALF = 1.0          # m, the cage is 2 x 2 m
CAGE_HEIGHT = 3.0        # m
LAB_HALF = 8.0           # m, the hall around the cage
LAB_HEIGHT = 5.0
PAD_SIZE = 0.5           # m
TILE = 0.6               # m, foam mat tiles
BORDER = 0.25            # m, darker strip along the net
NET_CELL = 0.10          # m, net mesh size
NET_STRAND = 0.012       # m, strand thickness

PAD_RED = np.array([232, 52, 88])
WHITE = np.array([245, 240, 242])
MAT = np.array([176, 166, 156])
MAT_BORDER = np.array([110, 108, 106])
NET = np.array([28, 28, 30])
LAB_WALL = np.array([215, 218, 220])
LAB_FLOOR = np.array([200, 200, 196])
LAB_CEIL = np.array([230, 232, 234])

# machines in the hall: (x0, x1, y0, y1, height, colour) - only their outline matters
MACHINES = [(-6.0, -4.5, 3.0, 4.5, 1.6, (235, 190, 30)),    # yellow robot cell
            (4.0, 5.5, -2.0, -0.5, 1.9, (40, 70, 160)),     # blue cabinet
            (2.0, 3.0, 5.0, 6.0, 1.4, (90, 200, 60)),       # green robot arm
            (-5.0, -3.0, -6.0, -5.0, 0.9, (205, 160, 90)),  # wooden bench
            (5.5, 7.0, 4.0, 5.0, 1.2, (230, 120, 40))]      # orange bench


@dataclass
class Camera:
    width: int = 320
    height: int = 180
    hfov_deg: float = 74.4
    gimbal_deg: float = -50.0       # negative = looking down


@dataclass
class Scene:
    pad_x: float = 0.0
    pad_y: float = 0.0
    pad_yaw: float = 0.0            # deg
    show_pad: bool = True
    brightness: float = 1.0
    seed: int = 0
    tile_shade: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.tile_shade = rng.uniform(-10, 10, size=(64, 64))


def _rays(cam, heading_deg):
    h = math.radians(heading_deg)
    g = math.radians(cam.gimbal_deg)
    fh = np.array([math.sin(h), math.cos(h), 0.0])
    right = np.array([math.cos(h), -math.sin(h), 0.0])
    fwd = math.cos(g) * fh + np.array([0, 0, math.sin(g)])
    up = -math.sin(g) * fh + np.array([0, 0, math.cos(g)])
    tx = math.tan(math.radians(cam.hfov_deg) / 2)
    ty = tx * cam.height / cam.width
    px = (np.arange(cam.width) + 0.5) / cam.width * 2 - 1
    py = 1 - (np.arange(cam.height) + 0.5) / cam.height * 2
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


def _floor_colour(scene, x, y):
    col = np.broadcast_to(MAT, x.shape + (3,)).astype(float).copy()
    ti = (np.floor(x / TILE).astype(int) % 64, np.floor(y / TILE).astype(int) % 64)
    col += scene.tile_shade[ti][..., None]
    fx, fy = np.mod(x, TILE), np.mod(y, TILE)
    seam = (np.minimum(fx, TILE - fx) < 0.006) | (np.minimum(fy, TILE - fy) < 0.006)
    col[seam] *= 0.85
    border = (np.abs(x) > CAGE_HALF - BORDER) | (np.abs(y) > CAGE_HALF - BORDER)
    col[border] = MAT_BORDER + scene.tile_shade[ti][border][..., None] * 0.5
    if scene.show_pad:
        a = math.radians(scene.pad_yaw)
        dx, dy = x - scene.pad_x, y - scene.pad_y
        u = dx * math.cos(a) - dy * math.sin(a)
        v = dx * math.sin(a) + dy * math.cos(a)
        on = (np.abs(u) <= PAD_SIZE / 2) & (np.abs(v) <= PAD_SIZE / 2)
        r = np.hypot(u, v)
        s = PAD_SIZE / 0.5
        ring = (r > 0.165 * s) & (r < 0.178 * s)
        h_bars = (np.abs(np.abs(u) - 0.062 * s) < 0.013 * s) & (np.abs(v) < 0.09 * s)
        h_cross = (np.abs(u) < 0.075 * s) & (np.abs(v) < 0.012 * s)
        white = on & (ring | h_bars | h_cross)
        col[on] = PAD_RED
        col[white] = WHITE
    return col


def _net_mask(a, b):
    """Diamond net on a wall with surface coordinates a, b (metres)."""
    p, q = (a + b) / NET_CELL, (a - b) / NET_CELL
    dp, dq = np.abs(p - np.round(p)), np.abs(q - np.round(q))
    w = NET_STRAND / NET_CELL
    return (dp < w) | (dq < w)


def _lab_colour(o, d):
    t, face = _box_exit(o, d, LAB_HALF, LAB_HEIGHT)
    p = o + d * t[..., None]
    col = np.empty(t.shape + (3,))
    col[:] = LAB_WALL
    col[face == 0] = LAB_FLOOR
    col[face == 3] = LAB_CEIL
    # ceiling light strips
    lights = (face == 3) & (np.mod(p[..., 0], 3.0) < 0.25)
    col[lights] = (255, 255, 250)
    # machines: seen where the ray passes their footprint below their height
    for x0, x1, y0, y1, hgt, c in MACHINES:
        hit = _ray_hits_block(o, d, t, x0, x1, y0, y1, hgt)
        col[hit] = c
    # windows band on the walls
    win = (face != 0) & (face != 3) & (p[..., 2] > 2.2) & (p[..., 2] < 3.4)
    col[win] = col[win] * 0.6 + np.array([120, 150, 190]) * 0.4
    return col


def _ray_hits_block(o, d, tmax, x0, x1, y0, y1, h):
    with np.errstate(divide="ignore", invalid="ignore"):
        tlo = np.full(tmax.shape, 0.0)
        thi = tmax.copy()
        for k, (lo, hi) in enumerate(((x0, x1), (y0, y1), (0.0, h))):
            t1 = (lo - o[k]) / d[..., k]
            t2 = (hi - o[k]) / d[..., k]
            tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
            tn = np.where(np.isnan(tn), -np.inf, tn)
            tf = np.where(np.isnan(tf), np.inf, tf)
            tlo, thi = np.maximum(tlo, tn), np.minimum(thi, tf)
    return tlo < thi


def render(x, y, z, heading, scene=None, cam=None):
    """RGB uint8 array (H, W, 3) seen from the drone at (x, y, z) m with this heading (deg)."""
    scene = scene or Scene()
    cam = cam or Camera()
    o = np.array([x, y, max(z, 0.05)])
    d = _rays(cam, heading)
    t, face = _box_exit(o, d, CAGE_HALF, CAGE_HEIGHT)
    p = o + d * t[..., None]
    col = np.empty(t.shape + (3,))
    floor = face == 0
    col[floor] = _floor_colour(scene, p[..., 0][floor], p[..., 1][floor])
    wall = ~floor
    a = np.where(face == 1, p[..., 1], p[..., 0])
    b = np.where(face == 3, p[..., 1], p[..., 2])
    strand = wall & _net_mask(a, b)
    behind = wall & ~strand
    if behind.any():
        col[behind] = _lab_colour(o, d[behind]) * 0.92        # seen through the net: a bit darker
    col[strand] = NET
    # distance shading so the floor near the drone is a little brighter
    col *= (1.0 - np.clip(t, 0, 6)[..., None] * 0.02) * scene.brightness
    return np.clip(col, 0, 255).astype(np.uint8)


def to_jpeg(rgb, quality=85):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    return buf.getvalue()
