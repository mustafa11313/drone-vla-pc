"""
render.py  -  the drone camera's view of the flight cage, drawn with numpy only (no GPU).

Realistic version: real textures cut from your own pictures by sim/make_textures.py

    sim/textures/pad.png     the landing pad, from a drone frame
    sim/textures/mat.png     the foam mats seen from above, from drone frames
    sim/textures/lab_*.jpg   the lab, from the walk-around video, as walls of the hall

plus the things a real camera adds: soft overhead light, vignetting, slight blur, sensor
noise, JPEG. When a texture file is missing, plain colours are used instead.

The scene (checked against the lab video):
    floor        black foam mats everywhere inside the cage, worn to a dirty grey in places
    landing pad  PAD_SIZE square with rounded corners
    cage         2 x 2 m, 3 m high: a black, slightly irregular net on the sides and top, down
                 to the floor, black posts in the corners
    lab          the hall around the cage: each wall shows photos of that side from the video
                 (north: yellow robot cell; south: the desk; east/west: the open lab)

Camera like the Mini 3 in the recordings: 16:9, 62 deg horizontal field of view, gimbal 50 deg
down, stabilised, and auto-exposure: the floor is black, so the camera brightens the picture a
lot (that is why the real frames show the mats grey-beige). Each frame is rendered at 2 x the
size and scaled down, which smooths the edges like a real lens.

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
# where the lab "walls" (photos from the video) stand, in metres from the cage centre.
# x = east, y = north. North: the yellow robot cell right behind the net; south: the desk where
# you stand; east and west: the open lab. Distances are guesses from the video.
LAB_BOUNDS = {"west": -2.5, "east": 3.0, "south": -1.8, "north": 1.5}
LAB_HEIGHT = 3.6
PAD_SIZE = 0.5           # m, side of the pad (landing_score.py uses 50 cm too)
PAD_CORNER = 0.04        # m, corner radius
MAT_M_PER_PX = 0.002     # mat.png: 2 mm per pixel (make_textures.ipm)
NET_CELL = 0.09          # m, net mesh size
NET_STRAND = 0.0035      # m, strand half-thickness
POST = 0.04              # m, black corner posts

TEX_DIR = os.path.join(os.path.dirname(__file__), "textures")

# The scene is described in "physical" brightness and the camera then brightens it like the
# drone's auto-exposure does (EXPOSURE_TARGET). The floor is black foam mats: the drone frames
# show them light grey-beige only because the camera brightens the dark picture. Textures cut
# from drone frames are therefore divided back by the brightening they had in those frames.
FLOOR_DIV = 5.2          # mat.png came from a floor-only drone frame, strongly brightened
PAD_DIV = 3.0            # pad.png came from a frame with the pad (less brightening)
LAB_DIV = 1.7            # lab photos came from the phone, exposed for the bright lab
EXPOSURE_TARGET = 200.0  # mean grey level the camera aims for (tuned with sim.realism)
MAX_GAIN = 7.0
EDGE_DARK = 0.62         # how much blacker the mats along the net are than the worn middle
KNEE = 170.0             # above this grey level the camera compresses highlights
FLOOR_TINT = np.array([1.12, 0.97, 0.87], np.float32)   # drone camera white balance on the black mats

PAD_RED = np.array([76, 16, 29])
MAT = np.array([34, 34, 34])
NET = np.array([12, 12, 12])
POST_COL = np.array([20, 20, 21])
LAB_FLOOR = np.array([105, 105, 102])
LAB_CEIL = np.array([125, 126, 128])
LAB_WALL = np.array([120, 121, 122])
SIDES = ("north", "east", "south", "west")


def _load(name):
    p = os.path.join(TEX_DIR, name)
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) if os.path.exists(p) else None


class Textures:
    _cache = None

    def __init__(self):
        pad = _load("pad.png")
        self.pad = pad / PAD_DIV if pad is not None else None
        mat = _load("mat.png")
        if mat is not None:
            # black mats, worn to a dirty grey in places by walking: lighter scuffed patches
            rng = np.random.default_rng(7)
            wear = Image.fromarray((rng.random((4, 5)) * 255).astype(np.uint8)).resize(
                (mat.shape[1], mat.shape[0]), Image.BICUBIC).filter(ImageFilter.GaussianBlur(40))
            wear = 0.85 + 0.3 * np.asarray(wear, np.float32)[..., None] / 255.0
            # neutral black: the warm tint in the drone frames is the camera's (FLOOR_TINT)
            grey = mat @ np.array([0.299, 0.587, 0.114], np.float32)
            self.mat = (grey[..., None] / FLOOR_DIV * wear).repeat(3, -1)
        else:
            self.mat = None
        self.lab = {s: [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / LAB_DIV
                        for p in sorted(glob.glob(os.path.join(TEX_DIR, f"lab_{s}_*.jpg")))]
                    for s in SIDES}

    @classmethod
    def get(cls):
        if cls._cache is None:
            cls._cache = cls()
        return cls._cache

    @classmethod
    def flat(cls):
        t = cls.__new__(cls)
        t.pad = t.mat = None
        t.lab = {s: [] for s in SIDES}
        return t


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
    net_phase: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
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


def _box_exit(o, d, bounds, top):
    """Distance along each ray to where it leaves the box xmin..xmax, ymin..ymax, 0..top, and
    which face: 0 floor, 1 side x, 2 side y, 3 top. bounds = (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = bounds
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[..., 0] > 0, (xmax - o[0]) / d[..., 0], (xmin - o[0]) / d[..., 0])
        ty = np.where(d[..., 1] > 0, (ymax - o[1]) / d[..., 1], (ymin - o[1]) / d[..., 1])
        tz = np.where(d[..., 2] > 0, (top - o[2]) / d[..., 2], (0 - o[2]) / d[..., 2])
    tx = np.where(np.isfinite(tx) & (tx > 0), tx, np.inf)
    ty = np.where(np.isfinite(ty) & (ty > 0), ty, np.inf)
    tz = np.where(np.isfinite(tz) & (tz > 0), tz, np.inf)
    t = np.minimum(np.minimum(tx, ty), tz)
    face = np.where(t == tz, np.where(d[..., 2] > 0, 3, 0), np.where(t == tx, 1, 2))
    return t, face


CAGE_BOUNDS = (-CAGE_HALF, CAGE_HALF, -CAGE_HALF, CAGE_HALF)
LAB_BOX = (LAB_BOUNDS["west"], LAB_BOUNDS["east"], LAB_BOUNDS["south"], LAB_BOUNDS["north"])


def _sample(tex, u, v):
    """Nearest texel; u, v in pixels, mirrored outside the texture (no visible seams)."""
    h, w = tex.shape[:2]
    u = np.mod(u, 2 * w)
    v = np.mod(v, 2 * h)
    u = np.where(u >= w, 2 * w - 1 - u, u).astype(int).clip(0, w - 1)
    v = np.where(v >= h, 2 * h - 1 - v, v).astype(int).clip(0, h - 1)
    return tex[v, u]


def _floor_colour(scene, x, y, tex):
    if tex.mat is not None:
        col = _sample(tex.mat, x / MAT_M_PER_PX, -y / MAT_M_PER_PX).copy()
    else:
        col = np.broadcast_to(MAT, x.shape + (3,)).astype(np.float32).copy()
    # less walked-on along the net: the mats there stayed blacker (the darker band in the frames)
    edge = CAGE_HALF - np.maximum(np.abs(x), np.abs(y))
    col *= (EDGE_DARK + (1 - EDGE_DARK) * np.clip((edge - 0.12) / 0.3, 0, 1))[..., None]
    outside = edge < 0
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


def lab_wall_colour(tex, p, side):
    """Colour of the lab wall `side` at points p (N x 3): the side's photos side by side, as
    seen from inside the cage facing that wall (left to right), floor to LAB_HEIGHT."""
    b = LAB_BOUNDS
    if side == "north":
        along, length = p[:, 0] - b["west"], b["east"] - b["west"]
    elif side == "south":
        along, length = b["east"] - p[:, 0], b["east"] - b["west"]
    elif side == "east":
        along, length = b["north"] - p[:, 1], b["north"] - b["south"]
    else:
        along, length = p[:, 1] - b["south"], b["north"] - b["south"]
    out = np.empty((len(p), 3), np.float32)
    out[:] = LAB_WALL
    photos = tex.lab.get(side) or []
    if not photos:
        return out
    n = len(photos)
    seg = np.clip((along / length * n).astype(int), 0, n - 1)
    fu = np.clip(along / length * n - seg, 0, 1)
    fv = 1 - np.clip(p[:, 2] / LAB_HEIGHT, 0, 1)
    for k, im in enumerate(photos):
        m = seg == k
        if m.any():
            h, w = im.shape[:2]
            out[m] = im[(fv[m] * (h - 1)).astype(int), (fu[m] * (w - 1)).astype(int)]
    return out


def _lab_colour(o, d, tex):
    t, face = _box_exit(o, d, LAB_BOX, LAB_HEIGHT)
    p = o + d * t[..., None]
    col = np.empty(t.shape + (3,), np.float32)
    col[:] = LAB_WALL
    col[face == 0] = LAB_FLOOR
    col[face == 3] = LAB_CEIL
    lights = (face == 3) & (np.mod(p[..., 0], 2.5) < 0.2) & (np.mod(p[..., 1], 3.0) < 1.6)
    col[lights] = (255, 255, 252)
    for side, m in (("east", (face == 1) & (p[..., 0] > 0)), ("west", (face == 1) & (p[..., 0] <= 0)),
                    ("north", (face == 2) & (p[..., 1] > 0)), ("south", (face == 2) & (p[..., 1] <= 0))):
        if m.any():
            col[m] = lab_wall_colour(tex, p[m], side)
    return col


def expose(col, cam):
    """The drone camera's auto-exposure, sensor and lens: brighten to EXPOSURE_TARGET, scale down
    (anti-aliasing), slight blur, vignetting, noise."""
    lum = float((col @ np.array([0.299, 0.587, 0.114], np.float32)).mean())
    gain = min(MAX_GAIN, max(0.7, EXPOSURE_TARGET / max(lum, 1.0)))
    v = col * gain
    # soft highlights: the bright lab behind the net keeps some detail instead of turning white
    v = np.where(v > KNEE, KNEE + (255 - KNEE) * np.tanh((v - KNEE) / (255 - KNEE)), v)
    img = Image.fromarray(np.clip(v, 0, 255).astype(np.uint8))
    if img.size != (cam.width, cam.height):
        img = img.resize((cam.width, cam.height), Image.BOX)
    if cam.realistic and cam.blur:
        img = img.filter(ImageFilter.GaussianBlur(cam.blur))
    return np.asarray(img, np.float32), gain


def render(x, y, z, heading, scene=None, cam=None):
    """RGB uint8 array (H, W, 3) seen from the drone at (x, y, z) m with this heading (deg)."""
    scene = scene or Scene()
    cam = cam or Camera()
    tex = Textures.get() if cam.realistic else Textures.flat()
    ss = cam.supersample if cam.realistic else 1
    W, H = cam.width * ss, cam.height * ss
    o = np.array([x, y, max(z, 0.05)])
    d = _rays(cam, heading, W, H)
    t, face = _box_exit(o, d, CAGE_BOUNDS, CAGE_HEIGHT)
    p = o + d * t[..., None]
    col = np.empty(t.shape + (3,), np.float32)
    floor = face == 0
    col[floor] = _floor_colour(scene, p[..., 0][floor], p[..., 1][floor], tex) * FLOOR_TINT
    wall = ~floor
    a = np.where(face == 1, p[..., 1], p[..., 0])
    b = np.where(face == 3, p[..., 1], p[..., 2])
    side = (face == 1) | (face == 2)
    strand = wall & _net_strand(scene, a, b)
    post = side & (np.abs(a) > CAGE_HALF - POST)
    behind = wall & ~strand & ~post
    if behind.any():
        col[behind] = _lab_colour(o, d[behind], tex) * 0.92      # seen through the net
    col[strand] = NET
    col[post] = POST_COL
    # light: overhead lamps make the floor near the centre a bit brighter, far things dimmer
    r = np.hypot(p[..., 0], p[..., 1])
    light = 1.04 - 0.06 * np.clip(r - 0.5, 0, 2) - 0.012 * np.clip(t, 0, 6)
    col *= light[..., None] * scene.brightness
    out, _ = expose(col, cam)
    if not cam.realistic:
        return out.astype(np.uint8)
    yy, xx = np.mgrid[0:cam.height, 0:cam.width]
    rr = np.hypot((xx - cam.width / 2) / (cam.width / 2), (yy - cam.height / 2) / (cam.height / 2))
    out = out * (1 - 0.12 * rr ** 2)[..., None]
    if cam.noise:
        out += scene.rng.normal(0, cam.noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_jpeg(rgb, quality=80):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    return buf.getvalue()
