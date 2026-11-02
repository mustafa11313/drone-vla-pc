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


PROBES_FILE = os.environ.get("SIM_PROBES", os.path.join(os.path.dirname(__file__), "blender", "probes.npz"))
FACE_AXES = {"px": ((1, 0, 0), (0, 0, 1)), "nx": ((-1, 0, 0), (0, 0, 1)),
             "py": ((0, 1, 0), (0, 0, 1)), "ny": ((0, -1, 0), (0, 0, 1)),
             "pz": ((0, 0, 1), (0, 1, 0)), "nz": ((0, 0, -1), (0, 1, 0))}


MAX_PROBES = 4          # blend at most this many of the nearest probes
PARALLAX_STEPS = 1
LAB_PROBE_GAIN = 0.4    # lab vs black mats as the drone camera sees them (tuned on real frames)


class Probes:
    """The lab as rendered by Blender (sim/blender/build_lab.py --probes): cube pictures with
    depth from a grid of points in the cage. Looked up for the rays that pass the net."""
    _cache = {}

    def __init__(self, path):
        z = np.load(path)
        self.pos = z["pos"].astype(np.float32)
        self.rgb = z["rgb"]                             # (P, 6, R, R, 3) linear, float16
        self.depth = z["depth"].astype(np.float32)[..., None]   # (P, 6, r, r, 1)
        faces = [str(f) for f in z["faces"]]
        ax = [FACE_AXES[f] for f in faces]
        self.fwd = np.array([a[0] for a in ax], np.float32)
        self.up = np.array([a[1] for a in ax], np.float32)
        self.right = np.cross(self.fwd, self.up)
        self.xs, self.ys, self.zs = (np.unique(self.pos[:, i]) for i in range(3))
        self.mat_radiance = float(z["mat_radiance"])

    @classmethod
    def get(cls, path=None):
        path = path or PROBES_FILE
        if path not in cls._cache:
            cls._cache[path] = cls(path) if os.path.exists(path) else None
        return cls._cache[path]

    def _lookup(self, img, i, d):
        """Sample cube i of `img` (P, 6, n, n, C) in directions d (N x 3), nearest texel."""
        a = np.abs(d)
        m = np.argmax(a, 1)
        sign = d[np.arange(len(d)), m] > 0
        face = np.where(m == 0, np.where(sign, 0, 1), np.where(m == 1, np.where(sign, 2, 3), np.where(sign, 4, 5)))
        f, r, u = self.fwd[face], self.right[face], self.up[face]
        k = np.einsum("ij,ij->i", d, f)
        x = np.einsum("ij,ij->i", d, r) / k
        y = np.einsum("ij,ij->i", d, u) / k
        n = img.shape[2]
        col = np.clip(((x + 1) / 2 * n).astype(int), 0, n - 1)
        row = np.clip(((1 - y) / 2 * n).astype(int), 0, n - 1)
        return img[i, face, row, col]

    def _weights(self, o):
        out = []
        idx = []
        for axis, vals in zip(o, (self.xs, self.ys, self.zs)):
            v = np.clip(axis, vals[0], vals[-1])
            j = int(np.clip(np.searchsorted(vals, v) - 1, 0, len(vals) - 2)) if len(vals) > 1 else 0
            t = (v - vals[j]) / (vals[j + 1] - vals[j]) if len(vals) > 1 else 0.0
            idx.append((j, min(j + 1, len(vals) - 1)))
            out.append((1 - t, t))
        res = []
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    w = out[0][a] * out[1][b] * out[2][c]
                    if w > 1e-3:
                        p = (self.xs[idx[0][a]], self.ys[idx[1][b]], self.zs[idx[2][c]])
                        i = int(np.argmin(np.abs(self.pos - np.array(p, np.float32)).sum(1)))
                        res.append((i, w))
        res = sorted(res, key=lambda r: -r[1])[:MAX_PROBES]
        tot = sum(w for _, w in res)
        return [(i, w / tot) for i, w in res]

    def colour(self, o, d):
        """Linear colour seen from o along d, blended from the nearest probes with parallax
        correction: where the probe's ray hits, seen from the drone."""
        o = np.asarray(o, np.float32)
        d = d.astype(np.float32)
        out = np.zeros((len(d), 3), np.float32)
        for i, w in self._weights(o):
            off = o - self.pos[i]
            dep = self.depth
            b = d @ off                                       # d . off
            c = float(off @ off)
            t = self._lookup(dep, i, d)[:, 0]
            for _ in range(PARALLAX_STEPS):   # find t with |off + d t| = probe depth in that direction
                q = d * t[:, None] + off
                dist = self._lookup(dep, i, q / np.linalg.norm(q, axis=1, keepdims=True))[:, 0]
                t = -b + np.sqrt(np.maximum(b * b - c + dist * dist, 0))
            q = d * t[:, None] + off
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            out += w * self._lookup(self.rgb, i, q).astype(np.float32)
        return out


def _lab_colour(o, d, tex):
    pr = Probes.get() if tex.mat is not None else None
    if pr is not None:
        flat = d.reshape(-1, 3)
        scale = LAB_PROBE_GAIN * float(tex.mat.mean()) / max(pr.mat_radiance, 1e-6)   # Blender light -> our units
        return (pr.colour(o, flat) * scale).reshape(d.shape)
    return _lab_colour_photos(o, d, tex)


def _lab_colour_photos(o, d, tex):
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
        if ss > 1 and Probes.get() is not None and tex.mat is not None:
            # the lab behind the net is blurry anyway: look it up at output size, then enlarge
            dl = d[ss // 2::ss, ss // 2::ss]
            lab = _lab_colour(o, dl.reshape(-1, 3), tex).reshape(dl.shape)
            lab = lab.repeat(ss, 0).repeat(ss, 1)
            col[behind] = lab[behind] * 0.92
        else:
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
