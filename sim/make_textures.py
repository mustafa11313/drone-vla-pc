"""
make_textures.py  -  cut real textures for the simulator from your own pictures.

    python3 -m sim.make_textures --pad-frame <episode>/frames/0001_000223.jpg \
        --floor-frames <episode>/frames/0001_0000{40,60,80}.jpg --floor-height 0.6 \
        --lab-video lab.mp4

Writes sim/textures/:
  pad.png          the landing pad, straightened to a square (from one frame where the whole
                   pad is in view; its four corners are found from the red pixels)
  mat.png          about 1 x 1 m of the beige foam mats seen from straight above, made by
                   projecting floor-only frames onto the floor (camera height + 50 deg gimbal)
  lab_00.jpg ...   frames of the lab video, used as the walls of the hall behind the net

Only numpy and Pillow (plus ffmpeg for the video). Run it again with other frames if a
texture looks wrong; render.py falls back to drawn colours when a file is missing.
"""
import argparse, glob, math, os, subprocess, tempfile

import numpy as np
from PIL import Image, ImageFilter

from brain.landing_score import red_mask

OUT = os.path.join(os.path.dirname(__file__), "textures")
HFOV = 62.0          # deg, see render.Camera


def homography(src, dst):
    """3x3 H with dst ~ H @ src for 4 point pairs."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def pad_corners(img):
    m = red_mask(img)
    ys, xs = np.nonzero(m)
    s, d = xs + ys, xs - ys
    return [(xs[s.argmin()], ys[s.argmin()]), (xs[d.argmax()], ys[d.argmax()]),
            (xs[s.argmax()], ys[s.argmax()]), (xs[d.argmin()], ys[d.argmin()])]


def make_pad(frame, size=512):
    img = Image.open(frame).convert("RGB")
    c = pad_corners(img)          # top-left, top-right, bottom-right, bottom-left in the image
    # far edge of the pad (top of the picture) = top of the texture
    H = homography([(0, 0), (size, 0), (size, size), (0, size)], c)     # texture -> image
    coeffs = H.flatten()[:8]
    pad = img.transform((size, size), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC)
    return pad.filter(ImageFilter.UnsharpMask(radius=2, percent=60))


def ipm(frame, height, gimbal=-50.0, hfov=HFOV, ahead=(0.25, 1.25), side=0.5, mm=2.0):
    """Floor seen from above: a (side*2) x (ahead) metre patch, mm millimetres per pixel."""
    img = np.asarray(Image.open(frame).convert("RGB")).astype(np.float32)
    H, W = img.shape[:2]
    nx, ny = int(2 * side * 1000 / mm), int((ahead[1] - ahead[0]) * 1000 / mm)
    X = np.linspace(-side, side, nx)
    Y = np.linspace(ahead[1], ahead[0], ny)            # far at the top
    gx, gy = np.meshgrid(X, Y)
    g = math.radians(gimbal)
    # camera axes (camera looks along +y, tilted down)
    f = np.array([0, math.cos(g), math.sin(g)])
    up = np.array([0, -math.sin(g), math.cos(g)])
    r = np.array([1.0, 0, 0])
    p = np.stack([gx, gy, np.full_like(gx, -height)], -1)
    zc = p @ f
    tx = math.tan(math.radians(hfov) / 2)
    ty = tx * H / W
    u = (p @ r) / zc / tx
    v = (p @ up) / zc / ty
    px = ((u + 1) / 2 * W).clip(0, W - 1).astype(int)
    py = ((1 - v) / 2 * H).clip(0, H - 1).astype(int)
    return img[py, px]


def make_mat(frames, height):
    patches = [ipm(f, height) for f in frames]
    mat = np.median(np.stack(patches), 0) if len(patches) > 2 else patches[0]
    return Image.fromarray(mat.clip(0, 255).astype(np.uint8))


def lab_frames(video, n=12):
    tmp = tempfile.mkdtemp()
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", video]).decode().strip())
    out = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        p = os.path.join(tmp, f"{i:02d}.jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", p], check=True)
        out.append(Image.open(p).convert("RGB"))
    return out


def main():
    ap = argparse.ArgumentParser(description="cut simulator textures from real pictures")
    ap.add_argument("--pad-frame", help="a drone frame with the whole pad in view")
    ap.add_argument("--floor-frames", nargs="*", help="drone frames that show only the beige mats")
    ap.add_argument("--floor-height", type=float, default=0.6, help="drone height (m) in those frames")
    ap.add_argument("--lab-video", help="a video walking around the lab")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.pad_frame:
        make_pad(args.pad_frame).save(os.path.join(args.out, "pad.png"))
        print("pad.png")
    if args.floor_frames:
        make_mat(args.floor_frames, args.floor_height).save(os.path.join(args.out, "mat.png"))
        print("mat.png")
    if args.lab_video:
        for old in glob.glob(os.path.join(args.out, "lab_*.jpg")):
            os.remove(old)
        for i, im in enumerate(lab_frames(args.lab_video)):
            im.thumbnail((360, 640))
            im.save(os.path.join(args.out, f"lab_{i:02d}.jpg"), quality=85)
        print("lab_*.jpg")


if __name__ == "__main__":
    main()
