"""
landing_score.py  -  how far from the pad centre did the drone land? (no tape measure)

Setup: a phone (or any camera) on a tripod or taped to the ceiling, looking straight down at
the pad, NOT moved during a session. Per flight, two photos:

    before.jpg  - the pad alone (drone not in the picture)
    after.jpg   - the drone sitting where it landed

    python3 -m brain.landing_score before.jpg after.jpg --pad-cm 50
    python3 -m brain.landing_score before.jpg after.jpg --pad-cm 50 --debug debug.png

How: the pad = the red pixels of before.jpg (centre + size; the known pad width turns pixels
into centimetres). The drone = what changed between the two photos. Result: distance in cm
between the pad centre and the drone centre, and whether the drone centre is on the pad.

Only numpy + Pillow, no OpenCV. Keep the light constant between the two photos.

Blind A/B flights (brain/ab_test.py): add --flight 7 --scores logs/ab_scores.csv to append the
result to the score sheet; the A/B key is not needed (and should not be looked at) for this.
"""
import argparse, csv, json, math, os

import numpy as np
from PIL import Image, ImageFilter

RED_MIN = 120           # red channel at least this
RED_MARGIN = 50         # and this much above green and blue
DIFF_MIN = 40           # grey-level change that counts as "something is here now"


def load(path, width=800):
    img = Image.open(path).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    return img


def red_mask(img):
    a = np.asarray(img).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r >= RED_MIN) & (r - g >= RED_MARGIN) & (r - b >= RED_MARGIN)


def blob(mask, min_px=50):
    """Centre (x, y) and pixel count of a mask, or None when too few pixels."""
    ys, xs = np.nonzero(mask)
    if len(xs) < min_px:
        return None
    return float(xs.mean()), float(ys.mean()), int(len(xs))


def score(before_path, after_path, pad_cm, square=True):
    before, after = load(before_path), load(after_path)
    if before.size != after.size:
        raise SystemExit("the two photos must come from the same, unmoved camera")
    pad = blob(red_mask(before))
    if pad is None:
        raise SystemExit("no red pad found in the 'before' photo")
    px, py, area = pad
    # pad width in pixels from its area: square pad -> sqrt(area), round pad -> 2 sqrt(area / pi)
    pad_px = math.sqrt(area) if square else 2 * math.sqrt(area / math.pi)
    cm_per_px = pad_cm / pad_px

    g0 = np.asarray(before.convert("L").filter(ImageFilter.MedianFilter(5))).astype(int)
    g1 = np.asarray(after.convert("L").filter(ImageFilter.MedianFilter(5))).astype(int)
    changed = np.abs(g1 - g0) >= DIFF_MIN
    changed = np.asarray(Image.fromarray((changed * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(7))) > 0
    drone = blob(changed, min_px=200)
    if drone is None:
        raise SystemExit("no drone found: the two photos look the same")
    dx, dy, _ = drone
    dist_cm = math.hypot(dx - px, dy - py) * cm_per_px
    on_pad = (abs(dx - px) <= pad_px / 2 and abs(dy - py) <= pad_px / 2) if square \
        else math.hypot(dx - px, dy - py) <= pad_px / 2
    result = {"distance_cm": round(dist_cm, 1), "on_pad": bool(on_pad),
              "pad_centre_px": [round(px), round(py)], "drone_centre_px": [round(dx), round(dy)],
              "cm_per_px": round(cm_per_px, 3)}
    return result, (after, changed, (px, py), (dx, dy), pad_px)


def save_debug(path, after, changed, pad_c, drone_c, pad_px):
    a = np.asarray(after).copy()
    a[changed] = (0.5 * a[changed] + [0, 0, 127]).astype(np.uint8)       # changed area tinted blue
    img = Image.fromarray(a)
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    (px, py), (dx, dy), h = pad_c, drone_c, pad_px / 2
    d.rectangle([px - h, py - h, px + h, py + h], outline=(255, 0, 0), width=2)
    d.line([px, py, dx, dy], fill=(255, 255, 0), width=2)
    d.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], outline=(0, 255, 0), width=2)
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description="landing distance from an overhead camera")
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--pad-cm", type=float, required=True, help="pad width (square) or diameter (round), cm")
    ap.add_argument("--round", action="store_true", help="the pad is round, not square")
    ap.add_argument("--debug", help="save a picture with the pad, the drone and the distance drawn in")
    ap.add_argument("--flight", type=int, help="blind A/B flight number (see brain/ab_test.py)")
    ap.add_argument("--scores", default="logs/ab_scores.csv", help="score sheet the --flight result goes to")
    args = ap.parse_args()
    result, dbg = score(args.before, args.after, args.pad_cm, square=not args.round)
    if args.debug:
        save_debug(args.debug, *dbg)
    print(json.dumps(result))
    if args.flight is not None:
        new = not os.path.exists(args.scores)
        os.makedirs(os.path.dirname(args.scores) or ".", exist_ok=True)
        with open(args.scores, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["flight", "distance_cm", "on_pad"])
            w.writerow([args.flight, result["distance_cm"], int(result["on_pad"])])
        print(f"flight {args.flight} added to {args.scores}")


if __name__ == "__main__":
    main()
