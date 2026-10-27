"""
phases.py  -  automatic phase labels for every step of an episode (idea D5).

    python3 -m brain.phases episode_0001_* episode_0002_*            # summary per episode
    python3 -m brain.phases episode_0001_* --csv                       # also writes <ep>/phases.csv

Each flown step (same trimming as csv_to_hdf5.py) gets one label, from the pilot's sticks and
from where the red pad is in the camera frame:

    climb      throttle up
    search     pad not in the picture
    approach   pad in the picture, pilot moving toward it (pitch or roll stick)
    align      pad in the picture, sticks almost centred (fine correction / settling)
    descend    throttle down

Why: one MAE over a whole episode hides WHERE the model is wrong. With labels, rehearsal.py
reports the error per phase ("good at approach, bad at final descent"), and the thesis can show
how much of each flight is search vs. approach vs. descent. The pad position (pad_x, pad_y,
-1..1 from the image centre) is also written: it is a free, simple "where is the target" signal.

Only numpy + Pillow + pandas. Thresholds are first guesses; check a few episodes with --csv.
"""
import argparse, io, os

import numpy as np
from PIL import Image

from brain.landing_score import red_mask
from brain.rehearsal import load_episode

PHASES = ["climb", "search", "approach", "align", "descend"]
MOVE_PCT = 15          # stick percent that counts as "the pilot is steering"
THROTTLE_PCT = 10      # stick percent that counts as climbing / descending
PAD_MIN_FRAC = 0.002   # share of red pixels that counts as "pad visible"
SMALL = 160            # frames are scaled down to this width for speed


def pad_in_frame(jpeg):
    """(visible, pad_x, pad_y, red_fraction); pad_x/pad_y in -1..1 from the image centre."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    img = img.resize((SMALL, max(1, round(img.height * SMALL / img.width))), Image.BILINEAR)
    m = red_mask(img)
    frac = float(m.mean())
    if frac < PAD_MIN_FRAC:
        return False, 0.0, 0.0, frac
    ys, xs = np.nonzero(m)
    return True, float(xs.mean() / m.shape[1] * 2 - 1), float(ys.mean() / m.shape[0] * 2 - 1), frac


def label(sticks, visible):
    """sticks: [pitch, roll, throttle, yaw] percent of the pilot."""
    pitch, roll, throttle, _ = sticks
    if throttle > THROTTLE_PCT:
        return "climb"
    if throttle < -THROTTLE_PCT:
        return "descend"
    if not visible:
        return "search"
    if max(abs(pitch), abs(roll)) >= MOVE_PCT:
        return "approach"
    return "align"


def label_episode(ep_dir, steps=None):
    """[{row, phase, pad_visible, pad_x, pad_y, red_frac}] for the flown part of one episode."""
    steps = steps if steps is not None else load_episode(ep_dir)
    out = []
    for i, row, jpeg in steps:
        visible, x, y, frac = pad_in_frame(jpeg)
        sticks = [float(row[c]) for c in ("pitch_cmd", "roll_cmd", "throttle_cmd", "yaw_cmd")]
        out.append({"row": int(i), "phase": label(sticks, visible), "pad_visible": visible,
                    "pad_x": round(x, 3), "pad_y": round(y, 3), "red_frac": round(frac, 4)})
    return out


def summary(labels, step_s=0.1):
    """{phase: seconds} in PHASES order."""
    return {p: round(sum(l["phase"] == p for l in labels) * step_s, 1) for p in PHASES}


def main():
    ap = argparse.ArgumentParser(description="automatic phase labels per episode step")
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--csv", action="store_true", help="write phases.csv into each episode folder")
    args = ap.parse_args()
    import pandas as pd
    total = {p: 0.0 for p in PHASES}
    for ep in args.episodes:
        labels = label_episode(ep)
        s = summary(labels)
        for p in PHASES:
            total[p] += s[p]
        print(f"{os.path.basename(os.path.normpath(ep))}: " + "  ".join(f"{p} {s[p]} s" for p in PHASES))
        if args.csv:
            pd.DataFrame(labels).to_csv(os.path.join(ep, "phases.csv"), index=False)
    if len(args.episodes) > 1:
        print("all: " + "  ".join(f"{p} {round(total[p], 1)} s" for p in PHASES))


if __name__ == "__main__":
    main()
