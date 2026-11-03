#!/usr/bin/env python3
"""
check_search_episodes.py  -  finds search episodes where the clockwise turn did not stop at the pad.

An early version of the autopilot sometimes turned past a pad that showed near the bottom of the
picture and stopped only after a 15 s limit, with the pad out of view; it then flew to a pad it
could not see. Such episodes teach the wrong thing. This script replays every search episode's
turn from its telemetry (heading and height per row; the drone does not move sideways while it
turns) and the drone / pad positions in setup.json, and reports:

    BAD   the pad was in the picture during the turn but the turn went on past it,
          or the pad is not in the picture when the turn ends
    ok    otherwise (also episodes without a search turn)

Usage:
    python3 check_search_episodes.py ~/mini3_sim_episodes_search            # report
    python3 check_search_episodes.py ~/mini3_sim_episodes_search --move-bad  # move BAD ones to bad/
"""
import csv
import glob
import json
import math
import os
import shutil
import sys

CAM_FWD, CAM_DOWN = 0.1, 0.015
CAM_PITCH, HFOV, VFOV = 50.0, 62.0, 37.3
FLOOR = 0.03
REACTION_S = 0.6      # the autopilot lets go 0.15-0.45 s after the pad is clearly in view


def in_view(fwd, right, z, margin=0.0):
    f, d = fwd - CAM_FWD, z - CAM_DOWN
    p = math.radians(CAM_PITCH)
    depth = f * math.cos(p) + d * math.sin(p)
    if depth <= 0.05:
        return False
    x = math.degrees(math.atan(right / depth))
    y = math.degrees(math.atan((d * math.cos(p) - f * math.sin(p)) / depth))
    return abs(x) < HFOV / 2 - margin and abs(y) < VFOV / 2 - margin


def check(ep):
    setup = json.load(open(glob.glob(os.path.join(ep, "*_setup.json"))[0]))
    if not setup.get("search"):
        return None, "not a search episode"
    rows = list(csv.DictReader(open(glob.glob(os.path.join(ep, "*_telemetry_*.csv"))[0])))
    (dx, dy), (px, py) = setup["drone_xy"], setup["pad_xy"]

    def visible(r, margin=0.0):
        h = math.radians(float(r["heading"]))
        ex, ny = px - dx, py - dy
        fwd = ex * math.sin(h) + ny * math.cos(h)
        right = ex * math.cos(h) - ny * math.sin(h)
        z = float(r["height_above_takeoff"]) / 3.28084 + FLOOR
        return in_view(fwd, right, z, margin)

    yaw = [int(r["yaw_cmd"]) for r in rows]
    start = next((i for i, v in enumerate(yaw) if v > 0), None)
    if start is None or (start > 0 and visible(rows[start])):
        return True, "no search turn"
    # the turn ends where its stick is let go. A "face the pad" turn can follow straight away,
    # its stick ramping up again before the rows show a 0, so end at the bottom of the first dip
    peak = max(range(start, min(start + 20, len(yaw))), key=lambda i: yaw[i])
    end = peak
    while end + 1 < len(yaw) and yaw[end] > 0 and (yaw[end + 1] < yaw[end] or yaw[end + 1] == yaw[end] == yaw[peak]):
        end += 1
    t = [int(r["time_ms"]) / 1000 for r in rows]
    seen = next((i for i in range(start, end) if visible(rows[i], margin=3.0)), None)
    turn = t[end] - t[start]
    if not visible(rows[end]):
        return False, f"turn of {turn:.1f} s ended with the pad out of view"
    if seen is not None and t[end] - t[seen] > REACTION_S + 1.0:
        return False, f"pad in view {t[end] - t[seen]:.1f} s before the turn ended"
    return True, f"turn of {turn:.1f} s, stopped on the pad"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = os.path.expanduser(sys.argv[1])
    move = "--move-bad" in sys.argv
    eps = sorted(d for d in glob.glob(os.path.join(root, "episode_*")) if os.path.isdir(d))
    bad = []
    for ep in eps:
        try:
            ok, why = check(ep)
        except Exception as e:                     # broken folder: report, do not crash
            ok, why = False, f"could not check: {e}"
        if ok is None:
            continue
        print(f"{'ok ' if ok else 'BAD'}  {os.path.basename(ep)}  {why}")
        if not ok:
            bad.append(ep)
    print(f"\n{len(bad)} of {len(eps)} episodes are BAD")
    if move and bad:
        dest = os.path.join(root, "bad")
        os.makedirs(dest, exist_ok=True)
        for ep in bad:
            shutil.move(ep, dest)
        print(f"moved them to {dest}")


if __name__ == "__main__":
    main()
