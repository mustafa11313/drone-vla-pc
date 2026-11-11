#!/usr/bin/env python3
"""
check_scenario_episodes.py  -  checks the yaw and roll scenario episodes and compares their sticks
with the real flights.

For every episode in the folder (setups:=scenarios), it is BAD if:
    - it is not marked success, or the phone check is not OK
    - yaw and roll are pushed in the same row (never done in the real flights)
    - the scenario's main stick was not used:
        yaw_small, yaw_edge, search, yaw_then_roll  ->  some yaw
        roll_side, diagonal, yaw_then_roll          ->  some roll
        straight                                    ->  no yaw at all
Then it prints, per scenario, how many episodes are good, and a stick table (share of rows with
each stick, left / right balance, typical size) next to the real flights, if given.

Usage:
    python3 check_scenario_episodes.py ~/mini3_sim_episodes_scenarios
    python3 check_scenario_episodes.py ~/mini3_sim_episodes_scenarios --real ~/drone_database/episodes
    python3 check_scenario_episodes.py ~/mini3_sim_episodes_scenarios --move-bad   # BAD ones to bad/

--real takes a folder of real .h5 episodes (csv_to_hdf5.py output) or of real episode folders
with their telemetry CSV.
"""
import collections
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np

STICKS = ("pitch_cmd", "roll_cmd", "throttle_cmd", "yaw_cmd")
NEEDS_YAW = {"yaw_small", "yaw_edge", "search", "yaw_then_roll"}
NEEDS_ROLL = {"roll_side", "diagonal", "yaw_then_roll"}


def read_csv_sticks(path):
    rows = list(csv.DictReader(open(path)))
    return np.array([[int(r[k]) for k in STICKS] for r in rows], float).reshape(-1, 4)


def read_real(folder):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*.h5"))):
        import h5py
        with h5py.File(f) as h:
            out.append(np.asarray(h["telemetry"][()])[:, 7:11])     # pitch, roll, throttle, yaw
    for f in sorted(glob.glob(os.path.join(folder, "episode_*", "*_telemetry_*.csv"))):
        out.append(read_csv_sticks(f))
    return out


def stats(eps):
    a = np.concatenate(eps) if eps else np.zeros((0, 4))
    n = max(len(a), 1)
    res = {}
    for i, k in enumerate(("pitch", "roll", "throttle", "yaw")):
        v = a[:, i]
        nz = np.abs(v[v != 0])
        res[k] = (100.0 * len(nz) / n, int((v > 0).sum()), int((v < 0).sum()),
                  float(np.median(nz)) if len(nz) else 0.0)
    res["episodes"], res["rows"] = len(eps), len(a)
    return res


def table(title, s):
    print(f"\n{title}: {s['episodes']} episodes, {s['rows']} rows")
    print("  stick      rows used   right/fwd   left/back   median size")
    for k in ("pitch", "roll", "throttle", "yaw"):
        share, pos, neg, med = s[k]
        print(f"  {k:9s}  {share:7.1f} %  {pos:9d}  {neg:10d}  {med:8.0f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = os.path.expanduser(sys.argv[1])
    real = None
    if "--real" in sys.argv:
        real = os.path.expanduser(sys.argv[sys.argv.index("--real") + 1])
    bad, good_eps = [], []
    per = collections.defaultdict(lambda: [0, 0])
    for ep in sorted(glob.glob(os.path.join(root, "episode_*"))):
        if not os.path.isdir(ep):
            continue
        name = os.path.basename(ep)
        try:
            setup = json.load(open(glob.glob(os.path.join(ep, "*_setup.json"))[0]))
            sticks = read_csv_sticks(glob.glob(os.path.join(ep, "*_telemetry_*.csv"))[0])
        except (IndexError, ValueError, KeyError) as e:
            bad.append((name, f"unreadable ({e})"))
            continue
        sc = setup.get("scenario", "?")
        res = (open(glob.glob(os.path.join(ep, "*_result.txt"))[0]).read().strip()
               if glob.glob(os.path.join(ep, "*_result.txt")) else "no result")
        pc = glob.glob(os.path.join(ep, "*_phone_check.txt"))
        pc = open(pc[0]).read() if pc else "no phone check"
        why = []
        if res != "success":
            why.append(res)
        if "| OK" not in pc:
            why.append("phone check: " + pc.split("|")[-1].strip())
        if ((sticks[:, 1] != 0) & (sticks[:, 3] != 0)).any():
            why.append("yaw and roll in the same row")
        if sc in NEEDS_YAW and not (sticks[:, 3] != 0).any():
            why.append("no yaw")
        if sc in NEEDS_ROLL and not (sticks[:, 1] != 0).any():
            why.append("no roll")
        if sc == "straight" and (sticks[:, 3] != 0).any():
            why.append("yaw in a straight episode")
        per[sc][0 if why else 1] += 1
        if why:
            bad.append((name, f"{sc}: " + ", ".join(why)))
        else:
            good_eps.append(sticks)
    print(f"{len(bad)} of {len(bad) + len(good_eps)} episodes are BAD")
    for n, w in bad:
        print(f"  BAD  {n}  {w}")
    print("\nper scenario (good / bad):")
    for sc in sorted(per):
        print(f"  {sc:14s} {per[sc][1]:4d} / {per[sc][0]}")
    table("simulator, good episodes", stats(good_eps))
    if real:
        table("real flights", stats(read_real(real)))
    if "--move-bad" in sys.argv and bad:
        os.makedirs(os.path.join(root, "bad"), exist_ok=True)
        for n, _ in bad:
            shutil.move(os.path.join(root, n), os.path.join(root, "bad", n))
        print(f"\nmoved {len(bad)} episodes to {os.path.join(root, 'bad')}; "
              f"renumber_episodes.py closes the gaps")


if __name__ == "__main__":
    main()
