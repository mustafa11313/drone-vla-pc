#!/usr/bin/env python3
"""
renumber_episodes.py  -  closes the gaps in episode numbers after episodes were deleted.

episode_0001_..., episode_0003_..., episode_0007_...  ->  episode_0001_..., episode_0002_..., episode_0003_...

The number is renamed everywhere it appears: the folder, the NNNN_ files inside it, the frames
(frames/NNNN_000000.jpg ...) and the folder name written in NNNN_phone_check.txt. The date stamp
and everything in setup.json (plan_number, positions) stay as they are.

Usage:
    python3 renumber_episodes.py ~/mini3_sim_episodes_search            # only shows what it would do
    python3 renumber_episodes.py ~/mini3_sim_episodes_search --apply    # does it
"""
import glob
import os
import re
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = os.path.expanduser(sys.argv[1])
    apply = "--apply" in sys.argv
    eps = []
    for d in glob.glob(os.path.join(root, "episode_*")):
        m = re.fullmatch(r"episode_(\d{4})_(.+)", os.path.basename(d))
        if m and os.path.isdir(d):
            eps.append((int(m.group(1)), m.group(1), m.group(2), d))
    eps.sort()
    changes = [(new, e) for new, e in enumerate(eps, start=1) if new != e[0]]
    print(f"{len(eps)} episodes, {len(changes)} to renumber")
    for new, (_, old, stamp, d) in changes:
        nn = f"{new:04d}"
        new_d = os.path.join(root, f"episode_{nn}_{stamp}")
        print(f"  {os.path.basename(d)}  ->  {os.path.basename(new_d)}")
        if not apply:
            continue
        # new numbers only go down and are taken in order, so a target never exists yet
        for f in glob.glob(os.path.join(d, "frames", f"{old}_*")) + glob.glob(os.path.join(d, f"{old}_*")):
            base = os.path.basename(f)
            os.rename(f, os.path.join(os.path.dirname(f), nn + base[len(old):]))
        pc = os.path.join(d, f"{nn}_phone_check.txt")
        if os.path.exists(pc):
            text = open(pc).read().replace(os.path.basename(d), os.path.basename(new_d))
            open(pc, "w").write(text)
        os.rename(d, new_d)
    if changes and not apply:
        print("\nnothing changed yet; run again with --apply to rename")
    elif changes:
        print("\ndone")


if __name__ == "__main__":
    main()
