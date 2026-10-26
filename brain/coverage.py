"""
coverage.py  -  which layouts does the dataset cover, and which are missing? (idea D1)

    python3 -m brain.coverage ~/drone_database/episodes/episode_*
    python3 -m brain.coverage ~/drone_database/episodes/episode_* --svg coverage.svg

Reads <nr>_setup.json and <nr>_result.txt of every episode folder (written by the app's
EpisodePlan) and prints:

  1. Where the pad was RELATIVE to the drone's start (grid steps east / north). This is what
     the model must generalise over: a 5 x 5 map, each cell = number of successful episodes,
     T/V = how many of them are in train / val (val = last 10 % by name, like the builder).
  2. The absolute layout (pad cell x drone cell) counts, and the layouts with no episode yet.
  3. Flown seconds (same trimming as csv_to_hdf5.py) and failed episodes.

The SVG is the same relative map, ready for the thesis. Empty cells = no data: the model is
guessing there, so record those layouts next or keep the test flights out of them.
"""
import argparse, glob, json, os

import pandas as pd

from brain.rehearsal import trim

NAMES = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
VAL_SPLIT_FRACTION = 0.1     # same as drone_dataset_dataset_builder.py


def read_episode(ep):
    def one(pattern):
        hits = sorted(glob.glob(os.path.join(ep, pattern)))
        return hits[0] if hits else None
    setup_p, result_p, csv_p = one("*_setup.json"), one("*_result.txt"), one("*telemetry_*.csv")
    setup = json.load(open(setup_p)) if setup_p else {}
    result = open(result_p).read().strip() if result_p else "unknown"
    flown_s = round(len(trim(pd.read_csv(csv_p))) * 0.1, 1) if csv_p else 0.0
    return {"name": os.path.basename(os.path.normpath(ep)), "pad": setup.get("pad"), "drone": setup.get("drone"),
            "east": setup.get("pad_east_grid"), "north": setup.get("pad_north_grid"),
            "result": result, "flown_s": flown_s}


def collect(episodes):
    eps = sorted((read_episode(e) for e in episodes if os.path.isdir(e)), key=lambda r: r["name"])
    ok = [e for e in eps if e["result"] == "success" and e["east"] is not None]
    n_val = max(1, int(len(ok) * VAL_SPLIT_FRACTION)) if len(ok) > 1 else 0
    for i, e in enumerate(ok):
        e["split"] = "val" if i >= len(ok) - n_val else "train"
    return eps, ok


def relative_map(ok):
    """{(east, north): [train count, val count]} for -2..2 x -2..2."""
    m = {(e, n): [0, 0] for e in range(-2, 3) for n in range(-2, 3)}
    for ep in ok:
        m[(ep["east"], ep["north"])][ep["split"] == "val"] += 1
    return m


def print_report(eps, ok):
    m = relative_map(ok)
    print(f"{len(eps)} episodes, {len(ok)} successful with a setup, "
          f"{sum(e['result'] != 'success' for e in eps)} failed / unknown")
    print("\n1. Pad position relative to the drone start (north up, D = drone). Cell = train/val.")
    for n in range(2, -3, -1):
        cells = []
        for e in range(-2, 3):
            t, v = m[(e, n)]
            cells.append("  D  " if (e, n) == (0, 0) else ("  .  " if t + v == 0 else f"{t:2d}/{v:<2d}"))
        print("   " + " ".join(cells))
    empty = [k for k, (t, v) in m.items() if k != (0, 0) and t + v == 0]
    no_val = [k for k, (t, v) in m.items() if k != (0, 0) and t > 0 and v == 0]
    print(f"   offsets with no data: {len(empty)} of 24" + (f"   (east, north): {empty}" if len(empty) <= 8 else ""))
    print(f"   offsets never validated: {len(no_val)}")

    print("\n2. Absolute layouts (rows = pad cell, columns = drone cell), successful episodes:")
    print("        " + " ".join(f"{d:>3s}" for d in NAMES))
    missing = []
    for p in NAMES:
        row = []
        for d in NAMES:
            if p == d:
                row.append("  -")
                continue
            c = sum(1 for e in ok if e["pad"] == p and e["drone"] == d)
            row.append(f"{c:3d}" if c else "  .")
            if not c:
                missing.append(f"{p}/{d}")
        print(f"   {p:>3s}  " + " ".join(row))
    print(f"   layouts not flown yet: {len(missing)} of 72")

    secs = [e["flown_s"] for e in ok]
    if secs:
        print(f"\n3. Flown time per episode: min {min(secs)} s, mean {sum(secs) / len(secs):.1f} s, "
              f"max {max(secs)} s, total {sum(secs) / 60:.1f} min")
    bad = [e["name"] for e in eps if e["result"] != "success"]
    if bad:
        print("   not used (failed / no result): " + ", ".join(bad))


def write_svg(path, ok):
    m = relative_map(ok)
    top = max([t + v for t, v in m.values()] + [1])
    cell, pad = 64, 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{5 * cell + 2 * pad}" height="{5 * cell + 2 * pad + 20}" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="13">',
           f'<text x="{pad}" y="24" font-size="14">Pad position relative to drone start (north up)</text>']
    for n in range(2, -3, -1):
        for e in range(-2, 3):
            x, y = pad + (e + 2) * cell, pad + 10 + (2 - n) * cell
            t, v = m[(e, n)]
            if (e, n) == (0, 0):
                fill, label = "#ffffff", "drone"
            else:
                share = (t + v) / top
                fill = "#f1f3f5" if t + v == 0 else f"rgba(37, 99, 235, {0.15 + 0.85 * share:.2f})"
                label = "no data" if t + v == 0 else f"{t} / {v}"
            out.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{fill}" stroke="#c9ced6"/>')
            color = "#ffffff" if t + v and (t + v) / top > 0.6 else "#222222"
            out.append(f'<text x="{x + cell / 2 - 1}" y="{y + cell / 2 + 4}" text-anchor="middle" fill="{color}">{label}</text>')
    out.append(f'<text x="{pad}" y="{5 * cell + pad + 28}" fill="#555">cell = train / val episodes</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description="dataset coverage map from the episode setup files")
    ap.add_argument("episodes", nargs="+", help="episode folders")
    ap.add_argument("--svg", help="also write the relative map as an SVG picture")
    args = ap.parse_args()
    eps, ok = collect(args.episodes)
    print_report(eps, ok)
    if args.svg:
        write_svg(args.svg, ok)
        print(f"\nwrote {args.svg}")


if __name__ == "__main__":
    main()
