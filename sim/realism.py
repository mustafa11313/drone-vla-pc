"""
realism.py  -  how "real" do the simulated frames look? A number instead of a feeling.

    python3 -m sim.realism --ref <real training episodes> --real <other real episodes>

Uses the unfamiliar-room detector (brain/ood.py): a reference is built from real episodes,
then three sets of frames are scored against it:

    real      frames of other real episodes (what "familiar" looks like)
    flat      the simulator with plain colours (Camera(realistic=False))
    textured  the simulator with the real textures and camera effects

For each set: the median score (lower = closer to the real frames) and the share of frames
the bridge would call unfamiliar (above the threshold). The sim frames are taken from random
drone poses in the cage (height 0.5-1.5 m, any heading), with the pad placed randomly.
"""
import argparse, math, os, tempfile

import numpy as np

from brain.ood import build, features, knn_score
from brain.rehearsal import load_episode
from sim.render import CAGE_HALF, Camera, Scene, render, to_jpeg


def sim_frames(n, realistic, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sc = Scene(pad_x=rng.uniform(-0.5, 0.5), pad_y=rng.uniform(-0.5, 0.5), pad_yaw=rng.uniform(-20, 20), seed=i)
        lim = CAGE_HALF - 0.3
        x, y = rng.uniform(-lim, lim), rng.uniform(-lim, lim)
        # mostly facing the pad, like a flight that is searching / approaching
        to_pad = math.degrees(math.atan2(sc.pad_x - x, sc.pad_y - y))
        heading = to_pad + rng.normal(0, 40) if rng.random() < 0.7 else rng.uniform(0, 360)
        rgb = render(x, y, rng.uniform(0.5, 1.5), heading, sc, Camera(realistic=realistic))
        out.append(to_jpeg(rgb))
    return out


def stats(ref, thr, jpegs):
    sc = np.array([knn_score(ref, features(j)) for j in jpegs])
    return float(np.median(sc)), float((sc > thr).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description="score simulated frames against real ones")
    ap.add_argument("--ref", nargs="+", required=True, help="real episodes for the reference")
    ap.add_argument("--real", nargs="*", default=[], help="other real episodes, scored for comparison")
    ap.add_argument("--n", type=int, default=150, help="simulated frames per set")
    args = ap.parse_args(argv)
    path = os.path.join(tempfile.mkdtemp(), "ref.npz")
    build(args.ref, path)
    z = np.load(path)
    ref, thr = z["feats"], float(z["threshold"])
    rows = []
    if args.real:
        jp = [j for ep in args.real for _, _, j in load_episode(ep)]
        rows.append(("real (other episodes)", *stats(ref, thr, jp), len(jp)))
    for name, realistic in (("sim, plain colours", False), ("sim, real textures", True)):
        jp = sim_frames(args.n, realistic)
        rows.append((name, *stats(ref, thr, jp), len(jp)))
    print(f"\nthreshold {thr:.3f}")
    print(f"{'frames':26s} {'median score':>13s} {'unfamiliar':>11s} {'n':>5s}")
    for name, med, frac, n in rows:
        print(f"{name:26s} {med:13.3f} {100 * frac:10.0f} % {n:5d}")
    return rows


if __name__ == "__main__":
    main()
