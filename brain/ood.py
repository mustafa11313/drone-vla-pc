"""
ood.py  -  "this room looks unfamiliar" (idea X12): a simple out-of-distribution detector.

    # once, on the laptop, from the TRAINING episode folders (not the validation ones)
    python3 -m brain.ood build ~/drone_database/incoming/episode_00* --out ood_ref.npz

    # then the bridge checks every frame against it
    python3 -m brain.bridge_server --policy http://<gpu>:8000 --ood ood_ref.npz

    # or just score some folders (e.g. a new room) without flying
    python3 -m brain.ood score ood_ref.npz <episode folders>

Each frame becomes a small "look" vector: colour histograms (RGB and HSV) and a histogram of
edge strengths (floor texture). No neural network, ~2 ms, and it does not care WHERE things
are in the picture, so a pad in a new position is not "unfamiliar"; a new floor, colour or
light level is. The reference keeps these vectors for every 3rd training frame. A live
frame's score is its distance to the 5 nearest reference frames. The threshold is 1.5 x the
score that 99 % of training frames from OTHER episodes stay below (leave one episode out), so
a normal flight in the training room is almost never flagged.

When 5 frames in a row score above the threshold, the bridge marks the scene unfamiliar: it
logs it, lowers the model's authority (--ood-authority, default 0.5 = half the stick range)
and tells the phone, which says "Room looks unfamiliar". It ends after 5 normal frames.
A new floor, a moved lamp or evening light can trigger it: that is the point.
"""
import argparse, io, os

import numpy as np
from PIL import Image

from brain.rehearsal import load_episode

SIZE = (96, 54)        # frames are compared at this size
BINS = 16              # histogram bins per channel
MARGIN = 1.5           # threshold = MARGIN x the 99th percentile of normal scores
K = 5                  # nearest neighbours
STRIDE = 3             # use every 3rd training frame
RUN = 5                # frames in a row before the state flips


def features(jpeg):
    img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize(SIZE, Image.BILINEAR)
    parts = []
    for a in (np.asarray(img), np.asarray(img.convert("HSV"))):
        for c in range(3):
            parts.append(np.histogram(a[..., c], bins=BINS, range=(0, 255))[0])
    g = np.asarray(img.convert("L"), dtype=np.float32)
    edges = np.concatenate([np.abs(np.diff(g, axis=1)).ravel(), np.abs(np.diff(g, axis=0)).ravel()])
    parts.append(np.histogram(edges, bins=BINS, range=(0, 64))[0])
    return np.concatenate([p / max(p.sum(), 1) for p in parts]).astype(np.float32) * 4


def knn_score(ref, f, k=K):
    d = np.sqrt(((ref - f) ** 2).sum(1))
    k = min(k, len(d))
    return float(np.sort(d)[:k].mean())


def build(episodes, out, stride=STRIDE):
    feats, groups = [], []
    for g, ep in enumerate(episodes):
        for _, _, jpeg in load_episode(ep)[::stride]:
            feats.append(features(jpeg))
            groups.append(g)
    feats, groups = np.stack(feats), np.array(groups)
    if len(set(groups)) > 1:
        scores = [knn_score(feats[groups != groups[i]], feats[i]) for i in range(len(feats))]
    else:                                   # one episode only: leave one frame out
        scores = [knn_score(np.delete(feats, i, 0), feats[i]) for i in range(len(feats))]
    threshold = MARGIN * float(np.percentile(scores, 99))
    np.savez_compressed(out, feats=feats, threshold=threshold)
    print(f"reference: {len(feats)} frames from {len(episodes)} episodes, threshold {threshold:.3f} -> {out}")
    return threshold


class Familiarity:
    """Live checker for the bridge. update(jpeg) -> (score, unfamiliar, changed)."""

    def __init__(self, ref_path):
        z = np.load(ref_path)
        self.ref, self.threshold = z["feats"], float(z["threshold"])
        self.unfamiliar = False
        self.run = 0

    def update(self, jpeg):
        score = knn_score(self.ref, features(jpeg))
        above = score > self.threshold
        self.run = self.run + 1 if above != self.unfamiliar else 0
        changed = False
        if self.run >= RUN:
            self.unfamiliar, self.run, changed = above, 0, True
        return score, self.unfamiliar, changed


def main():
    ap = argparse.ArgumentParser(description="unfamiliar-scene detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("episodes", nargs="+")
    b.add_argument("--out", default="ood_ref.npz")
    s = sub.add_parser("score")
    s.add_argument("ref")
    s.add_argument("episodes", nargs="+")
    args = ap.parse_args()
    if args.cmd == "build":
        build([e for e in args.episodes if os.path.isdir(e)], args.out)
        return
    f = Familiarity(args.ref)
    for ep in args.episodes:
        sc = [knn_score(f.ref, features(j)) for _, _, j in load_episode(ep)]
        flagged = sum(x > f.threshold for x in sc)
        print(f"{os.path.basename(os.path.normpath(ep))}: median score {np.median(sc):.3f} "
              f"(threshold {f.threshold:.3f}), {flagged}/{len(sc)} frames above")


if __name__ == "__main__":
    main()
