"""
make_pad.py  -  recolours the landing pad texture to the real pad's red (seen in the Nov 2026
lab video: saturated red with a white H, white ring and white corner marks).

The design (H, ring, ticks, corner marks) is kept from the old texture, which came from drone
frames; only the colours change: everything that was white stays white, the rest becomes red.

    python3 tools/make_pad.py
"""
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "models", "pad", "materials", "textures", "pad.png")
RED = np.array([210, 32, 64], np.float32)       # sRGB, crimson as in the real Mini 3 frames
WHITE = np.array([238, 236, 234], np.float32)


def main():
    a = np.asarray(Image.open(TEX).convert("RGB"), np.float32)
    # whiteness: the old pad was pink (high red, mid green/blue); white parts have high green/blue
    w = np.clip((a[..., 1:].mean(-1) - 110) / (200 - 110), 0, 1)[..., None]
    # keep a little of the old texture's grain
    grain = (a.mean(-1, keepdims=True) - a.mean()) * 0.15
    out = RED * (1 - w) + WHITE * w + grain
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(TEX)
    print("pad ->", TEX)


if __name__ == "__main__":
    main()
