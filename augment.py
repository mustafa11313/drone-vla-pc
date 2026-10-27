"""
augment.py  -  extra training data without extra flights (used by drone_dataset_dataset_builder.py).

1. Recovery views (idea X10).  The pilot data only shows the drone ON its path, so the model
   never sees what being slightly off looks like. Here every training frame is also cropped a
   little to the left or right, as if the camera looked a few degrees beside the path, and the
   roll stick is corrected to steer back. The same trick was used for NVIDIA's early
   self-driving work with side cameras.

   Shifting the view to the RIGHT means the drone sits too far right of its path: it must
   roll LEFT, so  roll_cmd -= ROLL_GAIN * shift.  ROLL_GAIN is a first guess (a 5 % shift
   gives -10 % roll); check it against shadow-mode flights before trusting it.

2. Paraphrased instructions (idea V2).  Each training episode gets one of many phrasings of
   the task instead of the single fixed sentence, picked deterministically from the episode
   name, so a rebuild gives the same dataset. Validation keeps the original sentence;
   paraphrases_test.txt holds phrasings never used in training, for eval_val.py --instruction.

Validation episodes are never augmented.
"""
import hashlib
import os

import numpy as np
from PIL import Image

CROP = 0.9              # the shifted window is 90 % of the frame, like OpenVLA's test-time crop
ROLL_GAIN = 200.0       # stick percent per unit of shift (fraction of the frame width)
ROLL_INDEX = 1          # action order: pitch, roll, throttle, yaw
HERE = os.path.dirname(os.path.abspath(__file__))


def shifts(n):
    """n shift values (fraction of frame width), symmetric around 0, never 0 itself."""
    if n <= 0:
        return []
    step = (1.0 - CROP) / 2 / ((n + 1) // 2)        # largest shift keeps the window inside the frame
    out = []
    for k in range(1, (n + 1) // 2 + 1):
        out += [k * step, -k * step]
    return out[:n]


def shifted_image(path, shift, shape=(224, 224)):
    """The frame seen through a 90 % window moved right by `shift` (fraction of the width)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        cw, ch = w * CROP, h * CROP
        left = (w - cw) / 2 + shift * w
        top = (h - ch) / 2
        left = min(max(left, 0), w - cw)
        img = img.crop((left, top, left + cw, top + ch)).resize((shape[1], shape[0]), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)


def corrected_actions(actions, shift):
    """actions: (T, 4) in -1..1. Returns a copy with the roll steered back toward the path."""
    a = np.array(actions, dtype=np.float32, copy=True)
    a[:, ROLL_INDEX] = np.clip(a[:, ROLL_INDEX] - ROLL_GAIN * shift / 100.0, -1.0, 1.0)
    return a


def load_paraphrases(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def paraphrase_for(key, phrases):
    """Same key -> same phrase, every time (no random state)."""
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    return phrases[h % len(phrases)]
