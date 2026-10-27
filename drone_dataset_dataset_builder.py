"""
drone_dataset_dataset_builder.py

RLDS dataset builder for the DJI Mini 3 flight episodes, for the original
openvla/openvla fine-tuning pipeline.

Reads every episode .h5 file (made by csv_to_hdf5.py) in H5_DIR and builds ONE
dataset with a train/val split. Each .h5 holds:
    telemetry   N x 14 float32  (column names are in the 'columns' attribute)
    frame_file  N strings, e.g. "frames/000041.jpg"
    instruction one string
    attrs: episode_dir (folder with the frames); telemetry.attrs["columns"]

Per step:
    image   = camera frame i, resized to 224 x 224
    state   = [height_above_takeoff, x_speed, y_speed, z_speed, heading]
              (OpenVLA ignores it; kept for checks)
    action  = [pitch_cmd, roll_cmd, throttle_cmd, yaw_cmd] / 100  -> -1 .. 1
              the stick values the pilot held while frame i was captured

Usage:
  1. Put this file in a folder named exactly drone_dataset/ (TFDS requirement),
     together with augment.py and paraphrases.txt (it imports them from its own folder):
         drone_dataset/drone_dataset_dataset_builder.py
         drone_dataset/augment.py
         drone_dataset/paraphrases.txt
  2. Set H5_DIR below, or the environment variable DRONE_H5_DIR.
  3. cd drone_dataset && tfds build
     Output: ~/tensorflow_datasets/drone_dataset/1.0.0/

Test without TensorFlow (checks the .h5 files and prints the action ranges):
     python3 drone_dataset_dataset_builder.py <folder with .h5 files>

Optional extra training data (see augment.py), both OFF unless set before `tfds build`:
     DRONE_RECOVERY_VIEWS=2   each train episode also appears 2x with the view shifted
                              left / right and the roll corrected back (idea X10)
     DRONE_PARAPHRASES=1      train episodes get varied instructions from paraphrases.txt (V2)
The validation split is always the same real, unchanged episodes, so scores stay comparable.
"""

import glob
import os
import sys

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment  # noqa: E402  (next to this file; copy it into drone_dataset/ too)

# --- Config ----------------------------------------------------------------
H5_DIR = os.environ.get("DRONE_H5_DIR", os.path.expanduser("~/drone_database/episodes"))
IMAGE_SHAPE = (224, 224, 3)   # OpenVLA input size
VAL_SPLIT_FRACTION = 0.1      # last 10 % of episodes (sorted by name) are validation
RECOVERY_VIEWS = int(os.environ.get("DRONE_RECOVERY_VIEWS", "0"))
PARAPHRASES = os.environ.get("DRONE_PARAPHRASES", "0") == "1"

ACTION_COLS = ["pitch_cmd", "roll_cmd", "throttle_cmd", "yaw_cmd"]
STATE_COLS = ["height_above_takeoff", "x_speed", "y_speed", "z_speed", "heading"]


def load_image(path, shape=IMAGE_SHAPE):
    with Image.open(path) as img:
        img = img.convert("RGB").resize((shape[1], shape[0]), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)


def _text(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def find_h5_files(h5_dir):
    return sorted(glob.glob(os.path.join(h5_dir, "**", "*.h5"), recursive=True))


def split_jobs(h5_files, recovery_views=0, paraphrases=False):
    """Train / val lists of (h5_path, shift, instruction or None, key).
    Val = the last 10 % of the REAL episodes, never augmented, original sentence.
    Train = the rest, plus their recovery views, with paraphrases when switched on."""
    n_val = max(1, int(len(h5_files) * VAL_SPLIT_FRACTION)) if len(h5_files) > 1 else 0
    train_files = h5_files[:len(h5_files) - n_val]
    val_files = h5_files[len(h5_files) - n_val:]
    phrases = augment.load_paraphrases(os.path.join(augment.HERE, "paraphrases.txt")) if paraphrases else None
    train = []
    for p in train_files:
        for shift in [0.0] + augment.shifts(recovery_views):
            key = p if shift == 0 else f"{p}#shift{shift:+.3f}"
            train.append((p, shift, augment.paraphrase_for(key, phrases) if phrases else None, key))
    val = [(p, 0.0, None, p) for p in val_files]
    return train, val


def load_episode(h5_path, with_images=True, shift=0.0, instruction=None):
    """Return (steps, instruction, episode_dir). steps is a list of dicts.
    shift != 0: recovery view (augment.py). instruction: replaces the recorded sentence."""
    with h5py.File(h5_path, "r") as f:
        telemetry = np.array(f["telemetry"], dtype=np.float32)
        frame_files = [_text(x) for x in f["frame_file"][:]]
        recorded = _text(f["instruction"][()])
        columns = [_text(c) for c in f["telemetry"].attrs["columns"]]
        episode_dir = _text(f.attrs["episode_dir"])

    # The folder path stored in the .h5 may belong to another computer:
    # fall back to the folder the .h5 file itself sits in.
    if not os.path.isdir(episode_dir):
        episode_dir = os.path.dirname(os.path.abspath(h5_path))

    idx = {name: i for i, name in enumerate(columns)}
    actions = telemetry[:, [idx[c] for c in ACTION_COLS]] / 100.0
    if shift:
        actions = augment.corrected_actions(actions, shift)
    instruction = instruction or recorded
    states = telemetry[:, [idx[c] for c in STATE_COLS]]
    T = len(telemetry)

    steps = []
    for i in range(T):
        step = {
            "observation": {
                "image": (augment.shifted_image(os.path.join(episode_dir, frame_files[i]), shift, IMAGE_SHAPE)
                          if shift else load_image(os.path.join(episode_dir, frame_files[i])))
                if with_images else None,
                "state": states[i].astype(np.float32),
            },
            "action": actions[i].astype(np.float32),
            "discount": 1.0,
            "reward": float(i == T - 1),
            "is_first": i == 0,
            "is_last": i == T - 1,
            "is_terminal": i == T - 1,
            "language_instruction": instruction,
        }
        steps.append(step)
    return steps, instruction, episode_dir


# --- TFDS builder (only defined when tensorflow_datasets is installed) ------
try:
    import tensorflow_datasets as tfds
except ImportError:
    tfds = None

if tfds is not None:

    class DroneDataset(tfds.core.GeneratorBasedBuilder):
        """DatasetBuilder for the drone flight dataset."""

        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Stick-command actions from app-recorded episodes."}

        def _info(self) -> tfds.core.DatasetInfo:
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict({
                    "steps": tfds.features.Dataset({
                        "observation": tfds.features.FeaturesDict({
                            "image": tfds.features.Image(
                                shape=IMAGE_SHAPE, dtype=np.uint8, encoding_format="jpeg",
                                doc="Main onboard camera RGB observation."),
                            "state": tfds.features.Tensor(
                                shape=(len(STATE_COLS),), dtype=np.float32,
                                doc="[height_above_takeoff ft, x_speed mph (north), "
                                    "y_speed mph (east), z_speed mph (down), heading deg]."),
                        }),
                        "action": tfds.features.Tensor(
                            shape=(4,), dtype=np.float32,
                            doc="[pitch, roll, throttle, yaw] stick values, -1 to 1 "
                                "(raw stick percent / 100)."),
                        "discount": tfds.features.Scalar(dtype=np.float32),
                        "reward": tfds.features.Scalar(dtype=np.float32),
                        "is_first": tfds.features.Scalar(dtype=np.bool_),
                        "is_last": tfds.features.Scalar(dtype=np.bool_),
                        "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                        "language_instruction": tfds.features.Text(),
                    }),
                    "episode_metadata": tfds.features.FeaturesDict({
                        "file_path": tfds.features.Text(doc="Path of the episode .h5 file."),
                    }),
                }))

        def _split_generators(self, dl_manager):
            h5_files = find_h5_files(H5_DIR)
            if not h5_files:
                raise FileNotFoundError(f"No .h5 files found under {H5_DIR}")
            train, val = split_jobs(h5_files, RECOVERY_VIEWS, PARAPHRASES)
            print(f"Found {len(h5_files)} episodes: {len(train)} train (incl. {RECOVERY_VIEWS} recovery "
                  f"views each, paraphrases {'on' if PARAPHRASES else 'off'}) / {len(val)} val")
            splits = {"train": self._generate_examples(train)}
            if val:
                splits["val"] = self._generate_examples(val)
            return splits

        def _generate_examples(self, jobs):
            for h5_path, shift, instruction, key in jobs:
                steps, _, _ = load_episode(h5_path, shift=shift, instruction=instruction)
                yield key, {"steps": steps, "episode_metadata": {"file_path": key}}


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else H5_DIR
    files = find_h5_files(folder)
    print(f"{len(files)} .h5 file(s) in {folder}")
    allA = []
    for p in files:
        steps, instr, ep_dir = load_episode(p, with_images=False)
        A = np.stack([s["action"] for s in steps])
        allA.append(A)
        print(f"  {os.path.basename(p)}: {len(steps)} steps, frames from {ep_dir}")
        print(f"    instruction: {instr}")
        print(f"    action min {A.min(0).round(2)}  max {A.max(0).round(2)}")
