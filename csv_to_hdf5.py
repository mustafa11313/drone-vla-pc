"""
csv_to_hdf5.py

Converts an episode's telemetry.csv (recorded by the app; row i belongs to
frames/00000i.jpg, counting from 0) into a single HDF5 file.

The episode is trimmed first: it starts at the first stick input and ends
when AUTO_LANDING begins (hover before and the landing itself are cut).
The frame file name is built from the row number, so it stays correct
after trimming. Stores the numeric telemetry columns as one dataset and
the frame filenames as a separate string dataset, both indexed the same way
(row i in each corresponds to the same frame).

Usage:
    python3 csv_to_hdf5.py telemetry.csv output.h5 [episode_folder_with_frames]
"""

import sys
import os
import re
import pandas as pd
import h5py
import numpy as np

# Single, fixed instruction written into every .h5 file -- no more pulling
# a varied phrasing per-episode from log.json.
INSTRUCTION = "Take off, locate the red landing pad, and land on it."


def find_instruction(csv_path):
    """Every episode gets the exact same instruction now."""
    return INSTRUCTION


TELEMETRY_COLS = [
    "height_above_takeoff",
    "x_speed",
    "y_speed",
    "z_speed",
    "heading",
    "drone_pitch",
    "drone_roll",
    "pitch_cmd",
    "roll_cmd",
    "throttle_cmd",
    "yaw_cmd",
    "gimbal_heading",
    "gimbal_pitch",
    "gimbal_roll",
]


CMD_COLS = ["pitch_cmd", "roll_cmd", "throttle_cmd", "yaw_cmd"]


def trim(df):
    """Keep first stick input .. start of AUTO_LANDING (exclusive)."""
    moving = (df[CMD_COLS] != 0).any(axis=1)
    start = int(moving.idxmax()) if moving.any() else 0
    landing = df.index[df["flight_mode"] == "AUTO_LANDING"]
    end = int(landing[0]) if len(landing) else len(df)
    return df.iloc[start:end]


def convert(csv_path, h5_path, episode_dir=None):
    df = pd.read_csv(csv_path)
    if "frame_file" not in df.columns:
        # frame i is stored as frames/<episode number>_00000i.jpg (same row number);
        # the number comes from the csv name: 0037_telemetry_<date>_<time>.csv
        m = re.match(r"^(\d+)_telemetry", os.path.basename(csv_path))
        pre = f"{m.group(1)}_" if m else ""
        df["frame_file"] = [f"frames/{pre}{i:06d}.jpg" for i in range(len(df))]
    n_all = len(df)
    df = trim(df)
    print(f"Trimmed {n_all} -> {len(df)} rows "
          f"(rows {df.index[0]}..{df.index[-1]}, {len(df) * 0.1:.1f} s)")

    missing = [c for c in TELEMETRY_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: CSV is missing expected columns: {missing}")
        sys.exit(1)

    telemetry_data = df[TELEMETRY_COLS].to_numpy(dtype=np.float32)
    frame_files = df["frame_file"].astype(str).to_numpy()

    instruction = find_instruction(csv_path)
    # Folder that contains frames/ (default: the folder of the csv)
    episode_dir = os.path.abspath(episode_dir) if episode_dir else os.path.dirname(os.path.abspath(csv_path))

    with h5py.File(h5_path, "w", libver="earliest") as f:
        f.create_dataset("telemetry", data=telemetry_data)
        f.create_dataset("frame_file", data=frame_files,
                          dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset("instruction", data=instruction,
                          dtype=h5py.string_dtype(encoding="utf-8"))

        # Store column names and the source folder as attributes, so this
        # file is self-describing -- includes where to find the real
        # images later, needed for RLDS conversion.
        f["telemetry"].attrs["columns"] = TELEMETRY_COLS
        f.attrs["episode_dir"] = episode_dir

    print(f"Wrote {h5_path}")
    print(f"  telemetry dataset shape: {telemetry_data.shape}  (columns: {TELEMETRY_COLS})")
    print(f"  frame_file dataset shape: {frame_files.shape}")
    print(f"  instruction: \"{instruction}\"")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 csv_to_hdf5.py <input.csv> <output.h5> [episode_folder_with_frames]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
