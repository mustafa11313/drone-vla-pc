"""
patch_openvla.py <path to the openvla repository>

Adds the drone dataset to OpenVLA. Three small edits, all inside prismatic/vla/datasets/rlds/oxe/:
  configs.py     new action type DRONE_STICKS (4 stick values) and the entry "drone_dataset"
  transforms.py  transform for "drone_dataset" (the data is already in the right format)
  materialize.py accepts the new action type (all 4 values normalised, none "absolute")
The script can be run twice safely (it skips edits that are already there).
"""
import os, sys

root = os.path.expanduser(sys.argv[1])
d = os.path.join(root, "prismatic/vla/datasets/rlds/oxe")

def edit(fname, old, new, marker):
    p = os.path.join(d, fname)
    s = open(p).read()
    if marker in s:
        print(fname, ": already patched"); return
    if s.count(old) != 1:
        sys.exit(f"{fname}: expected text not found exactly once ({s.count(old)}); not patched")
    open(p, "w").write(s.replace(old, new))
    print(fname, ": patched")

# 1. configs.py ---------------------------------------------------------------
edit("configs.py",
     "    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)\n",
     "    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)\n"
     "    DRONE_STICKS = 5        # Pitch (1) + Roll (1) + Throttle (1) + Yaw (1) stick values, -1 .. 1\n",
     "DRONE_STICKS")
edit("configs.py",
     '    "libero_10_no_noops": {',
     '    "drone_dataset": {\n'
     '        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},\n'
     '        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},\n'
     '        "state_obs_keys": ["state"],\n'
     '        "state_encoding": StateEncoding.NONE,\n'
     '        "action_encoding": ActionEncoding.DRONE_STICKS,\n'
     '    },\n'
     '    "libero_10_no_noops": {',
     '"drone_dataset"')

# 2. transforms.py ------------------------------------------------------------
edit("transforms.py",
     "# === Registry ===\n",
     "def drone_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:\n"
     "    # Our RLDS data already holds action = [pitch, roll, throttle, yaw] in -1..1, image and instruction.\n"
     "    return trajectory\n\n\n"
     "# === Registry ===\n",
     "drone_dataset_transform")
edit("transforms.py",
     '    "libero_10_no_noops": libero_dataset_transform,\n',
     '    "libero_10_no_noops": libero_dataset_transform,\n'
     '    ### Drone landing dataset\n'
     '    "drone_dataset": drone_dataset_transform,\n',
     '"drone_dataset": drone_dataset_transform')

# 3. materialize.py -----------------------------------------------------------
edit("materialize.py",
     'if dataset_kwargs["action_encoding"] not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6]:',
     'if dataset_kwargs["action_encoding"] not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6, ActionEncoding.DRONE_STICKS]:',
     "ActionEncoding.DRONE_STICKS]:")
edit("materialize.py",
     '    elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:\n'
     '        dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]\n'
     '        dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]\n',
     '    elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:\n'
     '        dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]\n'
     '        dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]\n'
     '    elif dataset_kwargs["action_encoding"] is ActionEncoding.DRONE_STICKS:\n'
     '        dataset_kwargs["absolute_action_mask"] = [False] * 4\n'
     '        dataset_kwargs["action_normalization_mask"] = [True] * 4\n',
     "is ActionEncoding.DRONE_STICKS:")
print("done")
