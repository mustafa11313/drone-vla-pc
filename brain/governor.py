"""
Safety governor on the PC side: the model may only ask for what the pilot did in the data.

The phone has its own, independent limits (StickController.kt); this is the first layer.
  - every stick is clamped to the range seen in training (OpenVLA's q01 .. q99 bounds),
    optionally widened or narrowed by a factor,
  - a stick may change by at most MAX_STEP_PCT per command, so one wild prediction
    cannot jerk the drone,
  - the model's yaw can be switched off (it was not learned; yaw stays with the pilot).
"""
import json
import os

MAX_STEP_PCT = 25.0


def limits_from_stats(stats_path, key="drone_dataset"):
    """[(low, high)] per stick in percent, from a checkpoint's dataset_statistics.json."""
    stats = json.load(open(os.path.expanduser(stats_path)))[key]["action"]
    return [(100.0 * lo, 100.0 * hi) for lo, hi in zip(stats["q01"], stats["q99"])]


class Governor:
    def __init__(self, limits, scale=1.0, max_step=MAX_STEP_PCT, allow_yaw=False):
        self.limits = [(lo * scale, hi * scale) for lo, hi in limits]
        self.max_step = max_step
        self.allow_yaw = allow_yaw
        self.last = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        self.last = [0.0, 0.0, 0.0, 0.0]

    def apply(self, sticks):
        out = []
        for i, (v, (lo, hi), prev) in enumerate(zip(sticks, self.limits, self.last)):
            if i == 3 and not self.allow_yaw:
                v = 0.0
            v = min(max(v, lo), hi)
            v = min(max(v, prev - self.max_step), prev + self.max_step)
            out.append(round(v, 1))
        self.last = out
        return out

    def limits_abs(self):
        """Largest allowed magnitude per stick, sent to the phone as an extra cap."""
        return [round(max(abs(lo), abs(hi)), 1) for lo, hi in self.limits]
