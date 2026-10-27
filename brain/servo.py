"""
servo.py  -  a classic pad-centring controller, and the hybrid brain (idea X7).

The VLA searches and approaches; for the last metre a plain visual-servoing rule takes over:

    roll   = steer the pad to the image centre line (pad right -> roll right)
    pitch  = move until the pad sits at TARGET_Y in the picture (low in the frame, because the
             camera looks 50 deg down and the spot under the drone is out of view)
    throttle = descend only while the pad is close to that target spot

    python3 -m brain.bridge_server --policy http://<gpu>:8000 --hybrid 1.5   # VLA, servo below 1.5 m
    python3 -m brain.bridge_server --policy servo                            # classic only, no model
    python3 -m brain.rehearsal --policy servo <episodes>                     # classic vs. the pilot

That gives the thesis three arms on the same course: pure VLA, pure classic, hybrid.

Pad position: the red pixels, as in brain/phases.py (x, y in -1..1 from the image centre).
When the pad has just slid out of the bottom of the picture after being centred, the
controller keeps descending straight for LOST_KEEP_S (the phone's auto-landing takes over
near the ground). Gains and TARGET_Y are first guesses: check them in shadow mode
(steps.csv, column source = servo) before flying active.
"""
import time

from brain.phases import pad_in_frame

GAIN_X = 30.0          # roll stick % for the pad at the left / right image edge
GAIN_Y = 30.0          # pitch stick % per unit of pad offset from TARGET_Y
TARGET_Y = 0.55        # where the pad should sit (-1 top .. 1 bottom) when the drone is above it
ALIGN_TOL = 0.2        # |offset| below which the pad counts as "under us"
DESCEND_PCT = -30.0    # throttle stick % while aligned
MAX_PCT = 25.0         # servo never asks for more than this on pitch / roll
LOST_KEEP_S = 2.0      # keep descending this long after an aligned pad left the picture


def clamp(v, m):
    return max(-m, min(m, v))


class Servo:
    def __init__(self, gain_x=GAIN_X, gain_y=GAIN_Y, target_y=TARGET_Y):
        self.gain_x, self.gain_y, self.target_y = gain_x, gain_y, target_y
        self.aligned_at = 0.0
        self.last_pad = None          # (visible, x, y, red_fraction) of the last frame

    def reset(self):
        self.aligned_at = 0.0

    def command(self, jpeg, now=None):
        """Stick % [pitch, roll, throttle, yaw], or None when the servo has nothing to say
        (pad not in view and not just lost below the drone)."""
        now = time.monotonic() if now is None else now
        visible, x, y, frac = self.last_pad = pad_in_frame(jpeg)
        if not visible:
            if now - self.aligned_at < LOST_KEEP_S:
                return [0.0, 0.0, DESCEND_PCT, 0.0]
            return None
        dy = self.target_y - y                      # pad above the target spot -> fly forward
        roll = clamp(self.gain_x * x, MAX_PCT)
        pitch = clamp(self.gain_y * dy, MAX_PCT)
        aligned = abs(x) < ALIGN_TOL and abs(dy) < ALIGN_TOL
        if aligned:
            self.aligned_at = now
        return [round(pitch, 1), round(roll, 1), DESCEND_PCT if aligned else 0.0, 0.0]


class ServoPolicy:
    """Classic controller with the same act() as PolicyClient (for the bridge and rehearsal)."""
    last_confidence = None

    def __init__(self):
        self.servo = Servo()

    def act(self, jpeg, instruction=None):
        t0 = time.perf_counter()
        cmd = self.servo.command(jpeg)
        ms = (time.perf_counter() - t0) * 1000
        return (cmd or [0.0, 0.0, 0.0, 0.0]), ms, ms
