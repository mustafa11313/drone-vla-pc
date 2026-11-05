"""
(Copy of drone-vla-pc/sim/dynamics.py and sim/mini3_params.json, so the ROS package stands alone.
Re-fit with sim/fit_dynamics.py and copy both files here again.)

dynamics.py  -  how the Mini 3 moves for given stick values (indoors, vision positioning).

DJI's flight controller turns the sticks into velocities; the drone then reaches that velocity
with a short delay. So the model is a first-order velocity response per axis, not rigid-body
physics:

    target forward speed = K_FWD  * curve(pitch %)     (m/s)
    target right speed   = K_RIGHT * curve(roll %)     (m/s)
    target climb speed   = K_UP   * throttle % / 100   (m/s, linear, like replay.py)
    target turn rate     = K_YAW  * curve(yaw %)       (deg/s)
    dv/dt = (target - v) / TAU                          (TAU_YAW for the turn rate)

curve(x) = 0.12 x + 0.88 x^3 is the stick curve fitted earlier to the recorded flights
(replay.py). The constants are fitted from recorded episodes with fit_dynamics.py; the
defaults below come from the two sample episodes and are only a starting point.

Frame: x east, y north, z up (metres), heading in degrees clockwise from north like the
telemetry. The telemetry speeds of the recordings are NED (x_speed north, y_speed east,
z_speed down) in mph; to_telemetry() returns what the phone app sends (m/s, protocol.py).
"""
import json
import math
import os
from dataclasses import asdict, dataclass

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "mini3_params.json")


def curve(pct):
    x = max(-1.0, min(1.0, float(pct) / 100.0))
    return 0.12 * x + 0.88 * x ** 3


@dataclass
class Params:
    k_fwd: float = 3.0       # m/s at full pitch stick (after the curve)
    k_right: float = 3.0     # m/s at full roll stick
    k_up: float = 0.3        # m/s climb at full throttle stick (linear)
    k_yaw: float = 60.0      # deg/s at full yaw stick
    tau: float = 0.4         # s, horizontal and vertical speed response
    tau_yaw: float = 0.2     # s, turn-rate response
    noise: float = 0.02      # m/s of random drift per axis (vision positioning is not perfect)

    @classmethod
    def load(cls, path=None):
        path = path or PARAMS_FILE
        if path and os.path.exists(path):
            d = json.load(open(path))
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, path=None):
        json.dump(asdict(self), open(path or PARAMS_FILE, "w"), indent=2)


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0
    heading: float = 0.0     # deg, 0 = north, clockwise
    vx: float = 0.0          # east, m/s
    vy: float = 0.0          # north
    vz: float = 0.0          # up
    yaw_rate: float = 0.0    # deg/s


class Mini3:
    def __init__(self, params=None, rng=None):
        self.p = params or Params.load()
        self.rng = rng
        self.s = State()

    def step(self, sticks, dt):
        """sticks: [pitch, roll, throttle, yaw] in stick percent. Advances the state by dt."""
        p, s = self.p, self.s
        pitch, roll, throttle, yaw = sticks
        h = math.radians(s.heading)
        fwd, right = p.k_fwd * curve(pitch), p.k_right * curve(roll)
        # body -> world (x east, y north)
        tvx = fwd * math.sin(h) + right * math.cos(h)
        tvy = fwd * math.cos(h) - right * math.sin(h)
        tvz = p.k_up * max(-100.0, min(100.0, throttle)) / 100.0
        a = min(1.0, dt / max(p.tau, 1e-3))
        s.vx += (tvx - s.vx) * a
        s.vy += (tvy - s.vy) * a
        s.vz += (tvz - s.vz) * a
        if self.rng is not None and p.noise > 0:
            s.vx += self.rng.normal(0, p.noise) * math.sqrt(dt)
            s.vy += self.rng.normal(0, p.noise) * math.sqrt(dt)
        s.yaw_rate += (p.k_yaw * curve(yaw) - s.yaw_rate) * min(1.0, dt / max(p.tau_yaw, 1e-3))
        s.x += s.vx * dt
        s.y += s.vy * dt
        s.z += s.vz * dt
        s.heading = (s.heading + s.yaw_rate * dt) % 360.0
        return s

    def to_telemetry(self, pilot=(0, 0, 0, 0)):
        """The telemetry block the phone sends (protocol.py): metres, m/s, NED speeds."""
        s = self.s
        return {"height_m": round(s.z, 3), "vx": round(s.vy, 3), "vy": round(s.vx, 3),
                "vz": round(-s.vz, 3), "heading": round(s.heading, 1),
                "pilot": list(pilot), "flying": s.z > 0.02, "vs_enabled": True}
