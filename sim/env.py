"""
env.py  -  one simulated flight in the cage: drone model + camera + the rules of a mission.

    env = CageEnv(seed=3)
    obs = env.reset()                       # random pad and start
    while not env.done:
        obs = env.step([pitch, roll, throttle, yaw])     # stick percent, 0.1 s per step
    print(env.result())

obs = {"jpeg": bytes, "telemetry": {...same block the phone sends...}, "t": seconds}

Rules (the same as on the real drone, where they exist):
  - DJI auto-landing: below LAND_HEIGHT while descending, the drone lands straight down by
    itself (the app does this; the training data ends there too). auto_land() starts it at
    any height (the "land" skill).
  - The net: closer than DRONE_RADIUS to a side or the top = "hit_net" and the flight ends.
  - The floor: reaching it ends the flight; on_pad = the drone's centre is on the pad.
  - Timeout after max_time seconds.
"""
import math

import numpy as np

from sim.dynamics import Mini3, Params
from sim.render import CAGE_HALF, CAGE_HEIGHT, PAD_SIZE, Camera, Scene, render, to_jpeg

DT = 0.1                 # s, one step = one camera frame (10 Hz like the recordings)
LAND_HEIGHT = 0.6        # m, the app's --land-height
AUTO_LAND_SPEED = 0.4    # m/s, DJI auto-landing descent
DRONE_RADIUS = 0.13      # m, Mini 3 with propellers is about 25 cm across
TAKEOFF_HEIGHT = 1.2     # m, auto take-off height


class CageEnv:
    def __init__(self, seed=0, params=None, camera=None, max_time=90.0, noise=True):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.params = params or Params.load()
        self.camera = camera or Camera()
        self.max_time = max_time
        self.noise = noise
        self.reset()

    def reset(self, pad=None, start=None, heading=None, scene=None):
        """pad = (x, y, yaw); start = (x, y); heading in deg. Missing ones are random."""
        r = self.rng
        lim = CAGE_HALF - 0.45
        if pad is None:
            pad = (r.uniform(-lim, lim), r.uniform(-lim, lim), r.uniform(-20, 20))
        if start is None:
            while True:
                start = (r.uniform(-lim, lim), r.uniform(-lim, lim))
                if math.hypot(start[0] - pad[0], start[1] - pad[1]) > 0.7:
                    break
        if heading is None:
            heading = float(r.choice([0, 90, 180, 270]) + r.uniform(-15, 15))
        self.scene = scene or Scene(pad_x=pad[0], pad_y=pad[1], pad_yaw=pad[2], seed=self.seed)
        self.drone = Mini3(self.params, self.rng if self.noise else None)
        s = self.drone.s
        s.x, s.y, s.z, s.heading = start[0], start[1], TAKEOFF_HEIGHT, heading % 360
        self.t = 0.0
        self.done = False
        self.outcome = None
        self.auto_landing = False
        self.min_net_m = 99.0
        self.path = []
        self.last_sticks = [0.0, 0.0, 0.0, 0.0]
        return self.observe()

    def observe(self):
        s = self.drone.s
        rgb = render(s.x, s.y, s.z, s.heading, self.scene, self.camera)
        self.last_rgb = rgb
        return {"jpeg": to_jpeg(rgb), "telemetry": self.drone.to_telemetry(), "t": round(self.t, 2)}

    def auto_land(self):
        self.auto_landing = True

    def step(self, sticks):
        if self.done:
            return self.observe()
        s = self.drone.s
        if not self.auto_landing and sticks[2] < -5 and s.z < LAND_HEIGHT:
            self.auto_landing = True
        cmd = [0.0, 0.0, 0.0, 0.0] if self.auto_landing else [float(v) for v in sticks]
        self.last_sticks = cmd
        z0 = s.z
        self.drone.step(cmd, DT)
        if self.auto_landing:                 # DJI lands straight down; horizontal speed dies out
            s.vz = -AUTO_LAND_SPEED
            s.z = z0 - AUTO_LAND_SPEED * DT
        self.t += DT
        net = min(CAGE_HALF - abs(s.x), CAGE_HALF - abs(s.y), CAGE_HEIGHT - s.z)
        self.min_net_m = min(self.min_net_m, net)
        self.path.append((round(self.t, 2), round(s.x, 3), round(s.y, 3), round(s.z, 3), round(s.heading, 1)))
        if net < DRONE_RADIUS:
            self._end("hit_net")
        elif s.z <= 0.0:
            s.z = 0.0
            self._end("landed_on_pad" if self.on_pad() else "landed_off_pad")
        elif self.t >= self.max_time:
            self._end("timeout")
        return self.observe()

    def on_pad(self):
        s, sc = self.drone.s, self.scene
        a = math.radians(sc.pad_yaw)
        dx, dy = s.x - sc.pad_x, s.y - sc.pad_y
        u = dx * math.cos(a) - dy * math.sin(a)
        v = dx * math.sin(a) + dy * math.cos(a)
        return abs(u) <= PAD_SIZE / 2 and abs(v) <= PAD_SIZE / 2

    def pad_distance(self):
        s, sc = self.drone.s, self.scene
        return math.hypot(s.x - sc.pad_x, s.y - sc.pad_y)

    def _end(self, outcome):
        self.done = True
        self.outcome = outcome

    def result(self):
        return {"seed": self.seed, "outcome": self.outcome or "running", "time_s": round(self.t, 1),
                "pad_distance_m": round(self.pad_distance(), 3), "min_net_m": round(self.min_net_m, 3),
                "pad": [round(self.scene.pad_x, 2), round(self.scene.pad_y, 2)]}
