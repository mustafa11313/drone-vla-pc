"""
autopilot  -  flies and records the plan's episodes by itself, with stick input like a person's.

For each setup of the plan (the recorder places the pad and the drone, nose to the pad):

    take off  ->  hover a moment  ->  start recording  ->  fly to the pad  ->  land on it
    ->  recording stops after touchdown  ->  success if the drone stands on the pad, else fail
    ->  the recorder places the next setup (or the same one again after a fail)

Only the sticks are used, through /drone/sticks, exactly as from the keyboard, so the recorded
pitch_cmd / roll_cmd / throttle_cmd / yaw_cmd are what a VLA would have to output.

The way it moves the sticks copies your real episodes (episode_0001/0002 of 2026-10-12):
  - one stick at a time most of the time, sometimes pitch and roll together;
  - pushes of 10-45 %, held 0.5-3 s, pushed in and let go over about 0.3 s, easing off a little
    as the pad comes closer and never perfectly still;
  - short pauses with the sticks centred between pushes while the drone brakes;
  - sometimes a short descent with the throttle right after takeoff;
  - the landing is done by holding the throttle fully down over the pad, which starts the
    DJI auto landing below 0.6 m (as in your episodes), not with a land button.
Every episode draws its own pilot "mood" (how hard, how patient, how precise), so the 216
episodes are not 216 copies of one flight.

The autopilot knows where the pad is (from the simulator); the VLA trained on the episodes
only gets the camera frames, the instruction and the telemetry.

    ros2 launch mini3_sim sim.launch.py autopilot:=true            # with the windows
    ros2 launch mini3_sim sim.launch.py autopilot:=true gui:=false camera_view:=false   # faster
"""
import json
import math
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from mini3_sim.scenarios import NEAR_Y, YAW_X, in_picture, pad_angles

PAD_HALF = 0.25          # m, the pad is 50 x 50 cm
TIMEOUT_S = 60.0         # a flight that takes longer is landed and marked fail
MAX_TRIES = 3            # the same setup is flown at most this many times, then skipped


CAM_FWD, CAM_DOWN = 0.1, 0.015         # camera position on the drone (m)
CAM_PITCH, HFOV, VFOV = 50.0, 62.0, 37.3   # deg, as in models/mini3/model.sdf (640 x 360)


def pad_in_view(fwd, right, z, margin=4.0, vmargin=None):
    """True if the pad centre is inside the camera picture, `margin` degrees in from the edges
    (a negative margin also accepts a centre just outside, with most of the pad showing)."""
    f, d = fwd - CAM_FWD, z - CAM_DOWN               # forward and down from the camera
    p = math.radians(CAM_PITCH)
    depth = f * math.cos(p) + d * math.sin(p)        # along the camera's axis (pinhole camera)
    if depth <= 0.05:
        return False
    x = math.degrees(math.atan(right / depth))       # left-right angle in the picture
    y = math.degrees(math.atan((d * math.cos(p) - f * math.sin(p)) / depth))   # up-down
    vmargin = margin if vmargin is None else vmargin
    return abs(x) < HFOV / 2 - margin and abs(y) < VFOV / 2 - vmargin


class Pilot:
    """The stick logic for one flight. step() is called at 20 Hz and returns the 4 sticks."""

    def __init__(self, rng, search=False):
        self.rng = rng
        # this episode's pilot mood
        self.gain = rng.uniform(55, 95)               # stick % per metre of distance
        self.max_push = rng.uniform(30, 45)
        self.min_push = rng.uniform(8, 14)
        self.ramp = rng.uniform(120, 220)             # stick % per second when pushing in / letting go
        self.precision = rng.uniform(0.05, 0.10)      # m: close enough to start the landing
        self.combine_p = rng.uniform(0.1, 0.35)       # chance of pitch and roll together
        self.sticks = [0.0, 0.0, 0.0, 0.0]
        self.target = [0.0, 0.0, 0.0, 0.0]
        self.phase = "start"
        self.until = 0.0
        self.landing = False
        self.search = search
        # search episodes: turn clockwise until the pad is in view (the only rule a model can learn
        # from the camera), sometimes turn to face it, and use roll for sideways offsets
        self.found = not search
        self.seen_at = None
        self.react = rng.uniform(0.15, 0.45)          # s from "pad in view" to letting go of yaw
        self.turn_push = rng.uniform(55, 80)        # 55-80 % turns at about 15-35 deg/s
        self.face_pad = rng.random() < 0.5            # turn to face the pad once found, or not
        self.side_first_p = rng.uniform(0.25, 0.5) if search else 0.0

    def ramp_to(self, dt):
        for i in range(4):
            d = self.target[i] - self.sticks[i]
            step = (400.0 if i == 2 and self.landing else self.ramp) * dt
            self.sticks[i] += max(-step, min(step, d))

    def step(self, t, dt, fwd, right, v_fwd, v_right, z, visible=True, clear=True):
        """visible: the pad is (at least half) in the picture; clear: well inside it."""
        """fwd/right: pad offset in the drone's frame (m); v_*: speeds in that frame (m/s)."""
        dist = math.hypot(fwd, right)
        speed = math.hypot(v_fwd, v_right)
        if self.phase == "start":
            # sometimes come down a bit first, as in your first episode
            if self.rng.random() < 0.5:
                self.target = [0, 0, -self.rng.uniform(40, 65), 0]
                self.phase, self.until = "descend", t + self.rng.uniform(0.4, 0.9)
            else:
                self.phase, self.until = "pause", t + self.rng.uniform(0.3, 1.0)
        elif self.phase == "descend":
            if t >= self.until or z < 0.75:
                self.target = [0, 0, 0, 0]
                self.phase, self.until = "pause", t + self.rng.uniform(0.4, 1.0)
        elif self.phase == "turn":
            # clockwise until the pad shows up, then let go after a human reaction time
            if clear and self.seen_at is None:        # it comes in from the right edge
                self.seen_at = t
            if self.seen_at is None and t - self.turn_start > 30.0:    # never came into view
                self.seen_at = t - self.react
            if self.seen_at is not None and t - self.seen_at >= self.react:
                self.found = True
                self.target = [0, 0, 0, 0]
                self.phase, self.until = "pause", t + self.rng.uniform(0.3, 0.8)
        elif self.phase == "face":
            bearing = math.degrees(math.atan2(right, fwd))
            if not visible or abs(bearing) < self.rng.uniform(3, 7) or t >= self.until:
                self.target = [0, 0, 0, 0]
                self.face_pad = False
                self.phase, self.until = "pause", t + self.rng.uniform(0.3, 0.8)
            else:
                push = min(55.0, max(25.0, abs(bearing) * 1.5))
                self.target = [0, 0, 0, math.copysign(push, bearing)]
        elif self.phase == "pause":
            self.target = [0, 0, 0, 0]
            if t >= self.until and speed < self.rng.uniform(0.08, 0.2):
                bearing = math.degrees(math.atan2(right, fwd))
                if not self.found:
                    if visible:
                        self.found = True
                    else:
                        self.target = [0, 0, 0, self.turn_push * self.rng.uniform(0.9, 1.1)]
                        self.phase, self.seen_at, self.turn_start = "turn", None, t
                        return self.finish(dt)
                if self.face_pad and visible and abs(bearing) > 10 and dist > 0.4:
                    self.phase, self.until = "face", t + 4.0
                    return self.finish(dt)
                if dist < self.precision:
                    self.phase, self.landing = "land", True
                else:
                    self.choose_push(fwd, right)
                    self.phase = "push"
                    self.until = t + 3.5
        elif self.phase == "push":
            # let go when the drone would coast the rest of the way (it brakes over ~0.75 s)
            coast_f = v_fwd * 0.6
            coast_r = v_right * 0.6
            done = True
            for i, err, coast in ((0, fwd, coast_f), (1, right, coast_r)):
                if self.target[i] != 0 and abs(err) > abs(coast) + 0.03 and err * self.target[i] > 0:
                    done = False
            # like a thumb on the stick: eases off as the pad comes closer, never perfectly still
            for i, err in ((0, fwd), (1, right)):
                if self.target[i] != 0:
                    want = max(self.min_push, min(abs(self.push0[i]), self.gain * 1.3 * abs(err)))
                    want += self.rng.gauss(0, 0.6)
                    self.target[i] = math.copysign(max(self.min_push * 0.7, want), self.target[i])
            if done or t >= self.until:
                self.target = [0, 0, 0, 0]
                self.phase, self.until = "pause", t + self.rng.uniform(0.3, 1.0)
        if self.phase == "land":
            self.target = [0, 0, -100, 0]
        return self.finish(dt)

    def finish(self, dt):
        self.ramp_to(dt)
        return list(self.sticks)

    def choose_push(self, fwd, right):
        def push(err):
            p = min(self.max_push, max(self.min_push, self.gain * abs(err)))
            return math.copysign(p * self.rng.uniform(0.85, 1.15), err)
        both = abs(fwd) > 0.12 and abs(right) > 0.12 and self.rng.random() < self.combine_p
        if not both and abs(right) > 0.12 and self.rng.random() < self.side_first_p:
            self.target = [0, push(right), 0, 0]          # sideways first, as you often did
            self.push0 = list(self.target)
            return
        self.target = [0, 0, 0, 0]
        if both:
            self.target = [push(fwd), push(right), 0, 0]
        elif abs(fwd) >= abs(right):
            self.target = [push(fwd), 0, 0, 0]
        else:
            self.target = [0, push(right), 0, 0]
        self.push0 = list(self.target)


class ScenarioPilot(Pilot):
    """The stick logic for the yaw and roll scenarios (setups:=scenarios, see scenarios.py).

    Every decision is taken from where the pad is in the camera picture, so the model can learn it:
        not in the picture   -> clockwise turn until it shows up
        far, well to a side  -> yaw towards it (strong, 45-100 %, as in the real flights)
        near, well to a side -> roll towards it, heading kept (steady 12-40 %)
        about centred        -> pitch forward, with roll for a small sideways offset
    Yaw and roll are never pushed at the same time. Before that, the throttle brings the drone to
    the scenario's hover height (0.8-1.4 m)."""

    def __init__(self, rng, scenario):
        super().__init__(rng, search=True)
        self.sc = scenario
        self.height = float(scenario.get("height", 1.0))
        self.yaw_stop = float(scenario.get("yaw_stop", 3.0))
        self.yaw_gain = rng.uniform(2.5, 4.0)          # yaw % per degree off-centre
        self.yaw_min = rng.uniform(45, 60)
        self.yaw_max = rng.uniform(80, 100)
        self.turn_push = rng.uniform(60, 90)
        self.max_push = rng.uniform(32, 42)            # roll and pitch: as in the real flights
        self.min_push = rng.uniform(12, 18)
        self.combine_p = 1.0
        self.seen = False
        self.phase = "height"
        self.climb = rng.uniform(40, 65)
        self.prev_z = None

    def step(self, t, dt, fwd, right, v_fwd, v_right, z, visible=True, clear=True):
        a = pad_angles(fwd, right, z)
        vis = in_picture(a, -2.0)
        self.seen = self.seen or vis
        dist = math.hypot(fwd, right)
        speed = math.hypot(v_fwd, v_right)
        if self.phase == "height":
            err = self.height - z
            vz = (z - self.prev_z) / dt if self.prev_z is not None else 0.0
            self.prev_z = z
            # let go early: the drone keeps moving up or down for ~0.75 s after the stick is centred
            if abs(err) < 0.04 or (err * vz > 0 and abs(err) < abs(vz) * 0.6):
                self.target = [0, 0, 0, 0]
                self.phase, self.until = "pause", t + self.rng.uniform(0.3, 0.8)
            else:
                self.target = [0, 0, math.copysign(self.climb, err) * min(1.0, abs(err) / 0.15 + 0.4), 0]
        elif self.phase == "turn":
            # clearly in: well in from the side it comes in from, anywhere up-down
            if a and abs(a[0]) < HFOV / 2 - 6 and abs(a[1]) < VFOV / 2 + 2 and self.seen_at is None:
                self.seen_at = t
            if self.seen_at is None and t - self.turn_start > 30.0:
                self.seen_at = t - self.react
            if self.seen_at is not None and t - self.seen_at >= self.react:
                self.target = [0, 0, 0, 0]
                self.phase, self.until = "pause", t + self.rng.uniform(0.3, 0.8)
        elif self.phase == "face":
            x = a[0] if a else None
            if x is None or abs(x) < self.yaw_stop or x * self.target[3] < 0 or t >= self.until:
                self.target = [0, 0, 0, 0]
                self.phase, self.until = "pause", t + self.rng.uniform(0.3, 0.8)
            else:
                push = min(self.yaw_max, max(self.yaw_min, abs(x) * self.yaw_gain))
                # let go a little early: the turn goes on for ~0.25 s after the stick is centred
                if abs(x) < self.yaw_stop + 6:
                    push = self.yaw_min * 0.8
                self.target = [0, 0, 0, math.copysign(push, x)]
        elif self.phase == "pause":
            self.target = [0, 0, 0, 0]
            centred = max(abs(v) for v in self.sticks) < 0.5        # never yaw and roll at once
            if t >= self.until and centred and speed < self.rng.uniform(0.08, 0.2):
                if dist < self.precision:
                    self.phase, self.landing = "land", True
                elif not vis and not (self.seen and dist < 0.7):
                    self.target = [0, 0, 0, self.turn_push * self.rng.uniform(0.95, 1.05)]
                    self.phase, self.seen_at, self.turn_start = "turn", None, t
                elif vis and abs(a[0]) >= YAW_X and a[1] < NEAR_Y:
                    self.phase, self.until = "face", t + 6.0
                elif vis and abs(a[0]) >= YAW_X:
                    self.target = [0, self.push(right), 0, 0]        # near and to the side: roll only
                    self.push0 = list(self.target)
                    self.phase, self.until = "push", t + 3.5
                else:
                    # centred, or just slid out of the bottom of the picture (it is right below)
                    self.target = [self.push(fwd) if abs(fwd) > 0.04 else 0, 0, 0, 0]
                    if abs(right) > 0.08:
                        self.target[1] = self.push(right)
                    if not any(self.target):
                        self.phase, self.landing = "land", True
                    else:
                        self.push0 = list(self.target)
                        self.phase, self.until = "push", t + 3.5
            return self.finish(dt)
        elif self.phase == "push":
            return super().step(t, dt, fwd, right, v_fwd, v_right, z, visible, clear)
        if self.phase == "land":
            self.target = [0, 0, -100, 0]
        return self.finish(dt)

    def push(self, err):
        p = min(self.max_push, max(self.min_push, self.gain * abs(err)))
        return math.copysign(p * self.rng.uniform(0.9, 1.1), err)


class ModelPilot:
    """A trained model flies: every 0.1 s of simulated time the newest camera frame goes to the
    inference server (inference_server.py: POST /act, one JPEG -> {"action": [pitch, roll,
    throttle, yaw]} in -1..1) and the answer is put on the sticks until the next one.

    lockstep: Gazebo stands still while the model thinks and is then stepped exactly 0.1 s
    further, so a slow model or a slow connection does not change the flight (the model always
    acts on the frame of "now", 10 times per simulated second, as in the recorded data).
    Without it the last sticks are held while waiting, as on the real drone.

    The frame is sent as the recorder saves it (1280 x 720 JPEG, quality 90), so the model sees
    what it saw in training."""

    def __init__(self, node):
        self.n = node
        self.landing = False
        self.phase = "model"
        self.sticks = [0.0, 0.0, 0.0, 0.0]
        self.next_query = 0.0
        self.steps, self.model_ms, self.wall_ms, self.frame_age = 0, 0.0, 0.0, 0.0
        self.stalled, self.last_t = 0, None

    def step(self, t, dt, *unused):
        if self.landing:                       # time limit: land where it is
            self.n.resume()
            return [0.0, 0.0, -100.0, 0.0]
        if self.n.tel.get("flight_mode") == "AUTO_LANDING":      # the drone lands by itself now
            self.n.resume()
            return [0.0, 0.0, 0.0, 0.0]
        # (the clock standing still for a few ticks means the 0.1 s step is done)
        self.stalled = self.stalled + 1 if t == self.last_t else 0
        self.last_t = t
        if t + 0.006 >= self.next_query or self.stalled >= 3:
            self.stalled = 0
            self.next_query = t + self.n.model_dt
            action = self.n.ask_model(t)
            if action is not None:
                self.sticks = [max(-100.0, min(100.0, 100.0 * float(v))) for v in action[:4]]
                self.steps += 1
        return list(self.sticks)


class Autopilot(Node):
    def __init__(self):
        super().__init__("autopilot")
        self.declare_parameter("episodes", 216)         # how many to record in this run
        self.declare_parameter("seed", 0)               # 0 = different every run
        seed = self.get_parameter("seed").value
        self.rng = random.Random(seed or None)
        # a trained model flies instead of the scripted pilot (see ModelPilot)
        self.declare_parameter("model_url", "")           # e.g. https://host/rnode/node/8000
        self.declare_parameter("cookie_file", os.path.expanduser("~/.ondemand_cookie"))
        self.declare_parameter("lockstep", True)
        self.declare_parameter("model_hz", 10.0)
        self.declare_parameter("model_timeout_s", 45.0)   # simulated seconds per flight
        self.declare_parameter("world", "cage_lab")
        self.declare_parameter("results", "")             # CSV of the flights (default: none)
        self.model_url = str(self.get_parameter("model_url").value or "").rstrip("/")
        if self.model_url.lower() in ("off", "none", "false"):
            self.model_url = ""
        self.model_dt = 1.0 / float(self.get_parameter("model_hz").value)
        self.declare_parameter("physics_step_s", 0.004)   # <max_step_size> of the world
        self.sim_steps = max(1, round(self.model_dt / float(self.get_parameter("physics_step_s").value)))
        self.frame, self.frame_lock = None, threading.Lock()
        self.results, self.timed_out, self.cookie = [], False, ""
        self.ctl, self.held = None, False            # Gazebo held still for the model (lockstep)
        if self.model_url:
            import cv2                                   # only needed when a model flies
            from rclpy.callback_groups import ReentrantCallbackGroup
            from sensor_msgs.msg import Image
            self.cv2 = cv2
            cf = os.path.expanduser(self.get_parameter("cookie_file").value)
            if os.path.exists(cf):
                self.cookie = open(cf).read().strip()
                if self.cookie.lower().startswith("cookie:"):
                    self.cookie = self.cookie[7:].strip()
            self.create_subscription(Image, "/drone/camera/image_raw", self.on_image, 5,
                                     callback_group=ReentrantCallbackGroup())
            self.check_model()
        self.sticks_pub = self.create_publisher(Float32MultiArray, "/drone/sticks", 10)
        self.cmd_pub = self.create_publisher(String, "/drone/command", 10)
        self.rec_pub = self.create_publisher(String, "/drone/record", 10)
        self.create_subscription(String, "/drone/telemetry", self.on_tel, 10)
        self.create_subscription(String, "/drone/recorder/status", self.on_status, 10)
        self.tel, self.status = None, None
        self.state, self.t_state = "wait_setup", 0.0
        self.seen_seq = 0
        self.done, self.ok_count = 0, 0
        self.tries = 0
        self.pilot = None
        self.dt = 0.05
        if self.model_url:
            # wall-clock ticks: they must go on while Gazebo is held still for the model
            from rclpy.clock import Clock, ClockType
            self.create_timer(self.dt, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        else:
            self.create_timer(self.dt, self.tick)
        self.get_logger().info("autopilot ready")

    def on_tel(self, m):
        self.tel = json.loads(m.data)

    # ------------------------------------------------------------------ a model flies
    def on_image(self, m):
        import numpy as np
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == "rgb8":
            a = a[..., ::-1]                              # OpenCV wants BGR
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.frame_lock:
            self.frame = (stamp, a)

    def http(self, path, data=None, timeout=30.0):
        req = urllib.request.Request(self.model_url + path, data=data, method="POST" if data else "GET")
        if self.cookie:
            req.add_header("Cookie", self.cookie)
        if data:
            req.add_header("Content-Type", "image/jpeg")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        try:
            return json.loads(body)
        except ValueError:
            raise RuntimeError("the answer is not from the inference server (a login page?): the cookie in "
                               f"{self.get_parameter('cookie_file').value} is missing or has expired")

    def check_model(self):
        try:
            info = self.http("/")
            self.get_logger().info(f"model server: {info.get('model', info)}")
        except Exception as e:
            self.get_logger().error(f"cannot reach the model server at {self.model_url}: {e}")

    def world_control(self, req, wait):
        """a request to Gazebo's world control service (pause, continue, step)."""
        world = self.get_parameter("world").value
        if self.ctl is not None:                       # let the last request finish first
            try:
                self.ctl.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.ctl.kill()
            self.ctl = None
        for tool in ("ign", "gz"):
            cmd = [tool, "service", "-s", f"/world/{world}/control", "--reqtype",
                   "ignition.msgs.WorldControl", "--reptype", "ignition.msgs.Boolean",
                   "--timeout", "2000", "--req", req]
            try:
                if not wait:
                    self.ctl = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
                if "true" in r.stdout:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return False

    def resume(self):
        """Gazebo runs freely again (after a model's flight)."""
        if self.held:
            self.held = False
            self.world_control("pause: false", wait=True)

    def ask_model(self, t):
        """one model step: newest frame -> server -> sticks (-1..1); with lockstep Gazebo stands
        still meanwhile and is then stepped one model period further."""
        lock = bool(self.get_parameter("lockstep").value)
        t0 = time.time()
        if lock and not self.held:
            self.held = self.world_control("pause: true", wait=True)
        elif self.held and self.ctl is not None:       # the last step request has to be through
            try:
                self.ctl.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.ctl.kill()
            self.ctl = None
        action = None
        try:
            # the picture of "now" reaches ROS a moment after it was taken: wait for it briefly
            end = time.time() + (1.5 if self.held else 0.0)
            while True:
                now = self.now() if self.held else t
                with self.frame_lock:
                    fr = self.frame
                if fr is not None and (fr[0] >= now - 0.08 or time.time() >= end):
                    break
                if time.time() >= end:
                    return None
                time.sleep(0.01)
            self.pilot.frame_age += max(0.0, now - fr[0])
            img = self.cv2.resize(fr[1], (1280, 720), interpolation=self.cv2.INTER_LINEAR)
            ok, jpg = self.cv2.imencode(".jpg", img, [self.cv2.IMWRITE_JPEG_QUALITY, 90])
            ans = self.http("/act", jpg.tobytes())
            if "action" not in ans:
                raise RuntimeError(str(ans))
            self.pilot.model_ms += float(ans.get("model_ms", 0.0))
            self.pilot.wall_ms += (time.time() - t0) * 1000
            action = ans["action"]
            # on the sticks before the simulation goes on
            self.send_sticks([max(-100.0, min(100.0, 100.0 * float(v))) for v in action[:4]])
        except Exception as e:
            self.get_logger().error(f"model step failed: {e}", throttle_duration_sec=5.0)
        finally:
            if self.held:                              # one model period further, then still again
                time.sleep(0.02)                       # (the sticks have to reach the drone first)
                self.world_control(f"pause: true, multi_step: {self.sim_steps}", wait=False)
        return action

    def write_results(self, summary=False):
        path = str(self.get_parameter("results").value or "")
        if path.lower() in ("", "off", "none") or not self.results:
            return
        path = os.path.expanduser(path)
        keys = list(self.results[0])
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in self.results:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        if not summary:
            return
        per = {}
        for r in self.results:
            p = per.setdefault(r["scenario"], [0, 0])
            p[0] += r["landed_on_pad"]
            p[1] += 1
        self.get_logger().info("landed on the pad, per scenario: " +
                               ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(per.items())))
        self.get_logger().info(f"results written to {path}")

    def on_status(self, m):
        self.status = json.loads(m.data)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def go(self, state):
        self.state, self.t_state = state, self.now()

    def send_sticks(self, s):
        self.sticks_pub.publish(Float32MultiArray(data=[float(v) for v in s]))

    def pad_offset(self):
        """pad position in the drone's frame (forward, right) and the drone's speed in it."""
        t = self.tel
        px, py = self.status["pad_xy"]
        dx, dy = px - t["sim_x"], py - t["sim_y"]
        h = math.radians(t["heading"])
        fwd = dx * math.sin(h) + dy * math.cos(h)
        right = dx * math.cos(h) - dy * math.sin(h)
        ve, vn = t["y_speed"] / 2.23694, t["x_speed"] / 2.23694     # telemetry: mph, NED
        v_fwd = ve * math.sin(h) + vn * math.cos(h)
        v_right = ve * math.cos(h) - vn * math.sin(h)
        return fwd, right, v_fwd, v_right

    def tick(self):
        if self.tel is None or self.status is None:
            return
        t, st, tel = self.now(), self.status, self.tel
        age = t - self.t_state
        if self.state == "wait_setup":
            self.send_sticks([0, 0, 0, 0])
            if self.done >= self.get_parameter("episodes").value or st.get("plan_done"):
                self.get_logger().info(f"finished: {self.ok_count} of {self.done} episodes landed on the pad")
                self.write_results(summary=True)
                self.go("finished")
                return
            # nothing placed for a while (a lost Gazebo reply): ask the recorder to place again
            if (age > 20 and not st["recording"] and not st["waiting_for_result"]
                    and tel.get("on_ground")):
                self.get_logger().warn("no new setup after 20 s, asking the recorder to place it again")
                self.rec_pub.publish(String(data="place"))
                self.go("wait_setup")
                return
            if (st["placed_seq"] > self.seen_seq and st.get("pad_xy") and tel.get("on_ground")
                    and not st["recording"] and not st["waiting_for_result"]):
                self.seen_seq = st["placed_seq"]
                self.go("settle")
        elif self.state == "settle" and age > 1.0:
            self.cmd_pub.publish(String(data="takeoff"))
            self.go("takeoff")
        elif self.state == "takeoff":
            self.send_sticks([0, 0, 0, 0])
            if tel["flight_mode"] == "GPS_NORMAL" and not tel["on_ground"]:
                self.hover = self.rng.uniform(0.5, 1.5)
                self.go("hover")
            elif age > 15:
                self.cmd_pub.publish(String(data="takeoff"))
                self.go("takeoff")
        elif self.state == "hover":
            self.send_sticks([0, 0, 0, 0])
            if age > self.hover:
                self.rec_pub.publish(String(data="start"))
                self.go("rec_wait")
        elif self.state == "rec_wait":
            self.send_sticks([0, 0, 0, 0])
            if st["recording"]:
                self.timed_out = False
                if self.model_url:
                    self.pilot = ModelPilot(self)
                elif st.get("scenario"):
                    self.pilot = ScenarioPilot(self.rng, st["scenario"])
                else:
                    self.pilot = Pilot(self.rng, search=bool(st.get("search")))
                self.go("fly")
            elif age > 5:
                self.rec_pub.publish(String(data="start"))
                self.go("rec_wait")
        elif self.state == "fly":
            fwd, right, vf, vr = self.pad_offset()
            s = self.pilot.step(t, self.dt, fwd, right, vf, vr, tel["sim_z"],
                                pad_in_view(fwd, right, tel["sim_z"], margin=-2.0),
                                # "clearly in view": well in from the side it enters from, but
                                # anywhere up-down (a close pad shows near the bottom edge)
                                pad_in_view(fwd, right, tel["sim_z"], margin=6.0, vmargin=0.0))
            limit = self.get_parameter("model_timeout_s").value if self.model_url else TIMEOUT_S
            if age > limit and not self.pilot.landing:
                self.pilot.phase, self.pilot.landing = "land", True
                self.timed_out = True
            self.send_sticks(s)
            if tel["on_ground"]:
                self.send_sticks([0, 0, 0, 0])
                self.flight_s = age
                self.resume()
                self.go("wait_stop")
        elif self.state == "wait_stop":
            self.send_sticks([0, 0, 0, 0])
            if not st["recording"] and st["waiting_for_result"]:
                fwd, right, _, _ = self.pad_offset()
                ok = abs(fwd) < PAD_HALF - 0.05 and abs(right) < PAD_HALF - 0.05
                check_ok = "CHECK" not in st.get("message", "")
                ok = ok and check_ok
                if self.model_url:
                    # a model's flight counts once: landed on the pad by itself within the time, or not
                    ok = ok and not self.timed_out
                    sc = st.get("scenario") or {}
                    p = self.pilot
                    self.results.append({
                        "setup": self.done + 1, "scenario": sc.get("scenario", ""), "pad_side": sc.get("pad_side", ""),
                        "landed_on_pad": int(ok), "off_centre_cm": round(math.hypot(fwd, right) * 100, 1),
                        "timed_out": int(self.timed_out), "flight_s": round(self.flight_s, 1), "model_steps": p.steps,
                        "model_ms": round(p.model_ms / max(p.steps, 1), 1),
                        "step_wall_ms": round(p.wall_ms / max(p.steps, 1), 1),
                        "frame_age_ms": round(1000 * p.frame_age / max(p.steps, 1), 1), "episode": st.get("episode", "")})
                    self.done += 1
                    self.ok_count += ok
                    self.rec_pub.publish(String(data="success" if ok else "fail_skip"))
                    self.get_logger().info(
                        f"model flight {self.done}: {'LANDED ON THE PAD' if ok else 'missed'}"
                        f"{' (time limit)' if self.timed_out else ''}, {math.hypot(fwd, right) * 100:.0f} cm "
                        f"from the pad centre; {self.ok_count}/{self.done} so far")
                    self.write_results()
                    self.go("wait_setup")
                    return
                self.done += 1
                self.ok_count += ok
                self.tries = 0 if ok else self.tries + 1
                if not ok and self.tries >= MAX_TRIES and st.get("scenario"):
                    self.get_logger().warn("the same setup failed 3 times; going on with the next one")
                    self.tries = 0
                    self.rec_pub.publish(String(data="fail_skip"))
                    self.go("wait_setup")
                    return
                if not ok and self.tries >= MAX_TRIES:
                    # give up on this setup: mark it success-less but move on is not possible
                    # through the recorder (fail repeats the setup), so stop here and say so
                    self.get_logger().warn("the same setup failed 3 times; stopping the autopilot")
                    self.rec_pub.publish(String(data="fail"))
                    self.go("finished")
                    return
                self.rec_pub.publish(String(data="success" if ok else "fail"))
                self.get_logger().info(f"episode done: {'success' if ok else 'fail'} "
                                       f"(off the pad centre by {math.hypot(fwd, right) * 100:.0f} cm); "
                                       f"{self.ok_count}/{self.done} so far")
                self.go("wait_setup")


def main():
    from rclpy.executors import MultiThreadedExecutor
    rclpy.init()
    ex = MultiThreadedExecutor(num_threads=3)     # the camera keeps arriving while a model step waits
    ex.add_node(Autopilot())
    ex.spin()


if __name__ == "__main__":
    main()
