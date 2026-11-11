"""
skills.py  -  the "hands" the high-level pilot (brain/pilot.py) can choose from.

Each skill turns one observation (camera JPEG + telemetry, the same block the phone sends)
into one stick command, 10 times per second, and says when it is finished. They only use
what the real drone also has: the picture, height, speeds and heading. Nothing here knows the
drone's position in the room.

    hover(seconds)                 sticks centred
    rotate(degrees)                turn on the spot (+ = clockwise), uses the heading
    climb(meters)                  change height (- = down), uses the height
    move(forward_m, right_m)       short straight move, distance from the measured speeds
    vla(instruction, seconds)      the fine-tuned VLA flies (PolicyClient / servo / mock)
    servo(seconds)                 the classic pad-centring controller (brain/servo.py)
    land                           DJI auto-landing straight down (the app does this)

Scripted skills get their own, small stick limits (SKILL_LIMITS); the VLA keeps the training
range of brain/governor.py. Every command still goes through a governor, so a skill cannot
jerk the drone either.
"""
import math

from brain.governor import Governor
from brain.phases import pad_in_frame

SKILL_LIMITS = [(-30.0, 30.0), (-30.0, 30.0), (-60.0, 40.0), (-60.0, 60.0)]   # stick %
MIN_HEIGHT = 0.4        # m, climb() never goes lower (use land for that)
MAX_HEIGHT = 2.2        # m, climb() never goes higher (cage is 3 m, keep away from the top net)
MAX_MOVE = 0.8          # m per move
MAX_SECONDS = 10.0
DT = 0.1


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Skill:
    name = "skill"
    source = "skill"

    def __init__(self):
        self.t = 0.0
        self.finished = False
        self.status = "running"     # running | done | timeout | failed: <why>

    def step(self, obs):
        """Stick percent [pitch, roll, throttle, yaw], or the string "auto_land"."""
        raise NotImplementedError

    def _tick(self, limit):
        self.t += DT
        if self.t >= limit and not self.finished:
            self._finish("timeout")

    def _finish(self, status="done"):
        self.finished = True
        self.status = status

    def describe(self):
        return self.name


class Hover(Skill):
    name = "hover"

    def __init__(self, seconds=1.0):
        super().__init__()
        self.seconds = _clamp(float(seconds or 1.0), 0.1, MAX_SECONDS)

    def step(self, obs):
        self.t += DT
        if self.t >= self.seconds:
            self._finish()
        return [0.0, 0.0, 0.0, 0.0]

    def describe(self):
        return f"hover({self.seconds:.1f} s)"


class Rotate(Skill):
    name = "rotate"
    TOL = 4.0

    def __init__(self, degrees):
        super().__init__()
        self.degrees = _clamp(float(degrees or 0.0), -180.0, 180.0)
        self.target = None

    def step(self, obs):
        heading = float(obs["telemetry"].get("heading", 0.0))
        if self.target is None:
            self.target = (heading + self.degrees) % 360
        err = (self.target - heading + 540) % 360 - 180
        self._tick(4.0 + abs(self.degrees) / 10)
        if abs(err) < self.TOL:
            self._finish()
            return [0.0, 0.0, 0.0, 0.0]
        yaw = math.copysign(60.0 if abs(err) > 30 else 40.0 if abs(err) > 10 else 30.0, err)   # the stick curve is cubic
        return [0.0, 0.0, 0.0, yaw]

    def describe(self):
        return f"rotate({self.degrees:+.0f} deg)"


class Climb(Skill):
    name = "climb"
    GAIN = 120.0        # throttle % per metre left
    TOL = 0.06

    def __init__(self, meters):
        super().__init__()
        self.meters = float(meters or 0.0)
        self.target = None

    def step(self, obs):
        h = float(obs["telemetry"].get("height_m", 1.0))
        if self.target is None:
            self.target = _clamp(h + self.meters, MIN_HEIGHT, MAX_HEIGHT)
        err = self.target - h
        self._tick(3.0 + abs(self.meters) * 6)
        if abs(err) < self.TOL:
            self._finish()
            return [0.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, _clamp(err * self.GAIN, -60.0, 40.0), 0.0]

    def describe(self):
        return f"climb({self.meters:+.2f} m)"


class Move(Skill):
    """Straight move by dead reckoning from the measured speeds (drifts a little, like on the drone)."""
    name = "move"
    GAIN = 60.0         # stick % per metre left
    TOL = 0.05

    def __init__(self, forward_m=0.0, right_m=0.0):
        super().__init__()
        self.fwd = _clamp(float(forward_m or 0.0), -MAX_MOVE, MAX_MOVE)
        self.right = _clamp(float(right_m or 0.0), -MAX_MOVE, MAX_MOVE)
        self.done_f = self.done_r = 0.0
        self.h0 = None

    def step(self, obs):
        tel = obs["telemetry"]
        if self.h0 is None:
            self.h0 = math.radians(float(tel.get("heading", 0.0)))
        vn, ve = float(tel.get("vx", 0.0)), float(tel.get("vy", 0.0))      # NED: north, east
        h = self.h0
        self.done_f += (vn * math.cos(h) + ve * math.sin(h)) * DT
        self.done_r += (-vn * math.sin(h) + ve * math.cos(h)) * DT
        ef, er = self.fwd - self.done_f, self.right - self.done_r
        self._tick(3.0 + 6 * math.hypot(self.fwd, self.right))
        if abs(ef) < self.TOL and abs(er) < self.TOL:
            self._finish()
            return [0.0, 0.0, 0.0, 0.0]
        return [_clamp(ef * self.GAIN, -30, 30), _clamp(er * self.GAIN, -30, 30), 0.0, 0.0]

    def describe(self):
        return f"move(forward {self.fwd:+.2f} m, right {self.right:+.2f} m)"


class VLA(Skill):
    name = "vla"
    source = "model"

    def __init__(self, policy, instruction=None, seconds=5.0):
        super().__init__()
        self.policy = policy
        self.instruction = instruction or None
        self.seconds = _clamp(float(seconds or 5.0), 0.5, MAX_SECONDS * 2)
        self.model_ms = []

    def step(self, obs):
        self._tick(self.seconds)
        try:
            sticks, model_ms, _ = self.policy.act(obs["jpeg"], self.instruction)
        except Exception as e:                     # no answer: hover, report it
            self._finish(f"failed: {e}")
            return [0.0, 0.0, 0.0, 0.0]
        self.model_ms.append(model_ms)
        return sticks

    def describe(self):
        return f"vla('{self.instruction or 'training sentence'}', {self.seconds:.0f} s)"


class ServoSkill(Skill):
    name = "servo"
    source = "servo"

    target_y = None     # override brain/servo.py TARGET_Y (where the pad sits when it is below us)

    def __init__(self, seconds=8.0):
        super().__init__()
        from brain.servo import Servo, TARGET_Y
        self.servo = Servo(target_y=self.target_y if self.target_y is not None else TARGET_Y)
        self.seconds = _clamp(float(seconds or 8.0), 0.5, MAX_SECONDS * 2)
        self.sim_t = 0.0

    def step(self, obs):
        self._tick(self.seconds)
        self.sim_t += DT
        cmd = self.servo.command(obs["jpeg"], now=self.sim_t)
        if cmd is None:
            self._finish("failed: pad not in view")
            return [0.0, 0.0, 0.0, 0.0]
        return cmd

    def describe(self):
        return f"servo({self.seconds:.0f} s)"


class Land(Skill):
    name = "land"

    def step(self, obs):
        self._finish()
        return "auto_land"


def make_skill(decision, policy):
    """Skill from a pilot decision (brain/pilot.Decision or a dict with the same fields)."""
    d = decision if isinstance(decision, dict) else decision.model_dump()
    s = d.get("skill")
    if s == "hover":
        return Hover(d.get("seconds") or 1.0)
    if s == "rotate":
        return Rotate(d.get("degrees") or 0.0)
    if s == "climb":
        return Climb(d.get("meters") or 0.0)
    if s == "move":
        return Move(d.get("forward_m") or 0.0, d.get("right_m") or 0.0)
    if s == "vla":
        return VLA(policy, d.get("instruction"), d.get("seconds") or 5.0)
    if s == "servo":
        return ServoSkill(d.get("seconds") or 8.0)
    if s in ("land", "abort"):
        return Land()
    raise ValueError(f"unknown skill {s!r}")


class Hands:
    """Runs the current skill and passes every command through a governor: the VLA through the
    training-range governor, scripted skills through SKILL_LIMITS."""

    def __init__(self, model_governor=None):
        self.model_gov = model_governor or Governor([(0.0, 45.0), (-38.0, 35.0), (-100.0, 0.0), (-37.0, 54.0)])
        self.skill_gov = Governor(SKILL_LIMITS, allow_yaw=True)
        self.skill = None

    def set(self, skill):
        self.skill = skill
        self.model_gov.reset()

    def step(self, obs):
        """(sticks or "auto_land", source)."""
        if self.skill is None or self.skill.finished:
            return self.skill_gov.apply([0.0, 0.0, 0.0, 0.0]), "hover"
        cmd = self.skill.step(obs)
        if cmd == "auto_land":
            return cmd, self.skill.source
        gov = self.model_gov if self.skill.source in ("model", "servo") else self.skill_gov
        out = gov.apply(cmd)
        (self.skill_gov if gov is self.model_gov else self.model_gov).last = list(out)
        return out, self.skill.source


def pad_info(jpeg):
    """Short text about the red pad in the picture, for the pilot and the logs."""
    visible, x, y, frac = pad_in_frame(jpeg)
    if not visible:
        return {"visible": False}
    return {"visible": True, "x": round(x, 2), "y": round(y, 2), "area_pct": round(100 * frac, 1)}
