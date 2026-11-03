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
import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

PAD_HALF = 0.25          # m, the pad is 50 x 50 cm
TIMEOUT_S = 60.0         # a flight that takes longer is landed and marked fail
MAX_TRIES = 3            # the same setup is flown at most this many times, then skipped


class Pilot:
    """The stick logic for one flight. step() is called at 20 Hz and returns the 4 sticks."""

    def __init__(self, rng):
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

    def ramp_to(self, dt):
        for i in range(4):
            d = self.target[i] - self.sticks[i]
            step = (400.0 if i == 2 and self.landing else self.ramp) * dt
            self.sticks[i] += max(-step, min(step, d))

    def step(self, t, dt, fwd, right, v_fwd, v_right, z):
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
        elif self.phase == "pause":
            self.target = [0, 0, 0, 0]
            if t >= self.until and speed < self.rng.uniform(0.08, 0.2):
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
        self.ramp_to(dt)
        return list(self.sticks)

    def choose_push(self, fwd, right):
        def push(err):
            p = min(self.max_push, max(self.min_push, self.gain * abs(err)))
            return math.copysign(p * self.rng.uniform(0.85, 1.15), err)
        both = abs(fwd) > 0.12 and abs(right) > 0.12 and self.rng.random() < self.combine_p
        self.target = [0, 0, 0, 0]
        if both:
            self.target = [push(fwd), push(right), 0, 0]
        elif abs(fwd) >= abs(right):
            self.target = [push(fwd), 0, 0, 0]
        else:
            self.target = [0, push(right), 0, 0]
        self.push0 = list(self.target)


class Autopilot(Node):
    def __init__(self):
        super().__init__("autopilot")
        self.declare_parameter("episodes", 216)         # how many to record in this run
        self.declare_parameter("seed", 0)               # 0 = different every run
        seed = self.get_parameter("seed").value
        self.rng = random.Random(seed or None)
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
        self.create_timer(self.dt, self.tick)
        self.get_logger().info("autopilot ready")

    def on_tel(self, m):
        self.tel = json.loads(m.data)

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
                self.pilot = Pilot(self.rng)
                self.go("fly")
            elif age > 5:
                self.rec_pub.publish(String(data="start"))
                self.go("rec_wait")
        elif self.state == "fly":
            fwd, right, vf, vr = self.pad_offset()
            s = self.pilot.step(t, self.dt, fwd, right, vf, vr, tel["sim_z"])
            if age > TIMEOUT_S and not self.pilot.landing:
                self.pilot.phase, self.pilot.landing = "land", True
            self.send_sticks(s)
            if tel["on_ground"]:
                self.send_sticks([0, 0, 0, 0])
                self.go("wait_stop")
        elif self.state == "wait_stop":
            self.send_sticks([0, 0, 0, 0])
            if not st["recording"] and st["waiting_for_result"]:
                fwd, right, _, _ = self.pad_offset()
                ok = abs(fwd) < PAD_HALF - 0.05 and abs(right) < PAD_HALF - 0.05
                check_ok = "CHECK" not in st.get("message", "")
                ok = ok and check_ok
                self.done += 1
                self.ok_count += ok
                self.tries = 0 if ok else self.tries + 1
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
    rclpy.init()
    rclpy.spin(Autopilot())


if __name__ == "__main__":
    main()
