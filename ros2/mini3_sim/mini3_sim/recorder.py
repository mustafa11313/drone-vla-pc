"""
recorder  -  records simulator episodes in exactly the format the phone app writes.

It copies the app's recording code (TelemetryRecorder.kt, FrameRecorder.kt, EpisodePlan.kt,
EpisodeCheck.kt and RecordingController.kt in dji-mini3-vla-app):

    <out_dir>/episode_0007_20261102_221500/
        0007_instruction.txt        "Take off, locate the red landing pad, and land on it."
        0007_setup.json             plan number, pad and drone grid cells, gimbal, speed mode
        0007_telemetry_20261102_221500.csv   the same 18 columns, units and number format
        frames/0007_000000.jpg ...  one 1280 x 720 JPEG (quality 90) per CSV row, 10 Hz
        0007_stick_check.txt        as the app writes it
        0007_phone_check.txt        the app's after-landing check (rows = frames, gaps, length...)
        0007_result.txt             "success" or "fail", when you press O or F

How an episode goes (as with the real drone):

    1. The pad and the drone are placed at the cells of the current plan setup (3 x 3 grid of the
       2 x 2 m cage, the same 72 pad/drone combinations as the app's plan).
    2. T takes off. R starts recording (the record button on the controller).
    3. Fly to the pad and land (L, or hold S low). Recording stops by itself after touchdown,
       and the episode is checked.
       The drone faces the pad, as in the real episodes.
    4. O marks it a success: the next setup of the plan is placed.
       F marks it a fail: the same setup is placed again so you can fly it again.

Commands on /drone/record (std_msgs/String): toggle, start, stop, success, fail, place.
Status on /drone/recorder/status (JSON String), shown in the keyboard window.

Differences from the real recording, on purpose:
  - Rows are 100 ms apart in simulated time (Gazebo may run slower than real time on a laptop).
  - The Gazebo camera renders 640 x 360 (fast enough on a laptop without a GPU); each frame is
    scaled up to 1280 x 720 before saving, so the files have the real size.
  - latitude / longitude are 0, as the app writes indoors without GPS.
  - The plan uses its own fixed shuffle (seed 2026), so its order differs from the app's.
"""
import json
import os
import random
import subprocess
import threading
from datetime import datetime
from collections import deque
from queue import Queue

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

HEADER = ("time_ms,latitude,longitude,height_above_takeoff,x_speed,y_speed,z_speed,"
          "heading,drone_pitch,drone_roll,pitch_cmd,roll_cmd,throttle_cmd,yaw_cmd,"
          "gimbal_heading,gimbal_pitch,gimbal_roll,flight_mode")
INSTRUCTION = "Take off, locate the red landing pad, and land on it."
NAMES = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
GIMBAL_PITCH_DEG = -50.0
PLAN_SIZE = 216               # 72 layouts x 3 repeats
STOP_AFTER_LANDED_S = 0.7     # real episodes go on ~0.7 s after touchdown until the camera stops
FRAME_MAX_AGE_S = 0.1         # FrameRecorder.MAX_AGE_MS
MAX_FRAME_LAG_S = 3.0         # wait at most this long (sim time) for a frame still being rendered
SAVE_SIZE = (1280, 720)
JPEG_QUALITY = 90


def make_plan(seed=2026):
    """72 (pad, drone) layouts, every block of 72 shuffled, 3 repeats (EpisodePlan.setups)."""
    rnd = random.Random(seed)
    out = []
    for rep in (1, 2, 3):
        block = [(p, d) for p in range(9) for d in range(9) if p != d]
        rnd.shuffle(block)
        out += [{"number": len(out) + i + 1, "pad": p, "drone": d, "repeat": rep}
                for i, (p, d) in enumerate(block)]
    return out


def cell_xy(i, step):
    """grid cell -> (x east, y north) of its centre; row 0 is the north row."""
    return ((i % 3) - 1) * step, (1 - i // 3) * step


def f6(x):
    return f"{x + 0.0:.6f}"      # + 0.0 turns -0.0 into 0.0


def f1(x):
    return f"{x:.1f}"


class Recorder(Node):
    def __init__(self):
        super().__init__("recorder")
        self.declare_parameter("out_dir", os.path.expanduser("~/mini3_sim_episodes"))
        self.declare_parameter("instruction", INSTRUCTION)
        self.declare_parameter("grid_step", 2.0 / 3.0)       # m between grid cell centres
        self.declare_parameter("heading_jitter_deg", 5.0)    # the pilot points the nose at the pad by eye
        self.declare_parameter("world", "cage_lab")
        self.declare_parameter("place_on_start", True)
        self.out_dir = self.get_parameter("out_dir").value
        os.makedirs(self.out_dir, exist_ok=True)
        self.plan = make_plan()
        self.progress_file = os.path.join(self.out_dir, "plan_progress.json")
        self.next_index = self.load_progress()

        self.tel = None
        self.frame = None            # newest (rgb array, sim time it was taken)
        self.frames = deque(maxlen=90)   # recent (sim time, rgb), so each row gets the right frame
        self.tels = deque(maxlen=200)     # recent (sim time, telemetry)
        self.lock = threading.Lock()
        self.recording = False
        self.ep_dir = None
        self.last_dir = None         # finished episode waiting for success / fail
        self.stop_at = None
        self.last_check = ""
        self.message = ""
        self.placed = False
        self.placed_seq = 0          # counts placements, so the autopilot knows a new setup is ready
        self.pad_xy = None
        self.saver_q = Queue(maxsize=4)
        threading.Thread(target=self.saver, daemon=True).start()

        self.create_subscription(Image, "/drone/camera/image_raw", self.on_image, 5)
        self.create_subscription(String, "/drone/telemetry", self.on_tel, 10)
        self.create_subscription(String, "/drone/event", self.on_event, 10)
        self.create_subscription(String, "/drone/record", self.on_command, 10)
        self.status_pub = self.create_publisher(String, "/drone/recorder/status", 10)
        self.tick_timer = None
        self.create_timer(0.5, self.publish_status)
        self.create_timer(0.1, self.housekeeping)
        self.get_logger().info(f"episodes go to {self.out_dir}; {self.describe()}")

    # ------------------------------------------------------------------ plan
    def load_progress(self):
        try:
            return int(json.load(open(self.progress_file))["next_index"])
        except Exception:
            return 0

    def save_progress(self):
        json.dump({"next_index": self.next_index}, open(self.progress_file, "w"))

    def current(self):
        return self.plan[self.next_index] if self.next_index < PLAN_SIZE else None

    def describe(self):
        s = self.current()
        if s is None:
            return f"plan finished: all {PLAN_SIZE} episodes done"
        e = (s["pad"] % 3) - (s["drone"] % 3)
        n = (s["drone"] // 3) - (s["pad"] // 3)
        parts = ([f"{abs(n)} {'north' if n > 0 else 'south'}"] if n else []) + \
                ([f"{abs(e)} {'east' if e > 0 else 'west'}"] if e else [])
        return (f"episode {s['number']}/{PLAN_SIZE} (repeat {s['repeat']}): pad {NAMES[s['pad']]}, "
                f"drone {NAMES[s['drone']]}, pad is {', '.join(parts)}")

    def place(self):
        """Move the pad and the drone (on the ground) to the current setup's cells."""
        s = self.current()
        if s is None:
            return
        if self.tel is not None and not self.tel.get("on_ground", True):
            self.message = "land first: the setup is placed only while the drone is on the ground"
            return
        step = self.get_parameter("grid_step").value
        px, py = cell_xy(s["pad"], step)
        dx, dy = cell_xy(s["drone"], step)
        # As in the real episodes, the drone starts facing the pad, so the camera sees it after
        # takeoff. The pilot aims by eye, so a few degrees of error are added.
        jitter = self.get_parameter("heading_jitter_deg").value
        self.start_heading = (np.degrees(np.arctan2(px - dx, py - dy))
                              + random.uniform(-jitter, jitter)) % 360.0
        # Gazebo yaw is counter-clockwise from east; heading is clockwise from north
        yaw = np.radians(90.0 - self.start_heading)
        ok = self.set_pose("pad", px, py, 0.0, 0.0) and self.set_pose("mini3", dx, dy, 0.05, yaw)
        if ok:
            self.pad_xy = (px, py)
            self.placed_seq += 1
        self.message = ("placed: " if ok else "could not move the models: ") + self.describe()
        self.get_logger().info(self.message)

    def set_pose(self, name, x, y, z, yaw):
        req = (f'name: "{name}", position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}, '
               f'orientation: {{x: 0, y: 0, z: {np.sin(yaw / 2):.6f}, w: {np.cos(yaw / 2):.6f}}}')
        world = self.get_parameter("world").value
        for tool in (["ign", "service"], ["gz", "service"]):
            try:
                r = subprocess.run(tool + ["-s", f"/world/{world}/set_pose", "--reqtype", "ignition.msgs.Pose",
                                           "--reptype", "ignition.msgs.Boolean", "--timeout", "3000",
                                           "--req", req], capture_output=True, text=True, timeout=10)
                if "true" in r.stdout:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return False

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ inputs
    def on_image(self, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == "bgr8":
            a = a[..., ::-1]
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            self.frame = (a, stamp or self.now())
            self.frames.append((self.frame[1], a))

    def on_tel(self, m):
        self.tel = json.loads(m.data)
        with self.lock:
            self.tels.append((self.now(), self.tel))

    def on_event(self, m):
        if m.data == "landed" and self.recording:
            self.stop_at = self.now() + STOP_AFTER_LANDED_S

    def on_command(self, m):
        c = m.data.strip().lower()
        if c == "toggle":
            c = "stop" if self.recording else "start"
        if c == "start":
            self.start()
        elif c == "stop" and self.recording and self.stop_at is None:
            self.stop_at = self.now()          # rows up to now are still written, then it stops
        elif c in ("success", "fail"):
            self.finish(c == "success")
        elif c == "place":
            self.place()

    # ------------------------------------------------------------------ recording
    def start(self):
        if self.recording:
            return
        if self.tel is None or self.frame is None:
            self.message = "no telemetry or camera yet: is the simulator running?"
            return
        if self.last_dir is not None:
            self.message = "the last episode has no result yet: press O (success) or F (fail) first"
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        last = max([int(d.split("_")[1]) for d in os.listdir(self.out_dir)
                    if d.startswith("episode_") and d.split("_")[1].isdigit()] or [0])
        self.ep_no = f"{last + 1:04d}"
        self.ep_dir = os.path.join(self.out_dir, f"episode_{self.ep_no}_{stamp}")
        os.makedirs(os.path.join(self.ep_dir, "frames"))
        open(os.path.join(self.ep_dir, f"{self.ep_no}_instruction.txt"), "w").write(
            self.get_parameter("instruction").value)
        self.write_setup(stamp)
        self.csv = open(os.path.join(self.ep_dir, f"{self.ep_no}_telemetry_{stamp}.csv"), "w")
        self.csv.write(HEADER + "\n")
        self.rows = 0
        self.frame_idx = 0
        self.max_stick = 0
        self.first_row_skipped = False
        self.t0 = self.now()
        self.stop_at = None
        self.recording = True
        self.next_tick = self.t0
        self.message = f"recording episode {self.ep_no}"
        self.get_logger().info(f"recording {self.ep_dir}")

    def write_setup(self, stamp):
        s = self.current()
        d = {"started": stamp}
        if s is not None:
            d.update({"plan_number": s["number"], "repeat": s["repeat"], "pad": NAMES[s["pad"]],
                      "drone": NAMES[s["drone"]], "pad_east_grid": (s["pad"] % 3) - (s["drone"] % 3),
                      "pad_north_grid": (s["drone"] // 3) - (s["pad"] // 3)})
        d.update({"gimbal_pitch_target": GIMBAL_PITCH_DEG, "speed_mode": "Normal", "start_heading": "north",
                  "start_heading_deg": round(getattr(self, "start_heading", 0.0), 1),
                  "simulator": True})
        with open(os.path.join(self.ep_dir, f"{self.ep_no}_setup.json"), "w") as f:
            json.dump(d, f, indent=2)
            f.write("\n")

    def housekeeping(self):
        if self.placed is False and self.tel is not None and self.get_parameter("place_on_start").value:
            self.placed = True
            threading.Thread(target=self.place, daemon=True).start()
        if not self.recording:
            return
        # a row every 100 ms of simulated time (the app: every 100 ms). Simulated, not wall time,
        # so the data is right even when the laptop runs Gazebo slower than real time.
        now = self.now()
        # Camera frames reach ROS later than the simulated moment they show (rendering takes time:
        # about 0.75 s with the Gazebo window open on a laptop). A row is therefore written only
        # once a frame taken at or after its tick has arrived, so no row misses its frame.
        end = self.stop_at if self.stop_at is not None else now
        with self.lock:
            newest = self.frames[-1][0] if self.frames else float("-inf")
        while (self.recording and self.next_tick <= end
               and (newest >= self.next_tick or now - self.next_tick > MAX_FRAME_LAG_S)):
            self.write_row(self.next_tick)
            self.next_tick += 0.1
        if self.stop_at is not None and self.next_tick > self.stop_at:
            self.stop()

    @staticmethod
    def newest_before(items, tick):
        best = None
        for stamp, v in items:
            if stamp <= tick + 1e-6:
                best = (stamp, v)
        return best

    def write_row(self, tick):
        with self.lock:
            tl = self.newest_before(self.tels, tick)
            fr = self.newest_before(self.frames, tick)
        if tl is None:
            return
        t = tl[1]
        for k in ("pitch_cmd", "roll_cmd", "throttle_cmd", "yaw_cmd"):
            self.max_stick = max(self.max_stick, abs(int(t[k])))
        time_ms = int(round((tick - self.t0) * 1000))
        if not self.first_row_skipped:          # TelemetryRecorder skips the very first row
            self.first_row_skipped = True
            return
        # FrameRecorder.onTick: only write the row if a fresh frame can be saved for it
        if fr is None or tick - fr[0] > FRAME_MAX_AGE_S or self.saver_q.full():
            return
        name = f"{self.ep_no}_{self.frame_idx:06d}.jpg"
        self.frame_idx += 1
        self.saver_q.put((os.path.join(self.ep_dir, "frames", name), fr[1]))
        row = [str(time_ms), "0.00000000", "0.00000000",
               f6(t["height_above_takeoff"]), f6(t["x_speed"]), f6(t["y_speed"]), f6(t["z_speed"]),
               f1(t["heading"]), f1(t["drone_pitch"]), f1(t["drone_roll"]),
               str(int(t["pitch_cmd"])), str(int(t["roll_cmd"])), str(int(t["throttle_cmd"])),
               str(int(t["yaw_cmd"])),
               f1(t["gimbal_heading"]), f1(t["gimbal_pitch"]), f1(t["gimbal_roll"]), t["flight_mode"]]
        self.csv.write(",".join(row) + "\n")
        self.rows += 1
        if self.rows % 10 == 0:
            self.csv.flush()

    def saver(self):
        while True:
            path, rgb = self.saver_q.get()
            img = cv2.resize(rgb, SAVE_SIZE, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(path, img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            self.saver_q.task_done()

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        self.saver_q.join()
        self.csv.close()
        raw = int(round(self.max_stick * 6.6))
        open(os.path.join(self.ep_dir, f"{self.ep_no}_stick_check.txt"), "w").write(
            f"max |raw stick| seen: {raw} (expected 660 at full deflection)\n")
        self.last_check = self.check(self.ep_dir)
        self.last_dir = self.ep_dir
        self.message = f"episode {self.ep_no} stopped: {self.last_check}. O = success, F = fail"
        self.get_logger().info(self.message)

    def check(self, d):
        """EpisodeCheck.kt: rows = frames, gaps, flown part >= 3 s, camera angle, stick range."""
        import csv
        problems, notes = [], []
        name = [f for f in os.listdir(d) if "_telemetry_" in f and f.endswith(".csv")][0]
        rows = list(csv.DictReader(open(os.path.join(d, name))))
        frames = len([f for f in os.listdir(os.path.join(d, "frames")) if f.endswith(".jpg")])
        if frames != len(rows):
            problems.append(f"rows {len(rows)} != frames {frames}")
        if len(rows) < 2:
            problems.append("telemetry CSV is empty")
        else:
            t = [int(r["time_ms"]) for r in rows]
            gaps = np.diff(t)
            if gaps.max() > 500:
                problems.append(f"gap of {gaps.max()} ms")
            elif (gaps > 150).any():
                notes.append(f"{int((gaps > 150).sum())} small gaps (max {gaps.max()} ms)")
            sticks = [[int(r[k]) for k in ("pitch_cmd", "roll_cmd", "throttle_cmd", "yaw_cmd")] for r in rows]
            start = next((i for i, s in enumerate(sticks) if any(s)), 0)
            end = next((i for i, r in enumerate(rows) if r["flight_mode"] == "AUTO_LANDING"), len(rows))
            if end == len(rows):
                notes.append("no AUTO_LANDING: the whole flight is kept")
            flown = max(0, end - start)
            if flown < 30:
                problems.append(f"only {flown} flown rows ({flown / 10:.1f} s)")
            notes.append(f"{len(rows)} rows, flown {flown / 10:.1f} s")
        summary = ("OK" if not problems else "CHECK: " + "; ".join(problems)) + \
            (" (" + "; ".join(notes) + ")" if notes else "")
        open(os.path.join(d, f"{self.ep_no}_phone_check.txt"), "w").write(
            f"{os.path.basename(d)} | {summary}\n")
        return summary

    def finish(self, ok):
        if self.recording:
            self.message = "still recording: land first (or R to stop)"
            return
        if self.last_dir is None:
            self.message = "no finished episode to mark"
            return
        no = os.path.basename(self.last_dir).split("_")[1]
        open(os.path.join(self.last_dir, f"{no}_result.txt"), "w").write("success\n" if ok else "fail\n")
        self.last_dir = None
        if ok:
            self.next_index += 1
            self.save_progress()
        self.message = f"episode {no} marked {'success' if ok else 'fail'}"
        threading.Thread(target=self.place, daemon=True).start()

    # ------------------------------------------------------------------ status
    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps({
            "recording": self.recording,
            "episode": os.path.basename(self.ep_dir) if self.ep_dir else "",
            "rows": self.rows if self.ep_dir else 0,
            "waiting_for_result": self.last_dir is not None,
            "plan": self.describe(),
            "message": self.message,
            "placed_seq": self.placed_seq,
            "pad_xy": self.pad_xy,
            "plan_done": self.current() is None,
        })))


def main():
    rclpy.init()
    rclpy.spin(Recorder())


if __name__ == "__main__":
    main()
