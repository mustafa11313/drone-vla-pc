"""
keyboard  -  fly the simulated Mini 3 with the keyboard, laid out like the DJI controller (Mode 2).

A small window opens; click on it once so it has the keyboard, then:

    W / S        left stick up / down       climb / descend
    A / D        left stick left / right    turn left / right
    Up / Down    right stick up / down      forward / back
    Left / Right right stick left / right   left / right
    T            take off (DJI auto takeoff to 1.2 m)
    L            land (DJI auto landing)
    1 / 2 / 3    stick strength: gentle 30 % / normal 60 % / full 100 %
    Space        let go of all sticks (hover)

Holding a key pushes the stick gradually, like a thumb on a real stick; letting go returns it to
the centre. The sticks are published on /drone/sticks at 20 Hz in stick percent (-100..100), the
same values your telemetry CSV logs as pitch_cmd, roll_cmd, throttle_cmd and yaw_cmd. Holding S
below 0.6 m starts the auto landing, as with the real app.
"""
import json
import time
import tkinter as tk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

AXES = {  # key -> (axis index, direction); axes: 0 pitch, 1 roll, 2 throttle, 3 yaw
    "w": (2, 1), "s": (2, -1), "a": (3, -1), "d": (3, 1),
    "Up": (0, 1), "Down": (0, -1), "Left": (1, -1), "Right": (1, 1),
}
STRENGTH = {"1": 30.0, "2": 60.0, "3": 100.0}
RISE, FALL = 250.0, 400.0          # stick percent per second while held / after letting go
RATE = 20                          # Hz


class Keyboard(Node):
    def __init__(self):
        super().__init__("keyboard")
        self.pub = self.create_publisher(Float32MultiArray, "/drone/sticks", 10)
        self.cmd = self.create_publisher(String, "/drone/command", 10)
        self.create_subscription(String, "/drone/telemetry", self.on_tel, 10)
        self.create_subscription(String, "/drone/event", self.on_event, 10)
        self.tel, self.last_event = {}, ""

    def on_tel(self, m):
        self.tel = json.loads(m.data)

    def on_event(self, m):
        self.last_event = m.data


class App:
    def __init__(self, node):
        self.node = node
        self.sticks = [0.0, 0.0, 0.0, 0.0]
        self.held = {}                 # key -> time pressed
        self.release_pending = {}
        self.strength = 60.0
        self.root = tk.Tk()
        self.root.title("Mini 3 keyboard")
        self.root.configure(bg="#202326")
        self.canvas = tk.Canvas(self.root, width=440, height=210, bg="#202326", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(10, 0))
        self.info = tk.Label(self.root, font=("Monospace", 11), fg="#e8e8e8", bg="#202326", justify="left")
        self.info.pack(padx=10, pady=6, anchor="w")
        help_text = ("W/S climb/descend   A/D turn   arrows forward/back/left/right\n"
                     "T take off   L land   1/2/3 stick strength   Space let go")
        tk.Label(self.root, text=help_text, font=("Monospace", 9), fg="#9aa0a6", bg="#202326",
                 justify="left").pack(padx=10, pady=(0, 10), anchor="w")
        self.root.bind("<KeyPress>", self.on_press)
        self.root.bind("<KeyRelease>", self.on_release)
        self.root.after(int(1000 / RATE), self.loop)
        self.last = time.time()

    def on_press(self, e):
        k = e.keysym if e.keysym in AXES else e.keysym.lower()
        self.release_pending.pop(k, None)          # key auto-repeat: press right after release
        if k in AXES:
            self.held.setdefault(k, time.time())
        elif k == "t":
            self.node.cmd.publish(String(data="takeoff"))
        elif k == "l":
            self.node.cmd.publish(String(data="land"))
        elif k in STRENGTH:
            self.strength = STRENGTH[k]
        elif k == "space":
            self.held.clear()
            self.sticks = [0.0, 0.0, 0.0, 0.0]

    def on_release(self, e):
        k = e.keysym if e.keysym in AXES else e.keysym.lower()
        if k in self.held:
            self.release_pending[k] = time.time()   # confirmed in loop() unless the key comes back

    def loop(self):
        now = time.time()
        dt, self.last = now - self.last, now
        for k, t in list(self.release_pending.items()):
            if now - t > 0.06:
                self.held.pop(k, None)
                self.release_pending.pop(k, None)
        target = [0.0, 0.0, 0.0, 0.0]
        for k in self.held:
            axis, sign = AXES[k]
            target[axis] += sign * self.strength
        for i in range(4):
            step = (RISE if abs(target[i]) > abs(self.sticks[i]) else FALL) * dt
            d = target[i] - self.sticks[i]
            self.sticks[i] += max(-step, min(step, d))
        self.node.pub.publish(Float32MultiArray(data=[float(v) for v in self.sticks]))
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self.draw()
        self.root.after(int(1000 / RATE), self.loop)

    def draw(self):
        c = self.canvas
        c.delete("all")
        p, r, th, y = self.sticks
        for cx, sx, sy, title in ((110, y, th, "left: throttle / yaw"), (330, r, p, "right: pitch / roll")):
            c.create_rectangle(cx - 80, 20, cx + 80, 180, outline="#5f6368", width=2)
            c.create_line(cx, 20, cx, 180, fill="#3c4043")
            c.create_line(cx - 80, 100, cx + 80, 100, fill="#3c4043")
            px, py = cx + sx / 100 * 80, 100 - sy / 100 * 80
            c.create_oval(px - 12, py - 12, px + 12, py + 12, fill="#8ab4f8", outline="")
            c.create_text(cx, 196, text=title, fill="#9aa0a6", font=("Monospace", 9))
        t = self.node.tel
        h = t.get("height_above_takeoff", 0.0) / 3.28084
        self.info.config(text=(f"mode {t.get('flight_mode', 'waiting for the simulator'):13s} height {h:4.2f} m   "
                               f"heading {t.get('heading', 0):6.1f}\n"
                               f"sticks  pitch {p:5.0f}  roll {r:5.0f}  throttle {th:5.0f}  yaw {y:5.0f}   "
                               f"strength {self.strength:.0f} %\n"
                               f"last event: {self.node.last_event or '-'}"
                               + ("   NET!" if t.get("touching_net") else "")))

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    App(Keyboard()).run()


if __name__ == "__main__":
    main()
