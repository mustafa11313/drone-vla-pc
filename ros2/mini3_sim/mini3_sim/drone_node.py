"""
drone_node  -  flies the Gazebo drone like a DJI Mini 3.

Subscribes
    /drone/sticks    std_msgs/Float32MultiArray  [pitch, roll, throttle, yaw] in stick percent
                     (-100..100, like pitch_cmd ... yaw_cmd in your telemetry CSV)
    /drone/command   std_msgs/String  "takeoff" | "land"
    /drone/odometry  nav_msgs/Odometry (from Gazebo: where the drone really is)
Publishes
    /drone/cmd_vel   geometry_msgs/Twist (to Gazebo, 50 Hz)
    /drone/telemetry std_msgs/String, JSON with the same fields as your telemetry CSV (20 Hz)
    /drone/event     std_msgs/String  "took_off" | "landing" | "landed" | "hit_net"

Behaviour, as on the real drone with the phone app:
    landed       motors off, sticks ignored; "takeoff" -> auto takeoff to 1.2 m
    flying       sticks -> speeds through the Mini 3 model fitted from your flights
                 (dynamics.py: stick curve, response time 0.75 s)
    auto landing below 0.6 m with the throttle stick down, or "land": straight down, sticks ignored
    the net      closer than 13 cm to a side or the top: "hit_net", and the drone is stopped
"""
import json
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from mini3_sim.dynamics import Mini3, Params

TAKEOFF_HEIGHT = 1.2     # m, DJI auto takeoff
LAND_HEIGHT = 0.6        # m, below this with throttle down the drone lands by itself
CAGE_HALF, CAGE_TOP = 1.0, 2.0
ROOF_SAG = 0.15         # m, the roof net hangs down in the middle (tools/make_net.py)
DRONE_RADIUS = 0.13      # m, centre to propeller tips
FLOOR = 0.03             # m, height of the drone's centre when it stands on the floor


def net_wall(along, z):
    """distance of a net wall from the cage centre: the net sags inwards up to 7 cm mid-wall
    (same shape as tools/make_net.py, which draws it)."""
    a = max(-1.0, min(1.0, along / CAGE_HALF))
    return CAGE_HALF - 0.06 * (1 - a * a) * math.sin(math.pi * max(0.0, min(z, CAGE_TOP)) / CAGE_TOP) - 0.01


M_TO_FT, MS_TO_MPH = 3.28084, 2.23694


class DroneNode(Node):
    def __init__(self):
        super().__init__("drone_node")
        self.model = Mini3(Params.load())
        self.sticks = [0.0, 0.0, 0.0, 0.0]
        self.mode = "landed"
        self.have_odom = False
        self.touching_net = False
        self.prev_fwd = self.prev_right = 0.0
        self.tilt = [0.0, 0.0]
        self.t0 = None
        self.cmd_pub = self.create_publisher(Twist, "/drone/cmd_vel", 10)
        self.tel_pub = self.create_publisher(String, "/drone/telemetry", 10)
        self.evt_pub = self.create_publisher(String, "/drone/event", 10)
        self.create_subscription(Float32MultiArray, "/drone/sticks", self.on_sticks, 10)
        self.create_subscription(String, "/drone/command", self.on_command, 10)
        self.create_subscription(Odometry, "/drone/odometry", self.on_odom, 10)
        self.dt = 0.02
        self.create_timer(self.dt, self.tick)
        self.create_timer(0.05, self.publish_telemetry)   # 20 Hz, so each 10 Hz recorder row is fresh
        self.get_logger().info("drone ready: send 'takeoff' on /drone/command (key T in the keyboard window)")

    # ------------------------------------------------------------------ inputs
    def on_sticks(self, m):
        if len(m.data) >= 4:
            self.sticks = [max(-100.0, min(100.0, float(v))) for v in m.data[:4]]

    def on_command(self, m):
        c = m.data.strip().lower()
        if c == "takeoff" and self.mode == "landed":
            self.mode = "takeoff"
            self.event("taking_off")
        elif c == "land" and self.mode in ("flying", "takeoff"):
            self.start_landing()

    def on_odom(self, m):
        s = self.model.s
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        s.x, s.y, s.z = p.x, p.y, p.z
        s.heading = (90.0 - math.degrees(yaw)) % 360.0        # Gazebo yaw (from east, ccw) -> heading
        self.have_odom = True

    # ------------------------------------------------------------------ flight logic
    def event(self, name):
        self.evt_pub.publish(String(data=name))
        self.get_logger().info(name)

    def start_landing(self):
        self.mode = "landing"
        self.event("landing")

    def tick(self):
        if not self.have_odom:
            return
        s = self.model.s
        if self.mode == "landed":
            s.vx = s.vy = s.vz = s.yaw_rate = 0.0
        elif self.mode == "takeoff":
            self.model.step([0, 0, 0, 0], self.dt)
            s.vz = 0.8 if s.z < TAKEOFF_HEIGHT - 0.05 else 0.0
            if s.z >= TAKEOFF_HEIGHT - 0.05:
                self.mode = "flying"
                self.event("took_off")
        elif self.mode == "flying":
            if self.sticks[2] < -5 and s.z < LAND_HEIGHT:
                self.start_landing()
            self.model.step(self.sticks, self.dt)
            if s.z < FLOOR + 0.05 and s.vz < 0:
                s.vz = 0.0
        if self.mode == "landing":
            self.model.step([0, 0, 0, 0], self.dt)
            s.vz = -0.5
            if s.z <= FLOOR + 0.01:
                s.vx = s.vy = s.vz = s.yaw_rate = 0.0
                self.mode = "landed"
                self.event("landed")
        self.keep_off_net()
        self.send_velocity()

    def keep_off_net(self):
        s = self.model.s
        gaps = {"x": net_wall(s.y, s.z) - abs(s.x), "y": net_wall(s.x, s.z) - abs(s.y),
                "z": CAGE_TOP - ROOF_SAG * (1 - s.x ** 2) * (1 - s.y ** 2) - s.z}
        gap = min(gaps.values())
        if gap < DRONE_RADIUS + 0.005 and not self.touching_net:      # props touch the net
            self.touching_net = True
            self.event("hit_net")
        elif gap > DRONE_RADIUS + 0.03:                                # clearly away again
            self.touching_net = False
        # the net stops the drone: no speed further into it. The speed towards the net is capped so
        # that the drone can close at most the remaining gap in 0.1 s, which stops it on the net
        # without overshooting through it between two odometry messages.
        def cap(v, toward, g):
            room = max(0.0, g - DRONE_RADIUS) / 0.1
            return math.copysign(min(abs(v), room), v) if v * toward > 0 else v
        s.vx = cap(s.vx, s.x or 1.0, gaps["x"])
        s.vy = cap(s.vy, s.y or 1.0, gaps["y"])
        s.vz = cap(s.vz, 1.0, gaps["z"])

    def send_velocity(self):
        s = self.model.s
        h = math.radians(s.heading)
        fwd = s.vx * math.sin(h) + s.vy * math.cos(h)
        right = s.vx * math.cos(h) - s.vy * math.sin(h)
        # a tilt like the real drone's, from the acceleration (for the telemetry only)
        a_fwd, a_right = (fwd - self.prev_fwd) / self.dt, (right - self.prev_right) / self.dt
        self.prev_fwd, self.prev_right = fwd, right
        self.tilt[0] += 0.1 * (max(-25, min(25, -6.0 * a_fwd)) - self.tilt[0])
        self.tilt[1] += 0.1 * (max(-25, min(25, 6.0 * a_right)) - self.tilt[1])
        t = Twist()
        t.linear.x, t.linear.y, t.linear.z = fwd, -right, s.vz     # Gazebo body frame: x forward, y left
        t.angular.z = -math.radians(s.yaw_rate)                    # heading turns clockwise
        self.cmd_pub.publish(t)

    def publish_telemetry(self):
        if not self.have_odom:
            return
        s = self.model.s
        now = self.get_clock().now().nanoseconds / 1e6
        if self.t0 is None:
            self.t0 = now
        flying = self.mode != "landed"
        tel = {
            "time_ms": int(now - self.t0),
            "latitude": 0.0, "longitude": 0.0,
            "height_above_takeoff": round(max(0.0, s.z - FLOOR) * M_TO_FT, 6),
            "x_speed": round(s.vy * MS_TO_MPH, 6),       # NED: north
            "y_speed": round(s.vx * MS_TO_MPH, 6),       # east
            "z_speed": round(-s.vz * MS_TO_MPH, 6),      # down
            "heading": round(s.heading, 1) % 360.0,           # compass 0..360, as the app writes it
            "drone_pitch": round(self.tilt[0], 1), "drone_roll": round(self.tilt[1], 1),
            "pitch_cmd": int(round(self.sticks[0])) if flying else 0,
            "roll_cmd": int(round(self.sticks[1])) if flying else 0,
            "throttle_cmd": int(round(self.sticks[2])) if flying else 0,
            "yaw_cmd": int(round(self.sticks[3])) if flying else 0,
            "gimbal_heading": round(s.heading, 1) % 360.0,
            "gimbal_pitch": -50.0, "gimbal_roll": 0.0,
            # DJI FlightMode names; on the ground the real drone reports GPS_NORMAL too
            "flight_mode": {"landed": "GPS_NORMAL", "takeoff": "AUTO_TAKE_OFF", "flying": "GPS_NORMAL",
                            "landing": "AUTO_LANDING"}[self.mode],
            # extra fields for the simulator (not in the real CSV)
            "sim_x": round(s.x, 3), "sim_y": round(s.y, 3), "sim_z": round(s.z, 3),
            "touching_net": self.touching_net,
            "on_ground": self.mode == "landed",
        }
        self.tel_pub.publish(String(data=json.dumps(tel)))


def main():
    rclpy.init()
    rclpy.spin(DroneNode())


if __name__ == "__main__":
    main()
