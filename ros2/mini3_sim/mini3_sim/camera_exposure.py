"""
camera_exposure  -  the Mini 3 camera's auto-exposure and colour, applied to Gazebo's picture.

Gazebo renders the scene with fixed exposure, so the black mats come out almost black. The real
drone camera brightens the picture until it is about mid-grey, which is why the mats look
grey-beige in your real frames, and it renders warm. This node does the same:

    /drone/camera/raw  (from Gazebo)  ->  /drone/camera/image_raw  (what everything else uses)

The gain follows the scene smoothly (like a real camera, it takes about half a second to adapt),
bright parts are compressed instead of clipped, and a warm white balance is applied.

Scenario episodes vary the look a little per episode (brightness, white balance, softness,
sensor noise), sent by the recorder as JSON on /drone/camera/look, so a model trained on them does
not learn one fixed simulator look.
"""
import json

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

WHITE_BALANCE = np.array([1.10, 1.00, 0.93], np.float32)   # tuned so the mats match the real frames (Nov 2026)
SOFTEN = 0.6   # the Mini 3 picture is softer than Gazebo's (compression, small lens): blur weight


class CameraExposure(Node):
    def __init__(self):
        super().__init__("camera_exposure")
        self.declare_parameter("target", 165.0)      # mean grey level the camera aims for (real frames: 155-190)
        self.declare_parameter("max_gain", 6.0)
        self.declare_parameter("adapt", 0.25)         # share of the way to the new gain per frame
        self.gain = 2.0
        self.wb, self.soften, self.noise, self.target = WHITE_BALANCE, SOFTEN, 0.0, None
        self.rng = np.random.default_rng()
        self.create_subscription(String, "/drone/camera/look", self.on_look, 10)
        self.pub = self.create_publisher(Image, "/drone/camera/image_raw", 10)
        self.create_subscription(Image, "/drone/camera/raw", self.on_image, 10)

    def on_look(self, m):
        d = json.loads(m.data)
        self.wb = np.array(d.get("wb", WHITE_BALANCE), np.float32)
        self.soften = float(d.get("soften", SOFTEN))
        self.noise = float(d.get("noise", 0.0))
        self.target = d.get("target")

    def on_image(self, m):
        if m.encoding not in ("rgb8", "bgr8"):
            self.pub.publish(m)
            return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).astype(np.float32)
        if m.encoding == "bgr8":
            a = a[..., ::-1]
        target = self.target or self.get_parameter("target").value
        lum = float((a @ np.array([0.299, 0.587, 0.114], np.float32)).mean())
        want = min(self.get_parameter("max_gain").value, max(0.7, target / max(lum, 1.0)))
        self.gain += self.get_parameter("adapt").value * (want - self.gain)
        # soften: blend with a 3x3 box blur (real frames have about half the edge strength)
        p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blur = sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0
        a = a + self.soften * (blur - a)
        v = a * self.gain * self.wb
        if self.noise > 0:
            v = v + self.rng.normal(0.0, self.noise, v.shape).astype(np.float32)
        knee = 170.0
        v = np.where(v > knee, knee + (255 - knee) * np.tanh((v - knee) / (255 - knee)), v)
        out = Image()
        out.header = m.header
        out.height, out.width, out.encoding, out.is_bigendian = m.height, m.width, "rgb8", 0
        out.step = m.width * 3
        out.data = np.clip(v, 0, 255).astype(np.uint8).tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(CameraExposure())


if __name__ == "__main__":
    main()
