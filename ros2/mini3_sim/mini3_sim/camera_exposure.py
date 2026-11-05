"""
camera_exposure  -  the Mini 3 camera's auto-exposure and colour, applied to Gazebo's picture.

Gazebo renders the scene with fixed exposure, so the black mats come out almost black. The real
drone camera brightens the picture until it is about mid-grey, which is why the mats look
grey-beige in your real frames, and it renders warm. This node does the same:

    /drone/camera/raw  (from Gazebo)  ->  /drone/camera/image_raw  (what everything else uses)

The gain follows the scene smoothly (like a real camera, it takes about half a second to adapt),
bright parts are compressed instead of clipped, and a warm white balance is applied.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

WHITE_BALANCE = np.array([1.10, 0.97, 0.86], np.float32)


class CameraExposure(Node):
    def __init__(self):
        super().__init__("camera_exposure")
        self.declare_parameter("target", 165.0)      # mean grey level the camera aims for (real frames: 155-190)
        self.declare_parameter("max_gain", 6.0)
        self.declare_parameter("adapt", 0.25)         # share of the way to the new gain per frame
        self.gain = 2.0
        self.pub = self.create_publisher(Image, "/drone/camera/image_raw", 10)
        self.create_subscription(Image, "/drone/camera/raw", self.on_image, 10)

    def on_image(self, m):
        if m.encoding not in ("rgb8", "bgr8"):
            self.pub.publish(m)
            return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).astype(np.float32)
        if m.encoding == "bgr8":
            a = a[..., ::-1]
        target = self.get_parameter("target").value
        lum = float((a @ np.array([0.299, 0.587, 0.114], np.float32)).mean())
        want = min(self.get_parameter("max_gain").value, max(0.7, target / max(lum, 1.0)))
        self.gain += self.get_parameter("adapt").value * (want - self.gain)
        v = a * self.gain * WHITE_BALANCE
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
