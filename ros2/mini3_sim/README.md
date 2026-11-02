# mini3_sim: the flight cage in Gazebo (ROS 2 Humble)

Your DJI Mini 3 in the lab's 2 x 2 x 3 m net cage, in Gazebo Fortress with ROS 2 Humble, on
Ubuntu 22.04. Runs on a laptop without an NVIDIA GPU.

**Step 1 (this version):** the lab, the cage with its net, the pad and the drone in Gazebo; the
drone camera on a ROS 2 topic. Flying, keyboard, recording and OpenVLA come in the next steps.

## Install (once)

```
sudo apt install ros-humble-ros-gz ros-humble-rqt-image-view python3-numpy
mkdir -p ~/mini3_ws/src
ln -s ~/drone-vla-pc/ros2/mini3_sim ~/mini3_ws/src/mini3_sim
cd ~/mini3_ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install
```

## Run

```
source /opt/ros/humble/setup.bash && source ~/mini3_ws/install/setup.bash
ros2 launch mini3_sim sim.launch.py
```

Two windows open: Gazebo (third-person view: drag with the left mouse button to turn, right
button to move, wheel to zoom) and `rqt_image_view` with the drone camera.

## Topics

| Topic | Type | What |
|---|---|---|
| `/drone/camera/image_raw` | sensor_msgs/Image | drone camera, 640 x 360, 62 deg wide, 50 deg down, with the drone's auto-brightening |
| `/drone/camera/raw` | sensor_msgs/Image | the same before auto-brightening (straight from Gazebo) |
| `/drone/odometry` | nav_msgs/Odometry | drone position and speed (x east, y north, z up, cage centre = 0) |
| `/drone/cmd_vel` | geometry_msgs/Twist | moves the drone (used by the drone node in step 2) |

## Files

- `worlds/cage_lab.sdf`: the world: lab, cage, pad at (0.2, 0.3), drone at (0, -0.6) facing north.
- `models/cage_lab`: the lab exported from `sim/blender/build_lab.py --export-obj`, and the net
  from `tools/make_net.py` (a real 3D rope mesh, 5.5 cm cells).
- `models/mini3`: the drone and its camera. `models/pad`: the 50 cm pad.
- `mini3_sim/camera_exposure.py`: the camera's auto-exposure and warm colour. The mats are black,
  so like the real camera it brightens the picture a lot.
