# mini3_sim: the flight cage in Gazebo (ROS 2 Humble)

Your DJI Mini 3 in the lab's 2 x 2 x 3 m net cage, in Gazebo Fortress with ROS 2 Humble, on
Ubuntu 22.04. Runs on a laptop without an NVIDIA GPU.

**Steps 1 to 4 (this version):** the lab, the cage with its net, the pad and the drone in Gazebo;
the drone camera on a ROS 2 topic; the drone flies like your Mini 3 (motion fitted from your real
flights, DJI auto takeoff and landing, the net stops it); you fly it with the keyboard; you record episodes in the same format as the phone app.
OpenVLA and Claude come in the next steps.

## Install (once)

```
sudo apt install ros-humble-ros-gz ros-humble-rqt-image-view python3-numpy python3-tk python3-opencv
mkdir -p ~/mini3_ws/src
ln -s ~/drone-vla-pc/ros2/mini3_sim ~/mini3_ws/src/mini3_sim
cd ~/mini3_ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install
```

## Run

```
source /opt/ros/humble/setup.bash && source ~/mini3_ws/install/setup.bash
ros2 launch mini3_sim sim.launch.py
```

Three windows open: Gazebo (third-person view: drag with the left mouse button to turn, right
button to move, wheel to zoom), `rqt_image_view` with the drone camera, and the small
"Mini 3 keyboard" window.

## Fly

Click once on the "Mini 3 keyboard" window so it gets the keys. The keys sit like the sticks of
the DJI controller (Mode 2):

| Key | Stick | Drone |
|---|---|---|
| T | (take off button) | DJI auto takeoff to 1.2 m |
| L | (land button) | DJI auto landing |
| W / S | left stick up / down | climb / descend |
| A / D | left stick left / right | turn left / right |
| Up / Down | right stick up / down | forward / back |
| Left / Right | right stick left / right | sideways left / right |
| 1 / 2 / 3 | | stick strength: gentle 30 % / normal 60 % (start) / full 100 % |
| Space | | let go of all sticks |
| R | (record button) | start / stop recording an episode |
| O / F | (app's OK / BAD) | mark the last episode success / fail |

Holding a key pushes the stick gradually and letting go brings it back to the centre, where the
drone brakes and hovers like the Mini 3. Holding S below 0.6 m lands it, like the real app. The
window shows both sticks, the height, the heading and the flight mode. When the drone touches
the net it stops there (the window shows NET!).

## Record episodes

Episodes are saved in `~/mini3_sim_episodes` (change with `episodes:=/some/folder` on the launch
line), in exactly the layout the phone app writes:

```
episode_0001_20261102_221808/
  0001_instruction.txt   0001_setup.json   0001_telemetry_20261102_221808.csv
  0001_stick_check.txt   0001_phone_check.txt   0001_result.txt
  frames/0001_000000.jpg ...   (1280 x 720, one per CSV row, 10 Hz)
```

1. When the simulator starts, the pad and the drone are placed on the cells of the current
   plan setup (3 x 3 grid of the cage, the app's 72 pad/drone layouts x 3 repeats). The keyboard
   window shows it, for example "pad NW, drone NE, pad is 2 west". As in your real episodes,
   the drone faces the pad (within about 5 degrees, like aiming by eye).
2. T to take off, R to start recording.
3. Fly to the pad and land (L, or hold S below 0.6 m). Recording stops by itself just after
   touchdown, and the app's episode check runs (shown in the window).
4. O if it was good: the next setup is placed. F if not: the same setup is placed again.

`csv_to_hdf5.py` reads these episodes as they are:

```
python3 csv_to_hdf5.py ~/mini3_sim_episodes/episode_0001_*/0001_telemetry_*.csv ep1.h5
```

`0001_setup.json` has `"simulator": true`, so sim and real episodes can always be told apart.
The plan's progress is kept in `~/mini3_sim_episodes/plan_progress.json`.

## Record all 216 episodes automatically

```
ros2 launch mini3_sim sim.launch.py autopilot:=true
```

The autopilot flies the plan by itself: take off, start recording, fly to the pad, land on it,
mark success (on the pad) or fail (the same setup is flown again), next setup. It moves only the
sticks, like the keyboard, in the style of your real episodes: mostly one stick at a time,
pushes of 10-45 % held for 0.5-3 s, pauses while the drone brakes, sometimes a short descent
first, and landing by holding the throttle down over the pad. Each episode draws its own
"pilot mood" (how hard, how patient, how precise), so the flights differ.

About 15-20 s per episode, so 216 take about an hour. Faster, with no windows:

```
ros2 launch mini3_sim sim.launch.py autopilot:=true gui:=false camera_view:=false
```

`autopilot_episodes:=20` flies only 20. It continues the plan where it stopped
(`plan_progress.json`), so you can stop with Ctrl+C and start again later.

## Topics

| Topic | Type | What |
|---|---|---|
| `/drone/camera/image_raw` | sensor_msgs/Image | drone camera, 640 x 360, 62 deg wide, 50 deg down, with the drone's auto-brightening |
| `/drone/camera/raw` | sensor_msgs/Image | the same before auto-brightening (straight from Gazebo) |
| `/drone/odometry` | nav_msgs/Odometry | drone position and speed (x east, y north, z up, cage centre = 0) |
| `/drone/sticks` | std_msgs/Float32MultiArray | the sticks [pitch, roll, throttle, yaw] in stick percent, -100..100 (from the keyboard; later from OpenVLA) |
| `/drone/command` | std_msgs/String | `takeoff` or `land` |
| `/drone/telemetry` | std_msgs/String (JSON) | 20 Hz, the same fields as the real `*_telemetry_*.csv` (height in ft, speeds in mph NED, heading, stick commands, flight_mode) plus `sim_x/y/z` and `touching_net` |
| `/drone/record` | std_msgs/String | recorder commands: `toggle`, `start`, `stop`, `success`, `fail`, `place` |
| `/drone/recorder/status` | std_msgs/String (JSON) | recording or not, rows, plan setup, last check |
| `/drone/event` | std_msgs/String | `taking_off`, `took_off`, `landing`, `landed`, `hit_net` |
| `/drone/cmd_vel` | geometry_msgs/Twist | the drone node moves the Gazebo model with this (body frame) |

## Files

- `worlds/cage_lab.sdf`: the world: lab, cage, pad at (0.2, 0.3), drone at (0, -0.6) facing north.
- `models/cage_lab`: the lab exported from `sim/blender/build_lab.py --export-obj`, and the net
  from `tools/make_net.py` (a real 3D rope mesh, 5.5 cm cells).
- `models/mini3`: the drone and its camera. `models/pad`: the 50 cm pad.
- `mini3_sim/drone_node.py`: turns sticks into Mini 3 motion with `dynamics.py` and
  `mini3_params.json` (copies of `sim/dynamics.py` and `sim/mini3_params.json`, fitted from your
  real flights), does takeoff and landing, keeps the drone off the net and publishes telemetry.
- `mini3_sim/keyboard.py`: the keyboard window.
- `mini3_sim/recorder.py`: records episodes like the app's TelemetryRecorder, FrameRecorder,
  EpisodePlan and EpisodeCheck. Rows are 100 ms apart in simulated time, so the data is right
  even if Gazebo runs slower than real time. Gazebo renders 640 x 360 (fast on a laptop); frames
  are scaled to 1280 x 720 when saved.
- `mini3_sim/camera_exposure.py`: the camera's auto-exposure and warm colour. The mats are black,
  so like the real camera it brightens the picture a lot.
