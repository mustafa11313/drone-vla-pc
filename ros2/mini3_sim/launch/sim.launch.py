"""
ros2 launch mini3_sim sim.launch.py            # Gazebo window + camera window
ros2 launch mini3_sim sim.launch.py gui:=false camera_view:=false   # no windows (servers only)

Starts Gazebo (Fortress) with the lab and the cage, the ROS <-> Gazebo bridge for the drone's
topics, and rqt_image_view on /drone/camera/image_raw.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("mini3_sim")
    world = os.path.join(share, "worlds", "cage_lab.sdf")
    models = os.path.join(share, "models")
    resource = os.pathsep.join(p for p in (models, os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")) if p)
    gui = LaunchConfiguration("gui")
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="open the Gazebo window"),
        DeclareLaunchArgument("camera_view", default_value="true", description="open rqt_image_view"),
        DeclareLaunchArgument("keyboard", default_value="true", description="open the keyboard flying window"),
        SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", resource),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource),
        ExecuteProcess(cmd=["ign", "gazebo", "-r", world], output="screen", condition=IfCondition(gui)),
        ExecuteProcess(cmd=["ign", "gazebo", "-r", "-s", "--headless-rendering", world], output="screen",
                       condition=UnlessCondition(gui)),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             parameters=[{"config_file": os.path.join(share, "config", "bridge.yaml"), "use_sim_time": True}]),
        Node(package="mini3_sim", executable="camera_exposure", output="screen",
             parameters=[{"use_sim_time": True}]),
        Node(package="mini3_sim", executable="drone_node", output="screen",
             parameters=[{"use_sim_time": True}]),
        Node(package="mini3_sim", executable="keyboard", output="screen",
             condition=IfCondition(LaunchConfiguration("keyboard"))),
        Node(package="rqt_image_view", executable="rqt_image_view", arguments=["/drone/camera/image_raw"],
             condition=IfCondition(LaunchConfiguration("camera_view"))),
    ])
