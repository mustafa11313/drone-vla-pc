import os
from glob import glob

from setuptools import setup

PKG = "mini3_sim"


def tree(folder):
    """(install dir, files) for every folder below `folder`, keeping the layout."""
    out = []
    for root, _, files in os.walk(folder):
        if files:
            out.append((os.path.join("share", PKG, root), [os.path.join(root, f) for f in files]))
    return out


setup(
    name=PKG,
    version="0.1.0",
    packages=[PKG],
    package_data={PKG: ["mini3_params.json"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PKG]),
        ("share/" + PKG, ["package.xml"]),
        ("share/" + PKG + "/launch", glob("launch/*.py")),
        ("share/" + PKG + "/worlds", glob("worlds/*.sdf")),
        ("share/" + PKG + "/config", glob("config/*.yaml")),
    ] + tree("models"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ali",
    maintainer_email="mustafaalmohammad123@gmail.com",
    description="DJI Mini 3 flight cage simulator for ROS 2 Humble + Gazebo Fortress",
    license="MIT",
    entry_points={"console_scripts": [
        "camera_exposure = mini3_sim.camera_exposure:main",
        "drone_node = mini3_sim.drone_node:main",
        "keyboard = mini3_sim.keyboard:main",
        "recorder = mini3_sim.recorder:main",
    ]},
)
