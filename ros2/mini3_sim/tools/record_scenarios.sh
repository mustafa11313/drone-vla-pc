#!/usr/bin/env bash
# record_scenarios.sh  -  records the yaw and roll scenario episodes with the autopilot, no windows.
#
#   bash record_scenarios.sh            # the first 300 episodes
#   bash record_scenarios.sh 660        # all of them (also continues after a stop: progress is saved)
#   OUT=~/other_folder bash record_scenarios.sh
#
# Episodes go to ~/mini3_sim_episodes_scenarios (a new folder; the older sim episodes are not
# touched). Stop any time with Ctrl+C; running it again continues where it stopped.
set -e
N=${1:-300}
OUT=${OUT:-$HOME/mini3_sim_episodes_scenarios}
WS=${WS:-$HOME/mini3_ws}
source /opt/ros/humble/setup.bash
if [ -f "$WS/install/setup.bash" ]; then source "$WS/install/setup.bash"; fi
mkdir -p "$OUT"
DONE=$(ls -d "$OUT"/episode_* 2>/dev/null | wc -l)
echo "recording into $OUT (episodes already there: $DONE), up to setup $N of 660"
ros2 launch mini3_sim sim.launch.py setups:=scenarios plan_size:="$N" autopilot:=true \
    autopilot_episodes:=100000 gui:=false camera_view:=false episodes:="$OUT"
