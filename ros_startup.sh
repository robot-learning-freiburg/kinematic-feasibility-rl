#!/usr/bin/env bash


if [ -z ${1+x} ]; then echo "no robot model provided. Choose pr2, tiago or hsr" && exit 1; fi

cd "catkin_ws_${1}" || echo "Could not cd in catkin_ws. Already in there?"
source /opt/ros/melodic/setup.bash
source devel/setup.bash

echo "Killing existing ros whatever"
rosnode kill -a && killall -9 roscore rosmaster gzserver gazebo || killall -9 roscore rosmaster gzserver gazebo || echo "Nothing to kill, moving on"

echo "Starting roscore"
(roscore &) &> /dev/null
sleep 10

# Start launchfiles through python2. If env is "" stop them immediately again to reduce resource usage
python2 src/modulation_rl/scripts/modulation/handle_launchfiles.py --env "$2"

echo "All done"
