#!/usr/bin/env bash

if [ -z ${ROBOT+x} ]; then echo "no robot model provided. Set the $ROBOT env var." && exit 1; fi

bash ros_startup.sh $ROBOT $STARTUP

cd "catkin_ws_${ROBOT}" || echo "_incl_train could not cd into catkin_ws. Already in there?"

source /opt/ros/melodic/setup.bash
source devel/setup.bash

echo "Starting command ${1}"
# conda run buffers all stdout -.-: https://github.com/conda/conda/issues/9412
cd src/modulation_rl && source activate base && ${1}
