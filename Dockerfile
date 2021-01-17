# FROM ros:melodic-ros-core-bionic
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# ENV LD_LIBRARY_PATH /usr/local/cuda-10.2/lib64:/usr/local/cuda-10.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV HOME /root

#####################
# ROS CORE
#####################
# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO melodic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-core=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*


#####################
# ROS BASE & APPLICATION SPECIFIC PACKAGES
#####################
# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
    rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-base=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# APPLICATION SPECIFIC ROS PACKAGES
RUN apt-get update && apt-get install -y \
    ros-melodic-pybind11-catkin \
    ros-melodic-moveit \
    ros-melodic-pr2-simulator \
    ros-melodic-moveit-pr2 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc


#####################
# INSTALL CONDA
#####################
RUN apt-get update --fix-missing && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH


#####################
# PYTORCH FROM SOURCE
#####################
ENV ENV_NAME=base
ENV TORCH_VERSION=v1.6.0
# FOR SOME REASON TORCH FAILS ("NO CUDA DRIVERS ...) IF CHANGING THE PYTHON VERSION HERE!
# not sure how much the above stuff depends on the same version / whether I have to or can define it further above
#ENV PYTHON_VERSION=3.6

#RUN conda install -n ${ENV_NAME} python=${PYTHON_VERSION} numpy ninja pyyaml mkl mkl-include setuptools cmake cffi \
RUN conda install -n ${ENV_NAME} python=3.7 numpy ninja pyyaml mkl mkl-include setuptools cmake cffi \
    && conda install -n ${ENV_NAME} -c pytorch magma-cuda102 \
    && conda clean -afy

RUN git clone --recursive --branch ${TORCH_VERSION} https://github.com/pytorch/pytorch \
    && cd pytorch \
    && git submodule sync \
    && git submodule update --init --recursive \
    && TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
       CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
       conda run -n ${ENV_NAME} python setup.py install \
    && cd .. \
    && rm -rf pytorch/ \
    && conda install -n ${ENV_NAME} cudatoolkit=10.2 -c pytorch \
    && conda clean -afy


#####################
# CONDA DEPENDENCIES FROM ENV.YAML
#####################
# to use git, solve key issue: https://vsupalov.com/build-docker-image-clone-private-repo-ssh-key/
# only want to get the environment.yml at this point, so as not to recreate everytime some code changes
# RUN git clone git@github.com:dHonerkamp/dllab_modulation_rl.git .
COPY environment_docker.yml src/modulation_rl/
RUN conda env update -n ${ENV_NAME} -f src/modulation_rl/environment_docker.yml \
    && conda clean -afy


######################
## CREATE CATKIN WORKSPACE WITH PYTHON3
######################
## python3: https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674
RUN conda install -n ${ENV_NAME} -c conda-forge rospkg catkin_pkg \
    && apt-get update \
    && apt-get install -y python-catkin-tools python3-dev python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $HOME/catkin_ws_pr2
RUN catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so

WORKDIR $HOME/catkin_ws_tiago
RUN catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so

WORKDIR $HOME/catkin_ws_hsr
RUN catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so


######################
## HSR
######################
WORKDIR $HOME/catkin_ws_hsr

# HSR: install couchdb for ros-melodic-tmc-desktop-full to work: https://docs.couchdb.org/en/latest/setup/single-node.html
#RUN apt-get update && apt-get install -y gnupg ca-certificates \
#    && echo "deb https://apache.bintray.com/couchdb-deb bionic main" | tee /etc/apt/sources.list.d/couchdb.list \
#    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 8756C4F765C9AC3CB6B85D62379CE192D401AB61 \
#    && apt-get update \
#    && echo "couchdb couchdb/mode select standalone" | debconf-set-selections \
#    && echo "couchdb couchdb/bindaddress select 0.0.0.0" | debconf-set-selections \
#    && echo "couchdb couchdb/adminpass select ''" | debconf-set-selections \
#    && echo "couchdb couchdb/adminpass_again select ''" | debconf-set-selections \
#    && apt-get install -y couchdb \
#    && rm -rf /var/lib/apt/lists/*

# INSTALL PROPRIETARY HSR SIMULATOR: IF YOU HAVE A TOYOTA / HSR ACCOUNT PLEASE FOLLOW THESE STEPS TO INSTALL ros-melodic-tmc-desktop-full: https://docs.hsr.io/manual_en/howto/pc_install.html?highlight=install


#RUN git clone https://github.com/hsr-project/hsrb_moveit_config.git src/hsrb_moveit_config
#RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build"
#
## update the configs
#COPY gazebo_world/hsr/modified_hsrb_kinematics.yaml src/hsrb_moveit_config/config/kinematics.yaml
#COPY gazebo_world/hsr/modified_CMakeLists.txt src/hsrb_moveit_controller_manager/CMakeLists.txt

######################
## Tiago
######################
WORKDIR $HOME/catkin_ws_tiago
# install tiago sdk following http://wiki.ros.org/Robots/TIAGo/Tutorials/Installation/TiagoSimulation
# https://hub.docker.com/r/jacknlliu/tiago-ros/dockerfile
RUN wget -O tiago_public.rosinstall https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/kinetic-devel/tiago_public-melodic.rosinstall \
    && yes | rosinstall src /opt/ros/melodic tiago_public.rosinstall \
    && rosdep update \
    && apt-get update \
    && rosdep install -y -r -q --from-paths src --ignore-src --rosdistro melodic --skip-keys="opencv2 opencv2-nonfree pal_laser_filters speed_limit_node sensor_to_cloud hokuyo_node libdw-dev python-graphitesend-pip python-statsd pal_filters pal_vo_server pal_usb_utils pal_pcl pal_pcl_points_throttle_and_filter pal_karto pal_local_joint_control camera_calibration_files pal_startup_msgs pal-orbbec-openni2 dummy_actuators_manager pal_local_planner gravity_compensation_controller current_limit_controller dynamic_footprint dynamixel_cpp tf_lookup opencv3 hsrb_moveit_plugins hsrb_moveit_config hsrb_description hsrc_description" \
    # && apt-get install -y ros-melodic-base-local-planner ros-melodic-people-msgs ros-melodic-roslint ros-melodic-four-wheel-steering-controller ros-melodic-twist-mux \
    && rm -rf /var/lib/apt/lists/*

# Controllers rely on tf2 -> need to compile with python3: https://github.com/ros/geometry2/issues/259
#RUN git clone https://github.com/ros/geometry src/geometry && git clone https://github.com/ros/geometry2 src/geometry2

RUN catkin config --blacklist tiago_pcl_tutorial # combined_robot_hw_tests # force_torque_sensor_controller mode_state_controller
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build"

# update the moveit configs with the global joint
COPY gazebo_world/tiago/modified_tiago.srdf.em src/tiago_moveit_config/config/srdf/tiago.srdf.em
COPY gazebo_world/tiago/modified_tiago_pal-gripper.srdf src/tiago_moveit_config/config/srdf/tiago_pal-gripper.srdf
COPY gazebo_world/tiago/modified_tiago_kinematics_kdl.yaml src/tiago_moveit_config/config/kinematics_kdl.yaml
COPY gazebo_world/tiago/modified_gripper.urdf.xacro src/pal_gripper/pal_gripper_description/urdf/gripper.urdf.xacro
COPY gazebo_world/tiago/modified_wsg_gripper.urdf.xacro src/pal_wsg_gripper/pal_wsg_gripper_description/urdf/gripper.urdf.xacro
#COPY gazebo_world/tiago/modified_gazebo.urdf.xacro src/tiago_robot/tiago_description/gazebo/gazebo.urdf.xacro

######################
## PR2
######################
WORKDIR $HOME/catkin_ws_pr2
RUN git clone https://github.com/PR2/pr2_mechanism.git src/pr2_mechanism
#    && wget -O src/ros_control.rosinstall https://raw.github.com/ros-controls/ros_control/melodic-devel/ros_control.rosinstall \
#    && rosdep install --from-paths . --ignore-src --rosdistro melodic -y
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build"

COPY gazebo_world/pr2/modified_pr2_kinematics.yaml /opt/ros/melodic/share/pr2_moveit_config/config/kinematics.yaml

######################
## UPGRADE TO LATEST GAZEBO VERSION FOR MELODIC FOR POINTER ISSUE IF RUNNING LONG
######################
RUN apt-get update && apt upgrade -y libignition-math2 && rm -rf /var/lib/apt/lists/*
# prevent potential erros with wandb
#RUN mv /opt/ros/melodic/lib/python2.7/dist-packages/cv2.so /opt/ros/melodic/lib/python2.7/dist-packages/cv2_renamed.so

#####################
# COPY FILES AND BUILD OUR ROS PACKAGE -> don't use conda python, but the original!
#####################
# get killall cmd, vim
RUN apt-get update && apt-get install -y psmisc vim && rm -rf /var/lib/apt/lists/*
# libgp
WORKDIR $HOME
RUN git clone https://github.com/mblum/libgp.git \
    && cd libgp \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF .. \
    && make \
    && make install

#WORKDIR $HOME/catkin_ws_hsr
## build our package: copy only files required for compilation use caching whenever possible
#COPY include/ src/modulation_rl/include/
#COPY src/ src/modulation_rl/src/
#COPY CMakeLists.txt package.xml src/modulation_rl/
#RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build modulation_rl"

WORKDIR $HOME/catkin_ws_tiago
# build our package: copy only files required for compilation use caching whenever possible
COPY include/ src/modulation_rl/include/
COPY src/ src/modulation_rl/src/
COPY CMakeLists.txt package.xml src/modulation_rl/
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build modulation_rl"

WORKDIR $HOME/catkin_ws_pr2
# build our package: copy only files required for compilation use caching whenever possible
COPY include/ src/modulation_rl/include/
COPY src/ src/modulation_rl/src/
COPY CMakeLists.txt package.xml src/modulation_rl/
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build modulation_rl"

######################
## COPY FILES
######################
WORKDIR $HOME
# copy our object models into gazebo
COPY gazebo_world/models/ $HOME/.gazebo/models/
# copy our launch files and configs
COPY gazebo_world/hsr catkin_ws_hsr/src/modulation_rl/gazebo_world/hsr
COPY gazebo_world/tiago catkin_ws_tiago/src/modulation_rl/gazebo_world/tiago
COPY gazebo_world/pr2 catkin_ws_pr2/src/modulation_rl/gazebo_world/pr2
# gmm models
COPY GMM_models/ catkin_ws_hsr/src/modulation_rl/GMM_models
COPY GMM_models/ catkin_ws_tiago/src/modulation_rl/GMM_models
COPY GMM_models/ catkin_ws_pr2/src/modulation_rl/GMM_models/
COPY Ellipse_modulation_models/ catkin_ws_pr2/src/modulation_rl/Ellipse_modulation_models/
# our task world includes assets from the publicly available pal gazebo worlds
RUN echo "export GAZEBO_MODEL_PATH=${HOME}/catkin_ws_tiago/src/modulation_rl/gazebo_world/models:${HOME}/catkin_ws_tiago/src/pal_gazebo_world/models:${GAZEBO_MODEL_PATH}" >> ~/.bashrc
RUN echo "export GAZEBO_RESOURCE_PATH=${HOME}/catkin_ws_tiago/src/modulation_rl/gazebo_world/worlds:${HOME}/catkin_ws_tiago/src/pal_gazebo_world:${GAZEBO_RESOURCE_PATH}" >> ~/.bashrc
# launch helper
COPY ros_startup.sh ros_startup_incl_train.sh ./

######################
## RUN TRAINING
######################
# ensure we take python2 to run the ros startup stuff
ENV PATH /usr/bin/:$PATH
ENV PYTHONPATH /usr/bin/:$PYTHONPATH
ENV PYTHONUNBUFFERED=1
ENV ROSCONSOLE_FORMAT='[${severity} ${node}] [${time}]: ${message}'
COPY gazebo_world/rosconsole.config /opt/ros/melodic/share/ros/config/rosconsole.config

# copy all remaining files
COPY scripts/ catkin_ws_hsr/src/modulation_rl/scripts
COPY scripts/ catkin_ws_tiago/src/modulation_rl/scripts
COPY scripts/ catkin_ws_pr2/src/modulation_rl/scripts

CMD bash ros_startup_incl_train.sh "python scripts/main.py --load_best_defaults"
