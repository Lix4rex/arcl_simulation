#!/bin/bash
set -e

source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
if [ -f src/ros2.repos ]; then
    envsubst < src/ros2.repos | vcs import src
fi
sudo apt-get update
rosdep update --rosdistro=$ROS_DISTRO
rosdep install --from-paths src --ignore-src -y --rosdistro=$ROS_DISTRO
