#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
if [ -f /ws/ros2_ws/install/setup.bash ]; then source /ws/ros2_ws/install/setup.bash; fi
# ament_python entry points run under the system interpreter; expose the venv's packages to it
# (the venv was created from the same /usr/bin/python3.12, so the packages are binary-compatible).
export PYTHONPATH=/ws:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH}
exec "$@"
