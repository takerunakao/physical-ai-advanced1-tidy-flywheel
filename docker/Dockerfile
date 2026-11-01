# ROS 2 Jazzy (Ubuntu 24.04, Python 3.12) + MuJoCo + LeRobot (CPU) — multi-arch (arm64 / amd64).
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-venv git ffmpeg \
      libosmesa6 libgl1 libglu1-mesa libegl1 \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Python deps live in a venv that can still see rclpy from /opt/ros.
RUN python3 -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venv/bin/pip install "mujoco==3.14.0" "lerobot[dataset]==0.6.1" imageio imageio-ffmpeg pyyaml pytest
ENV PATH=/opt/venv/bin:$PATH

# Language layer (vlm_node: Claude planner/pointer) and a CJK font for the video overlay ("片付けて").
# Separate layer so the torch/lerobot layer above stays cached. API keys are NOT part of the image:
# compose passes the host's .env as environment variables at run time.
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && /opt/venv/bin/pip install anthropic python-dotenv pydantic pillow

WORKDIR /ws
COPY pyproject.toml README.md LICENSE NOTICE.md /ws/
COPY tidy_flywheel /ws/tidy_flywheel
RUN /opt/venv/bin/pip install -e /ws

COPY ros2_ws/src /ws/ros2_ws/src
RUN . /opt/ros/jazzy/setup.sh && cd /ws/ros2_ws \
    && colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/opt/venv/bin/python

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
