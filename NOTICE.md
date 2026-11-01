# Third-party notices

This project bundles or depends on the following third-party work.

| Component | Where | License | Notes |
|---|---|---|---|
| MuJoCo Menagerie — `franka_emika_panda` | `tidy_flywheel/assets/panda/` | Apache-2.0 (`tidy_flywheel/assets/panda/LICENSE`) | Copied from commit `c96a32d2` (2026-09-23). **Modified**: arm joint actuators removed (the arm is driven by a mocap weld), `tcp` site and `wrist` camera added, gripper actuator gains changed, `meshdir` adjusted. See `tidy_flywheel/assets/panda/SOURCE.txt`. |
| MuJoCo | pip dependency | Apache-2.0 | Physics and rendering |
| LeRobot | pip dependency | Apache-2.0 | Dataset format, ACT policy, training |
| PyTorch / torchvision | pip dependency | BSD-3-Clause | |
| ROS 2 (Jazzy) | Docker base image `ros:jazzy-ros-base` | Apache-2.0 | |

All demonstration data used in this project is generated in simulation by this repository's scripted teacher; no external datasets are used.
