"""LeRobotDataset (v3) writer shared by the native recorder and the ROS 2 recorder node."""
from __future__ import annotations

from pathlib import Path

import numpy as np

TASK = "put the red cube in the bin"
FPS = 10
STATE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "tcp_yaw", "gripper_opening"]
# multi scene: + goal points (normalised front-camera image coords of the instructed object and box)
STATE_NAMES_MULTI = STATE_NAMES + ["u_obj", "v_obj", "u_box", "v_box"]
ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "grip"]
ENV_STATE_NAMES = ["cube_x", "cube_y", "cube_z", "cube_yaw"]


def features(image_size: int, cameras: tuple[str, ...] = ("front", "wrist"), use_videos: bool = True,
             state_names: list[str] = STATE_NAMES) -> dict:
    feats = {
        cam_key(c): {
            "dtype": "video" if use_videos else "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        }
        for c in cameras
    }
    feats["observation.state"] = {"dtype": "float32", "shape": (len(state_names),), "names": list(state_names)}
    # Privileged simulator state for analysis only. The key must NOT start with "observation." or LeRobot would
    # feed it to the policy as an input feature (it did, in the first smoke run).
    feats["privileged.cube_pose"] = {"dtype": "float32", "shape": (4,), "names": ENV_STATE_NAMES}
    feats["action"] = {"dtype": "float32", "shape": (5,), "names": ACTION_NAMES}
    return feats


def cam_key(camera: str) -> str:
    return f"observation.images.{camera}"


class DatasetWriter:
    """Thin wrapper: one `add_step` per transition, `end_episode` per episode, `close` at the end."""

    def __init__(self, repo_id: str, root: str | Path, image_size: int, cameras: tuple[str, ...] = ("front", "wrist"),
                 use_videos: bool = True, overwrite: bool = False, state_names: list[str] = STATE_NAMES):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(root)
        if overwrite and root.exists():
            import shutil

            shutil.rmtree(root)
        self.cameras = cameras
        self.ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=FPS,
            features=features(image_size, cameras, use_videos, state_names),
            root=root,
            robot_type="panda_sim",
            use_videos=use_videos,
        )
        self.episode_meta: list[dict] = []

    def add_step(self, images: dict[str, np.ndarray], state: np.ndarray, action: np.ndarray, env_state: np.ndarray,
                 task: str = TASK) -> None:
        frame = {cam_key(c): np.ascontiguousarray(images[c], dtype=np.uint8) for c in self.cameras}
        frame["observation.state"] = np.asarray(state, dtype=np.float32)
        frame["privileged.cube_pose"] = np.asarray(env_state, dtype=np.float32)
        frame["action"] = np.asarray(action, dtype=np.float32)
        frame["task"] = task
        self.ds.add_frame(frame)

    def end_episode(self, meta: dict | None = None) -> None:
        self.ds.save_episode()
        if meta is not None:
            self.episode_meta.append(meta)

    def close(self) -> None:
        self.ds.finalize()

    def push(self, private: bool = True) -> None:
        self.ds.push_to_hub(private=private)
