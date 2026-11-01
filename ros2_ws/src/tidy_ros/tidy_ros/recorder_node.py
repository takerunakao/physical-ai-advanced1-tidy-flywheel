"""Recorder: subscribes to /sim/frame and /episode/result and writes a LeRobotDataset.

Frames of one episode share a seed; the episode is closed when /episode/result arrives for that seed.
Episodes whose result says success=false (or robot_contact=true) are dropped unless keep_failures:=true.
"""
from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from tidy_flywheel.dataset import DatasetWriter
from tidy_interfaces.msg import Frame

from .common import image_to_array


class RecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("recorder_node")
        self.declare_parameter("repo_id", "takerunakao/tidy-ros")
        self.declare_parameter("root", "/ws/data/tidy-ros")
        self.declare_parameter("image_size", 128)
        self.declare_parameter("keep_failures", False)
        self.declare_parameter("overwrite", False)
        self.writer = DatasetWriter(
            str(self.get_parameter("repo_id").value),
            str(self.get_parameter("root").value),
            int(self.get_parameter("image_size").value),
            overwrite=bool(self.get_parameter("overwrite").value),
        )
        self.keep_failures = bool(self.get_parameter("keep_failures").value)
        self.buffer: list[dict] = []
        self.buffer_seed: int | None = None
        self.n_saved = 0
        self.create_subscription(Frame, "/sim/frame", self.on_frame, 100)
        self.create_subscription(String, "/episode/result", self.on_result, 10)
        self.get_logger().info("recorder ready")

    def on_frame(self, f: Frame) -> None:
        if self.buffer_seed is not None and f.seed != self.buffer_seed:
            self.get_logger().warn(f"frame for seed {f.seed} while seed {self.buffer_seed} is open; dropping the open episode")
            self.buffer = []
        self.buffer_seed = int(f.seed)
        self.buffer.append({
            "images": {"front": image_to_array(f.front), "wrist": image_to_array(f.wrist)},
            "state": np.asarray(f.state, dtype=np.float32),
            "action": np.asarray(f.action, dtype=np.float32),
            "env_state": np.asarray(f.cube_pose, dtype=np.float32),
        })

    def on_result(self, msg: String) -> None:
        res = json.loads(msg.data)
        if self.buffer_seed is None or int(res["seed"]) != self.buffer_seed:
            self.get_logger().warn(f"result for seed {res.get('seed')} but open seed is {self.buffer_seed}")
            return
        keep = self.keep_failures or (res.get("success") and not res.get("robot_contact"))
        if keep:
            for fr in self.buffer:
                self.writer.add_step(fr["images"], fr["state"], fr["action"], fr["env_state"])
            self.writer.end_episode(res)
            self.n_saved += 1
        self.get_logger().info(f"seed {self.buffer_seed}: {len(self.buffer)} frames, kept={bool(keep)}, saved={self.n_saved}")
        self.buffer, self.buffer_seed = [], None

    def close(self) -> None:
        self.writer.close()
        self.get_logger().info(f"dataset finalised: {self.n_saved} episodes")


def main() -> None:
    rclpy.init()
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
