"""Scripted-teacher client: drives /sim/reset and /sim/step for a list of seeds.

Publishes /episode/result (std_msgs/String, JSON) at the end of every episode so that a recorder can
close the episode with its outcome.
"""
from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from tidy_flywheel.teacher import ScriptedTeacher, TeacherParams
from tidy_interfaces.srv import Reset, Step


class TeacherNode(Node):
    def __init__(self) -> None:
        super().__init__("teacher_node")
        self.declare_parameter("seed_start", 0)
        self.declare_parameter("num_episodes", 5)
        self.declare_parameter("home_close_yaw", 0.0)
        self.declare_parameter("settle_steps", 5)
        self.pub_result = self.create_publisher(String, "/episode/result", 10)
        self.cli_reset = self.create_client(Reset, "/sim/reset")
        self.cli_step = self.create_client(Step, "/sim/step")
        for cli in (self.cli_reset, self.cli_step):
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f"waiting for {cli.srv_name} ...")
        # Give late subscribers (recorder, bag) time to connect before the first frame is published.
        self.declare_parameter("start_delay", 2.0)
        self._spin_for(float(self.get_parameter("start_delay").value))

    def _spin_for(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def call(self, cli, req):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()

    @staticmethod
    def info_from(resp, bin_pos, tcp_cmd, yaw, t) -> dict:
        cp = np.asarray(resp.cube_pose, dtype=np.float64)
        return {
            "t": t,
            "tcp_cmd": np.asarray(tcp_cmd, dtype=np.float64),
            "yaw": float(yaw),
            "cube_pos": cp[:3],
            "cube_yaw": float(cp[3]),
            "bin_pos": np.asarray(bin_pos, dtype=np.float64),
        }

    def run_episode(self, seed: int) -> dict:
        home_close_yaw = float(self.get_parameter("home_close_yaw").value)
        r = self.call(self.cli_reset, Reset.Request(seed=int(seed)))
        bin_pos = list(r.bin_pos)
        state = np.asarray(r.state)
        tcp_cmd = state[:3].copy()
        yaw = float(state[3])
        teacher = ScriptedTeacher(home_close_yaw, TeacherParams())
        info = self.info_from(r, bin_pos, tcp_cmd, yaw, 0)
        contact = False
        steps = 0
        t0 = time.time()
        while True:
            a = teacher.act(None, info)
            s = self.call(self.cli_step, Step.Request(action=[float(v) for v in a]))
            steps += 1
            contact = contact or s.robot_contact
            info = self.info_from(s, bin_pos, s.tcp_cmd, float(s.state[3]), s.t)
            if teacher.done or s.timeout:
                break
        for _ in range(int(self.get_parameter("settle_steps").value)):
            s = self.call(self.cli_step, Step.Request(action=[0.0] * 5))
        result = {
            "seed": int(seed),
            "success": bool(s.success),
            "robot_contact": bool(contact),
            "steps": steps,
            "final_phase": teacher.phase,
            "wall_time_s": round(time.time() - t0, 2),
            "source": "teacher",
        }
        msg = String()
        msg.data = json.dumps(result)
        self.pub_result.publish(msg)
        self.get_logger().info(json.dumps(result))
        return result

    def run(self) -> list[dict]:
        start = int(self.get_parameter("seed_start").value)
        n = int(self.get_parameter("num_episodes").value)
        results = [self.run_episode(seed) for seed in range(start, start + n)]
        ok = sum(r["success"] for r in results)
        self.get_logger().info(f"teacher: {ok}/{n} success, seeds {start}..{start + n - 1}")
        self._spin_for(1.5)
        return results


def main() -> None:
    rclpy.init()
    node = TeacherNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
