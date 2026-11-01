"""Learned-policy node: runs a LeRobot ACT checkpoint closed-loop through /sim/step.

Two modes (parameter `mode`):
  seeds  (default) a client that evaluates seeds on its own via /sim/reset and /sim/step.
         Publishes /episode/result (JSON) after each episode.
  serve  a skill server for the tidy orchestrator: /policy/execute (tidy_interfaces/Execute) takes the starting
         observation returned by /sim/retarget, resets the policy and runs it until success or timeout, then
         `settle_steps` zero actions (as scripts/tidy_all.py does), and returns the outcome.

Parameters
  checkpoint   local dir or Hub repo id of the checkpoint (required)
  mode         seeds | serve
  seeds        e.g. "1000-1099"                       (seeds mode)
  n_action_steps  override the chunk execution horizon (optional)
  results_json where to write the per-episode outcomes (seeds mode)
  settle_steps zero-action steps after the policy stops (serve mode, default 5)
Publishes /policy/action for introspection.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from tidy_flywheel.evaluate import classify, wilson_interval
from tidy_interfaces.srv import Execute, Reset, Step

from .common import obs_from_response, parse_seeds


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("seeds", "1000-1009")
        self.declare_parameter("n_action_steps", 0)
        self.declare_parameter("results_json", "")
        self.declare_parameter("tag", "policy")
        self.declare_parameter("mode", "seeds")
        self.declare_parameter("settle_steps", 5)
        self.mode = str(self.get_parameter("mode").value)
        if self.mode not in ("seeds", "serve"):
            raise ValueError(f"mode must be 'seeds' or 'serve', got {self.mode!r}")
        # serve: the Execute callback calls /sim/step itself, so the server and the client share a reentrant
        # group and the node runs in a MultiThreadedExecutor (see main()).
        self.cb_group = ReentrantCallbackGroup() if self.mode == "serve" else None
        self.pub_result = self.create_publisher(String, "/episode/result", 10)
        self.pub_action = self.create_publisher(Float32MultiArray, "/policy/action", 10)
        self.cli_reset = self.create_client(Reset, "/sim/reset", callback_group=self.cb_group)
        self.cli_step = self.create_client(Step, "/sim/step", callback_group=self.cb_group)
        for cli in (self.cli_reset, self.cli_step):
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f"waiting for {cli.srv_name} ...")
        # Give late subscribers (recorder, bag) time to connect before the first frame is published.
        self.declare_parameter("start_delay", 2.0)
        if self.mode == "seeds":
            self._spin_for(float(self.get_parameter("start_delay").value))
        ckpt = str(self.get_parameter("checkpoint").value)
        n_act = int(self.get_parameter("n_action_steps").value) or None
        from tidy_flywheel.policy import ActPolicy

        t0 = time.time()
        self.policy = ActPolicy(ckpt, n_action_steps=n_act)
        self.get_logger().info(f"loaded {ckpt} in {time.time() - t0:.1f}s (cameras {self.policy.cameras})")
        if self.mode == "serve":
            self._busy = threading.Lock()
            self.create_service(Execute, "/policy/execute", self.on_execute, callback_group=self.cb_group)
            self.get_logger().info("serving /policy/execute")

    def _spin_for(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def call(self, cli, req):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()

    def call_nested(self, cli, req, timeout: float = 120.0):
        """Service call from inside a callback (serve mode): the MultiThreadedExecutor delivers the response on
        another thread while this one waits on an event; spin_until_future_complete would re-enter the executor."""
        done = threading.Event()
        fut = cli.call_async(req)
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            cli.remove_pending_request(fut)
            raise TimeoutError(f"{cli.srv_name} did not answer within {timeout:.0f}s")
        return fut.result()

    def on_execute(self, req: Execute.Request, res: Execute.Response) -> Execute.Response:
        if not self._busy.acquire(blocking=False):  # one robot, one skill at a time
            res.ok, res.error = False, "busy"
            return res
        try:
            t0 = time.time()
            self.policy.reset()
            obs = obs_from_response(req)
            s = None
            while True:
                a = self.policy.act(obs)
                self.pub_action.publish(Float32MultiArray(data=[float(v) for v in a]))
                s = self.call_nested(self.cli_step, Step.Request(action=[float(v) for v in a]))
                if not s.ok:
                    raise RuntimeError("/sim/step refused (no episode?)")
                obs = obs_from_response(s)
                if s.success or s.timeout:
                    break
            timed_out = bool(s.timeout)
            for _ in range(int(self.get_parameter("settle_steps").value)):
                s = self.call_nested(self.cli_step, Step.Request(action=[0.0] * 5))
            res.ok = True
            res.success, res.timeout = bool(s.success), timed_out
            res.wrong_object, res.contact, res.steps = bool(s.wrong_object), bool(s.robot_contact), int(s.t)
            self.get_logger().info(f"execute '{req.task}': success={res.success} steps={res.steps} "
                                   f"({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            res.ok, res.error = False, f"{type(e).__name__}: {e}"
            self.get_logger().error(f"execute failed: {res.error}")
        finally:
            self._busy.release()
        return res

    def run_episode(self, seed: int) -> dict:
        r = self.call(self.cli_reset, Reset.Request(seed=int(seed)))
        self.policy.reset()
        obs = obs_from_response(r)
        cube0 = np.asarray(r.cube_pose[:2])
        ever_grasped = contact = False
        t0 = time.time()
        while True:
            a = self.policy.act(obs)
            self.pub_action.publish(Float32MultiArray(data=[float(v) for v in a]))
            s = self.call(self.cli_step, Step.Request(action=[float(v) for v in a]))
            obs = obs_from_response(s)
            ever_grasped = ever_grasped or s.grasped
            contact = contact or s.robot_contact
            if s.success or s.timeout:
                break
        cube = np.asarray(s.cube_pose)
        ep = {
            "seed": int(seed),
            "success": bool(s.success),
            "steps": int(s.t),
            "ever_grasped": bool(ever_grasped),
            "grasped_at_end": bool(s.grasped),
            "robot_contact": bool(contact),
            "cube_moved": bool(np.linalg.norm(cube[:2] - cube0) > 0.01),
            "cube_near_bin": bool(np.linalg.norm(cube[:2] - np.asarray(r.bin_pos[:2])) < 0.12),
            "wall_time_s": round(time.time() - t0, 2),
            "source": "policy",
        }
        ep["failure_mode"] = classify(ep)
        self.pub_result.publish(String(data=json.dumps(ep)))
        self.get_logger().info(json.dumps(ep))
        return ep

    def run(self) -> None:
        seeds = parse_seeds(self.get_parameter("seeds").value)
        episodes = [self.run_episode(s) for s in seeds]
        k, n = sum(e["success"] for e in episodes), len(episodes)
        lo, hi = wilson_interval(k, n)
        summary = {"tag": str(self.get_parameter("tag").value), "n": n, "success": k, "success_rate": k / n,
                   "wilson95": [lo, hi], "episodes": episodes}
        self.get_logger().info(f"policy: {k}/{n} success (95% CI {lo:.2f}-{hi:.2f})")
        out = str(self.get_parameter("results_json").value)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(summary, indent=1))
        self._spin_for(1.5)


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        if node.mode == "serve":
            executor = MultiThreadedExecutor(num_threads=4)
            executor.add_node(node)
            try:
                executor.spin()
            except (KeyboardInterrupt, ExternalShutdownException):
                pass
        else:
            node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
