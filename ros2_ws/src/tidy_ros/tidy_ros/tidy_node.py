"""Tidy-up orchestrator: one free-form command ("片付けて") -> plan (LLM) -> point (VLM) -> act (ACT), closed loop.

The ROS 2 counterpart of scripts/tidy_all.py. For every seed:

    /sim/reset
    repeat until the plan is empty or `max_subtasks` instructions were executed:
        /sim/go_home, /sim/render front+side 480px, /vlm/plan(command, views, attempted)
        for each planned step (object, box):
            /sim/go_home, /sim/render front 480px, /vlm/point(image, instruction)
            /sim/retarget(object, box, points, snap_drop)    # goal points for the policy
            /policy/execute(starting observation)            # until success/timeout, then 5 settle steps
    /sim/go_home -> in_box (privileged; used only to score the outcome)

The planner re-plans from what the cameras show, so a failed pick is retried; no simulator state is used to
decide. (tidy_all.py calls retarget before rendering for the pointer; retarget moves nothing, so pointing first
and passing the points to /sim/retarget is equivalent.)

Parameters
  command       default "片付けて"
  seeds         e.g. "1000-1009"
  max_subtasks  instruction budget over all passes (default 6 = 3 objects + retries)
  snap_drop     drop at the centre of the box half the VLM chose (default true)
  results_json  where to write per-seed results and the summary (same keys as tidy_all.py)
  executor      label stored in the summary (the launch file passes the checkpoint path)
Topics
  /tidy/status       std_msgs/String   human-readable progress lines (video_node overlays them)
  /tidy/goal_points  std_msgs/Float32MultiArray  current goal points (empty = none)
  /episode/result    std_msgs/String   JSON per seed (video_node closes the MP4 on it)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from tidy_flywheel.sim_multi import BOXES, OBJECTS, instruction
from tidy_interfaces.srv import Execute, GoHome, Plan, Point, Render, Reset, Retarget

from .common import parse_seeds

GROUND_RES = 480
JP = {"red cube": "赤いキューブ", "blue cube": "青いキューブ", "green cylinder": "緑の円柱",
      "white box": "白い箱", "black box": "黒い箱"}


def jp_step(obj: int, box: int) -> str:
    return f"{JP[OBJECTS[obj][1]]} → {JP[BOXES[box][1]]}"


class TidyNode(Node):
    def __init__(self) -> None:
        super().__init__("tidy_node")
        self.declare_parameter("command", "片付けて")
        self.declare_parameter("seeds", "1000-1009")
        self.declare_parameter("max_subtasks", 6)
        self.declare_parameter("snap_drop", True)
        self.declare_parameter("results_json", "")
        self.declare_parameter("executor", "policy_node")
        self.declare_parameter("start_delay", 2.0)
        self.declare_parameter("video_wait", 120.0)
        self.command = str(self.get_parameter("command").value)
        self.max_subtasks = int(self.get_parameter("max_subtasks").value)
        self.snap_drop = bool(self.get_parameter("snap_drop").value)
        self.pub_status = self.create_publisher(String, "/tidy/status", 10)
        self.pub_points = self.create_publisher(Float32MultiArray, "/tidy/goal_points", 10)
        self.pub_result = self.create_publisher(String, "/episode/result", 10)
        self.video_saved: list[str] = []
        self.create_subscription(String, "/video/saved", lambda m: self.video_saved.append(m.data), 10)
        self.cli = {
            "reset": self.create_client(Reset, "/sim/reset"),
            "go_home": self.create_client(GoHome, "/sim/go_home"),
            "render": self.create_client(Render, "/sim/render"),
            "retarget": self.create_client(Retarget, "/sim/retarget"),
            "plan": self.create_client(Plan, "/vlm/plan"),
            "point": self.create_client(Point, "/vlm/point"),
            "execute": self.create_client(Execute, "/policy/execute"),
        }
        for cli in self.cli.values():
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f"waiting for {cli.srv_name} ...")
        self._spin_for(float(self.get_parameter("start_delay").value))  # let video_node subscribe

    # -- plumbing --------------------------------------------------------------
    def _spin_for(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def call(self, name: str, req):
        fut = self.cli[name].call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()

    def status(self, *lines: str, points=None) -> None:
        self.pub_status.publish(String(data="\n".join(lines)))
        self.pub_points.publish(Float32MultiArray(data=[] if points is None else [float(v) for v in points]))
        self.get_logger().info(" | ".join(lines))

    def render(self, camera: str):
        r = self.call("render", Render.Request(camera=camera, size=GROUND_RES))
        if not r.ok:
            raise RuntimeError(f"/sim/render {camera}: {r.error}")
        return r.image

    # -- one scene -------------------------------------------------------------
    def run_seed(self, seed: int) -> dict:
        header = f"指示: 「{self.command}」"
        self.status(header)  # before reset: the reset frame must not carry the previous seed's text
        self.call("reset", Reset.Request(seed=int(seed)))
        subtasks: list[dict] = []
        passes: list[dict] = []
        t0 = time.time()
        while len(subtasks) < self.max_subtasks:
            self.call("go_home", GoHome.Request())
            self.status(header, f"Claude が計画中 ({len(passes) + 1} 回目)")
            p = self.call("plan", Plan.Request(command=self.command, views=[self.render("front"), self.render("side")],
                                                view_names=["front", "side"],
                                                attempted=[s["task"] for s in subtasks]))
            if not p.ok:  # tidy_all.py would stop with the exception; here the seed ends and the error is kept
                passes.append({"steps": [], "reasoning": "", "error": p.error})
                break
            obj_idx = {n: i for i, (_, n) in enumerate(OBJECTS)}
            box_idx = {n: j for j, (_, n) in enumerate(BOXES)}
            steps = [(obj_idx[o], box_idx[b]) for o, b in zip(p.objects, p.boxes)]
            passes.append({"steps": [instruction(o, b) for o, b in steps], "reasoning": p.reasoning})
            if not steps:
                break
            for obj, box in steps:
                if len(subtasks) >= self.max_subtasks:
                    break
                self.call("go_home", GoHome.Request())
                task = instruction(obj, box)
                rec: dict = {"task": task}
                self.status(header, f"手順 {len(subtasks) + 1}: {jp_step(obj, box)}  ({task})", "Claude が置き場所を指示中 ...")
                pt = self.call("point", Point.Request(image=self.render("front"), instruction=task))
                if not pt.ok:  # a failed call skips this step; the next pass re-plans
                    rec["error"] = pt.error
                    subtasks.append(rec)
                    continue
                r = self.call("retarget", Retarget.Request(obj=obj, box=box, vlm_points=list(pt.points),
                                                           snap_drop=self.snap_drop))
                if not r.ok:
                    rec["error"] = r.error
                    subtasks.append(rec)
                    continue
                pts = np.asarray(r.goal_points, dtype=np.float32)
                rec["ground_obj_err_px"] = round(float(r.ground_obj_err_px), 1)
                rec["drop_offset_mm"] = [round(float(v), 1) for v in r.drop_offset_mm]
                rec["points"] = pts.round(4).tolist()
                self.status(header, f"手順 {len(subtasks) + 1}: {jp_step(obj, box)}  ({task})", points=pts)
                ex = self.call("execute", Execute.Request(task=task, state=r.state, front=r.front, wrist=r.wrist))
                if not ex.ok:
                    rec["error"] = ex.error
                    subtasks.append(rec)
                    continue
                rec.update(success=bool(ex.success), wrong_object=bool(ex.wrong_object), contact=bool(ex.contact),
                           steps=int(ex.steps))
                subtasks.append(rec)
        final = json.loads(self.call("go_home", GoHome.Request()).in_box)  # privileged: scoring only
        final = {k: (v or None) for k, v in final.items()}
        tidied = sum(v is not None for v in final.values())
        row = {"seed": int(seed), "tidied": tidied, "final": final, "all_tidied": tidied == len(OBJECTS),
               "subtasks": subtasks, "passes": passes, "seconds": round(time.time() - t0, 1)}
        self.status(header, f"結果: {len(OBJECTS)} 個中 {tidied} 個が箱の中")
        self._spin_for(0.5)  # the status reaches video_node before the result closes the clip
        recording = self.count_subscribers("/episode/result") > 0
        n_saved = len(self.video_saved)
        self.pub_result.publish(String(data=json.dumps(
            {"seed": int(seed), "success": row["all_tidied"], "failure_mode": f"tidied{tidied}"})))
        if recording:  # wait until video_node has written the MP4 (it answers on /video/saved)
            end = time.time() + float(self.get_parameter("video_wait").value)
            while len(self.video_saved) == n_saved and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
            if len(self.video_saved) > n_saved:
                row["video"] = self.video_saved[-1]
        self.get_logger().info(json.dumps({k: v for k, v in row.items() if k != "subtasks"}, ensure_ascii=False))
        return row

    def run(self) -> dict:
        rows = [self.run_seed(s) for s in parse_seeds(self.get_parameter("seeds").value)]
        summary = {"command": self.command, "planner": "claude", "ground": "claude", "snap_drop": self.snap_drop,
                   "executor": str(self.get_parameter("executor").value), "n": len(rows),
                   "all_tidied_rate": sum(r["all_tidied"] for r in rows) / len(rows),
                   "mean_tidied": float(np.mean([r["tidied"] for r in rows])),
                   "mean_subtasks": float(np.mean([len(r["subtasks"]) for r in rows])), "episodes": rows}
        out = str(self.get_parameter("results_json").value)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        self.get_logger().info(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, ensure_ascii=False))
        self._spin_for(1.0)
        return summary


def main() -> None:
    rclpy.init()
    node = TidyNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
