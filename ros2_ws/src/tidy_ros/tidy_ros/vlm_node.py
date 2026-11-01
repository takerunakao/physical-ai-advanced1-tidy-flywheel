"""Language side of the tidy-up as ROS 2 services: LLM task planner and VLM pointer (Claude).

Services (servers only; this node never calls other nodes)
  /vlm/plan   (tidy_interfaces/Plan)   command + camera views (+ instructions already attempted)
                                       -> ordered [object, box] steps    (tidy_flywheel.vlm_plan.plan)
  /vlm/point  (tidy_interfaces/Point)  front image + instruction -> [u_obj, v_obj, u_drop, v_drop]
                                       (tidy_flywheel.vlm_ground.ground, backend "claude", view "front")
The API key is read from the environment (ANTHROPIC_API_KEY, optional ANTHROPIC_WORKSPACE_ID); in Docker it
comes from compose's env_file (.env on the host, never baked into the image). Key values are never logged.
"""
from __future__ import annotations

import os
import time

import numpy as np
import rclpy
from rclpy.node import Node

from tidy_interfaces.srv import Plan, Point

from .common import image_to_array


class VlmNode(Node):
    def __init__(self) -> None:
        super().__init__("vlm_node")
        self._client = None
        self.create_service(Plan, "/vlm/plan", self.on_plan)
        self.create_service(Point, "/vlm/point", self.on_point)
        have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.get_logger().info(f"vlm_node ready (ANTHROPIC_API_KEY {'set' if have_key else 'NOT set'}; "
                               f"workspace header {'on' if os.environ.get('ANTHROPIC_WORKSPACE_ID') else 'off'})")

    @property
    def client(self):
        if self._client is None:
            from tidy_flywheel.vlm_ground import anthropic_client

            self._client = anthropic_client()
        return self._client

    def on_plan(self, req: Plan.Request, res: Plan.Response) -> Plan.Response:
        from tidy_flywheel.sim_multi import BOXES, OBJECTS
        from tidy_flywheel.vlm_plan import plan

        t0 = time.time()
        try:
            names = list(req.view_names) or [f"view{k}" for k in range(len(req.views))]
            if len(names) != len(req.views):
                raise ValueError(f"{len(req.views)} views but {len(names)} view_names")
            views = {n: image_to_array(img) for n, img in zip(names, req.views)}
            steps, why = plan(views, req.command, client=self.client, attempted=list(req.attempted))
        except Exception as e:  # noqa: BLE001 - reported to the caller, who decides
            res.ok, res.error = False, f"{type(e).__name__}: {e}"
            self.get_logger().error(f"plan failed: {res.error}")
            return res
        res.ok = True
        res.objects = [OBJECTS[o][1] for o, _ in steps]
        res.boxes = [BOXES[b][1] for _, b in steps]
        res.reasoning = why
        self.get_logger().info(f"plan ({time.time() - t0:.1f}s): {list(zip(res.objects, res.boxes))} | {why}")
        return res

    def on_point(self, req: Point.Request, res: Point.Response) -> Point.Response:
        from tidy_flywheel.vlm_ground import ground

        t0 = time.time()
        try:
            pts = ground(image_to_array(req.image), req.instruction, "claude", client=self.client, view="front")
        except Exception as e:  # noqa: BLE001
            res.ok, res.error = False, f"{type(e).__name__}: {e}"
            self.get_logger().error(f"point failed: {res.error}")
            return res
        res.ok = True
        res.points = [float(v) for v in np.asarray(pts, dtype=np.float32)]
        self.get_logger().info(f"point ({time.time() - t0:.1f}s) '{req.instruction}': {np.round(pts, 3).tolist()}")
        return res


def main() -> None:
    rclpy.init()
    node = VlmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
