"""Lock-step MuJoCo simulator as a ROS 2 node.

Parameters
  scene        single | multi | seq (tidy_flywheel.make_env); default single
  prefill_p    multi scene: probability that an object starts in a box (0 = all objects on the table, as in
               scripts/tidy_all.py)
  image_size, max_steps, publish_video, video_camera
Services
  /sim/reset  (tidy_interfaces/Reset)  seed -> initial observation
  /sim/step   (tidy_interfaces/Step)   action -> next observation + flags
  multi / seq scenes only (language-instructed tidy-up, see tidy_node):
  /sim/go_home   (tidy_interfaces/GoHome)    arm back to the start pose -> observation + in_box JSON (privileged,
                                             for scoring only)
  /sim/render    (tidy_interfaces/Render)    camera, size -> image (planner / pointer views)
  /sim/retarget  (tidy_interfaces/Retarget)  next instruction + VLM points -> observation with the new goal points
Topics (published on every reset/step)
  /sim/frame                  tidy_interfaces/Frame  (obs_t + action_t; the record of one transition)
  /camera/front/image_raw     sensor_msgs/Image
  /camera/wrist/image_raw     sensor_msgs/Image
  /camera/video/image_raw     sensor_msgs/Image (hi-res, only if publish_video:=true)
  /robot/state                std_msgs/Float32MultiArray
"""
from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from tidy_flywheel import SCENES, make_env
from tidy_interfaces.msg import Frame
from tidy_interfaces.srv import GoHome, Render, Reset, Retarget, Step

GROUND_RES = 480  # the VLM sees the front camera at 480x480 (scripts/tidy_all.py)


def to_image_msg(arr: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    return msg


class SimNode(Node):
    def __init__(self) -> None:
        super().__init__("sim_node")
        self.declare_parameter("image_size", 128)
        self.declare_parameter("max_steps", 200)
        self.declare_parameter("publish_video", False)
        self.declare_parameter("video_camera", "front")
        self.declare_parameter("scene", "single")
        self.declare_parameter("prefill_p", 0.0)
        size = int(self.get_parameter("image_size").value)
        self.scene = str(self.get_parameter("scene").value)
        if self.scene not in SCENES:
            raise ValueError(f"scene must be one of {SCENES}, got {self.scene!r}")
        self.env = make_env(self.scene, size)
        cfg = self.env.config
        cfg.max_steps = int(self.get_parameter("max_steps").value)
        if self.scene != "single":
            cfg.prefill_p = float(self.get_parameter("prefill_p").value)
        self.publish_video = bool(self.get_parameter("publish_video").value)
        self.video_camera = str(self.get_parameter("video_camera").value)

        self.pub_frame = self.create_publisher(Frame, "/sim/frame", 10)
        self.pub_front = self.create_publisher(Image, "/camera/front/image_raw", 10)
        self.pub_wrist = self.create_publisher(Image, "/camera/wrist/image_raw", 10)
        self.pub_video = self.create_publisher(Image, "/camera/video/image_raw", 10)
        self.pub_state = self.create_publisher(Float32MultiArray, "/robot/state", 10)
        self.create_service(Reset, "/sim/reset", self.on_reset)
        self.create_service(Step, "/sim/step", self.on_step)
        if self.scene != "single":
            self.create_service(GoHome, "/sim/go_home", self.on_go_home)
            self.create_service(Render, "/sim/render", self.on_render)
            self.create_service(Retarget, "/sim/retarget", self.on_retarget)

        self.seed = -1
        self.obs = None
        self.info = None
        self.get_logger().info(f"sim_node ready (scene {self.scene}, image {size}px, max_steps {cfg.max_steps})")

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _cube_pose(info: dict) -> list[float]:
        return [float(v) for v in info["cube_pos"]] + [float(info["cube_yaw"])]

    def _image_msgs(self) -> tuple[Image, Image]:
        stamp = self.get_clock().now().to_msg()
        return (to_image_msg(self.obs["images"]["front"], stamp, "front"),
                to_image_msg(self.obs["images"]["wrist"], stamp, "wrist"))

    def _publish(self, action: np.ndarray | None) -> None:
        """Publish obs_t (cached) together with action_t (None at reset)."""
        stamp = self.get_clock().now().to_msg()
        obs, info = self.obs, self.info
        front = to_image_msg(obs["images"]["front"], stamp, "front")
        wrist = to_image_msg(obs["images"]["wrist"], stamp, "wrist")
        f = Frame()
        f.header.stamp = stamp
        f.header.frame_id = "sim"
        f.seed = int(self.seed)
        f.t = int(info["t"])
        f.front, f.wrist = front, wrist
        f.state = [float(v) for v in obs["state"]]
        f.action = [] if action is None else [float(v) for v in action]
        f.cube_pose = self._cube_pose(info)
        f.grasped = bool(info["grasped"])
        if action is not None:  # a frame is a transition; the reset observation is published once the first action arrives
            self.pub_frame.publish(f)
        self.pub_front.publish(front)
        self.pub_wrist.publish(wrist)
        st = Float32MultiArray()
        st.data = f.state
        self.pub_state.publish(st)
        self._publish_video(stamp)

    def _publish_video(self, stamp=None) -> None:
        if self.publish_video:
            stamp = stamp or self.get_clock().now().to_msg()
            self.pub_video.publish(to_image_msg(self.env.render(self.video_camera, (480, 640)), stamp, self.video_camera))

    # -- services --------------------------------------------------------------
    def on_reset(self, req: Reset.Request, res: Reset.Response) -> Reset.Response:
        self.seed = int(req.seed)
        self.obs, self.info = self.env.reset(seed=self.seed)
        self._publish(None)
        res.ok = True
        res.state = [float(v) for v in self.obs["state"]]
        res.front, res.wrist = self._image_msgs()
        res.cube_pose = self._cube_pose(self.info)
        res.bin_pos = [float(v) for v in self.info["bin_pos"]]
        return res

    def on_step(self, req: Step.Request, res: Step.Response) -> Step.Response:
        if self.obs is None:
            res.ok = False
            return res
        action = np.asarray(req.action, dtype=np.float32).reshape(5)
        self._publish(action)  # obs_t paired with action_t
        self.obs, self.info = self.env.step(action)
        info = self.info
        res.ok = True
        res.state = [float(v) for v in self.obs["state"]]
        res.front, res.wrist = self._image_msgs()
        res.cube_pose = self._cube_pose(info)
        res.tcp_cmd = [float(v) for v in info["tcp_cmd"]]
        res.grasped = bool(info["grasped"])
        res.success = bool(info["success"])
        res.robot_contact = bool(info["robot_contact"])
        res.timeout = bool(info["timeout"])
        res.t = int(info["t"])
        res.wrong_object = bool(info.get("wrong_object", False))
        return res

    # -- multi-object scenes ---------------------------------------------------
    def _fill_obs(self, res) -> None:
        res.state = [float(v) for v in self.obs["state"]]
        res.front, res.wrist = self._image_msgs()

    def on_go_home(self, req: GoHome.Request, res: GoHome.Response) -> GoHome.Response:
        from tidy_flywheel.sim_multi import BOXES, OBJECTS

        if self.obs is None:
            res.ok = False
            return res
        env = self.env
        self.obs = env.go_home()
        self.info = env._info()
        self._publish_video()  # one frame of the homed arm (the final frame of an episode)
        # Privileged: which box each object rests in. Only the orchestrator's scoring reads this.
        in_box = {name: next((BOXES[j][1] for j in range(len(BOXES)) if env._in_box(env.obj_pos(i), j)), "")
                  for i, (_, name) in enumerate(OBJECTS)}
        res.ok = True
        self._fill_obs(res)
        res.in_box = json.dumps(in_box)
        return res

    def on_render(self, req: Render.Request, res: Render.Response) -> Render.Response:
        size = int(req.size) or GROUND_RES
        try:
            img = self.env.render(str(req.camera), (size, size))
        except Exception as e:  # noqa: BLE001 - e.g. unknown camera name
            res.ok, res.error = False, repr(e)
            return res
        res.ok = True
        res.image = to_image_msg(img, self.get_clock().now().to_msg(), str(req.camera))
        return res

    def on_retarget(self, req: Retarget.Request, res: Retarget.Response) -> Retarget.Response:
        """Same bookkeeping as scripts/tidy_all.py: retarget, map the VLM's points to goal points, aim the drop."""
        env = self.env
        if self.obs is None:
            res.ok, res.error = False, "call /sim/reset first"
            return res
        if len(req.vlm_points) not in (0, 4):
            res.ok, res.error = False, f"vlm_points must have 0 or 4 values, got {len(req.vlm_points)}"
            return res
        obj, box = int(req.obj), int(req.box)
        self.obs, self.info = env.retarget(obj, box)
        res.ground_obj_err_px = -1.0
        if len(req.vlm_points) == 4:
            pts = env.to_goal_points(np.asarray(req.vlm_points, dtype=np.float32))
            true_obj = env.true_goal_points()[:2]
            res.ground_obj_err_px = round(float(np.linalg.norm(pts[:2] - true_obj) * GROUND_RES), 1)
            drop = env.unproject(pts[2:], env.box_pos(box)[2])  # where the VLM wants it, on the box floor
            env.place_offset = (drop - env.box_pos(box))[:2]
            if req.snap_drop:  # the demos drop 3 cm left/right of centre; map the VLM's point onto that
                side = np.sign(env.place_offset[0]) or -1.0
                env.place_offset = np.array([side * env.config.place_offset, 0.0])
                pts[2:] = env.project(env.bin_pos)
            self.obs = env.set_goal_points(pts)
            self.info = env._info()
        res.ok = True
        self._fill_obs(res)
        res.goal_points = [float(v) for v in env.goal_points]
        res.task = env.task
        res.drop_offset_mm = [float(v) for v in (env.place_offset * 1000).round(1)]
        return res


def main() -> None:
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.env.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
