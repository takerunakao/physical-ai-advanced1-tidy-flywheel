"""Tabletop tidy-up environment on MuJoCo.

The Panda arm is driven through a mocap body welded to the hand, so the action space is a
Cartesian end-effector delta. The environment is stepped in lock-step (no wall-clock dependence)
and is fully deterministic for a given seed.

Action (5,): [dx, dy, dz, dyaw, grip]
  dx/dy/dz  : TCP translation per control step [m], clipped to +-MAX_DPOS
  dyaw      : TCP yaw increment per control step [rad], clipped to +-MAX_DYAW
  grip      : > 0.5 closes the gripper, otherwise opens it

Observation:
  state (5,) : [tcp_x, tcp_y, tcp_z, tcp_yaw, gripper_opening]
  images     : dict camera_name -> uint8 (H, W, 3)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSET_DIR / "scene.xml"

TCP_OFFSET = np.array([0.0, 0.0, 0.1034])  # tcp site in the hand frame
MAX_DPOS = 0.02  # m per control step
MAX_DYAW = 0.2  # rad per control step
GRIP_OPEN = 255.0
GRIP_CLOSE = 0.0


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, q1, q2)
    return out


def quat_z(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class EpisodeConfig:
    """Randomisation ranges. Kept small on purpose: difficulty is a tuning knob (PLAN §4.4)."""

    cube_x: tuple[float, float] = (0.40, 0.60)
    cube_y: tuple[float, float] = (0.02, 0.28)
    cube_yaw: tuple[float, float] = (-np.pi, np.pi)
    max_steps: int = 200


@dataclass
class TidyEnv:
    image_size: tuple[int, int] = (128, 128)  # (height, width)
    cameras: tuple[str, ...] = ("front", "wrist")
    control_dt: float = 0.1
    config: EpisodeConfig = field(default_factory=EpisodeConfig)
    scene_xml: Path = SCENE_XML

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(self.control_dt / self.model.opt.timestep))
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

        m = self.model
        self.key_home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.hand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.target_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap_id = m.body_mocapid[self.target_bid]
        self.tcp_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.bin_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "bin_center")
        self.cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qadr = m.jnt_qposadr[self.cube_jid]
        self.grip_aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.finger_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.finger_qadr = m.jnt_qposadr[self.finger_jid]
        self.arm_qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 8)]

        # Bin geometry (inner half extents) from the wall geoms.
        self.bin_half = 0.06
        self.bin_wall_top = 0.06

        # Canonical "pointing down" hand orientation = hand orientation at the home keyframe.
        mujoco.mj_resetDataKeyframe(m, self.data, self.key_home)
        mujoco.mj_forward(m, self.data)
        self.q_home = self.data.xquat[self.hand_bid].copy()
        # Yaw of the gripper closing axis (hand y) in the world at the home pose.
        y_axis = quat_to_mat(self.q_home)[:, 1]
        self.home_close_yaw = float(np.arctan2(y_axis[1], y_axis[0]))

        self.rng = np.random.default_rng(0)
        self.t = 0
        self.yaw = 0.0
        self.grip_cmd = 0.0

    # ------------------------------------------------------------------ helpers
    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_sid].copy()

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_bid].copy()

    @property
    def cube_quat(self) -> np.ndarray:
        return self.data.xquat[self.cube_bid].copy()

    @property
    def cube_yaw(self) -> float:
        x_axis = quat_to_mat(self.cube_quat)[:, 0]
        return float(np.arctan2(x_axis[1], x_axis[0]))

    @property
    def bin_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.bin_sid].copy()

    @property
    def gripper_opening(self) -> float:
        return float(self.data.qpos[self.finger_qadr])

    def hand_quat(self, yaw: float) -> np.ndarray:
        return quat_mul(quat_z(yaw), self.q_home)

    def _set_mocap(self, tcp_target: np.ndarray, yaw: float) -> None:
        q = self.hand_quat(yaw)
        hand_pos = tcp_target - quat_to_mat(q) @ TCP_OFFSET
        self.data.mocap_pos[self.mocap_id] = hand_pos
        self.data.mocap_quat[self.mocap_id] = q

    def _is_grasped(self) -> bool:
        """Cube is between the fingers: closed on something and lifted with the tcp."""
        d = np.linalg.norm(self.cube_pos - self.tcp_pos)
        return self.grip_cmd > 0.5 and self.gripper_opening > 0.005 and d < 0.03

    def in_bin(self, pos: np.ndarray) -> bool:
        rel = pos - self.bin_pos
        return bool(abs(rel[0]) < self.bin_half and abs(rel[1]) < self.bin_half and 0.0 < rel[2] < self.bin_wall_top)

    def success(self) -> bool:
        return self.in_bin(self.cube_pos) and not self._is_grasped() and float(np.linalg.norm(self.data.qvel[self.cube_qadr : self.cube_qadr + 3])) < 0.05

    def robot_table_contact(self) -> bool:
        """True if any arm/hand geom (not the fingers) touches the table or bin."""
        m, d = self.model, self.data
        table_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        bin_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bin")
        finger_bids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_finger", "right_finger")}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for g, b, other_b in ((c.geom1, b1, b2), (c.geom2, b2, b1)):
                is_static = g == table_gid or b == bin_bid
                if is_static and other_b not in finger_bids and other_b not in (0, bin_bid) and other_b != self.cube_bid:
                    return True
        return False

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        m, d = self.model, self.data
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(m, d, self.key_home)
        cfg = self.config
        x = self.rng.uniform(*cfg.cube_x)
        y = self.rng.uniform(*cfg.cube_y)
        yaw = self.rng.uniform(*cfg.cube_yaw)
        d.qpos[self.cube_qadr : self.cube_qadr + 3] = [x, y, 0.02]
        d.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = quat_z(yaw)
        d.qvel[:] = 0
        self.yaw = 0.0
        self.grip_cmd = 0.0
        d.ctrl[self.grip_aid] = GRIP_OPEN
        mujoco.mj_forward(m, d)
        # Weld target = current hand pose so the arm starts at rest.
        d.mocap_pos[self.mocap_id] = d.xpos[self.hand_bid]
        d.mocap_quat[self.mocap_id] = d.xquat[self.hand_bid]
        self.start_tcp = self.tcp_pos
        self._tcp_cmd = self.tcp_pos
        self.t = 0
        # Let contacts settle.
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
        return self._obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[dict, dict]:
        a = np.asarray(action, dtype=np.float64).reshape(5)
        dpos = np.clip(a[:3], -MAX_DPOS, MAX_DPOS)
        dyaw = float(np.clip(a[3], -MAX_DYAW, MAX_DYAW))
        self._tcp_cmd = self._tcp_cmd + dpos
        # Workspace box keeps the weld from dragging the arm into the table or out of reach.
        self._tcp_cmd = np.clip(self._tcp_cmd, [0.25, -0.45, 0.005], [0.70, 0.45, 0.60])
        self.yaw = wrap_angle(self.yaw + dyaw)
        self.grip_cmd = 1.0 if a[4] > 0.5 else 0.0
        self.data.ctrl[self.grip_aid] = GRIP_CLOSE if self.grip_cmd > 0.5 else GRIP_OPEN
        self._set_mocap(self._tcp_cmd, self.yaw)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        return self._obs(), self._info()

    def _obs(self) -> dict:
        tcp = self.tcp_pos
        state = np.array([tcp[0], tcp[1], tcp[2], self.yaw, self.gripper_opening], dtype=np.float32)
        return {"state": state, "images": {cam: self.render(cam) for cam in self.cameras}}

    def _info(self) -> dict:
        return {
            "t": self.t,
            "tcp_pos": self.tcp_pos,
            "tcp_cmd": self._tcp_cmd.copy(),
            "yaw": self.yaw,
            "grip_cmd": self.grip_cmd,
            "cube_pos": self.cube_pos,
            "cube_yaw": self.cube_yaw,
            "bin_pos": self.bin_pos,
            "grasped": self._is_grasped(),
            "success": self.success(),
            "robot_contact": self.robot_table_contact(),
            "timeout": self.t >= self.config.max_steps,
        }

    def render(self, camera: str = "front", size: tuple[int, int] | None = None) -> np.ndarray:
        h, w = size or self.image_size
        r = self._renderers.get((h, w))
        if r is None:  # one renderer per output size; re-creating one per call costs ~100 ms
            r = self._renderers[(h, w)] = mujoco.Renderer(self.model, height=h, width=w)
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
