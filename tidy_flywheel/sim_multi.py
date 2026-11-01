"""Language-instructed tidy-up: three objects, two boxes, one instruction per episode.

    "put the blue cube in the black box"

The instruction reaches the policy as two goal points in the front camera image (normalised u, v):
the object to pick and the box to place it in. During data collection the points are the exact projections
of the simulator's poses. At deployment a VLM supplies them by pointing at the image from the text instruction
(see vlm_ground.py), which is the only place language enters. Low-level control stays a visuomotor policy.

The env exposes the *instructed* object and box under the names the single-object code already uses
(cube_pos, cube_yaw, bin_pos, ...), so the scripted teacher, recovery demos and evaluator work unchanged.

Sequential tidy-up (several instructions in one scene, no reset): `retarget(obj, box)` switches the instruction
after `go_home()`; objects already tidied (`done_objs`) are ignored by the success and wrong-object checks.

Observation state (9,): [tcp_x, tcp_y, tcp_z, tcp_yaw, gripper_opening, u_obj, v_obj, u_box, v_box]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .sim import ASSET_DIR, GRIP_OPEN, MAX_DPOS, EpisodeConfig, TidyEnv, quat_z

SCENE_MULTI_XML = ASSET_DIR / "scene_multi.xml"

OBJECTS = (("obj0", "red cube"), ("obj1", "blue cube"), ("obj2", "green cylinder"))
BOXES = (("bin0", "white box"), ("bin1", "black box"))
GOAL_CAMERA = "front"
OBJ_CENTRE_Z = 0.02  # objects are 4 cm tall
BOX_POINT_Z = 0.03  # a VLM pointing "into" an open box points at the middle of its volume, half the 6 cm wall


def instruction(obj: int, box: int) -> str:
    return f"put the {OBJECTS[obj][1]} in the {BOXES[box][1]}"


@dataclass
class MultiEpisodeConfig(EpisodeConfig):
    obj_x: tuple[float, float] = (0.38, 0.62)
    obj_y: tuple[float, float] = (-0.02, 0.30)
    min_dist: float = 0.11  # between object centres; leaves room for the open fingers
    place_offset: float = 0.03  # the drop point is sampled within +-this of the box centre (x and y)
    prefill_p: float = 0.5  # probability that one other object already rests in a box (mid-tidy-up scenes)
    prefill_side: float = 0.035  # prefilled object sits this far off-centre; the drop point takes the other side
    # "seq": states met while tidying the whole table -- 0, 1 or 2 objects already in boxes, anywhere on one side
    # of a box floor and at any yaw, as a learned policy leaves them (overrides prefill_p / prefill_side).
    prefill_mode: str = "one"  # "one" | "seq"
    seq_counts_p: tuple[float, float, float] = (0.25, 0.45, 0.30)
    max_steps: int = 200


@dataclass
class MultiTidyEnv(TidyEnv):
    config: MultiEpisodeConfig = field(default_factory=MultiEpisodeConfig)
    scene_xml: object = SCENE_MULTI_XML

    def __post_init__(self) -> None:
        super().__post_init__()
        m = self.model
        self.obj_bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n, _ in OBJECTS]
        self.obj_qadrs = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")] for n, _ in OBJECTS]
        self.box_bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n, _ in BOXES]
        self.box_sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{n}_center") for n, _ in BOXES]
        self.table_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.finger_bids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_finger", "right_finger")}
        self._select(0, 0)
        self.done_objs: set[int] = set()
        self.place_offset = np.zeros(2)
        self.goal_points = np.zeros(4, dtype=np.float32)

    def _select(self, obj: int, box: int) -> None:
        """Point the single-object attributes (cube_*, bin_*) at the instructed object and box."""
        self.target_obj, self.target_box = int(obj), int(box)
        self.cube_bid = self.obj_bids[obj]
        self.cube_qadr = self.obj_qadrs[obj]
        self.bin_sid = self.box_sids[box]

    @property
    def task(self) -> str:
        return instruction(self.target_obj, self.target_box)

    # ------------------------------------------------------------------ geometry
    def project(self, pos: np.ndarray, camera: str = GOAL_CAMERA) -> np.ndarray:
        """World point -> normalised image coordinates (u right, v down) of `camera`, any resolution."""
        m, d = self.model, self.data
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        rel = d.cam_xmat[cid].reshape(3, 3).T @ (np.asarray(pos) - d.cam_xpos[cid])  # camera frame, looks along -z
        f = 0.5 / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)  # focal length in units of image height
        h, w = self.image_size
        u = 0.5 + f * (rel[0] / -rel[2]) * h / w
        v = 0.5 - f * (rel[1] / -rel[2])
        return np.array([u, v], dtype=np.float32)

    def unproject(self, uv: np.ndarray, z: float, camera: str = GOAL_CAMERA) -> np.ndarray:
        """Normalised image point -> world point on the horizontal plane at height z (inverse of project)."""
        m, d = self.model, self.data
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        f = 0.5 / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
        h, w = self.image_size
        ray_cam = np.array([(uv[0] - 0.5) * w / h / f, -(uv[1] - 0.5) / f, -1.0])
        R, o = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
        ray = R @ ray_cam
        return o + ray * (z - o[2]) / ray[2]

    def pointed_world(self, points: np.ndarray, camera: str = GOAL_CAMERA) -> tuple[np.ndarray, np.ndarray]:
        """A VLM's [object, drop] points in `camera`'s image -> world points (object centre, drop point on the box
        floor). The drop point is read at mid-wall height, where an oblique view of the box puts its interior."""
        obj = self.unproject(points[:2], OBJ_CENTRE_Z, camera)
        drop = self.unproject(points[2:], self.box_pos(0)[2] + BOX_POINT_Z, camera)
        drop[2] = self.box_pos(0)[2]
        return obj, drop

    def to_goal_points(self, points: np.ndarray, camera: str = GOAL_CAMERA) -> np.ndarray:
        """A VLM's points -> the policy's goal points (exact projections of the implied object and floor drop point)."""
        obj, drop = self.pointed_world(points, camera)
        return np.concatenate([self.project(obj), self.project(drop)]).astype(np.float32)

    def obj_pos(self, i: int) -> np.ndarray:
        return self.data.xpos[self.obj_bids[i]].copy()

    def box_pos(self, j: int) -> np.ndarray:
        return self.data.site_xpos[self.box_sids[j]].copy()

    @property
    def bin_pos(self) -> np.ndarray:
        """Drop point: the instructed box's floor centre plus this episode's offset (what the teacher aims at)."""
        return self.box_pos(self.target_box) + np.array([*self.place_offset, 0.0])

    def in_bin(self, pos: np.ndarray) -> bool:
        rel = pos - self.box_pos(self.target_box)
        return bool(abs(rel[0]) < self.bin_half and abs(rel[1]) < self.bin_half and 0.0 < rel[2] < self.bin_wall_top)

    def true_goal_points(self) -> np.ndarray:
        """[object centre, drop point] in the goal camera; the box point says *where* in the box to put it."""
        return np.concatenate([self.project(self.obj_pos(self.target_obj)), self.project(self.bin_pos)])

    def free_offset(self, box: int) -> np.ndarray:
        """A drop offset away from whatever already rests in `box`; an empty box is filled from one side (-x) so a
        second object still fits next to the first."""
        rels = [self.obj_pos(i)[:2] - self.box_pos(box)[:2] for i in range(len(OBJECTS)) if self._in_box(self.obj_pos(i), box)]
        if not rels:
            return np.array([-self.config.place_offset, 0.0])
        c = np.mean(rels, axis=0)
        return -self.config.place_offset * c / max(np.linalg.norm(c), 1e-6)

    # ------------------------------------------------------------------ outcome
    def _in_box(self, pos: np.ndarray, j: int) -> bool:
        rel = pos - self.box_pos(j)
        return bool(abs(rel[0]) < self.bin_half and abs(rel[1]) < self.bin_half and 0.0 < rel[2] < self.bin_wall_top)

    def _in_any_box(self, pos: np.ndarray) -> bool:
        return any(self._in_box(pos, j) for j in range(len(BOXES)))

    def wrong_object_moved(self) -> bool:
        """A non-instructed object ended in a box or was pushed more than 3 cm."""
        for i in range(len(OBJECTS)):
            if i == self.target_obj or i in self.done_objs:
                continue
            p = self.obj_pos(i)
            if self._in_any_box(p) or np.linalg.norm(p[:2] - self.obj_start[i][:2]) > 0.03:
                return True
        return False

    def success(self) -> bool:
        return super().success() and not any(
            self._in_any_box(self.obj_pos(i)) for i in range(len(OBJECTS)) if i != self.target_obj and i not in self.done_objs)

    def robot_table_contact(self) -> bool:
        """True if any arm/hand geom (not the fingers) touches the table or a box."""
        m, d = self.model, self.data
        static_bodies = set(self.box_bids)
        allowed = self.finger_bids | static_bodies | set(self.obj_bids) | {0}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for g, b, other in ((c.geom1, b1, b2), (c.geom2, b2, b1)):
                if (g == self.table_gid or b in static_bodies) and other not in allowed:
                    return True
        return False

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, obj: int | None = None, box: int | None = None) -> tuple[dict, dict]:
        m, d = self.model, self.data
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(m, d, self.key_home)
        mujoco.mj_forward(m, d)  # box sites (static) for the prefill placement below
        cfg = self.config
        placed: list[np.ndarray] = []
        while len(placed) < len(OBJECTS):
            p = np.array([self.rng.uniform(*cfg.obj_x), self.rng.uniform(*cfg.obj_y)])
            if all(np.linalg.norm(p - q) >= cfg.min_dist for q in placed):
                placed.append(p)
        for qadr, p in zip(self.obj_qadrs, placed):
            d.qpos[qadr : qadr + 3] = [p[0], p[1], 0.02]
            d.qpos[qadr + 3 : qadr + 7] = quat_z(self.rng.uniform(-np.pi, np.pi))
        k = int(self.rng.integers(len(OBJECTS)))
        b = int(self.rng.integers(len(BOXES)))
        self._select(k if obj is None else obj, b if box is None else box)
        # Mid-tidy-up scenes: one other object already rests on one side of a box; the drop point avoids it.
        self.place_offset = self.rng.uniform(-cfg.place_offset, cfg.place_offset, 2)
        if cfg.prefill_mode == "seq":
            self._prefill_seq()
        elif self.rng.random() < cfg.prefill_p:
            others = [i for i in range(len(OBJECTS)) if i != self.target_obj]
            j, bj = int(self.rng.choice(others)), int(self.rng.integers(len(BOXES)))
            side = self.rng.choice([-1.0, 1.0])
            c = self.box_pos(bj)
            qadr = self.obj_qadrs[j]
            d.qpos[qadr : qadr + 3] = [c[0] + side * cfg.prefill_side, c[1] + self.rng.uniform(-0.02, 0.02), c[2] + 0.025]
            if bj == self.target_box:
                self.place_offset = np.array([-side * cfg.place_offset, self.rng.uniform(-0.02, 0.02)])
        d.qvel[:] = 0
        self.yaw = 0.0
        self.grip_cmd = 0.0
        d.ctrl[self.grip_aid] = GRIP_OPEN
        mujoco.mj_forward(m, d)
        d.mocap_pos[self.mocap_id] = d.xpos[self.hand_bid]
        d.mocap_quat[self.mocap_id] = d.xquat[self.hand_bid]
        self.start_tcp = self.tcp_pos
        self._tcp_cmd = self.tcp_pos
        self.t = 0
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
        self.done_objs = {i for i in range(len(OBJECTS)) if self._in_any_box(self.obj_pos(i))}
        self.obj_start = [self.obj_pos(i) for i in range(len(OBJECTS))]
        self.goal_points = self.true_goal_points()
        return self._obs(), self._info()

    def _prefill_seq(self) -> None:
        """Put 0-2 of the other objects into boxes (at most 2 per box, the target box keeps a free side) and aim the
        drop point at the side away from what is already there."""
        cfg, d = self.config, self.data
        n = int(self.rng.choice(3, p=cfg.seq_counts_p))
        others = [int(i) for i in self.rng.permutation([i for i in range(len(OBJECTS)) if i != self.target_obj])][:n]
        in_target = 0
        sides: dict[int, list[float]] = {j: [] for j in range(len(BOXES))}
        for j in others:
            free = [b for b in sides if len(sides[b]) < (1 if b == self.target_box else 2)]
            bj = int(self.rng.choice(free))
            side = -sides[bj][0] if sides[bj] else float(self.rng.choice([-1.0, 1.0]))
            sides[bj].append(side)
            in_target += bj == self.target_box
            c = self.box_pos(bj)
            qadr = self.obj_qadrs[j]
            d.qpos[qadr : qadr + 3] = [c[0] + side * self.rng.uniform(0.025, 0.04), c[1] + self.rng.uniform(-0.03, 0.03),
                                       c[2] + 0.025]
            d.qpos[qadr + 3 : qadr + 7] = quat_z(self.rng.uniform(-np.pi, np.pi))
        mujoco.mj_forward(self.model, d)
        if in_target:
            occ = [self.obj_pos(i)[:2] - self.box_pos(self.target_box)[:2] for i in others
                   if self._in_box(self.obj_pos(i), self.target_box)]
            c = np.mean(occ, axis=0) if occ else np.array([1.0, 0.0])
            u = -c / max(np.linalg.norm(c), 1e-6)
            self.place_offset = u * self.rng.uniform(0.025, 0.035) + self.rng.uniform(-0.008, 0.008, 2)

    def remaining(self) -> list[int]:
        """Objects not yet resting in a box."""
        return [i for i in range(len(OBJECTS)) if not self._in_any_box(self.obj_pos(i))]

    def go_home(self, max_steps: int = 60, tol: float = 0.01) -> dict:
        """Scripted return to the start pose with the gripper open (between instructions), like a skill primitive."""
        obs = self._obs()
        for _ in range(max_steps):
            delta = self.start_tcp - self._tcp_cmd
            if np.linalg.norm(delta) < tol and abs(self.yaw) < 0.05:
                break
            step = delta * min(1.0, MAX_DPOS / max(np.abs(delta).max(), 1e-9))
            obs, _ = self.step(np.array([*step, -self.yaw, 0.0]))
        return obs

    def retarget(self, obj: int, box: int, offset: np.ndarray | None = None) -> tuple[dict, dict]:
        """Next instruction in the same scene: objects already in a box count as done, the step budget restarts.
        The drop point defaults to free space in the box (the VLM's pointing replaces it at deployment)."""
        self.done_objs = {i for i in range(len(OBJECTS)) if self._in_any_box(self.obj_pos(i))}
        self._select(obj, box)
        self.place_offset = self.free_offset(box) if offset is None else np.asarray(offset, dtype=float)
        self.obj_start = [self.obj_pos(i) for i in range(len(OBJECTS))]
        self.goal_points = self.true_goal_points()
        self.t = 0
        return self._obs(), self._info()

    def set_goal_points(self, points: np.ndarray) -> dict:
        """Replace the goal points (e.g. with a VLM's pointing) and return the refreshed observation."""
        self.goal_points = np.asarray(points, dtype=np.float32).reshape(4)
        return self._obs()

    def _obs(self) -> dict:
        obs = super()._obs()
        obs["state"] = np.concatenate([obs["state"], self.goal_points]).astype(np.float32)
        return obs

    def _info(self) -> dict:
        info = super()._info()
        info.update(task=self.task, target_obj=self.target_obj, target_box=self.target_box,
                    wrong_object=self.wrong_object_moved(), goal_points=self.goal_points.copy())
        return info
