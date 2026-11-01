"""Scripted teacher with privileged state (object pose from the simulator).

A phase machine in TCP space: hover -> descend -> grasp -> lift -> carry -> lower -> release -> retreat.
It emits the same 5-d action the learned policy will emit, so teacher rollouts are directly usable as
demonstrations and as DAgger-style corrections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import MAX_DPOS, MAX_DYAW, TidyEnv, wrap_angle

PHASES = ("hover", "descend", "grasp", "lift", "carry", "lower", "release", "retreat", "done")


@dataclass
class TeacherParams:
    hover_height: float = 0.12  # above the cube top
    grasp_z: float = 0.022  # tcp height at grasp (cube centre is at 0.02)
    lift_z: float = 0.18
    place_z: float = 0.10  # release height above the bin floor
    pos_tol: float = 0.006
    yaw_tol: float = 0.05
    grasp_wait: int = 6
    release_wait: int = 5
    speed: float = 1.0  # fraction of MAX_DPOS per step


class ScriptedTeacher:
    """Only needs the hand's home closing-axis yaw (a constant of the robot model), not the env itself,
    so it can run in a separate ROS 2 process from the simulator."""

    def __init__(self, home_close_yaw: float, params: TeacherParams | None = None):
        self.home_close_yaw = float(home_close_yaw)
        self.p = params or TeacherParams()
        self.reset()

    def reset(self) -> None:
        self.phase = "hover"
        self.wait = 0
        self.grasp_yaw: float | None = None

    # -- geometry helpers ------------------------------------------------------
    def _target_yaw(self, info: dict) -> float:
        """Hand yaw that aligns the closing axis with a cube face, closest to the current yaw."""
        cube_yaw = info["cube_yaw"]
        cur = info["yaw"]
        best = None
        for k in range(4):
            cand = wrap_angle(cube_yaw + k * np.pi / 2 - self.home_close_yaw)
            if best is None or abs(wrap_angle(cand - cur)) < abs(wrap_angle(best - cur)):
                best = cand
        return float(best)

    def _move(self, info: dict, target: np.ndarray, yaw: float, grip: float) -> tuple[np.ndarray, bool]:
        cur = info["tcp_cmd"]
        delta = target - cur
        dist = float(np.linalg.norm(delta))
        step = MAX_DPOS * self.p.speed
        if dist > step:
            delta = delta / dist * step
        dyaw = wrap_angle(yaw - info["yaw"])
        dyaw = float(np.clip(dyaw, -MAX_DYAW, MAX_DYAW))
        reached = dist < self.p.pos_tol and abs(wrap_angle(yaw - info["yaw"])) < self.p.yaw_tol
        return np.array([delta[0], delta[1], delta[2], dyaw, grip], dtype=np.float32), reached

    # -- policy ---------------------------------------------------------------
    def act(self, obs: dict, info: dict) -> np.ndarray:
        p = self.p
        cube = info["cube_pos"]
        bin_pos = info["bin_pos"]
        if self.grasp_yaw is None:
            self.grasp_yaw = self._target_yaw(info)

        if self.phase == "hover":
            target = np.array([cube[0], cube[1], cube[2] + 0.02 + p.hover_height])
            a, reached = self._move(info, target, self.grasp_yaw, 0.0)
            if reached:
                self.phase = "descend"
            return a
        if self.phase == "descend":
            # Re-aim at the (possibly nudged) cube while descending.
            target = np.array([cube[0], cube[1], p.grasp_z])
            a, reached = self._move(info, target, self.grasp_yaw, 0.0)
            if reached:
                self.phase, self.wait = "grasp", 0
            return a
        if self.phase == "grasp":
            a, _ = self._move(info, info["tcp_cmd"], self.grasp_yaw, 1.0)
            self.wait += 1
            if self.wait >= p.grasp_wait:
                self.phase = "lift"
            return a
        if self.phase == "lift":
            target = np.array([info["tcp_cmd"][0], info["tcp_cmd"][1], p.lift_z])
            a, reached = self._move(info, target, self.grasp_yaw, 1.0)
            if reached:
                self.phase = "carry"
            return a
        if self.phase == "carry":
            target = np.array([bin_pos[0], bin_pos[1], p.lift_z])
            a, reached = self._move(info, target, self.grasp_yaw, 1.0)
            if reached:
                self.phase = "lower"
            return a
        if self.phase == "lower":
            target = np.array([bin_pos[0], bin_pos[1], p.place_z])
            a, reached = self._move(info, target, self.grasp_yaw, 1.0)
            if reached:
                self.phase, self.wait = "release", 0
            return a
        if self.phase == "release":
            a, _ = self._move(info, info["tcp_cmd"], self.grasp_yaw, 0.0)
            self.wait += 1
            if self.wait >= p.release_wait:
                self.phase = "retreat"
            return a
        if self.phase == "retreat":
            target = np.array([bin_pos[0], bin_pos[1], p.lift_z])
            a, reached = self._move(info, target, self.grasp_yaw, 0.0)
            if reached:
                self.phase = "done"
            return a
        return np.zeros(5, dtype=np.float32)

    @property
    def done(self) -> bool:
        return self.phase == "done"


def run_teacher_episode(env: TidyEnv, seed: int, params: TeacherParams | None = None, record_frames: bool = False,
                        video_camera: str = "front", video_size: tuple[int, int] = (480, 640)) -> dict:
    """Roll out one teacher episode. Returns transitions and the outcome."""
    obs, info = env.reset(seed=seed)
    teacher = ScriptedTeacher(env.home_close_yaw, params)
    states, actions, images, frames, phases, env_states = [], [], [], [], [], []
    contact = False
    while True:
        a = teacher.act(obs, info)
        states.append(obs["state"])
        images.append(obs["images"])
        actions.append(a)
        phases.append(teacher.phase)
        env_states.append(np.array([*info["cube_pos"], info["cube_yaw"]], dtype=np.float32))
        if record_frames:
            frames.append(env.render(video_camera, video_size))
        obs, info = env.step(a)
        contact = contact or info["robot_contact"]
        if teacher.done or info["timeout"]:
            break
    # A few settle steps so success reflects the resting state.
    for _ in range(5):
        obs, info = env.step(np.zeros(5))
    return {
        "seed": seed,
        "success": bool(info["success"]),
        "robot_contact": contact,
        "steps": len(actions),
        "final_phase": teacher.phase,
        "states": np.stack(states),
        "actions": np.stack(actions),
        "env_states": np.stack(env_states),
        "images": images,
        "frames": frames,
        "phases": phases,
        "final_info": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()},
    }
