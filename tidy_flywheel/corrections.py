"""Correction demonstrations for the data flywheel.

Two ways to turn a failed policy episode (seed) into training data:

* ``redemo``   — the teacher demonstrates the task from the same initial state. Targets the initial-state
                 distribution the policy fails on.
* ``recovery`` — run the *policy* for ``handover`` steps from that seed, then hand control to the teacher,
                 which recovers from wherever the arm and cube ended up. Only the teacher-controlled suffix is
                 recorded, so every recorded (obs, action) pair is a valid expert label on a state the policy
                 actually visits (HG-DAgger style). ``handover`` is drawn uniformly in [0, t_fail].
"""
from __future__ import annotations

import numpy as np

from .sim import TidyEnv
from .teacher import ScriptedTeacher, TeacherParams


def run_recovery_episode(env: TidyEnv, policy, seed: int, handover: int, params: TeacherParams | None = None,
                         max_teacher_steps: int = 200) -> dict:
    obs, info = env.reset(seed=seed)
    policy.reset()
    for _ in range(handover):
        obs, info = env.step(policy.act(obs))
        if info["success"] or info["timeout"]:
            break
    handover_info = {"tcp": info["tcp_pos"].tolist(), "cube": info["cube_pos"].tolist(), "grasped": bool(info["grasped"]),
                     "t": int(info["t"])}
    teacher = ScriptedTeacher(env.home_close_yaw, params)
    # If the policy is holding the cube, let the teacher start from the lift phase instead of re-hovering.
    if info["grasped"]:
        teacher.grasp_yaw = info["yaw"]
        teacher.phase = "lift"
    states, actions, images, env_states = [], [], [], []
    contact = False
    env.config.max_steps = info["t"] + max_teacher_steps
    for _ in range(max_teacher_steps):
        a = teacher.act(obs, info)
        states.append(obs["state"])
        images.append(obs["images"])
        actions.append(a)
        env_states.append(np.array([*info["cube_pos"], info["cube_yaw"]], dtype=np.float32))
        obs, info = env.step(a)
        contact = contact or info["robot_contact"]
        if teacher.done or info["timeout"]:
            break
    for _ in range(5):
        obs, info = env.step(np.zeros(5))
    env.config.max_steps = 200
    return {
        "seed": int(seed),
        "mode": "recovery",
        "handover": int(handover),
        "handover_info": handover_info,
        "success": bool(info["success"]),
        "robot_contact": contact,
        "steps": len(actions),
        "final_phase": teacher.phase,
        "states": np.stack(states) if states else np.zeros((0, 5), np.float32),
        "actions": np.stack(actions) if actions else np.zeros((0, 5), np.float32),
        "env_states": np.stack(env_states) if env_states else np.zeros((0, 4), np.float32),
        "images": images,
    }
