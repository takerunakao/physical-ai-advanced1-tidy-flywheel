"""Held-out evaluation of a policy in the simulator.

Outputs one JSON per run with per-episode outcomes, a failure-mode label per episode, and (optionally) an
MP4 per failed episode so failures can be inspected and re-used as DAgger seeds.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio
import numpy as np

from .sim import TidyEnv

FAILURE_MODES = ("success", "no_grasp", "dropped", "missed_bin", "timeout_holding", "contact", "wrong_object")


def classify(ep: dict) -> str:
    if ep["success"]:
        return "success"
    if ep["robot_contact"]:
        return "contact"
    if ep.get("wrong_object"):  # multi scene: a non-instructed object was moved or boxed
        return "wrong_object"
    if not ep["ever_grasped"]:
        return "no_grasp"
    if ep["grasped_at_end"]:
        return "timeout_holding"
    if ep["cube_moved"]:
        return "missed_bin" if ep["cube_near_bin"] else "dropped"
    return "no_grasp"


def run_episode(env: TidyEnv, policy, seed: int, record: bool = False, video_camera: str = "front",
                video_size: tuple[int, int] = (480, 640), grounder=None) -> dict:
    """grounder (multi scene): fn(image, instruction) -> points in the front image (e.g. a VLM), converted to goal
    points with env.to_goal_points and replacing the exact ones."""
    obs, info = env.reset(seed=seed)
    policy.reset()
    grounding = {}
    if grounder is not None:
        true_pts = info["goal_points"].copy()
        pts = env.to_goal_points(np.asarray(grounder(env.render("front", (480, 480)), info["task"]), dtype=np.float32))
        obs = env.set_goal_points(pts)
        grounding = {"goal_points": pts.round(4).tolist(),
                     "ground_err_px": [round(float(np.linalg.norm(pts[i:i + 2] - true_pts[i:i + 2]) * 480), 1) for i in (0, 2)]}
    cube0 = info["cube_pos"].copy()
    frames, ever_grasped, contact = [], False, False
    t0 = time.time()
    while True:
        if record:
            frames.append(env.render(video_camera, video_size))
        a = policy.act(obs)
        obs, info = env.step(a)
        ever_grasped = ever_grasped or info["grasped"]
        contact = contact or info["robot_contact"]
        if info["success"] or info["timeout"]:
            break
    if record:
        frames.append(env.render(video_camera, video_size))
    cube = info["cube_pos"]
    ep = {
        "seed": int(seed),
        "success": bool(info["success"]),
        "steps": int(info["t"]),
        "ever_grasped": bool(ever_grasped),
        "grasped_at_end": bool(info["grasped"]),
        "robot_contact": bool(contact),
        "cube_moved": bool(np.linalg.norm(cube[:2] - cube0[:2]) > 0.01),
        "cube_near_bin": bool(np.linalg.norm(cube[:2] - info["bin_pos"][:2]) < 0.12),
        "final_cube_pos": [float(v) for v in cube],
        **({"task": info["task"], "wrong_object": bool(info["wrong_object"])} if "task" in info else {}),
        **grounding,
        "wall_time_s": round(time.time() - t0, 2),
    }
    ep["failure_mode"] = classify(ep)
    ep["_frames"] = frames
    return ep


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def judge_agreement(episodes: list[dict]) -> dict:
    """Confusion of the VLM verdict against the privileged outcome (positive = success)."""
    c = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "errors": 0}
    for e in episodes:
        v = e["vlm"]["success"]
        if v is None:
            c["errors"] += 1
        else:
            c[("t" if v == e["success"] else "f") + ("p" if v else "n")] += 1
    return c


def evaluate(env: TidyEnv, policy, seeds: list[int], out_dir: str | Path, tag: str, save_failure_videos: bool = True,
             save_success_videos: int = 0, verbose: bool = True, grounder=None, judge=None) -> dict:
    """judge: fn([(step, frame), ...], task) -> Verdict (vlm_judge). Every episode is then also judged from its camera frames
    alone, alongside the simulation, and the verdict is stored under ep["vlm"] next to the privileged outcome."""
    out_dir = Path(out_dir)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    episodes, n_success_videos, verdicts = [], 0, []
    pool = ThreadPoolExecutor(4) if judge is not None else None
    t0 = time.time()
    for seed in seeds:
        record = save_failure_videos or save_success_videos > 0 or judge is not None
        ep = run_episode(env, policy, seed, record=record, grounder=grounder)
        frames = ep.pop("_frames")
        if pool is not None:
            from .vlm_judge import sample_frames

            verdicts.append(pool.submit(judge, sample_frames(frames), ep.get("task")))  # only what the judge sees
        keep = (not ep["success"] and save_failure_videos) or (ep["success"] and n_success_videos < save_success_videos)
        if keep and frames:
            path = out_dir / "videos" / f"{tag}_seed{seed:04d}_{ep['failure_mode']}.mp4"
            imageio.mimwrite(path, frames, fps=10, codec="libx264", quality=6)
            ep["video"] = str(path)
            n_success_videos += int(ep["success"])
        episodes.append(ep)
        if verbose:
            print(json.dumps({k: v for k, v in ep.items() if k != "final_cube_pos"}))
    if pool is not None:
        for ep, fut in zip(episodes, verdicts):
            try:
                ep["vlm"] = fut.result().model_dump()
            except Exception as e:  # noqa: BLE001 - an unanswered episode counts as a detected failure downstream
                ep["vlm"] = {"success": None, "error": repr(e)}
        pool.shutdown()
    k, n = sum(e["success"] for e in episodes), len(episodes)
    modes = {m: sum(e["failure_mode"] == m for e in episodes) for m in FAILURE_MODES}
    summary = {
        "tag": tag,
        "n": n,
        "success": k,
        "success_rate": k / n if n else 0.0,
        "wilson95": wilson_interval(k, n),
        "failure_modes": modes,
        "seeds": [int(s) for s in seeds],
        "wall_time_s": round(time.time() - t0, 1),
        **({"vlm_judge": judge_agreement(episodes)} if judge is not None else {}),
        "episodes": episodes,
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    if verbose:
        lo, hi = summary["wilson95"]
        print(f"[{tag}] success {k}/{n} = {k / n:.2f} (95% CI {lo:.2f}-{hi:.2f})  modes={modes}")
    return summary
