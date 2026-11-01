#!/usr/bin/env python
"""Tidy the whole table from one free-form command: plan (LLM) -> point (VLM) -> act (ACT policy), closed loop.

    python scripts/tidy_all.py --checkpoint outputs/act_m3-v2/checkpoints/last/pretrained_model \\
        --command "片付けて" --seeds 1000-1019 --video-dir outputs/tidy_all/videos --out outputs/tidy_all/run.json

After each pass of the plan the planner looks at the table again and plans whatever is left, so a failed pick is
retried from what the camera shows (no simulator state is used to decide). Scenes start with all three objects
on the table (no prefilled boxes). The simulator's state is used only to *score* the outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tidy_flywheel.sim_multi import BOXES, OBJECTS, MultiTidyEnv, instruction
from tidy_flywheel.teacher import ScriptedTeacher

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_teacher import parse_seeds  # noqa: E402

VIDEO_HW = (480, 640)
WHO = {"plan": "Claude がカメラ 2 台の画像から", "point": "VLM が指示"}  # overlay wording, set from the CLI
GROUND_RES = 480


def oracle_plan(env: MultiTidyEnv) -> list[tuple[int, int]]:
    """Rule for the command "cubes in the white box, the rest in the black box" (used without an LLM)."""
    return [(i, 0 if "cube" in OBJECTS[i][1] else 1) for i in env.remaining()]


JP = {"red cube": "赤いキューブ", "blue cube": "青いキューブ", "green cylinder": "緑の円柱",
      "white box": "白い箱", "black box": "黒い箱"}


def intervene(env: MultiTidyEnv, obj: int, box: int, path: Path, frames: list | None, header: str) -> dict:
    """The expert takes over after a judged failure: back to the start pose, then the scripted teacher carries out
    the same instruction from the table as it now is (object knocked over, against a wall, on another object...).
    The teacher's transitions are saved as a demonstration; it uses simulator state, as a human operator would use
    their own eyes."""
    env.go_home()
    obs, info = env.retarget(obj, box)
    teacher = ScriptedTeacher(env.home_close_yaw)
    states, actions, front, wrist, env_states, contact = [], [], [], [], [], False
    lines = [header, f"介入: 教師が引き継ぎ  {jp_step(obj, box)}"]
    while True:
        if frames is not None:
            frames.append(annotate(env.render("front", VIDEO_HW), lines, None))
        a = teacher.act(obs, info)
        states.append(obs["state"]); actions.append(a)
        front.append(obs["images"]["front"]); wrist.append(obs["images"]["wrist"])
        env_states.append(np.array([*info["cube_pos"], info["cube_yaw"]], dtype=np.float32))
        obs, info = env.step(a)
        contact = contact or info["robot_contact"]
        if teacher.done or info["timeout"]:
            break
    for _ in range(5):
        obs, info = env.step(np.zeros(5))
    out = {"success": bool(info["success"]), "contact": bool(contact), "steps": len(actions), "file": str(path)}
    np.savez_compressed(path, states=np.stack(states), actions=np.stack(actions), front=np.stack(front),
                        wrist=np.stack(wrist), env_states=np.stack(env_states), task=instruction(obj, box),
                        success=out["success"], contact=out["contact"])
    return out


def interventions_summary(rows: list[dict]) -> dict:
    subs = [s for r in rows for s in r["subtasks"] if "vlm" in s]
    iv = [s["intervention"] for s in subs if "intervention" in s]
    judged = [s for s in subs if s["vlm"]["success"] is not None]
    return {"policy_steps": len(subs), "judged_failures": len(iv),
            "judge_agrees": sum(s["vlm"]["success"] == s["success"] for s in judged), "judged": len(judged),
            "missed_failures": sum(s["vlm"]["success"] and not s["success"] for s in judged),
            "false_alarms": sum((not s["vlm"]["success"]) and s["success"] for s in judged),
            "teacher_rescued": sum(i["success"] and not i["contact"] for i in iv)}


def follows(in_box: dict, expect: dict | None) -> bool | None:
    """Final placement vs the command's expectation: a box name, "any" (in some box) or None (left on the table)."""
    if expect is None:
        return None
    return all(in_box[o] is not None if want == "any" else in_box[o] == want for o, want in expect.items())


def jp_step(obj: int, box: int) -> str:
    return f"{JP[OBJECTS[obj][1]]} → {JP[BOXES[box][1]]}"


def annotate(frame: np.ndarray, lines: list[str], points: np.ndarray | None) -> np.ndarray:
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 18, index=1)
    except OSError:
        font = ImageFont.load_default()
    dr.rectangle([0, 0, img.width, 8 + 24 * len(lines)], fill=(0, 0, 0, 200))
    for k, t in enumerate(lines):
        dr.text((8, 4 + 24 * k), t, fill=(255, 255, 255), font=font)
    if points is not None:  # goal points are normalised for a square image; the video is wider
        h, w = frame.shape[:2]
        for (u, v), col in ((points[:2], (255, 220, 0)), (points[2:], (0, 255, 120))):
            x, y = (0.5 + (u - 0.5) * h / w) * w, v * h
            dr.ellipse([x - 9, y - 9, x + 9, y + 9], outline=col, width=3)
        y0 = h - 30  # legend
        dr.rectangle([0, y0 - 4, 330, h], fill=(0, 0, 0, 160))
        dr.ellipse([10, y0 + 2, 26, y0 + 18], outline=(255, 220, 0), width=3)
        dr.text((32, y0), "物体", fill=(255, 255, 255), font=font)
        dr.ellipse([90, y0 + 2, 106, y0 + 18], outline=(0, 255, 120), width=3)
        dr.text((112, y0), f"置き場所({WHO['point']})", fill=(255, 255, 255), font=font)
    return np.asarray(img)


def plan_card(frame: np.ndarray, header: str, steps: list[tuple[int, int]], why: str, n_pass: int) -> np.ndarray:
    """The planner's output over the scene, shown for a moment before execution."""
    img = Image.fromarray(annotate(frame, [header, f"計画 {n_pass}({WHO['plan']})"], None))
    dr = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 20, index=1)
    except OSError:
        font = ImageFont.load_default()
    rows = [f"{k + 1}. {jp_step(o, b)}" for k, (o, b) in enumerate(steps)] or ["すべて片付いています"]
    h = 20 + 30 * len(rows)
    dr.rounded_rectangle([20, 70, 380, 70 + h], radius=10, fill=(0, 0, 0, 190))
    for k, r in enumerate(rows):
        dr.text((36, 80 + 30 * k), r, fill=(255, 235, 140), font=font)
    return np.asarray(img)


MODE_JP = {"success": "成功", "no_grasp": "掴めていない", "dropped": "途中で落とした", "missed_bin": "箱に入っていない",
           "timeout_holding": "持ったまま時間切れ", "contact": "衝突", "wrong_object": "違う物体を動かした"}


def judge_card(frame: np.ndarray, header: str, samples: list[tuple[int, np.ndarray]], ok: bool, mode: str) -> np.ndarray:
    """What the judge saw (the sampled front-camera frames) and what it concluded, shown before the next step."""
    img = Image.fromarray(annotate(frame, [header, "判定(Claude が正面カメラの 6 コマだけを見て)"], None))
    dr = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 26, index=1)
    except OSError:
        font = ImageFont.load_default()
    tw, th, y0 = 200, 150, 70
    dr.rectangle([0, y0 - 6, img.width, y0 + 2 * (th + 6) + 54], fill=(0, 0, 0, 200))
    for k, (_, f) in enumerate(samples[:6]):
        img.paste(Image.fromarray(f).resize((tw, th)), (10 + (k % 3) * (tw + 10), y0 + (k // 3) * (th + 6)))
    col = (120, 230, 140) if ok else (255, 110, 90)
    dr.text((14, y0 + 2 * (th + 6) + 10), f"{'成功' if ok else '失敗'}" + ("" if ok else f": {MODE_JP.get(mode, mode)}"),
            fill=col, font=font)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="ACT checkpoint; omit to execute with the scripted teacher")
    ap.add_argument("--command", default="片付けて")
    ap.add_argument("--commands", default=None,
                    help='JSON file: [{"command": ..., "expect": {"red cube": "white box" | "any" | null, ...}}]; '
                         "every command runs on every seed and is scored against its expectation")
    ap.add_argument("--planner", choices=["claude", "oracle"], default="claude")
    ap.add_argument("--ground", choices=["claude", "gemini", "exact"], default="claude")
    ap.add_argument("--seeds", default="1000-1009")
    ap.add_argument("--snap-drop", action="store_true",
                    help="keep the VLM's choice of box half but drop at that half's centre, as in the demos")
    ap.add_argument("--intervene", default=None, metavar="DIR",
                    help="deployment with an expert on call: the VLM judge watches every policy step; on a judged "
                         "failure the scripted teacher takes over from the current table and its demo is saved to DIR "
                         "(HG-DAgger interventions for build_dataset.py --interventions)")
    ap.add_argument("--max-subtasks", type=int, default=6, help="budget over all passes (3 objects + retries)")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.planner == "oracle":
        WHO["plan"] = "規則"
    if args.ground == "exact":
        WHO["point"] = "正解"
    env = MultiTidyEnv()
    env.config.prefill_p = 0.0
    policy = None
    if args.checkpoint:
        from tidy_flywheel.policy import ActPolicy

        policy = ActPolicy(args.checkpoint)
    if args.ground != "exact":
        from tidy_flywheel.vlm_ground import ground
    if args.planner == "claude":
        from tidy_flywheel.vlm_plan import plan
    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    if args.intervene:
        from tidy_flywheel.vlm_judge import judge_samples, make_client, sample_frames

        judge_client = make_client()
        Path(args.intervene).mkdir(parents=True, exist_ok=True)

    specs = json.loads(Path(args.commands).read_text()) if args.commands else [{"command": args.command, "expect": None}]
    rows = []
    for ci, spec, seed in ((ci, spec, seed) for ci, spec in enumerate(specs) for seed in parse_seeds(args.seeds)):
        command = spec["command"]
        obs, info = env.reset(seed=seed)
        frames, subtasks, passes, t0 = [], [], [], time.time()
        header = f"指示: 「{command}」"
        while len(subtasks) < args.max_subtasks:
            env.go_home()
            view = env.render("front", (GROUND_RES, GROUND_RES))
            if args.planner == "claude":
                steps, why = plan({"front": view, "side": env.render("side", (GROUND_RES, GROUND_RES))}, command,
                                  attempted=[s["task"] for s in subtasks])
            else:
                steps, why = oracle_plan(env), "rule"
            passes.append({"steps": [instruction(o, b) for o, b in steps], "reasoning": why})
            if args.video_dir:
                frames += [plan_card(env.render("front", VIDEO_HW), header, steps, why, len(passes))] * 15
            if not steps:
                break
            for obj, box in steps:
                if len(subtasks) >= args.max_subtasks:
                    break
                env.go_home()
                obs, info = env.retarget(obj, box)  # bookkeeping (scoring) uses the instructed pair
                task = instruction(obj, box)
                rec = {"task": task}
                if args.ground != "exact":
                    view = env.render("front", (GROUND_RES, GROUND_RES))
                    try:
                        pts = env.to_goal_points(np.asarray(ground(view, task, args.ground), dtype=np.float32))
                    except Exception as e:  # noqa: BLE001 - a failed call skips this step; the next pass re-plans
                        rec["error"] = repr(e)
                        subtasks.append(rec)
                        continue
                    true_obj = env.true_goal_points()[:2]
                    rec["ground_obj_err_px"] = round(float(np.linalg.norm(pts[:2] - true_obj) * GROUND_RES), 1)
                    drop = env.unproject(pts[2:], env.box_pos(box)[2])  # where the VLM wants it, on the box floor
                    env.place_offset = (drop - env.box_pos(box))[:2]  # teacher executor aims there; policy sees pts
                    if args.snap_drop:  # the demos drop 3 cm left/right of centre; map the VLM's point onto that
                        side = np.sign(env.place_offset[0]) or -1.0
                        env.place_offset = np.array([side * env.config.place_offset, 0.0])
                        pts[2:] = env.project(env.bin_pos)
                    rec["drop_offset_mm"] = (env.place_offset * 1000).round(1).tolist()
                    obs = env.set_goal_points(pts)
                else:
                    pts = env.goal_points.copy()
                rec["points"] = pts.round(4).tolist()
                lines = [header, f"手順 {len(subtasks) + 1}: {jp_step(obj, box)}   ({task})"]
                if policy is not None:
                    policy.reset()
                teacher = ScriptedTeacher(env.home_close_yaw) if policy is None else None
                watch = []  # judge frames (front camera) when an expert is on call
                while True:
                    if args.video_dir or args.intervene:
                        raw = env.render("front", VIDEO_HW)
                        if args.intervene:
                            watch.append(raw)
                        if args.video_dir:
                            frames.append(annotate(raw, lines, pts))
                    a = policy.act(obs) if policy is not None else teacher.act(obs, info)
                    obs, info = env.step(a)
                    if info["success"] or info["timeout"] or (teacher is not None and teacher.done):
                        break
                for _ in range(5):
                    obs, info = env.step(np.zeros(5))
                rec.update(success=bool(info["success"]), wrong_object=bool(info["wrong_object"]),
                           contact=bool(info["robot_contact"]), steps=int(info["t"]))
                if args.intervene and policy is not None:
                    watch.append(env.render("front", VIDEO_HW))
                    samples = sample_frames(watch)
                    try:
                        v = judge_samples(judge_client, samples, task=task)
                        rec["vlm"] = {"success": bool(v.success), "failure_mode": v.failure_mode}
                    except Exception as e:  # noqa: BLE001 - no verdict: no intervention
                        rec["vlm"] = {"success": None, "error": repr(e)}
                    if args.video_dir and rec["vlm"]["success"] is not None:
                        frames += [judge_card(watch[-1], header, samples, rec["vlm"]["success"],
                                              rec["vlm"]["failure_mode"])] * (12 if rec["vlm"]["success"] else 25)
                    if rec["vlm"]["success"] is False:
                        rec["intervention"] = intervene(env, obj, box, Path(args.intervene) / f"seed{seed}_k{len(subtasks)}.npz",
                                                        frames if args.video_dir else None, header)
                subtasks.append(rec)
            if args.planner == "oracle" and not env.remaining():
                break
        env.go_home()
        in_box = {OBJECTS[i][1]: next((BOXES[j][1] for j in range(len(BOXES)) if env._in_box(env.obj_pos(i), j)), None)
                  for i in range(len(OBJECTS))}
        row = {"seed": seed, "command": command, "follows": follows(in_box, spec.get("expect")),
               "tidied": sum(v is not None for v in in_box.values()), "final": in_box,
               "all_tidied": all(v is not None for v in in_box.values()), "subtasks": subtasks, "passes": passes,
               "seconds": round(time.time() - t0, 1)}
        if args.video_dir and frames:
            final = annotate(env.render("front", VIDEO_HW), [header, f"結果: 3 個中 {row['tidied']} 個が箱の中"], None)
            frames += [final] * 20
            path = Path(args.video_dir) / (f"tidy_all_seed{seed:04d}.mp4" if not args.commands
                                           else f"cmd{ci}_seed{seed:04d}.mp4")
            imageio.mimwrite(path, frames, fps=10, codec="libx264", quality=7)
            row["video"] = str(path)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("subtasks",)}, ensure_ascii=False))

    by_cmd = {}
    for r in rows:
        by_cmd.setdefault(r["command"], []).append(r)
    summary = {"command": args.command if not args.commands else [s["command"] for s in specs], "planner": args.planner, "ground": args.ground, "snap_drop": args.snap_drop,
               "executor": args.checkpoint or "teacher", "n": len(rows),
               "all_tidied_rate": sum(r["all_tidied"] for r in rows) / len(rows),
               "mean_tidied": float(np.mean([r["tidied"] for r in rows])),
               "mean_subtasks": float(np.mean([len(r["subtasks"]) for r in rows])),
               "interventions": interventions_summary(rows) if args.intervene else None,
               "follows_rate": {c: sum(r["follows"] for r in rs) / len(rs) for c, rs in by_cmd.items()}
               if args.commands else None, "episodes": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
