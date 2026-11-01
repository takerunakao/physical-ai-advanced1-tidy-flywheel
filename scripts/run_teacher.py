#!/usr/bin/env python
"""Roll out the scripted teacher on a range of seeds (no ROS 2 needed).

    python scripts/run_teacher.py --seeds 0-19 --video outputs/teacher_seed0.mp4 --json outputs/teacher.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio
import numpy as np

from tidy_flywheel.sim import TidyEnv
from tidy_flywheel.teacher import run_teacher_episode


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--video", default=None, help="save the first episode's front-camera video here")
    ap.add_argument("--json", default=None, help="write per-episode results here")
    args = ap.parse_args()

    env = TidyEnv()
    seeds = parse_seeds(args.seeds)
    results = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        r = run_teacher_episode(env, seed, record_frames=(i == 0 and args.video is not None))
        results.append({k: r[k] for k in ("seed", "success", "robot_contact", "steps", "final_phase")})
        print(json.dumps(results[-1]))
        if i == 0 and args.video:
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(args.video, r["frames"], fps=10, codec="libx264", quality=7)
    ok = sum(r["success"] for r in results)
    print(f"success {ok}/{len(seeds)}  ({(time.time() - t0) / len(seeds):.1f} s/episode)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
