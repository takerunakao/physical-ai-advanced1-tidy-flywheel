#!/usr/bin/env python
"""Record teacher demonstrations into a LeRobotDataset (no ROS 2 needed).

    python scripts/record_dataset.py --seeds 0-99 --repo-id takerunakao/tidy-v0 --root data/tidy-v0

Only successful, contact-free episodes are kept (teacher gate). Per-episode outcomes are written to
<root>/episodes_meta.json so that filtered seeds are auditable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tidy_flywheel.dataset import DatasetWriter
from tidy_flywheel.sim import TidyEnv
from tidy_flywheel.teacher import run_teacher_episode

from run_teacher import parse_seeds  # noqa: E402  (scripts/ is on sys.path when run as a script)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--keep-failures", action="store_true", help="also store failed teacher episodes")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--push", action="store_true", help="push to the Hugging Face Hub (private)")
    args = ap.parse_args()

    env = TidyEnv(image_size=(args.image_size, args.image_size))
    writer = DatasetWriter(args.repo_id, args.root, args.image_size, overwrite=args.overwrite)
    seeds = parse_seeds(args.seeds)
    kept, meta = 0, []
    t0 = time.time()
    for seed in seeds:
        r = run_teacher_episode(env, seed)
        ok = r["success"] and not r["robot_contact"]
        m = {k: r[k] for k in ("seed", "success", "robot_contact", "steps", "final_phase")}
        m["kept"] = bool(ok or args.keep_failures)
        meta.append(m)
        print(json.dumps(m))
        if not m["kept"]:
            continue
        for t in range(r["steps"]):
            writer.add_step(r["images"][t], r["states"][t], r["actions"][t], r["env_states"][t])
        writer.end_episode(m)
        kept += 1
    writer.close()
    Path(args.root, "episodes_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"kept {kept}/{len(seeds)} episodes in {args.root}  ({(time.time() - t0) / len(seeds):.1f} s/episode)")
    if args.push:
        writer.push(private=True)
        print("pushed")


if __name__ == "__main__":
    main()
