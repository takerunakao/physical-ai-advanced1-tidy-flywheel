#!/usr/bin/env python
"""Evaluate a checkpoint on held-out seeds (no ROS 2 needed).

    python scripts/evaluate.py --checkpoint outputs/act_v0/pretrained_model --seeds 1000-1099 --tag v0
    python scripts/evaluate.py --plumbing data/tidy-smoke --seeds 1000-1002 --tag plumbing   # untrained ACT
"""
from __future__ import annotations

import argparse

from tidy_flywheel import SCENES, make_env
from tidy_flywheel.evaluate import evaluate

from run_teacher import parse_seeds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="local dir or HF repo id of a LeRobot ACT checkpoint")
    ap.add_argument("--plumbing", default=None, help="dataset root: use an untrained ACT with that feature spec")
    ap.add_argument("--seeds", default="1000-1099")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--scene", choices=SCENES, default="single")
    ap.add_argument("--ground", choices=["gemini", "claude"], default=None,
                    help="multi scene: goal points from this VLM instead of the exact projections")
    ap.add_argument("--judge", action="store_true",
                    help="also judge every episode from its camera frames with the VLM success judge (vlm_judge.py)")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--success-videos", type=int, default=2)
    ap.add_argument("--no-failure-videos", action="store_true")
    args = ap.parse_args()

    if args.plumbing:
        from tidy_flywheel.policy import RandomInitActPolicy

        policy = RandomInitActPolicy(args.plumbing)
    else:
        from tidy_flywheel.policy import ActPolicy

        policy = ActPolicy(args.checkpoint, n_action_steps=args.n_action_steps)
    env = make_env(args.scene, args.image_size)
    grounder = None
    if args.ground:
        from functools import partial

        from tidy_flywheel.vlm_ground import ground

        grounder = partial(ground, backend=args.ground)
    judge = None
    if args.judge:
        from tidy_flywheel.vlm_judge import judge_samples, make_client

        client = make_client()

        def judge(samples, task):
            return judge_samples(client, samples, task=task)
    evaluate(env, policy, parse_seeds(args.seeds), args.out, args.tag, save_failure_videos=not args.no_failure_videos,
             save_success_videos=args.success_videos, grounder=grounder, judge=judge)


if __name__ == "__main__":
    main()
