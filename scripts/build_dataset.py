#!/usr/bin/env python
"""Build a dataset for one flywheel iteration, reproducibly, from seeds and an evaluation JSON.

    # K=0: teacher demos only
    python scripts/build_dataset.py --base-seeds 0-99 --repo-id takerunakao/tidy-v0 --root data/tidy-v0

    # K=1: same base + corrections for the seeds policy v0 failed on
    python scripts/build_dataset.py --base-seeds 0-99 --eval outputs/eval/v0.json --checkpoint outputs/act_v0 \
        --mode recovery --repo-id takerunakao/tidy-v1 --root data/tidy-v1

    # control: same base + the same *number* of fresh random teacher demos instead of corrections
    python scripts/build_dataset.py --base-seeds 0-99 --extra-seeds 200-2xx --repo-id takerunakao/tidy-v1c --root data/tidy-v1c

    # data audit (multi scene): mislabel 30% of the demos, or rebuild without the episodes an audit flagged
    python scripts/build_dataset.py --scene multi --base-seeds 0-99 --poison-frac 0.3 ... --root data/tidy-m3-p30
    python scripts/build_dataset.py --scene multi --base-seeds 0-99 --poison-frac 0.3 --exclude-seeds 3,17,... ...

Everything is deterministic (seeded sim, deterministic teacher and ACT), so a dataset is fully described by
its command line, which is stored in <root>/build_info.json together with per-episode outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from tidy_flywheel import SCENES, make_env
from tidy_flywheel.dataset import STATE_NAMES, STATE_NAMES_MULTI, TASK, DatasetWriter
from tidy_flywheel.teacher import run_teacher_episode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_teacher import parse_seeds  # noqa: E402


def mislabel(env, seed: int, rng: np.random.Generator) -> tuple[str, np.ndarray]:
    """A wrong instruction for this seed's scene: another object still on the table, a random box/drop point."""
    from tidy_flywheel.sim_multi import BOXES, instruction

    env.reset(seed=seed)  # the teacher's own reset reproduces this scene exactly
    others = [i for i in env.remaining() if i != env.target_obj]
    obj, box = int(rng.choice(others)), int(rng.integers(len(BOXES)))
    drop = env.box_pos(box) + np.array([*rng.uniform(-0.03, 0.03, 2), 0.0])
    return instruction(obj, box), np.concatenate([env.project(env.obj_pos(obj)), env.project(drop)]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-seeds", default="0-99", help="teacher demos from these seeds")
    ap.add_argument("--extra-seeds", default="", help="additional plain teacher demos (control arm)")
    ap.add_argument("--eval", default=None, help="evaluation JSON; failed seeds get corrections")
    ap.add_argument("--checkpoint", default=None, help="policy used to reach handover states (recovery mode)")
    ap.add_argument("--corrections", action="append", default=[], metavar="EVAL_JSON=CHECKPOINT",
                    help="repeatable; earlier rounds' corrections are regenerated from these pairs (in order)")
    ap.add_argument("--failure-source", choices=["privileged", "vlm"], default="privileged",
                    help="which episodes count as failed: the simulator's outcome, or the VLM judge's verdict "
                         "(evaluate.py --judge; an unanswered episode counts as failed)")
    ap.add_argument("--poison-frac", type=float, default=0.0,
                    help="multi scene: this fraction of base demos is labelled with a *different* instruction "
                         "(goal points and task text of another object/box) than the one the teacher carries out")
    ap.add_argument("--exclude-seeds", default="", help="comma-separated base seeds to leave out (e.g. audit flags)")
    ap.add_argument("--interventions", action="append", default=[], metavar="DIR",
                    help="directories of expert takeovers saved by tidy_all.py --intervene (*.npz), added as demos")
    ap.add_argument("--mode", choices=["redemo", "recovery"], default="recovery")
    ap.add_argument("--per-failure", type=int, default=1, help="corrections per failed seed")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--scene", choices=SCENES, default="single")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    env = make_env(args.scene, args.image_size)
    writer = DatasetWriter(args.repo_id, args.root, args.image_size, overwrite=args.overwrite,
                           state_names=STATE_NAMES_MULTI if args.scene != "single" else STATE_NAMES)
    rng = np.random.default_rng(args.rng_seed)
    meta: list[dict] = []
    t0 = time.time()

    def add(r: dict, source: str, label: tuple[str, np.ndarray] | None = None) -> None:
        m = {k: r[k] for k in ("seed", "success", "robot_contact", "steps", "final_phase")}
        task = getattr(env, "task", TASK)  # the env still holds the episode's instruction
        if label is not None:  # mislabelled demo: the recorded instruction differs from what was carried out
            m.update(poisoned=True, executed_task=task)
            task, pts = label
            r["states"] = r["states"].copy()
            r["states"][:, 5:9] = pts
        m.update({"source": source, "task": task,
                  "kept": bool(r["success"] and not r["robot_contact"] and r["steps"] > 0)})
        for k in ("mode", "handover", "handover_info"):
            if k in r:
                m[k] = r[k]
        meta.append(m)
        print(json.dumps(m))
        if not m["kept"]:
            return
        for t in range(r["steps"]):
            writer.add_step(r["images"][t], r["states"][t], r["actions"][t], r["env_states"][t], task)
        writer.end_episode(m)

    base = parse_seeds(args.base_seeds)
    excluded = {int(x) for x in args.exclude_seeds.split(",") if x.strip()}
    # chosen from the full base list with its own rng, so excluding seeds never changes which demos are poisoned
    n_poison = round(args.poison_frac * len(base))
    poisoned = set(np.random.default_rng(args.rng_seed + 1).choice(base, n_poison, replace=False).tolist()) if n_poison else set()
    for seed in base:
        if seed in excluded:
            meta.append({"seed": seed, "source": "teacher", "kept": False, "excluded": True, "poisoned": seed in poisoned})
            continue
        label = mislabel(env, seed, np.random.default_rng([args.rng_seed, seed])) if seed in poisoned else None
        add(run_teacher_episode(env, seed), "teacher", label)
    if args.extra_seeds:
        for seed in parse_seeds(args.extra_seeds):
            add(run_teacher_episode(env, seed), "teacher_extra")

    pairs = [tuple(c.split("=", 1)) for c in args.corrections]
    if args.eval:
        pairs.append((args.eval, args.checkpoint))
    for round_idx, (eval_json, checkpoint) in enumerate(pairs):
        ev = json.loads(Path(eval_json).read_text())
        if args.failure_source == "vlm":
            failed = [e for e in ev["episodes"] if not e["vlm"]["success"]]
        else:
            failed = [e for e in ev["episodes"] if not e["success"]]
        print(f"round {round_idx}: {len(failed)} failed seeds in {eval_json}")
        if args.mode == "redemo":
            for e in failed:
                add(run_teacher_episode(env, e["seed"]), f"redemo_r{round_idx}")
        else:
            from tidy_flywheel.corrections import run_recovery_episode
            from tidy_flywheel.policy import ActPolicy

            policy = ActPolicy(checkpoint)
            for e in failed:
                for _ in range(args.per_failure):
                    handover = int(rng.integers(0, max(1, e["steps"])))
                    add(run_recovery_episode(env, policy, e["seed"], handover), f"recovery_r{round_idx}")

    for d in args.interventions:
        files = sorted(Path(d).glob("*.npz"))
        print(f"{len(files)} interventions in {d}")
        for f in files:
            z = np.load(f)
            m = {"seed": f.stem, "source": f"intervention:{d}", "task": str(z["task"]), "success": bool(z["success"]),
                 "robot_contact": bool(z["contact"]), "steps": len(z["actions"]),
                 "kept": bool(z["success"]) and not bool(z["contact"]) and len(z["actions"]) > 0}
            meta.append(m)
            print(json.dumps(m))
            if not m["kept"]:
                continue
            for t in range(len(z["actions"])):
                writer.add_step({"front": z["front"][t], "wrist": z["wrist"][t]}, z["states"][t], z["actions"][t],
                                z["env_states"][t], m["task"])
            writer.end_episode(m)

    writer.close()
    kept = sum(m["kept"] for m in meta)
    info = {"argv": sys.argv[1:], "kept": kept, "total": len(meta), "seconds": round(time.time() - t0, 1),
            "episodes": meta}
    Path(args.root, "build_info.json").write_text(json.dumps(info, indent=1))
    print(f"kept {kept}/{len(meta)} episodes → {args.root} ({info['seconds']} s)")
    if args.push:
        writer.push(private=True)
        print("pushed", args.repo_id)


if __name__ == "__main__":
    main()
