#!/usr/bin/env python
"""Audit a LeRobot dataset for mislabelled demos with a VLM; score against build_info.json when it records them.

    python scripts/audit_dataset.py --repo-id takerunakao/tidy-m3-p30 --root data/tidy-m3-p30 --out outputs/audit/m3-p30.json

Prints the flagged seeds as a comma list for build_dataset.py --exclude-seeds.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tidy_flywheel.vlm_audit import audit_episode
from tidy_flywheel.vlm_judge import make_client, sample_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root)
    eps = ds.meta.episodes
    kept = [m for m in json.loads(Path(args.root, "build_info.json").read_text())["episodes"] if m["kept"]]
    assert len(kept) == ds.meta.total_episodes, "build_info and dataset disagree"

    jobs = []
    for e in range(ds.meta.total_episodes):  # decode serially, query in parallel
        lo, hi = int(eps["dataset_from_index"][e]), int(eps["dataset_to_index"][e])
        idx = np.unique(np.linspace(lo, hi - 1, args.frames).round().astype(int))
        items = [ds[int(i)] for i in idx]
        frames = [(int(it["frame_index"]), (it["observation.images.front"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                  for it in items]
        jobs.append((e, items[0]["task"], frames))

    client = make_client()

    def run(job):
        e, label, frames = job
        t0 = time.time()
        try:
            v = audit_episode(client, frames, label).model_dump()
        except Exception as ex:  # noqa: BLE001 - unanswered: keep the demo, record the error
            v = {"matches_label": True, "error": repr(ex)}
        m = kept[e]
        return {"episode": e, "seed": m["seed"], "label": label, "truth_poisoned": bool(m.get("poisoned")),
                "executed": m.get("executed_task", label), **v, "seconds": round(time.time() - t0, 1)}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run, jobs))

    flagged = [r for r in rows if not r["matches_label"]]
    tp = sum(r["truth_poisoned"] for r in flagged)
    n_pos = sum(r["truth_poisoned"] for r in rows)
    summary = {"dataset": args.root, "n": len(rows), "flagged": len(flagged), "poisoned": n_pos,
               "precision": tp / len(flagged) if flagged else None, "recall": tp / n_pos if n_pos else None,
               "errors": sum("error" in r for r in rows), "flagged_seeds": sorted(r["seed"] for r in flagged),
               "episodes": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("episodes", "flagged_seeds")}, indent=1))
    for r in rows:
        if r["truth_poisoned"] != (not r["matches_label"]):
            print(f"seed {r['seed']}: poisoned={r['truth_poisoned']} flagged={not r['matches_label']} "
                  f"label='{r['label']}' executed='{r['executed']}' saw={r.get('moved_object')}->{r.get('into_box')}: "
                  f"{r.get('rationale', r.get('error'))}")
    print("EXCLUDE", ",".join(str(s) for s in summary["flagged_seeds"]))


if __name__ == "__main__":
    main()
