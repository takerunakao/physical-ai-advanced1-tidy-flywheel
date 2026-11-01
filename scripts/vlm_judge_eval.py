"""Measure the VLM success judge against the simulator's privileged ground truth.

    python scripts/vlm_judge_eval.py --videos 'outputs/eval/videos/b25-v0_seed*.mp4' --out outputs/vlm/b25-v0.json

Ground truth comes from the file name written by evaluate.py: <tag>_seed<NNNN>_<failure_mode>.mp4.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tidy_flywheel.vlm_judge import MODEL, judge_video, make_client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="glob of evaluate.py videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.videos))
    client = make_client()

    def run(p: str) -> dict:
        m = re.search(r"_seed(\d+)_(\w+)\.mp4$", p)
        t0 = time.time()
        v = judge_video(client, p, args.frames, args.model)
        return {"video": p, "seed": int(m[1]), "truth": m[2], "truth_success": m[2] == "success",
                **v.model_dump(), "seconds": round(time.time() - t0, 1)}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run, paths))

    tp = sum(r["truth_success"] and r["success"] for r in rows)
    tn = sum(not r["truth_success"] and not r["success"] for r in rows)
    fp = sum(not r["truth_success"] and r["success"] for r in rows)
    fn = sum(r["truth_success"] and not r["success"] for r in rows)
    mode_hits = sum(r["truth"] == r["failure_mode"] for r in rows)
    summary = {"model": args.model, "frames": args.frames, "n": len(rows), "accuracy": (tp + tn) / len(rows),
               "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
               "failure_mode_accuracy": mode_hits / len(rows), "episodes": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1))
    for r in rows:
        if r["truth"] != r["failure_mode"]:
            print(f"seed {r['seed']}: truth={r['truth']} judge={r['failure_mode']} ({r['confidence']:.2f}) {r['rationale']}")


if __name__ == "__main__":
    main()
