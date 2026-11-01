"""Measure VLM grounding against the simulator on held-out multi-scene seeds.

    python scripts/vlm_ground_eval.py --backend claude --view top --seeds 1000-1049 --out outputs/vlm/ground_claude_top.json

The VLM points in `--view` (front: the policy's camera; top: an unoccluded overhead view, converted to the front
camera through the table plane; the drop point is read at mid-wall height, see MultiTidyEnv.pointed_world).
Per seed, in world coordinates:
  obj_correct  the object point is closest to the instructed object,       obj_err_mm  its distance to it
  box_correct  the drop point lies in the instructed box's footprint,
  drop_ok      ... and leaves room: >= 1 cm inside the walls and >= 5 cm from anything already in that box.
With --prefill 1 every scene has one object already resting in a box (as mid-way through a tidy-up).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tidy_flywheel.sim_multi import BOXES, OBJECTS, MultiTidyEnv
from tidy_flywheel.vlm_ground import ground

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_teacher import parse_seeds  # noqa: E402

RES = 480
WALL_MARGIN = 0.06 - 0.02 - 0.01  # half box - half object - 1 cm
CLEARANCE = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gemini", "claude"], required=True)
    ap.add_argument("--view", choices=["front", "top"], default="front")
    ap.add_argument("--seeds", default="1000-1049")
    ap.add_argument("--prefill", type=float, default=0.5, help="probability of an object already in a box")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    env = MultiTidyEnv()
    env.config.prefill_p = args.prefill
    scenes = []
    for seed in parse_seeds(args.seeds):  # render serially (one MuJoCo context), query the VLM in parallel
        _, info = env.reset(seed=seed)
        objs = np.stack([env.obj_pos(i) for i in range(len(OBJECTS))])
        boxes = np.stack([env.box_pos(j) for j in range(len(BOXES))])
        inside = [objs[i] for i in env.done_objs if env._in_box(objs[i], env.target_box)]
        scenes.append(dict(seed=seed, task=info["task"], img=env.render(args.view, (RES, RES)), objs=objs, boxes=boxes,
                           k=env.target_obj, b=env.target_box, inside=inside))

    def run(sc):
        t0 = time.time()
        try:
            p = ground(sc["img"], sc["task"], args.backend, view=args.view)
        except Exception as e:  # noqa: BLE001 - a failed call is a grounding failure
            return {"seed": sc["seed"], "task": sc["task"], "error": repr(e)}
        return {"seed": sc["seed"], "task": sc["task"], "points": p.round(4).tolist(), "occupied": bool(sc["inside"]),
                "seconds": round(time.time() - t0, 1)}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run, scenes))

    for r, sc in zip(rows, scenes):  # score in world coordinates (needs the env, so serially)
        if "error" in r:
            continue
        p = np.array(r["points"])
        obj_w, drop_w = env.pointed_world(p, args.view)
        d_obj = np.linalg.norm(sc["objs"][:, :2] - obj_w[:2], axis=1)
        rel = drop_w[:2] - sc["boxes"][sc["b"]][:2]
        clear = min((np.linalg.norm(drop_w[:2] - q[:2]) for q in sc["inside"]), default=np.inf)
        r.update(obj_correct=bool(np.argmin(d_obj) == sc["k"]), obj_err_mm=round(float(d_obj[sc["k"]]) * 1000, 1),
                 box_correct=bool(np.all(np.abs(rel) < 0.06)),
                 drop_ok=bool(np.all(np.abs(rel) < WALL_MARGIN) and clear >= CLEARANCE),
                 leaves_room=bool(sc["inside"] or np.linalg.norm(rel) >= 0.02),  # empty box: a 2nd object still fits
                 drop_rel_mm=(rel * 1000).round(1).tolist(), clearance_mm=None if np.isinf(clear) else round(clear * 1000, 1))

    ok = [r for r in rows if "error" not in r]
    occ = [r for r in ok if r["occupied"]]
    rate = lambda rs, k: sum(r[k] for r in rs) / max(len(rs), 1)  # noqa: E731
    summary = {"backend": args.backend, "view": args.view, "n": len(rows), "api_errors": len(rows) - len(ok),
               "obj_correct": rate(ok, "obj_correct"), "box_correct": rate(ok, "box_correct"),
               "drop_ok": rate(ok, "drop_ok"), "drop_ok_occupied_box": rate(occ, "drop_ok"), "n_occupied": len(occ),
               "leaves_room_empty_box": rate([r for r in ok if not r["occupied"]], "leaves_room"),
               "obj_err_mm_median": float(np.median([r["obj_err_mm"] for r in ok])) if ok else None,
               "episodes": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1))


if __name__ == "__main__":
    main()
