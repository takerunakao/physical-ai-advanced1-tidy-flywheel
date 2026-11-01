#!/usr/bin/env python
"""Redraw the burnt-in Japanese annotations of tidy_all.py clips with a Japanese font.

The clips were recorded with "Hiragino Sans GB" (Simplified-Chinese glyph shapes). No raw frames were kept, but the
run record (tidy_all.py --out JSON) fixes every overlay: 15 plan-card frames per pass, steps-5 frames per subtask
(the 5 settling steps after each subtask are counted in "steps" but not filmed), 12/25 judge-card frames, the
intervention's steps, and 20 result frames. Each frame's overlay regions are repainted and the text redrawn.

    uv run python scripts/refont_clips.py outputs/tidy_all/d200-v2.json --seeds 1002,1022 --out-dir outputs/video_src/clips_jp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
JP = {"red cube": "赤いキューブ", "blue cube": "青いキューブ", "green cylinder": "緑の円柱",
      "white box": "白い箱", "black box": "黒い箱"}
MODE_JP = {"success": "成功", "no_grasp": "掴めていない", "dropped": "途中で落とした", "missed_bin": "箱に入っていない",
           "timeout_holding": "持ったまま時間切れ", "contact": "衝突", "wrong_object": "違う物体を動かした"}
WHO = {"plan": "Claude がカメラ 2 台の画像から", "point": "VLM が指示"}
PLAN_N, JUDGE_OK, JUDGE_NG, FINAL_N, SETTLE = 15, 12, 25, 20, 5


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def jp_task(task: str) -> str:
    obj, box = task.removeprefix("put the ").split(" in the ")
    return f"{JP[obj]} → {JP[box]}"


def band(dr: ImageDraw.ImageDraw, w: int, lines: list[str]) -> None:
    dr.rectangle([0, 0, w, 8 + 24 * len(lines)], fill=(0, 0, 0, 255))
    for k, t in enumerate(lines):
        dr.text((8, 4 + 24 * k), t, fill=(255, 255, 255), font=font(18))


def legend(dr: ImageDraw.ImageDraw, h: int) -> None:
    y0 = h - 30
    dr.rectangle([0, y0 - 4, 330, h], fill=(0, 0, 0, 255))
    dr.ellipse([10, y0 + 2, 26, y0 + 18], outline=(255, 220, 0), width=3)
    dr.text((32, y0), "物体", fill=(255, 255, 255), font=font(18))
    dr.ellipse([90, y0 + 2, 106, y0 + 18], outline=(0, 255, 120), width=3)
    dr.text((112, y0), f"置き場所({WHO['point']})", fill=(255, 255, 255), font=font(18))


def timeline(row: dict) -> list[tuple[str, dict, int]]:
    """(kind, payload, n frames) in recording order, mirroring tidy_all.py's main loop."""
    header = f"指示: 「{row['command']}」"
    out, subs, idx = [], row["subtasks"], 0
    for n_pass, p in enumerate(row["passes"], 1):
        out.append(("plan", {"header": header, "n": n_pass, "steps": p["steps"]}, PLAN_N))
        for _ in p["steps"]:
            if idx >= len(subs):
                break
            s, idx = subs[idx], idx + 1
            if "error" in s:
                continue
            out.append(("step", {"lines": [header, f"手順 {idx}: {jp_task(s['task'])}   ({s['task']})"]}, s["steps"] - SETTLE))
            if "vlm" in s and s["vlm"].get("success") is not None:
                ok = s["vlm"]["success"]
                out.append(("judge", {"header": header, "ok": ok, "mode": s["vlm"].get("failure_mode", "")}, JUDGE_OK if ok else JUDGE_NG))
            if "intervention" in s:
                out.append(("iv", {"lines": [header, f"介入: 教師が引き継ぎ  {jp_task(s['task'])}"]}, s["intervention"]["steps"]))
    out.append(("final", {"lines": [header, f"結果: 3 個中 {row['tidied']} 個が箱の中"]}, FINAL_N))
    return out


def redraw(frame: np.ndarray, kind: str, p: dict) -> np.ndarray:
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    if kind == "plan":
        band(dr, w, [p["header"], f"計画 {p['n']}({WHO['plan']})"])
        rows = [f"{k + 1}. {jp_task(t)}" for k, t in enumerate(p["steps"])] or ["すべて片付いています"]
        dr.rounded_rectangle([20, 70, 380, 70 + 20 + 30 * len(rows)], radius=10, fill=(0, 0, 0, 255))
        for k, r in enumerate(rows):
            dr.text((36, 80 + 30 * k), r, fill=(255, 235, 140), font=font(20))
    elif kind == "step":
        band(dr, w, p["lines"])
        legend(dr, h)
    elif kind == "judge":
        band(dr, w, [p["header"], "判定(Claude が正面カメラの 6 コマだけを見て)"])
        y = 70 + 2 * (150 + 6)  # below the 2x3 thumbnails, as in tidy_all.judge_card
        dr.rectangle([0, y + 4, w, y + 54], fill=(0, 0, 0, 255))
        col = (120, 230, 140) if p["ok"] else (255, 110, 90)
        dr.text((14, y + 10), "成功" if p["ok"] else f"失敗: {MODE_JP.get(p['mode'], p['mode'])}", fill=col, font=font(26))
    else:  # iv, final
        band(dr, w, p["lines"])
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds to convert (default: every row with a video)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    seeds = {int(s) for s in args.seeds.split(",")} if args.seeds else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in json.loads(Path(args.json).read_text())["episodes"]:
        if "video" not in row or (seeds is not None and row["seed"] not in seeds):
            continue
        src = Path(row["video"])
        r = imageio.get_reader(src)
        frames = [f for f in r]
        r.close()
        tl = timeline(row)
        n = sum(k for _, _, k in tl)
        if n != len(frames):
            raise SystemExit(f"{src}: record implies {n} frames, video has {len(frames)}; not converting")
        out, i = [], 0
        for kind, p, k in tl:
            out += [redraw(f, kind, p) for f in frames[i:i + k]]
            i += k
        dst = out_dir / f"{src.parent.name}_{src.name}"
        imageio.mimwrite(dst, out, fps=10, codec="libx264", quality=7)
        print(f"{src} -> {dst} ({len(out)} frames)")


if __name__ == "__main__":
    main()
