#!/usr/bin/env python
"""Write the narration script (台本) of a storyboard as Markdown: one row per scene with the on-screen text, the
seconds the scene lasts, the seconds its pictures last on their own (from make_video.py's log) and the file name of
the recorded take that make_video.py picks up from NARR_DIR.

    uv run python scripts/narration_script.py docs/storyboard.json outputs/logs/make_video_draft9.log docs/NARRATION.md
"""
import json
import re
import sys
from pathlib import Path

LIMIT = 180.0  # YouTube video must be 1-3 min
board = json.loads(Path(sys.argv[1]).read_text())
log = Path(sys.argv[2]).read_text()
rows_log = re.findall(r"^\s*([\d.]+)s .*\[video ([\d.]+)s, voice ([\d.]+)s\]", log, re.M)
cum = [float(c) for c, _, _ in rows_log]
vid = [float(v) for _, v, _ in rows_log]
starts = [0.0] + cum[:-1]
narrated = [i for i, seg in enumerate(board) if seg.get("narration")]
fixed = sum(vid[i] for i in range(len(board)) if i not in narrated)  # jingles and silent scenes
budget = LIMIT - fixed - 2.0  # 2 s of margin for YouTube's rounding
rows = ["# ナレーション台本", "",
        f"動画全体は {LIMIT:.0f} 秒以内。ナレーションの無い場面(ジングル)が {fixed:.0f} 秒あるので、**ナレーション付きの "
        f"{len(narrated)} 場面の合計を {budget:.0f} 秒以内**に収める(場面ごとの長さ = max(画面の長さ, 録音 + 0.7 秒))。",
        "", "録音は場面ごとに 1 ファイル、「ファイル」列の名前で `outputs/narration/` に置く(形式は m4a / wav / mp3 など何でも可)。"
        "前後の無音は自動で切るので、間を気にせず録ってよい。「画面」の秒数より短い録音は画面が残るだけで問題なく、"
        "長い録音は最後のコマを止めて待つ。", "", "カタカナ表記は読み方の指定。", "",
        "| # | ファイル | 開始 | 画面(秒) | 今の長さ(秒) | 画面の内容 | ナレーション |", "|--:|---|--:|--:|--:|---|---|"]
for i, seg in enumerate(board):
    screen = seg.get("card") or seg.get("label") or seg.get("caption", "").split("\n")[0] or Path(str(seg.get("image") or seg.get("clip"))).stem
    f = f"`{i + 1:02d}.m4a`" if seg.get("narration") else "(なし)"
    rows.append(f"| {i + 1} | {f} | {int(starts[i])//60}:{int(starts[i])%60:02d} | {vid[i]:.0f} | {cum[i] - starts[i]:.0f} "
                f"| {screen.replace('|', '/')} | {seg.get('narration', '')} |")
rows += ["", f"今の合成音声版の合計: {cum[-1]:.0f} 秒(ナレーション付き場面 {sum(cum[i] - starts[i] for i in narrated):.0f} 秒)。"]
Path(sys.argv[3]).write_text("\n".join(rows) + "\n")
print(f"wrote {sys.argv[3]} ({len(board)} scenes, {cum[-1]:.0f} s, budget for narrated scenes {budget:.0f} s)")
