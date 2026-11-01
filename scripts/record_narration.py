#!/usr/bin/env python
"""Record the narration takes scene by scene into outputs/narration/NN.wav (what make_video.py's NARR_DIR reads).

    uv run --with sounddevice --with soundfile python scripts/record_narration.py [docs/storyboard.json] [--from N] [--only 3,15] [--mic "MacBook Pro"]

For each narrated scene: shows the text and the on-screen budget, Enter starts the recording, Enter stops it,
the take is played back, then Enter keeps it / r records again / s skips (keeps an existing file) / q quits.
Recording goes through CoreAudio (PortAudio via sounddevice) at the microphone's native rate; ffmpeg's AVFoundation
input was dropping buffers, which left clicks in every take. macOS asks for microphone permission the first time.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

OUT = Path("outputs/narration")


def mics() -> list[tuple[int, str, int]]:
    return [(i, d["name"], int(d["default_samplerate"])) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


def record(dev: int, rate: int, out: Path) -> float:
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []
    with sd.InputStream(device=dev, samplerate=rate, channels=1, dtype="float32", blocksize=0,
                        callback=lambda indata, frames, t, status: chunks.append(indata.copy())):
        input("  ● 録音中 … 読み終わって 1 秒おいてから Enter ")
    x = np.concatenate(chunks)[:, 0] if chunks else np.zeros(1, np.float32)
    x = x[int(0.3 * rate):max(int(0.3 * rate) + 1, len(x) - int(0.12 * rate))]  # drop the Enter key clicks at both ends
    sf.write(out, x, rate, subtype="PCM_24")
    return len(x) / rate


def main() -> None:
    args = sys.argv[1:]
    board = Path(args.pop(0)) if args and not args[0].startswith("--") else Path("docs/storyboard.json")
    start = int(args[args.index("--from") + 1]) if "--from" in args else 1
    only = {int(k) for k in args[args.index("--only") + 1].split(",")} if "--only" in args else None
    want = args[args.index("--mic") + 1] if "--mic" in args else None
    devs = mics()
    if not devs:
        sys.exit("no input device")
    if want is None:
        for i, name, rate in devs:
            print(f"  [{i}] {name}  ({rate} Hz)")
        s = input("マイクの番号 (Enter = 'MacBook Pro' を含むもの): ").strip()
        dev = int(s) if s else next((i for i, n, _ in devs if "MacBook" in n), devs[0][0])
    else:
        dev = next(i for i, n, _ in devs if want in n)
    name, rate = next((n, r) for i, n, r in devs if i == dev)
    print(f"マイク: [{dev}] {name}  {rate} Hz\n")
    segs = json.loads(board.read_text())
    for i, seg in enumerate(segs, 1):
        text = seg.get("narration")
        if not text or i < start or (only is not None and i not in only):
            continue
        out = OUT / f"{i:02d}.wav"
        while True:
            done = f"  (録音済み {sf.info(out).duration:.1f} 秒)" if out.exists() else ""
            print(f"\n=== 場面 {i:02d}  画面 {seg.get('sec', '?')} 秒{done}")
            print("  " + text)
            key = input("  Enter = 録音開始 / s = 飛ばす / q = 終了: ").strip().lower()
            if key == "q":
                return
            if key == "s":
                break
            took = record(dev, rate, out)
            print(f"  録音 {took:.1f} 秒(前後の無音は組み立て時に切られる)。再生します…")
            subprocess.run(["afplay", str(out)])
            key = input("  Enter = 採用 / r = 録り直し / q = 終了: ").strip().lower()
            if key == "q":
                return
            if key != "r":
                break
    print(f"\n{OUT}/ の録音: " + ", ".join(sorted(p.name for p in OUT.glob('[0-9][0-9].*'))))
    print("組み立て: NARR_DIR=outputs/narration uv run python scripts/make_video.py docs/storyboard.json outputs/video/draft15.mp4")


if __name__ == "__main__":
    main()
