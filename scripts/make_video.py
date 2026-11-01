#!/usr/bin/env python
"""Assemble the submission video (1280x720, 30 fps, H.264) from a storyboard of cards, clips, images and
side-by-side comparisons, with Japanese captions burnt in.

    uv run python scripts/make_video.py docs/storyboard.json outputs/video/final.mp4

Storyboard: a JSON list of segments.
  {"card": "title text", "sub": "smaller text", "sec": 4}
  {"clip": "path.mp4", "speed": 2, "caption": "...", "label": "corner label", "start": 0, "end": null}
  {"image": "docs/figures/flywheel.png", "sec": 6, "caption": "..."}
  {"side": ["left.mp4", "right.mp4"], "labels": ["v0", "v3"], "speed": 2, "caption": "...", "ends": [40, null]}
Clips are sped up by dropping frames (speed <= 2 keeps within the 2x rule); source fps is read from the file.
Any segment may carry "narration": it is spoken with macOS `say` (voice NARR_VOICE) from the segment start, and the
segment is stretched (holding its last frame) until the narration has finished.
With NARR_DIR=<dir>, a recorded take <dir>/NN.<ext> (NN = 1-based segment number, any ffmpeg-readable format) replaces
the synthetic voice of segment NN; leading/trailing silence is trimmed (NARR_DECLICK=1 also repairs clicks). Segments without a take fall back to `say`.
A segment with "clip" and no caption/label/narration is shown as is (opening and ending jingles).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG = (18, 22, 28)
SR, NARR_VOICE, NARR_RATE, LEAD = 22050, "Kyoko", 180, 0.2  # narration starts LEAD s into its segment
TAIL = float(os.environ.get("NARR_TAIL", "0.25"))  # air after the narration before the next segment (0.4 until draft16)
# Japanese Hiragino Sans (not "Hiragino Sans GB", whose kanji are the Simplified Chinese forms)
FONT = {False: "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", True: "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"}


def font(size: int, light: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT[light], size)
    except OSError:
        return ImageFont.load_default(size)


def text_center(dr: ImageDraw.ImageDraw, y: int, text: str, f, fill=(255, 255, 255)) -> None:
    for k, line in enumerate(text.split("\n")):
        w = dr.textlength(line, font=f)
        dr.text(((W - w) / 2, y + k * int(f.size * 1.45)), line, font=f, fill=fill)


def caption(img: Image.Image, text: str | None, label: str | None = None, label_y: int = 16) -> Image.Image:
    dr = ImageDraw.Draw(img, "RGBA")
    if text:
        lines = text.split("\n")
        size = 30
        while size > 18 and max(dr.textlength(l, font=font(size)) for l in lines) > W - 48:  # shrink to fit the width
            size -= 1
        f = font(size)
        h = 22 + int(30 * 1.45) * len(lines)
        dr.rectangle([0, H - h, W, H], fill=(0, 0, 0, 170))
        text_center(dr, H - h + 12, text, f)
    if label:
        f = font(24)
        w = dr.textlength(label, font=f)
        dr.rounded_rectangle([W - 36 - w, label_y, W - 16, label_y + 40], radius=8, fill=(0, 0, 0, 170))
        dr.text((W - 26 - w, label_y + 4), label, font=f, fill=(255, 230, 120))
    return img


def fit(frame: np.ndarray, w: int, h: int) -> Image.Image:
    im = Image.fromarray(frame)
    s = min(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), BG)
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def read_clip(path: str, speed: float, start: float = 0, end: float | None = None) -> list[np.ndarray]:
    r = imageio.get_reader(path)
    src_fps = r.get_meta_data().get("fps", 10)
    frames = [f for f in r]
    r.close()
    a, b = int(start * src_fps), len(frames) if end is None else int(end * src_fps)
    frames = frames[a:b]
    n_out = max(1, int(len(frames) / src_fps / speed * FPS))
    idx = np.minimum((np.arange(n_out) * len(frames) / n_out).astype(int), len(frames) - 1)
    return [frames[i] for i in idx]


def render(seg: dict) -> list[np.ndarray]:
    if "card" in seg:
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img)
        text_center(dr, H // 2 - 70, seg["card"], font(52))
        if seg.get("sub"):
            text_center(dr, H // 2 + 30, seg["sub"], font(30, light=True), fill=(190, 200, 210))
        return [np.asarray(img)] * int(seg.get("sec", 4) * FPS)
    if "image" in seg:
        img = caption(fit(np.asarray(Image.open(seg["image"]).convert("RGB")), W, H - 110), None)
        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(img, (0, 10))
        return [np.asarray(caption(canvas, seg.get("caption"), seg.get("label")))] * int(seg.get("sec", 5) * FPS)
    if "clip" in seg:
        frames = read_clip(seg["clip"], seg.get("speed", 1), seg.get("start", 0), seg.get("end"))
        hc = 22 + int(30 * 1.45) * len(seg["caption"].split("\n")) if seg.get("caption") else 0  # keep clear of the caption
        out = []
        for f in frames:
            canvas = Image.new("RGB", (W, H), BG)
            canvas.paste(fit(f, W, H - hc), (0, 0))
            out.append(np.asarray(caption(canvas, seg.get("caption"), seg.get("label"), label_y=84)))  # below the clip's own annotation band
        return out
    if "side" in seg:
        ends = seg.get("ends", [None] * len(seg["side"]))
        clips = [read_clip(p, seg.get("speed", 1), 0, e) for p, e in zip(seg["side"], ends)]
        n = max(len(c) for c in clips)
        out = []
        for i in range(n):
            canvas = Image.new("RGB", (W, H), BG)
            dr = ImageDraw.Draw(canvas)
            for k, c in enumerate(clips):
                tile = fit(c[min(i, len(c) - 1)], W // 2 - 8, H - 160)
                canvas.paste(tile, (k * (W // 2) + 4, 70))
                lab = seg.get("labels", ["", ""])[k]
                size = 30
                while size > 16 and dr.textlength(lab, font=font(size)) > W // 2 - 24:  # shrink to fit the tile
                    size -= 1
                w = dr.textlength(lab, font=font(size))
                dr.text((k * (W // 2) + (W // 2 - w) / 2, 20 + (30 - size) // 2), lab, font=font(size), fill=(255, 230, 120))
            out.append(np.asarray(caption(canvas, seg.get("caption"))))
        return out
    raise ValueError(f"unknown segment {seg}")


def load_wav(wav: Path) -> np.ndarray:
    with wave.open(str(wav)) as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def speak(text: str, tmp: Path) -> np.ndarray:
    wav = tmp / f"n{abs(hash(text))}.wav"
    subprocess.run(["say", "-v", NARR_VOICE, "-r", str(NARR_RATE), "--file-format=WAVE", f"--data-format=LEI16@{SR}",
                    "-o", str(wav), text], check=True)
    return load_wav(wav)


def take(index: int, tmp: Path) -> np.ndarray | None:
    """A recorded narration for segment `index` (1-based) from NARR_DIR, trimmed of silence at both ends."""
    d = os.environ.get("NARR_DIR")
    hits = sorted(Path(d).glob(f"{index:02d}.*")) if d else []
    if not hits:
        return None
    wav = tmp / f"t{index:02d}.wav"
    # -50 dB keeps devoiced sentence-final syllables (「〜ます」の「す」); the takes' noise floor is about -60 dB.
    trim = "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.2"
    # NARR_DECLICK=1 bridges step discontinuities (ffmpeg's AVFoundation capture used to drop buffers and click);
    # record_narration.py now captures through CoreAudio, so it is off by default.
    declick = "adeclick=t=1:w=20:o=75," if os.environ.get("NARR_DECLICK") == "1" else ""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(hits[0]), "-ac", "1", "-ar", str(SR),
                    "-af", f"{declick}{trim},areverse,{trim},areverse", str(wav)], check=True)
    print(f"  take {hits[0].name}")
    return load_wav(wav)


def main() -> None:
    board, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    silent = tmp / "video.mp4"
    w = imageio.get_writer(silent, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    total, voice = 0, []  # voice: (start sample, waveform)
    for i, seg in enumerate(board, 1):
        frames = render(seg)
        n_vid, n_voice = len(frames), 0.0
        if seg.get("narration"):
            v = take(i, tmp)
            if v is None:
                v = speak(seg["narration"], tmp)
            n_voice = len(v) / SR
            need = int((LEAD + n_voice + TAIL) * FPS)
            frames += [frames[-1]] * max(0, need - len(frames))  # hold the last frame until the voice is done
            voice.append((int((total / FPS + LEAD) * SR), v))
        for f in frames:
            w.append_data(f)
        total += len(frames)
        print(f"{total / FPS:6.1f}s  {next(iter(seg))}: {str(seg.get('card') or seg.get('clip') or seg.get('image') or seg.get('side'))[:60]}"
              f"  [video {n_vid / FPS:.1f}s, voice {n_voice:.1f}s]")
    w.close()
    track = np.zeros(int(total / FPS * SR) + SR, np.float32)
    for a, v in voice:
        track[a:a + len(v)] += v[:len(track) - a]
    with wave.open(str(tmp / "voice.wav"), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((np.clip(track * 0.95, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(tmp / "voice.wav"), "-c:v", "copy",
                    "-af", "loudnorm=I=-16:TP=-1.5", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", str(out)], check=True)
    print(f"wrote {out} ({total / FPS:.1f} s, narration {sum(len(v) for _, v in voice) / SR:.1f} s)")


if __name__ == "__main__":
    main()
