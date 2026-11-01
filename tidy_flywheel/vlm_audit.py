"""Demonstration audit: does a recorded demo actually carry out the instruction it is labelled with?

A mislabelled demo (the arm puts the red cube away while the label says "blue cube") teaches a goal-conditioned
policy to ignore its goal. The audit looks only at what a dataset holds -- the episode's front-camera frames
(128 px) and its task string -- and returns a verdict per episode, so flagged demos can be dropped before training.
"""
from __future__ import annotations

import base64
import io
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

from .sim_multi import BOXES, OBJECTS
from .vlm_judge import MODEL

ObjectOrNone = Literal[tuple([name for _, name in OBJECTS] + ["none"])]  # type: ignore[valid-type]
BoxOrNone = Literal[tuple([name for _, name in BOXES] + ["none"])]  # type: ignore[valid-type]

PROMPT = """You audit robot demonstration data before it is used for training. Each demonstration is labelled \
with an instruction; a label can be wrong. The scene has a red cube, a blue cube and a green cylinder on a table \
and two open boxes, a white one and a black one. The frames come from the fixed front camera at low resolution \
(upscaled), in time order, labelled with their step index.

Say which object the arm actually moved and which box it ended up in, then whether that matches the label. An \
object that was already inside a box at step 0 does not count as moved."""


class AuditVerdict(BaseModel):
    moved_object: ObjectOrNone
    into_box: BoxOrNone
    matches_label: bool
    confidence: float  # 0..1
    rationale: str  # one or two sentences


def _png_b64(frame: np.ndarray, size: int = 384) -> str:
    img = Image.fromarray(frame).resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def audit_episode(client, samples: list[tuple[int, np.ndarray]], label: str, model: str = MODEL) -> AuditVerdict:
    content: list[dict] = []
    for i, f in samples:
        content.append({"type": "text", "text": f"step {i}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(f)}})
    content.append({"type": "text", "text": f'Label: "{label}". What did the arm do, and does it match the label?'})
    resp = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=PROMPT,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
        output_format=AuditVerdict,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"no verdict (stop_reason={resp.stop_reason})")
    return resp.parsed_output
