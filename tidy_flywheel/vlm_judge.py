"""VLM success judge: decides from camera frames alone whether a tidy-up episode succeeded.

The simulator knows the true outcome (privileged state); a real robot does not. This judge replaces the
privileged check with what a deployed robot actually has, frames from its camera, so the data flywheel can
find its own failures. `scripts/vlm_judge_eval.py` measures its agreement with the privileged ground truth.
"""
from __future__ import annotations

import base64
import io
from typing import Literal

import imageio
import numpy as np
from PIL import Image
from pydantic import BaseModel

MODEL = "claude-opus-5-5"

FailureMode = Literal["success", "no_grasp", "dropped", "missed_bin", "timeout_holding", "contact", "wrong_object"]

PROMPT = """You are checking a robot arm's tidy-up attempt in simulation. Task: pick up the red cube and \
leave it resting inside the grey open box, with the gripper released.

The images are frames from the fixed front camera in time order, labelled with their step index; the last \
image is the final state. Judge from the images only.

Failure modes:
- no_grasp: the cube was never lifted off the table
- dropped: the cube was lifted but fell outside the box
- missed_bin: the cube was carried to the box but ended on the rim or next to it
- timeout_holding: the gripper still holds the cube at the end
- contact: the arm itself (not the fingers) hits the table or the box
Report success only if the final frame shows the cube inside the box and not held by the gripper."""

PROMPT_MULTI = """You are checking a robot arm's tidy-up attempt in simulation. The table holds a red cube, a blue \
cube and a green cylinder, and there are two open boxes, a white one and a black one. Instruction: "{task}". \
Success means the instructed object ends resting inside the instructed box, not held by the gripper, and no \
other object was put into a box.

The images are frames from the fixed front camera in time order, labelled with their step index; the last \
image is the final state. Judge from the images only.

Failure modes:
- no_grasp: the instructed object was never lifted off the table
- dropped: it was lifted but fell outside the instructed box
- missed_bin: it was carried to the box but ended on the rim or next to it
- timeout_holding: the gripper still holds it at the end
- contact: the arm itself (not the fingers) hits the table or a box
- wrong_object: a different object was picked up, pushed away or put into a box"""


def system_prompt(task: str | None = None) -> str:
    """Single scene (task None): the fixed prompt measured in EXPERIMENT_LOG. Multi scene: the instruction."""
    return PROMPT if task is None else PROMPT_MULTI.format(task=task)


class Verdict(BaseModel):
    success: bool
    failure_mode: FailureMode
    confidence: float  # 0..1
    rationale: str  # one or two sentences, citing the frames


def sample_frames(frames: list[np.ndarray], k: int = 6) -> list[tuple[int, np.ndarray]]:
    """k frames evenly spaced over the episode, always including the first and last."""
    idx = np.unique(np.linspace(0, len(frames) - 1, k).round().astype(int))
    return [(int(i), frames[i]) for i in idx]


def _png_b64(frame: np.ndarray, width: int = 480) -> str:
    img = Image.fromarray(frame)
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def make_client():
    from .vlm_ground import anthropic_client

    return anthropic_client()


def judge_samples(client, samples: list[tuple[int, np.ndarray]], model: str = MODEL, task: str | None = None) -> Verdict:
    """samples: (step index, frame) pairs in time order, e.g. from sample_frames."""
    content: list[dict] = []
    for i, f in samples:
        content.append({"type": "text", "text": f"step {i}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(f)}})
    content.append({"type": "text", "text": "Did the robot succeed? Classify the outcome."})
    resp = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=system_prompt(task),
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
        output_format=Verdict,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"no verdict (stop_reason={resp.stop_reason})")
    return resp.parsed_output


def judge_frames(client, frames: list[np.ndarray], k: int = 6, model: str = MODEL, task: str | None = None) -> Verdict:
    return judge_samples(client, sample_frames(frames, k), model, task)


def judge_video(client, path: str, k: int = 6, model: str = MODEL) -> Verdict:
    return judge_frames(client, list(imageio.mimread(path, memtest=False)), k, model)
