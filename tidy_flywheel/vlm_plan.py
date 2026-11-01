"""Task planning: a free-form command ("片付けて", "cubes in the white box, the rest in the black box") plus the
front-camera image -> an ordered list of single-object instructions, each executed by ground -> visuomotor policy.

The planner only chooses *what* goes *where* and in which order; it never outputs coordinates. Pointing is the
grounding step (vlm_ground.py), motion is the ACT policy.
"""
from __future__ import annotations

import base64
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .sim_multi import BOXES, OBJECTS, instruction
from .vlm_ground import CLAUDE_MODEL, _png, anthropic_client

ObjectName = Literal[tuple(name for _, name in OBJECTS)]  # type: ignore[valid-type]
BoxName = Literal[tuple(name for _, name in BOXES)]  # type: ignore[valid-type]

PROMPT = """You plan a tidy-up for a robot arm that moves one object at a time into an open box.
The images come from two fixed cameras (front and side) looking at the same table; use both, since an \
object can be hidden behind a box in one view. Objects that may be on the table: {objects}. \
Boxes: {boxes}. Objects lying flat on the floor of a box are done; do not move them again. An object resting \
on a box's rim or wall, tilted against it, stacked on another object, or anywhere outside a box is NOT done \
and must be planned again (into the box with room).

User's request: "{command}"

Return the steps to carry out the request, one object per step, in a sensible order (for example, nearer or \
unobstructed objects first). Each box holds at most two objects. If the request does not say which box to use, \
choose one and keep similar objects together."""


class Step(BaseModel):
    object: ObjectName
    box: BoxName


class Plan(BaseModel):
    reasoning: str = Field(description="one or two sentences: what is on the table and why this order")
    steps: list[Step]


def plan(views: dict[str, np.ndarray], command: str, client=None, model: str = CLAUDE_MODEL,
         attempted: list[str] | None = None) -> tuple[list[tuple[int, int]], str]:
    """views: {camera name: image}; attempted: instructions already executed in this scene (the
    images show which of them worked; no simulator state is passed).
    -> ([(object index, box index), ...], reasoning)."""
    client = client or anthropic_client()
    content: list[dict] = []
    for name, img in views.items():
        content.append({"type": "text", "text": f"{name} camera:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                    "data": base64.standard_b64encode(_png(img)).decode()}})
    prompt = PROMPT.format(objects=", ".join(n for _, n in OBJECTS), boxes=", ".join(n for _, n in BOXES), command=command)
    if attempted:
        prompt += ("\n\nSteps the robot has already executed, in order: " + "; ".join(f'"{a}"' for a in attempted)
                   + ". If the images show an object still not done after it was attempted twice, put it in a "
                   "different box with room instead.")
    resp = client.messages.parse(
        model=model,
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}],
        output_format=Plan,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"no plan (stop_reason={resp.stop_reason})")
    p = resp.parsed_output
    obj_idx = {n: i for i, (_, n) in enumerate(OBJECTS)}
    box_idx = {n: j for j, (_, n) in enumerate(BOXES)}
    return [(obj_idx[s.object], box_idx[s.box]) for s in p.steps], p.reasoning


def describe(steps: list[tuple[int, int]]) -> list[str]:
    return [instruction(o, b) for o, b in steps]
