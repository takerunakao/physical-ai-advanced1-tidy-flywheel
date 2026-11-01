"""VLM grounding: text instruction + front-camera image -> goal points for the visuomotor policy.

    "put the blue cube in the black box"  ->  [u_obj, v_obj, u_box, v_box]  (normalised image coordinates)

This is the only place language enters the system. The policy never sees text; it sees two points, which a
person could also give by tapping the image. Two backends are compared against the simulator's exact
projections (scripts/vlm_ground_eval.py): Gemini Robotics-ER, a model trained for embodied pointing, and
Claude, a general VLM.

API keys are read from the environment or from the project's .env (never committed).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

PROJECT_ENV = Path(__file__).resolve().parents[1] / ".env"
GEMINI_MODEL = "gemini-robotics-er-1.6-preview"
CLAUDE_MODEL = "claude-opus-5-5"


def load_dotenv(path: str | Path = PROJECT_ENV) -> None:
    """Minimal .env loader (KEY=VALUE lines). The project .env overrides the shell, whose key may be stale."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m and not line.lstrip().startswith("#"):
            os.environ[m[1]] = m[2].strip("'\"")


def anthropic_client():
    """Anthropic client from the environment/.env. An org-level key needs ANTHROPIC_WORKSPACE_ID as well."""
    import anthropic

    load_dotenv()
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)


def split_instruction(instruction: str) -> tuple[str, str]:
    """'put the blue cube in the black box' -> ('blue cube', 'black box')."""
    m = re.match(r"put the (.+) in the (.+)$", instruction.strip())
    if not m:
        raise ValueError(f"unexpected instruction: {instruction!r}")
    return m[1], m[2]


def _png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------- Gemini Robotics-ER
GEMINI_PROMPT = """Point to the {obj}, and to where it should be dropped in the {box}: the box floor fits two \
objects side by side (left half and right half in the image); point to the centre of a free half (the left \
half if the box is empty, otherwise the half away from the object already inside).
The answer should follow the json format: [{{"point": <point>, "label": <label>}}, ...]. \
The points are in [y, x] format normalized to 0-1000. Use the labels "object" and "box"."""


def ground_gemini(image: np.ndarray, instruction: str, client=None, thinking_budget: int = 0,
                  view: str = "front") -> np.ndarray:
    from google import genai
    from google.genai import types

    load_dotenv()
    client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    obj, box = split_instruction(instruction)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=_png(image), mime_type="image/png"), GEMINI_PROMPT.format(obj=obj, box=box)],
        config=types.GenerateContentConfig(temperature=0.0, thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)),
    )
    text = re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.MULTILINE).strip()
    pts = {d["label"]: d["point"] for d in json.loads(text)}
    (oy, ox), (by, bx) = pts["object"], pts["box"]
    return np.array([ox, oy, bx, by], dtype=np.float32) / 1000.0


# ----------------------------------------------------------------------------- Claude
class _Points(BaseModel):
    object_x: int = Field(description="x of the object's centre, 0 (left edge) to 1000 (right edge)")
    object_y: int = Field(description="y of the object's centre, 0 (top edge) to 1000 (bottom edge)")
    box_x: int = Field(description="x of the drop spot on the box floor, 0-1000")
    box_y: int = Field(description="y of the drop spot on the box floor, 0-1000")


CLAUDE_PROMPT = """The image is {w}x{h} pixels from a fixed camera {view} a table with a robot arm. \
Locate the centre of the {obj}, and choose where to drop it in the {box}. The box floor fits exactly two \
objects side by side, one in its left half and one in its right half (as seen in the image). Point to the \
centre of a free half: the left half if the box is empty, otherwise the half away from the object already \
inside. Give both points as coordinates normalised to 0-1000 (x from the left edge, y from the top edge)."""


VIEWS = {"front": "looking obliquely down at", "top": "looking straight down at"}


def ground_claude(image: np.ndarray, instruction: str, client=None, model: str = CLAUDE_MODEL,
                  view: str = "front") -> np.ndarray:
    client = client or anthropic_client()
    obj, box = split_instruction(instruction)
    h, w = image.shape[:2]
    resp = client.messages.parse(
        model=model,
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(_png(image)).decode()}},
            {"type": "text", "text": CLAUDE_PROMPT.format(w=w, h=h, obj=obj, box=box, view=VIEWS[view])},
        ]}],
        output_format=_Points,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"no points (stop_reason={resp.stop_reason})")
    p = resp.parsed_output
    return np.array([p.object_x, p.object_y, p.box_x, p.box_y], dtype=np.float32) / 1000.0


BACKENDS = {"gemini": ground_gemini, "claude": ground_claude}


def ground(image: np.ndarray, instruction: str, backend: str = "gemini", **kw) -> np.ndarray:
    return BACKENDS[backend](image, instruction, **kw)


def ground_env(env, instruction: str, backend: str = "claude", view: str = "front", res: int = 480) -> np.ndarray:
    """Render `view`, point there, and return goal points in the policy's goal camera (MultiTidyEnv)."""
    pts = ground(env.render(view, (res, res)), instruction, backend, view=view)
    return env.to_goal_points(pts, view)
