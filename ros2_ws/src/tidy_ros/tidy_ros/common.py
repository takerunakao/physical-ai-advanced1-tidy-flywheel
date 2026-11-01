"""Helpers shared by the ROS 2 client nodes."""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


def image_to_array(msg: Image) -> np.ndarray:
    return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, 3)


def obs_from_response(resp) -> dict:
    """Build the observation dict expected by tidy_flywheel policies from a Reset/Step response."""
    return {
        "state": np.asarray(resp.state, dtype=np.float32),
        "images": {"front": image_to_array(resp.front), "wrist": image_to_array(resp.wrist)},
    }


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out
