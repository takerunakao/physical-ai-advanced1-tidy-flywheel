"""Video recorder: writes one MP4 per episode from /camera/video/image_raw, split on /episode/result.

Parameters
  video_dir  output directory (required)
  tag        file-name prefix, e.g. "v1" -> v1_seed1000_success.mp4
  fps        playback rate (sim runs at 10 Hz; default 10 = real time)
  hold_frames  frames of the final (annotated) image appended at the end when an overlay is active
Frames arriving between two /episode/result messages belong to one episode (the reset frame starts it).

Overlay (tidy-up runs): if /tidy/status (std_msgs/String, one line per row) has been received, its latest text is
drawn in a black band at the top of every frame, and the latest /tidy/goal_points (object = yellow ring,
drop point = green ring) are marked, as in scripts/tidy_all.py. Without status messages frames are written
unchanged. After each MP4 is written its path is published on /video/saved.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio
import numpy as np
import rclpy
from PIL import Image, ImageDraw, ImageFont
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Float32MultiArray, String

from .common import image_to_array

CJK_FONTS = (  # fonts-noto-cjk (Docker image), then macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
)


def load_font(size: int = 18):
    for path in CJK_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()  # no CJK glyphs: Japanese text would render as boxes


def annotate(frame: np.ndarray, lines: list[str], points: np.ndarray | None, font) -> np.ndarray:
    """Same overlay as scripts/tidy_all.py: status lines on a black band, goal points as rings."""
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, img.width, 8 + 24 * len(lines)], fill=(0, 0, 0))
    for k, t in enumerate(lines):
        dr.text((8, 4 + 24 * k), t, fill=(255, 255, 255), font=font)
    if points is not None and len(points) == 4:  # goal points are normalised for a square image; the video is wider
        h, w = frame.shape[:2]
        for (u, v), col in ((points[:2], (255, 220, 0)), (points[2:], (0, 255, 120))):
            x, y = (0.5 + (u - 0.5) * h / w) * w, v * h
            dr.ellipse([x - 9, y - 9, x + 9, y + 9], outline=col, width=3)
    return np.asarray(img)


class VideoNode(Node):
    def __init__(self) -> None:
        super().__init__("video_node")
        self.declare_parameter("video_dir", "")
        self.declare_parameter("tag", "policy")
        self.declare_parameter("fps", 10)
        self.declare_parameter("hold_frames", 20)
        self.out = Path(str(self.get_parameter("video_dir").value))
        self.out.mkdir(parents=True, exist_ok=True)
        self.tag = str(self.get_parameter("tag").value)
        self.fps = int(self.get_parameter("fps").value)
        self.hold = int(self.get_parameter("hold_frames").value)
        self.frames: list = []
        self.last_raw: np.ndarray | None = None
        self.lines: list[str] | None = None  # None until the first /tidy/status message
        self.points: np.ndarray | None = None
        self.font = load_font()
        self.create_subscription(ImageMsg, "/camera/video/image_raw", self.on_image, 50)
        self.create_subscription(String, "/episode/result", self.on_result, 10)
        self.create_subscription(String, "/tidy/status", self.on_status, 10)
        self.create_subscription(Float32MultiArray, "/tidy/goal_points", self.on_points, 10)
        self.pub_saved = self.create_publisher(String, "/video/saved", 10)

    def on_status(self, msg: String) -> None:
        self.lines = msg.data.split("\n")

    def on_points(self, msg: Float32MultiArray) -> None:
        self.points = np.asarray(msg.data, dtype=np.float32) if len(msg.data) == 4 else None

    def _render(self, raw: np.ndarray) -> np.ndarray:
        return raw if self.lines is None else annotate(raw, self.lines, self.points, self.font)

    def on_image(self, msg: ImageMsg) -> None:
        self.last_raw = image_to_array(msg).copy()
        self.frames.append(self._render(self.last_raw))

    def on_result(self, msg: String) -> None:
        ep = json.loads(msg.data)
        mode = ep.get("failure_mode", "success" if ep.get("success") else "failure")
        path = self.out / f"{self.tag}_seed{int(ep['seed']):04d}_{mode}.mp4"
        if self.lines is not None and self.last_raw is not None:  # hold the final state with the latest status
            self.frames += [self._render(self.last_raw)] * self.hold
        if self.frames:
            imageio.mimwrite(path, self.frames, fps=self.fps, codec="libx264", quality=8)
            self.get_logger().info(f"wrote {path} ({len(self.frames)} frames)")
            self.pub_saved.publish(String(data=str(path)))
        self.frames = []


def main() -> None:
    rclpy.init()
    node = VideoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
