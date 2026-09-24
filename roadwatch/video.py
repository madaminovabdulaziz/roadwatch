"""Video decoding.

Contract:
- `probe(path)` returns `VideoMeta` read exactly like the harness (OpenCV CAP_PROP_FPS and
  CAP_PROP_FRAME_COUNT), so timestamps and duration match the metric. `t_sec = frame_idx / fps`.
- `FrameReader(path, stride)` yields `(frame_idx, t_sec, frame_bgr)` for every `stride`-th frame.
- `read_window(path, t0, t1, stride)` returns the frames in `[t0, t1)` for boundary refinement.

The samples are 4K H.264 4:2:2 10-bit at 29.97 fps and the T4 cannot decode them in hardware
(SPEC §12), so decoding is CPU-bound: readers must not convert frames they will not use.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from roadwatch.types import VideoMeta


def probe(video_path: str | Path) -> VideoMeta:
    """Read fps, size and frame count without decoding; raises if the file cannot be opened."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        return VideoMeta(
            video_id=path.name,
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


class FrameReader:
    """Sequential reader over every `stride`-th frame (RUNBOOK P0.2)."""

    def __init__(self, video_path: str | Path, stride: int = 1) -> None:
        self.video_path = Path(video_path)
        self.stride = stride

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        raise NotImplementedError("FrameReader lands in RUNBOOK P0.2")


def read_window(
    video_path: str | Path, t0: float, t1: float, stride: int = 1
) -> list[tuple[int, float, np.ndarray]]:
    """Frames with `t0 <= t_sec < t1` at the given stride (RUNBOOK P0.2)."""
    raise NotImplementedError("read_window lands in RUNBOOK P0.2")
