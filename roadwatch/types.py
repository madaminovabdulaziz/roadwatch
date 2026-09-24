"""Shared data types.

- `Frame` = `(frame_idx, t_sec, frame_bgr)`, the unit every reader yields; `Size` = `(width, height)`.
- `VideoMeta`: what we know about a video before decoding. `duration = n_frames / fps`, exactly as the
  harness computes it (SPEC §12), so our clipping matches the metric's.
- `FrameDetections`: detector output for one frame as parallel numpy arrays (no per-box objects).
- `FrameTracks`: tracker output for one frame, the same arrays plus a track id per box.
- `Segment`: one raw event from a rule, before post-processing.
- `TRACK_DTYPES` / `empty_track_table()`: the TrackTable schema (SPEC §3), one row per track per
  processed frame, footprint `(fx, fy)` = bottom-centre of the box in native pixels;
  `features.add_kinematics` appends the kinematic columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

Frame = tuple[int, float, np.ndarray]
Size = tuple[int, int]


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class FrameDetections:
    frame_idx: int
    t: float
    xyxy: np.ndarray  # (N, 4) float32, native-resolution pixels
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int64, COCO class ids


@dataclass(frozen=True)
class FrameTracks:
    frame_idx: int
    t: float
    track_id: np.ndarray  # (N,) int64, unique across tracker groups within one video
    cls: np.ndarray  # (N,) int64, COCO class id of this frame's detection
    conf: np.ndarray  # (N,) float32
    xyxy: np.ndarray  # (N, 4) float32, native-resolution pixels


@dataclass
class Segment:
    start: float
    end: float
    label: str
    score: float = 1.0
    track_ids: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


TRACK_DTYPES: dict[str, str] = {
    "frame": "int32",
    "t": "float64",
    "track_id": "int32",
    "cls": "category",  # class name, one per track (majority over the track's detections)
    "conf": "float32",
    "x1": "float32",
    "y1": "float32",
    "x2": "float32",
    "y2": "float32",
    "fx": "float32",
    "fy": "float32",
}
TRACK_COLUMNS: tuple[str, ...] = tuple(TRACK_DTYPES)


def empty_track_table() -> pd.DataFrame:
    """A TrackTable with the right columns and dtypes and no rows."""
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in TRACK_DTYPES.items()})
