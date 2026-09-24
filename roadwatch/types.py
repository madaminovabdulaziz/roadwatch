"""Shared data types.

- `VideoMeta`: what we know about a video before decoding. `duration = n_frames / fps`, exactly as the
  harness computes it (SPEC §12), so our clipping matches the metric's.
- `FrameDetections`: detector output for one frame as parallel numpy arrays (no per-box objects).
- `Segment`: one raw event from a rule, before post-processing.
- `TRACK_COLUMNS` / `empty_track_table()`: the TrackTable schema (SPEC §3), one row per track per
  processed frame; `features.add_kinematics` appends the kinematic columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


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


@dataclass
class Segment:
    start: float
    end: float
    label: str
    score: float = 1.0
    track_ids: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


TRACK_COLUMNS: tuple[str, ...] = (
    "frame",
    "t",
    "track_id",
    "cls",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "fx",
    "fy",
)


def empty_track_table() -> pd.DataFrame:
    """A TrackTable with the right columns and no rows."""
    return pd.DataFrame({col: pd.Series(dtype="float64") for col in TRACK_COLUMNS})
