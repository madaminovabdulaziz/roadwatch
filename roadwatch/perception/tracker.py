"""Online multi-object tracking (SPEC §3, RUNBOOK P0.3).

Contract:
- `OnlineTracker(frame_rate)` wraps `supervision.ByteTrack`, one tracker per class group from
  `perception.tracker_groups`, so ids never jump between object types. `frame_rate = fps / stride`.
- `update(dets)` consumes one frame's `FrameDetections` in time order and returns that frame's rows of
  the TrackTable (`types.TRACK_COLUMNS`); track ids are unique across groups.
- Fixed camera: no camera-motion compensation. Deterministic for identical input.
"""

from __future__ import annotations

import pandas as pd

from roadwatch.types import FrameDetections


class OnlineTracker:
    """Per-class-group ByteTrack."""

    def __init__(self, frame_rate: float) -> None:
        self.frame_rate = frame_rate

    def update(self, dets: FrameDetections) -> pd.DataFrame:
        raise NotImplementedError("OnlineTracker lands in RUNBOOK P0.3")
