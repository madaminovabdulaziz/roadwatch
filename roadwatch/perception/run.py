"""Offline perception for Part A (RUNBOOK P0.3).

Contract: `run_perception(video_path, stride)` decodes every `stride`-th frame, detects in batches with
the shared `Detector`, tracks with a fresh `OnlineTracker`, and returns the TrackTable sorted by
`(t, track_id)` (`types.TRACK_COLUMNS`; footprint `(fx, fy)` = bottom-centre of the box).
`scripts/cache_tracks.py` stores this table as cache/tracks/<video>.parquet for fast rule iteration.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_perception(video_path: str | Path, stride: int) -> pd.DataFrame:
    raise NotImplementedError("run_perception lands in RUNBOOK P0.3")
