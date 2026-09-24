"""red_light: A vehicle crosses its stop line while its signal is red.

Trigger: The signal of the vehicle's lane is RED (stable >= red_stable_sec) and the vehicle's front
point crosses the stop line in the lane direction at least min_after_red_onset_sec after red onset.
Start: crossing time (interpolated between processed frames).
End: footprint leaves the intersection polygon, or the vehicle leaves the frame.

Thresholds: configs/thresholds.yaml -> classes.red_light.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "red_light"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "lanes", "stop_lines", "signals", "intersection")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
