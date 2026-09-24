"""stop_line: A vehicle stops past the stop line on red without entering the intersection.

Trigger: Signal RED and the vehicle stopped (speed < stop_speed_mps for >= stopped_sec) with its
front point past the stop line by more than past_line_m, footprint not in the intersection.
Start: the time it stopped.
End: the signal turns GREEN; overlapping vehicles form one segment (union).

Thresholds: configs/thresholds.yaml -> classes.stop_line.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "stop_line"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "stop_lines", "signals", "intersection")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
