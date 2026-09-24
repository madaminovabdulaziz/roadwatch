"""wrong_way: A vehicle moves against the traffic direction of its lane (incl. the oncoming lane).

Trigger: Speed > min_speed_mps and cos(velocity, lane direction) < max_cos_to_lane for >= min_sec.
Start: time the footprint entered the lane where it is wrong (walked back).
End: returns to a correct lane or leaves the frame.

Thresholds: configs/thresholds.yaml -> classes.wrong_way.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "wrong_way"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "lanes")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
