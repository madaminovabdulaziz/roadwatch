"""jaywalking: A pedestrian on the carriageway outside a crossing.

Trigger: Person footprint on the carriageway, outside crosswalks and sidewalks, for >= min_sec.
Riders are excluded (box centre inside a two-wheeler box expanded by rider_box_expand, or IoU >
rider_iou with one), and so are persons at least in_vehicle_frac inside a vehicle box (SPEC
§12.5-12.6).
Start: the pedestrian steps onto the road.
End: the pedestrian leaves the road; simultaneous pedestrians form one segment (union).

Thresholds: configs/thresholds.yaml -> classes.jaywalking.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "jaywalking"
REQUIRED_LAYERS: tuple[str, ...] = ("carriageway", "crosswalks")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
