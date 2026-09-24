"""road_obstacle: Debris, an animal or a fallen object on the carriageway.

Trigger: (a) A COCO animal/bag class on the carriageway for >= coco_min_sec, or (b) a static
foreground blob (background = running median over background_window_sec) on the carriageway, area >=
min_area_m2, not overlapping any vehicle/person box expanded by box_expand, static for >=
static_sec.
Start: first appearance.
End: gone for >= gone_sec.

Thresholds: configs/thresholds.yaml -> classes.road_obstacle.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "road_obstacle"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "carriageway")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
