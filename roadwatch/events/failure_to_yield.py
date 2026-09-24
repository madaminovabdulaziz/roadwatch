"""failure_to_yield: A vehicle drives through a crossing while a pedestrian is on it or stepping onto it.

Trigger: Vehicle speed > min_vehicle_speed_mps with its footprint in a crosswalk polygon while at
least one person footprint is in the same crosswalk, or within ped_edge_dist_m of its edge and
moving toward it.
Start: the vehicle enters the crosswalk.
End: the vehicle leaves the crosswalk.

Thresholds: configs/thresholds.yaml -> classes.failure_to_yield.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "failure_to_yield"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "crosswalks")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
