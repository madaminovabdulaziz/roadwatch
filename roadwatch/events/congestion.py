"""congestion: Traffic at a standstill or crawling across all lanes of one direction.

Trigger: Per direction, every step_sec: mean speed < max_mean_speed_mps AND >= min_vehicles_per_lane
vehicles in every lane, sustained >= min_sustain_sec, and either lasting >= long_sec or persisting
>= green_persist_sec into a GREEN phase. A normal red-light queue is not congestion (SPEC §12.1).
Start: the queue stopped moving (backdated to when the condition began).
End: mean speed > clear_speed_mps sustained for clear_hold_sec.

Thresholds: configs/thresholds.yaml -> classes.congestion.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "congestion"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "directions", "lanes")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
