"""illegal_turn: A turn from the wrong lane or in a prohibited direction.

Trigger: The movement (entry lane at the stop line -> exit polygon) is not in the entry lane's
allowed_exits. U-turns are left to illegal_u_turn.
Start: the vehicle leaves its entry lane / yaw rate > start_yaw_rate_dps.
End: enters the exit polygon with heading stable within end_heading_tol_deg for end_stable_sec.

Thresholds: configs/thresholds.yaml -> classes.illegal_turn.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "illegal_turn"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "lanes", "exits", "stop_lines")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
