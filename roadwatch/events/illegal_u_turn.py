"""illegal_u_turn: A U-turn where road markings or signs prohibit it.

Trigger: Cumulative heading change >= min_heading_change_deg within window_sec, speed >
min_start_speed_mps at the start, and the turn starts inside a no-U-turn zone (or the scene sets
u_turn_prohibited_everywhere). Either of those two layers is enough, so detect() checks them itself.
Start: yaw rate first > start_yaw_rate_dps.
End: heading stable within end_heading_tol_deg for end_stable_sec.

Thresholds: configs/thresholds.yaml -> classes.illegal_u_turn.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "illegal_u_turn"
REQUIRED_LAYERS: tuple[str, ...] = ("homography",)


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
