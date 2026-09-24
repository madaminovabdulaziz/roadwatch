"""near_miss: Sharp braking or swerving to avoid a collision, with no contact.

Trigger: A pair with TTC < max_ttc_sec and closing speed > min_closing_speed_mps, followed by
evasive action (deceleration > evasive_decel_mps2 or yaw rate > evasive_yaw_rate_dps); separation
stays > min_separation_m and no accident follows within no_accident_within_sec.
Start: onset of the evasive action (deceleration first > onset_decel_mps2).
End: separation increasing and TTC > end_ttc_sec.

Thresholds: configs/thresholds.yaml -> classes.near_miss.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "near_miss"
REQUIRED_LAYERS: tuple[str, ...] = ("homography",)


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
