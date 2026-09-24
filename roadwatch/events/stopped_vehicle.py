"""stopped_vehicle: A vehicle stationary on the carriageway for 10 s or more, not queued at a signal.

Trigger: Speed < stop_speed_mps continuously for >= min_stopped_sec, footprint on the carriageway,
not in a parking zone, and NOT queued: not within queue_upstream_m upstream of a stop line whose
signal is red/yellow, and no stopped vehicle ahead in the same lane within queue_ahead_m. Position
persistence bridges id switches of stationary objects (SPEC §3).
Start: first time the speed dropped below stop_speed_mps (backdated).
End: speed > resume_speed_mps for resume_hold_sec, or the track is gone; still stopped at the last
frame -> video duration.

Thresholds: configs/thresholds.yaml -> classes.stopped_vehicle.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "stopped_vehicle"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "carriageway")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
