"""accident: Collision between road users, or between a road user and a fixed object.

Trigger: A pair (vehicle-vehicle, vehicle-person, vehicle-two-wheeler) with footprints closer than
contact_dist_m AND overlapping boxes, AND at least one kinematic shock within +/- shock_window_sec
(deceleration > shock_decel_mps2, heading change > shock_heading_deg in under
shock_heading_window_sec, or a person box flipping aspect as they fall); confirmed when the involved
objects stay stopped >= confirm_stopped_sec outside any queue, or leave. Single vehicle: speed >
single_speed_before_mps drops below single_speed_after_mps within single_window_sec, off-lane or
near the road edge. Box overlap alone is never enough (occlusion is the main false-positive source).
Start: first contact frame (refined at stride 1).
End: all involved speeds < end_speed_mps for end_hold_sec, or they leave the frame.

Thresholds: configs/thresholds.yaml -> classes.accident.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "accident"
REQUIRED_LAYERS: tuple[str, ...] = ("homography",)


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
