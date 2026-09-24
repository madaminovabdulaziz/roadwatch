"""fire_smoke: Visible fire or smoke from a vehicle or on the road.

Trigger: Disabled by default (SPEC §5): enable only if an open fire/smoke model is shipped, gives
zero detections on every sample, and fires on public fire clips.
Start: first detection.
End: last detection + end_pad_sec.

Thresholds: configs/thresholds.yaml -> classes.fire_smoke.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "fire_smoke"
REQUIRED_LAYERS: tuple[str, ...] = ()


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
