"""solid_line_crossing: A lane change or manoeuvre across a solid marking.

Trigger: Both bottom box corners start on one side of a solid polyline and end on the other, the
lateral move is >= min_lateral_m, and it happens outside the intersection.
Start: the first bottom corner crosses.
End: the second bottom corner crosses.

Thresholds: configs/thresholds.yaml -> classes.solid_line_crossing.params (SPEC §5).
Not implemented yet (RUNBOOK Phase 2): returns no events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "solid_line_crossing"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "solid_lines")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    return []
