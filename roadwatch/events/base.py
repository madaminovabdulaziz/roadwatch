"""Event-rule contract and shared per-video context (SPEC §5).

Every module in roadwatch/events/ defines:
    LABEL: str                        # one of solution.CLASSES
    REQUIRED_LAYERS: tuple[str, ...]  # scene layers it needs; any missing -> rule disabled, never guessed
    detect(tt, scene, ctx, cfg) -> list[Segment]
`tt` is the TrackTable with kinematics, `cfg` the class block from thresholds.yaml. Rules return raw
segments with a score in [0, 1]; no tracks means no segments. Merging, minimum durations, boundary
refinement, union and clipping belong to postprocess.py, not to rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from roadwatch.scene.light import SignalTimeline
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment, VideoMeta


@dataclass
class VideoContext:
    meta: VideoMeta
    stride: int
    signal_timeline: SignalTimeline = field(default_factory=dict)
    background: np.ndarray | None = None


class EventRule(Protocol):
    LABEL: str
    REQUIRED_LAYERS: tuple[str, ...]

    def detect(
        self, tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]
    ) -> list[Segment]: ...


def is_runnable(rule: EventRule, scene: Scene, cfg: dict[str, Any]) -> bool:
    """A rule runs only if its class is enabled and the scene has every layer it needs (SPEC §7)."""
    return bool(cfg.get("enabled", False)) and all(scene.has(layer) for layer in rule.REQUIRED_LAYERS)
