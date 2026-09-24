"""Segment post-processing (SPEC §6, RUNBOOK P2.1), applied in this order:

1. drop segments with score < the class's `min_score`;
2. per class, merge segments separated by less than `merge_gap`;
3. drop segments shorter than `min_dur`;
4. boundary refinement: re-read +/- `refine_window_sec` at stride 1 around starts/ends of
   `refine_classes`, within the `runtime.refine_budget_frac` frame budget;
5. same-class union (overlaps become one segment);
6. clip to [0, duration], drop start >= end, plain Python floats rounded to `round_decimals`.

Contract: `postprocess(segments, ctx, cfg)` runs steps 1-5; `to_events(segments, duration)` runs
step 6 and returns `[[start, end, label], ...]` that passes evaluate.py's format check.
"""

from __future__ import annotations

from typing import Any

from roadwatch.events.base import VideoContext
from roadwatch.types import Segment


def postprocess(segments: list[Segment], ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    raise NotImplementedError("Post-processing lands in RUNBOOK P2.1")


def to_events(segments: list[Segment], duration: float) -> list[list]:
    raise NotImplementedError("Post-processing lands in RUNBOOK P2.1")
