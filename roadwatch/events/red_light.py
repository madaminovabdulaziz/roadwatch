"""red_light: A vehicle crosses its stop line while its signal is red.

Trigger: The signal of the vehicle's lane is RED (stable >= red_stable_sec) and the vehicle's front
point crosses the stop line in the lane direction at least min_after_red_onset_sec after red onset.
Start: crossing time (interpolated between processed frames).
End: footprint leaves the intersection polygon, or the vehicle leaves the frame.

Details:
- Only the lamp-based signal timeline counts; without it (state unknown) nothing fires.
- The crossing must go the lane's way (image step along the lane direction) from one of the stop
  line's lanes. A vehicle that crosses but never enters the intersection crept past the line and
  stopped: that is stop_line, not red_light.
Score = min(1, 0.5 + (crossing - red onset) / 2 s): crossings right after the change are less certain.

Thresholds: configs/thresholds.yaml -> classes.red_light.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, crossings, in_group, max_gap
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "red_light"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "lanes", "stop_lines", "signals", "intersection")


def red_segment_at(segments: list[tuple[float, float, str]], t: float) -> tuple[float, float] | None:
    """(onset, end) of the red phase containing t, or None."""
    for t0, t1, state in segments:
        if state == "red" and t0 <= t <= t1:
            return t0, t1
    return None


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    dirs = scene.lane_directions()
    v = tt[in_group(tt, "vehicles", "two_wheelers")]
    gap = max_gap(ctx)
    segments = []
    for sl in scene.layers.get("stop_lines") or []:
        timeline = ctx.signal_timeline.get(str(sl.get("signal", "")), [])
        if not timeline:
            continue
        line = np.asarray(sl["line"], dtype=np.float64)
        lanes = set(sl.get("lanes", []))
        for tid, rows in by_track(v):
            t = rows["t"].to_numpy()
            front = rows[["front_x", "front_y"]].to_numpy(dtype=np.float64)
            lane_ids = rows["lane_id"].to_numpy()
            inter = rows["in_intersection"].to_numpy()
            for k, t_cross in crossings(t, front, line, gap):
                lane = lane_ids[k - 1]
                if lane not in lanes or lane not in dirs or (front[k] - front[k - 1]) @ dirs[lane] <= 0:
                    continue
                red = red_segment_at(timeline, t_cross)
                if red is None or red[1] - red[0] < p["red_stable_sec"]:
                    continue
                since = t_cross - red[0]
                if since < p["min_after_red_onset_sec"]:
                    continue
                inside = np.flatnonzero(inter[k:]) + k
                if not len(inside):
                    continue  # never entered the intersection: a stop_line case
                left = np.flatnonzero(~inter[inside[0] :]) + inside[0]
                end = t[left[0]] if len(left) else t[-1]
                score = float(min(1.0, 0.5 + since / 2.0))
                segments.append(Segment(t_cross, float(end), LABEL, score, (tid,), {"stop_line": sl["id"]}))
    return segments
