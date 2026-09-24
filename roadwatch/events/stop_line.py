"""stop_line: A vehicle stops past the stop line on red without entering the intersection.

Trigger: Signal RED and the vehicle stopped (speed < stop_speed_mps for >= stopped_sec) with its
front point past the stop line by more than past_line_m, footprint not in the intersection.
Start: the time it stopped.
End: the signal turns GREEN; overlapping vehicles form one segment (union).

Details:
- A vehicle belongs to a stop line if the last lane it was seen in is one of the line's lanes (past
  the line the footprint may already be outside the lane polygon).
- "Past the line" is measured on the road plane, perpendicular to the line, positive in the lane
  direction. Only the lamp-based signal timeline counts (unknown -> no event).
- The end is the first green after the stop; if the vehicle is gone before that, its last sighting.
Score = 1.0 (all tests are geometric; min_score filters nothing).

Thresholds: configs/thresholds.yaml -> classes.stop_line.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, in_group, max_gap, past_line_m, runs, signal_states
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "stop_line"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "stop_lines", "signals", "intersection")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    dirs = scene.lane_directions()
    v = tt[in_group(tt, "vehicles", "two_wheelers")]
    gap = max_gap(ctx)
    segments = []
    for sl in scene.layers.get("stop_lines") or []:
        sid = str(sl.get("signal", ""))
        timeline = ctx.signal_timeline.get(sid, [])
        lanes = [lane for lane in sl.get("lanes", []) if lane in dirs]
        if not timeline or not lanes:
            continue
        line = np.asarray(sl["line"], dtype=np.float64)
        for tid, rows in by_track(v):
            last_lane = rows["lane_id"].replace("", np.nan).ffill().fillna("").to_numpy()
            mine = np.isin(last_lane, lanes)
            if not mine.any():
                continue
            t = rows["t"].to_numpy()
            front = rows[["front_x", "front_y"]].to_numpy(dtype=np.float64)
            past = past_line_m(scene, line, front, dirs[lanes[0]])
            red = signal_states(ctx, sid, t) == "red"
            bad = mine & red & (rows["speed"].to_numpy() < p["stop_speed_mps"]) & (past > p["past_line_m"])
            bad &= ~rows["in_intersection"].to_numpy()
            for a, b in runs(t, bad, gap):
                if t[b] - t[a] < p["stopped_sec"]:
                    continue
                green = [t0 for t0, _, state in timeline if state == "green" and t0 > t[a]]
                end = min(green[0], t[-1]) if green else t[-1]
                segments.append(
                    Segment(float(t[a]), float(max(end, t[b])), LABEL, 1.0, (tid,), {"stop_line": sl["id"]})
                )
    return segments
