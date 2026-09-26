"""stop_line: A vehicle stops past the stop line on red without entering the intersection.

Trigger: signal RED and the vehicle stopped (speed < stop_speed_mps for >= stopped_sec) with its front
bumper more than past_line_m beyond the stop line and not inside the intersection.
Start: the vehicle stops (first stopped sample past the line on red).
End: the signal turns GREEN (annotator convention); earlier only if the vehicle moves again or backs
behind the line before that. Overlapping vehicles form one segment (union, in postprocess).

Details:
- Objects are followed by `obj_id` (persistence-merged), and a stopped vehicle may disappear for up to
  max_gap_sec (pedestrians walking in front of it) without ending the run: SPEC §12.36.
- A vehicle belongs to a stop line if the last lane it was seen in is one of the line's lanes (past the
  line the footprint may already be outside the lane polygon).
- "Past the line" is measured on the road plane, perpendicular to the line, positive in the lane
  direction, from the front bumper of the ground-plane vehicle model (SPEC §12.35). Only the
  lamp-based signal timeline counts (unknown -> no event). No green before the video ends -> duration.
Score = 1.0 (all tests are geometric).

Thresholds: configs/thresholds.yaml -> classes.stop_line.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, in_group, last_seen, past_line_m, runs, signal_states
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
    segments = []
    for sl in scene.layers.get("stop_lines") or []:
        sid = str(sl.get("signal", ""))
        timeline = ctx.signal_timeline.get(sid, [])
        lanes = [lane for lane in sl.get("lanes", []) if lane in dirs]
        if not timeline or not lanes:
            continue
        line = np.asarray(sl["line"], dtype=np.float64)
        greens = sorted(t0 for t0, _, state in timeline if state == "green")
        for oid, rows in by_track(v, key="obj_id"):
            last_lane = last_seen(rows["lane_id"].to_numpy())
            mine = np.isin(last_lane, lanes)
            if not mine.any():
                continue
            t = rows["t"].to_numpy()
            speed = rows["speed"].to_numpy(dtype=np.float64)
            front = rows[["front_x", "front_y"]].to_numpy(dtype=np.float64)
            past = past_line_m(scene, line, front, dirs[lanes[0]])
            red = signal_states(ctx, sid, t) == "red"
            outside = ~scene.point_in("intersection", front)
            bad = mine & red & (speed < p["stop_speed_mps"]) & (past > p["past_line_m"]) & outside
            for a, b in runs(t, bad, p["max_gap_sec"]):
                if t[b] - t[a] < p["stopped_sec"]:
                    continue
                green_t = next((g for g in greens if g > t[a]), ctx.meta.duration)
                left = np.flatnonzero(
                    (t > t[b])
                    & (t < green_t)
                    & ((speed > p["resume_speed_mps"]) | (past <= p["past_line_m"]))
                )
                end = float(t[left[0]]) if len(left) else float(green_t)
                segments.append(
                    Segment(float(t[a]), max(end, float(t[b])), LABEL, 1.0, (oid,), {"stop_line": sl["id"]})
                )
    return segments
