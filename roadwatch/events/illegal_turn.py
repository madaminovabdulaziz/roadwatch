"""illegal_turn: A turn from the wrong lane or in a prohibited direction.

Trigger: The movement (entry lane at the stop line -> exit polygon) is not in the entry lane's
allowed_exits. U-turns are left to illegal_u_turn.
Start: the vehicle leaves its entry lane / yaw rate > start_yaw_rate_dps.
End: enters the exit polygon with heading stable within end_heading_tol_deg for end_stable_sec.

Details:
- The entry lane is the lane the front point was in when it crossed a stop line in the lane
  direction. Lanes without `allowed_exits` are never judged (the rule is unknown, not guessed).
- The exit is the first exit polygon the footprint reaches after the crossing.
- Start = the earlier of: first sample after the crossing in another lane, first sample after the
  crossing with |yaw rate| > start_yaw_rate_dps (lane polygons end at the stop line, so merely
  entering the intersection is not "leaving the lane").
- A heading change of classes.illegal_u_turn min_heading_change_deg or more is a U-turn: skipped.
Score = 1.0 (the movement is either allowed or not).

Thresholds: configs/thresholds.yaml -> classes.illegal_turn.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.config import class_cfg
from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, crossings, in_group, max_gap, stable_from
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "illegal_turn"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "lanes", "exits", "stop_lines")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    u_turn_deg = class_cfg("illegal_u_turn")["params"]["min_heading_change_deg"]
    allowed = {
        str(lane["id"]): set(lane["allowed_exits"])
        for lane in scene.layers["lanes"]
        if lane.get("allowed_exits")
    }
    dirs = scene.lane_directions()
    v = tt[in_group(tt, "vehicles", "two_wheelers")]
    gap = max_gap(ctx)
    segments = []
    for tid, rows in by_track(v):
        t = rows["t"].to_numpy()
        front = rows[["front_x", "front_y"]].to_numpy(dtype=np.float64)
        lanes = rows["lane_id"].to_numpy()
        exits = scene.region_of("exits", rows[["fx", "fy"]].to_numpy(dtype=np.float64))
        heading = rows["heading_deg"].to_numpy(dtype=np.float64)
        yaw = np.abs(rows["yaw_rate"].to_numpy(dtype=np.float64))
        for sl in scene.layers.get("stop_lines") or []:
            for k, _ in crossings(t, front, np.asarray(sl["line"], dtype=np.float64), gap):
                entry = lanes[k - 1]
                if entry not in allowed or entry not in dirs or (front[k] - front[k - 1]) @ dirs[entry] <= 0:
                    continue
                reached = np.flatnonzero(exits[k:] != "")
                if not len(reached):
                    continue
                first_exit = k + int(reached[0])
                if exits[first_exit] in allowed[entry]:
                    continue
                known = heading[k - 1 : first_exit + 1]
                known = known[np.isfinite(known)]
                if len(known) and abs((known[-1] - known[0] + 180) % 360 - 180) >= u_turn_deg:
                    continue  # a U-turn: illegal_u_turn's business
                # lane polygons end at the stop line, so only a move into another lane counts as leaving
                left_lane = np.flatnonzero((lanes[k:] != entry) & (lanes[k:] != ""))
                turning = np.flatnonzero(yaw[k:] > p["start_yaw_rate_dps"])
                start = k + min([int(x[0]) for x in (left_lane, turning) if len(x)], default=0)
                end = stable_from(t, heading, first_exit, p["end_heading_tol_deg"], p["end_stable_sec"])
                end = len(t) - 1 if end is None else end
                segments.append(
                    Segment(
                        float(t[start]),
                        float(t[end]),
                        LABEL,
                        1.0,
                        (tid,),
                        {"from": entry, "to": exits[first_exit]},
                    )
                )
    return segments
