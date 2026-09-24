"""wrong_way: A vehicle moves against the traffic direction of its lane (incl. the oncoming lane).

Trigger: Speed > min_speed_mps and cos(velocity, lane direction) < max_cos_to_lane for >= min_sec.
Start: time the footprint entered the lane where it is wrong (walked back).
End: returns to a correct lane or leaves the frame.

Directions are compared on the road plane (metres): the lane's image direction is mapped through the
homography at the vehicle's footprint. Score = 0.5 + 0.5 * mean(-cos) over the wrong-way samples, so a
vehicle driving straight against the lane scores 1.

Thresholds: configs/thresholds.yaml -> classes.wrong_way.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, in_group, lane_world_dirs, max_gap, runs
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "wrong_way"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "lanes")


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    v = tt[in_group(tt, "vehicles", "two_wheelers")].reset_index(drop=True)
    if v.empty:
        return []
    lane_dir = lane_world_dirs(v, scene)
    vel = v[["vx", "vy"]].to_numpy(dtype=np.float64)
    speed = v["speed"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        v["cos"] = (vel * lane_dir).sum(axis=1) / speed
    v["wrong"] = v["kin_valid"] & (speed > p["min_speed_mps"]) & (v["cos"] < p["max_cos_to_lane"])

    gap = max_gap(ctx)
    segments = []
    for tid, rows in by_track(v):
        t = rows["t"].to_numpy()
        lanes = rows["lane_id"].to_numpy()
        for a, b in runs(t, rows["wrong"].to_numpy(), gap):
            if t[b] - t[a] < p["min_sec"]:
                continue
            # the stretch of this lane that contains the wrong-way run: from entering it to leaving it
            same_lane = lanes == lanes[a]
            first, last = next((s, e) for s, e in runs(t, same_lane, gap) if s <= a <= e)
            last = max(last, b)
            score = float(np.clip(0.5 + 0.5 * np.nanmean(-rows["cos"].to_numpy()[a : b + 1]), 0.0, 1.0))
            segments.append(
                Segment(float(t[first]), float(t[last]), LABEL, score, (tid,), {"lane": str(lanes[a])})
            )
    return segments
