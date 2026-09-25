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
from roadwatch.events.common import by_track, depth_in_polygon_m, in_group, lane_world_dirs, max_gap, runs
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "wrong_way"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "lanes")


def _depth_in_own_lane(v: pd.DataFrame, scene: Scene) -> np.ndarray:
    """Metres each footprint is inside the lane it is in (0 outside every lane).

    A car driving correctly next to the centre line can project a few tens of cm into the opposing lane
    in this oblique view; only being clearly inside it (lateral_margin_m) is wrong-way (SPEC §12.42).
    """
    depth = np.zeros(len(v))
    foot = v[["fx", "fy"]].to_numpy(dtype=np.float64)
    lanes = v["lane_id"].to_numpy()
    for lane in scene.layers.get("lanes") or []:
        rows = lanes == str(lane["id"])
        if rows.any():
            depth[rows] = depth_in_polygon_m(scene, lane["polygon"], foot[rows])
    return depth


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
    v["wrong"] = (
        v["kin_valid"]
        & (speed > p["min_speed_mps"])
        & (v["cos"] < p["max_cos_to_lane"])
        & (_depth_in_own_lane(v, scene) >= p["lateral_margin_m"])
    )

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
