"""failure_to_yield: A vehicle drives through a crossing while a pedestrian is on it or stepping onto it.

Trigger: Vehicle speed > min_vehicle_speed_mps with its footprint in a crosswalk polygon while at
least one person footprint is in the same crosswalk, or within ped_edge_dist_m of its edge and
moving toward it.
Start: the vehicle enters the crosswalk.
End: the vehicle leaves the crosswalk.

Details:
- Persons riding a two-wheeler or sitting in a vehicle are not pedestrians (jaywalking's exclusions).
- "Moving toward it": the person's velocity points at the crosswalk (positive component toward the
  nearest boundary point) and is faster than a standing sway.
- Start/end are the first and last samples of the vehicle's stay in that crosswalk.
Score = 1.0 if a pedestrian was inside the crosswalk at the conflict, 0.7 if only approaching it.

Thresholds: configs/thresholds.yaml -> classes.failure_to_yield.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.config import class_cfg
from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, dist_to_polygon_m, in_group, max_gap, runs
from roadwatch.events.jaywalking import excluded_persons
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "failure_to_yield"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "crosswalks")
_WALKING_MPS = 0.3  # slower than this a pedestrian is standing, not stepping onto the crossing


def _pedestrian_presence(tt: pd.DataFrame, scene: Scene, p: dict[str, Any]) -> dict[str, dict[int, float]]:
    """Per crosswalk id: {frame: 1.0 if a pedestrian is inside, 0.7 if one is approaching the edge}."""
    walker = in_group(tt, "persons") & ~excluded_persons(tt, class_cfg("jaywalking")["params"])
    ped = tt[walker]
    out: dict[str, dict[int, float]] = {}
    if ped.empty:
        return out
    foot = ped[["fx", "fy"]].to_numpy(dtype=np.float64)
    vel = ped[["vx", "vy"]].to_numpy(dtype=np.float64)
    frames = ped["frame"].to_numpy()
    world = scene.to_world(foot)
    for cw_id, poly in scene.polygons("crosswalks"):
        inside = ped["crosswalk_id"].to_numpy() == cw_id
        dist = dist_to_polygon_m(scene, poly, foot)
        centre = scene.to_world(poly).mean(axis=0)
        toward = ((centre - world) * vel).sum(axis=1) > 0
        moving = np.hypot(vel[:, 0], vel[:, 1]) > _WALKING_MPS
        approaching = ~inside & (dist <= p["ped_edge_dist_m"]) & toward & moving
        presence = {int(f): 0.7 for f in frames[approaching]}
        presence.update({int(f): 1.0 for f in frames[inside]})
        out[cw_id] = presence
    return out


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    presence = _pedestrian_presence(tt, scene, p)
    if not presence:
        return []
    v = tt[in_group(tt, "vehicles", "two_wheelers")]
    gap = max_gap(ctx)
    segments = []
    for tid, rows in by_track(v):
        t = rows["t"].to_numpy()
        cw = rows["crosswalk_id"].to_numpy()
        fast = rows["speed"].to_numpy() > p["min_vehicle_speed_mps"]
        frames = rows["frame"].to_numpy()
        for cw_id, seen in presence.items():
            for a, b in runs(t, cw == cw_id, gap):
                conflict = [seen.get(int(frames[i]), 0.0) for i in range(a, b + 1) if fast[i]]
                score = max(conflict, default=0.0)
                if score > 0:
                    segments.append(
                        Segment(float(t[a]), float(t[b]), LABEL, score, (tid,), {"crosswalk": cw_id})
                    )
    return segments
