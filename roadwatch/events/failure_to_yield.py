"""failure_to_yield: A vehicle drives through a crossing while a pedestrian is on it or stepping onto it.

Trigger: the vehicle (speed > min_vehicle_speed_mps) overlaps a crosswalk while a pedestrian in its
path is on that crosswalk, or within ped_edge_dist_m of its edge and moving toward it. "In its path":
within conflict_lateral_m of the vehicle's axis and between its rear bumper and conflict_ahead_m ahead
of its front bumper (SPEC §12.36): a pedestrian at the far end of a 20 m crossing is not a conflict.
Start: the vehicle enters the crosswalk (its front bumper reaches it).
End: the vehicle leaves the crosswalk (its rear bumper clears it).

Details:
- The vehicle's extent comes from the ground-plane vehicle model (front/rear bumpers, SPEC §12.35);
  "overlaps" = any of 5 points from front to rear bumper inside the polygon.
- Boundaries at sample transitions are timed at the midpoint of the two samples.
- Persons riding a two-wheeler or sitting in a vehicle are not pedestrians (jaywalking's exclusions).
- "Moving toward it": the person's velocity points at the crosswalk and is faster than standing sway.
Score = 1.0 if a pedestrian in the path was inside the crosswalk, 0.7 if one was only stepping onto it.

Thresholds: configs/thresholds.yaml -> classes.failure_to_yield.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.config import class_cfg
from roadwatch.events.base import VideoContext
from roadwatch.events.common import (
    along_vehicle,
    by_track,
    dist_to_polygon_m,
    in_group,
    max_gap,
    midpoint_after,
    midpoint_before,
    midpoint_gap,
    runs,
)
from roadwatch.events.jaywalking import excluded_persons
from roadwatch.scene.scene import Scene, points_in_polygon
from roadwatch.types import Segment

LABEL = "failure_to_yield"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "crosswalks")
_WALKING_MPS = 0.3  # slower than this a pedestrian is standing, not stepping onto the crossing


def _pedestrians(tt: pd.DataFrame, scene: Scene, p: dict[str, Any]) -> dict[str, dict[int, np.ndarray]]:
    """Per crosswalk id: {frame: (M, 3) array of pedestrian X, Y (metres) and score (1.0 in, 0.7 near)}."""
    walker = in_group(tt, "persons") & ~excluded_persons(tt, class_cfg("jaywalking")["params"])
    ped = tt[walker]
    out: dict[str, dict[int, np.ndarray]] = {}
    if ped.empty:
        return out
    foot = ped[["fx", "fy"]].to_numpy(dtype=np.float64)
    world = ped[["X", "Y"]].to_numpy(dtype=np.float64)
    vel = ped[["vx", "vy"]].to_numpy(dtype=np.float64)
    frames = ped["frame"].to_numpy()
    for cw_id, poly in scene.polygons("crosswalks"):
        inside = ped["crosswalk_id"].to_numpy() == cw_id
        centre = scene.to_world(poly).mean(axis=0)
        toward = ((centre - world) * vel).sum(axis=1) > 0
        moving = np.hypot(vel[:, 0], vel[:, 1]) > _WALKING_MPS
        near = dist_to_polygon_m(scene, poly, foot) <= p["ped_edge_dist_m"]
        score = np.where(inside, 1.0, np.where(near & toward & moving, 0.7, 0.0))
        keep = score > 0
        by_frame: dict[int, list[np.ndarray]] = {}
        for f, x, y, s in zip(frames[keep], world[keep, 0], world[keep, 1], score[keep], strict=True):
            by_frame.setdefault(int(f), []).append(np.array([x, y, s]))
        out[cw_id] = {f: np.stack(rows) for f, rows in by_frame.items()}
    return out


def _conflict(rows: pd.DataFrame, i: int, peds: np.ndarray, scene: Scene, p: dict[str, Any]) -> float:
    """Best pedestrian score among those in the vehicle's path at row i (0 if none)."""
    front = scene.to_world(rows[["front_x", "front_y"]].to_numpy(dtype=np.float64)[i : i + 1])[0]
    rear = scene.to_world(rows[["rear_x", "rear_y"]].to_numpy(dtype=np.float64)[i : i + 1])[0]
    axis = front - rear
    length = float(np.hypot(*axis))
    if length < 1e-6:  # no vehicle model: fall back to the motion direction
        axis = rows[["vx", "vy"]].to_numpy(dtype=np.float64)[i]
        length = float(np.hypot(*axis))
        if length < 1e-6:
            return 0.0
        rear = front
    h = axis / length
    rel = peds[:, :2] - rear
    along = rel @ h
    lateral = np.abs(rel @ np.array([-h[1], h[0]]))
    in_path = (
        (lateral <= p["conflict_lateral_m"])
        & (along >= 0)
        & (along <= float(np.hypot(*(front - rear))) + p["conflict_ahead_m"])
    )
    return float(peds[in_path, 2].max()) if in_path.any() else 0.0


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    peds_by_cw = _pedestrians(tt, scene, p)
    if not peds_by_cw:
        return []
    v = tt[in_group(tt, "vehicles", "two_wheelers")]
    gap, mid = max_gap(ctx), midpoint_gap(ctx)
    segments = []
    for tid, rows in by_track(v):
        t = rows["t"].to_numpy()
        fast = rows["speed"].to_numpy(dtype=np.float64) > p["min_vehicle_speed_mps"]
        frames = rows["frame"].to_numpy()
        body = along_vehicle(rows)
        for cw_id, poly in scene.polygons("crosswalks"):
            seen = peds_by_cw.get(cw_id)
            if not seen:
                continue
            overlap = points_in_polygon(body.reshape(-1, 2), poly).reshape(len(rows), -1).any(axis=1)
            for a, b in runs(t, overlap, gap):
                score = max(
                    (
                        _conflict(rows, i, seen[int(frames[i])], scene, p)
                        for i in range(a, b + 1)
                        if fast[i] and int(frames[i]) in seen
                    ),
                    default=0.0,
                )
                if score > 0:
                    start, end = midpoint_before(t, a, mid), midpoint_after(t, overlap, a, mid)
                    segments.append(Segment(start, end, LABEL, score, (tid,), {"crosswalk": cw_id}))
    return segments
