"""jaywalking: A pedestrian on the carriageway outside a crossing.

Trigger: a pedestrian on the carriageway (outside crosswalks and pavements) who is, for >= min_sec,
clearly on it: at least kerb_buffer_m inside it and from every traffic island, and at least
crosswalk_buffer_m from every crosswalk. Feet at the kerb, beside the stripes or on a refuge are not
jaywalking (footprints are noisy by tens of cm). The buffers only qualify an event; its boundaries are
where the pedestrian steps onto and off the road (annotator convention), so they cost no IoU.
Not pedestrians (SPEC §12.5-12.6, §12.42): riders (box centre inside a two-wheeler box expanded by
rider_box_expand, or IoU > rider_iou with one), persons at least in_vehicle_frac inside a vehicle box,
and riding rows: faster than max_walk_speed_mps along the lane (|cos| > along_lane_cos), or faster
than max_run_speed_mps in any direction. A track matched to a two-wheeler in >= rider_track_frac of
its frames, or riding in most of them, is a rider throughout: the bike under a rider is detected less
reliably than the rider. A jaywalker running straight across the road is still a pedestrian.
Start: the pedestrian steps onto the road.
End: the pedestrian leaves the road; simultaneous pedestrians form one segment (union, postprocess).

Score is 1.0: every test here is geometric and binary, so precision is controlled by min_sec and the
exclusions, not by min_score.

Thresholds: configs/thresholds.yaml -> classes.jaywalking.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import (
    by_track,
    depth_in_polygon_m,
    dist_to_polygon_m,
    in_group,
    lane_world_dirs,
    max_gap,
    runs,
)
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "jaywalking"
REQUIRED_LAYERS: tuple[str, ...] = ("carriageway", "crosswalks")
_BOX = ["x1", "y1", "x2", "y2"]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Na, Nb) IoU of xyxy boxes."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _inside_frac(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Na, Nb) share of each box in `a` that lies inside each box in `b`."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-9)[:, None]


def excluded_persons(tt: pd.DataFrame, p: dict[str, Any]) -> np.ndarray:
    """Per row of `tt`: True for persons riding a two-wheeler or sitting inside a vehicle."""
    out = np.zeros(len(tt), dtype=bool)
    person = in_group(tt, "persons")
    rider_host = in_group(tt, "two_wheelers")
    vehicle = in_group(tt, "vehicles")
    boxes = tt[_BOX].to_numpy(dtype=np.float64)
    for _, idx in tt.groupby("frame", sort=False).indices.items():
        ppl = idx[person[idx]]
        if not len(ppl):
            continue
        pb = boxes[ppl]
        hit = np.zeros(len(ppl), dtype=bool)
        hosts = idx[rider_host[idx]]
        if len(hosts):
            hb = boxes[hosts].copy()
            w, h = hb[:, 2] - hb[:, 0], hb[:, 3] - hb[:, 1]
            e = p["rider_box_expand"]
            grown = np.stack([hb[:, 0] - e * w, hb[:, 1] - e * h, hb[:, 2] + e * w, hb[:, 3] + e * h], axis=1)
            cx, cy = (pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2
            centre_in = (cx[:, None] >= grown[None, :, 0]) & (cx[:, None] <= grown[None, :, 2])
            centre_in &= (cy[:, None] >= grown[None, :, 1]) & (cy[:, None] <= grown[None, :, 3])
            hit |= centre_in.any(axis=1) | (_iou(pb, hb) > p["rider_iou"]).any(axis=1)
        cars = idx[vehicle[idx]]
        if len(cars):
            hit |= (_inside_frac(pb, boxes[cars]) >= p["in_vehicle_frac"]).any(axis=1)
        out[ppl] = hit
    return out


def _clear_of_edges(foot: np.ndarray, scene: Scene, p: dict[str, Any]) -> np.ndarray:
    """Footprints well inside the carriageway: kerb_buffer_m from its edge and from every island,
    crosswalk_buffer_m from every crosswalk (a missing homography disables the buffers)."""
    ok = np.ones(len(foot), dtype=bool)
    if not len(foot) or not scene.has("homography"):
        return ok
    ok &= depth_in_polygon_m(scene, scene.layers["carriageway"], foot) >= p["kerb_buffer_m"]
    for island in scene.layers.get("islands") or []:
        ok &= dist_to_polygon_m(scene, island, foot) >= p["kerb_buffer_m"]
    for _, crosswalk in scene.polygons("crosswalks"):
        ok &= dist_to_polygon_m(scene, crosswalk, foot) >= p["crosswalk_buffer_m"]
    return ok


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    person = in_group(tt, "persons")
    excluded = excluded_persons(tt, p)
    riding = _riding(tt, scene, p)
    tid = tt["track_id"].to_numpy()
    per_track = pd.DataFrame({"tid": tid[person], "matched": excluded[person], "riding": riding[person]})
    by_tid = per_track.groupby("tid", sort=True)
    riders = set(by_tid["matched"].mean().loc[lambda f: f >= p["rider_track_frac"]].index)
    riders |= set(by_tid["riding"].mean().loc[lambda f: f > 0.5].index)
    walker = person & ~excluded & ~riding & ~np.isin(tid, list(riders))
    on_road = (tt["on_road"] & ~tt["in_crosswalk"] & ~tt["on_sidewalk"]).to_numpy()
    clear = np.zeros(len(tt), dtype=bool)
    clear[walker] = on_road[walker] & _clear_of_edges(tt[walker][["fx", "fy"]].to_numpy(np.float64), scene, p)
    persons = tt[walker].assign(on_road=on_road[walker], clear=clear[walker])
    gap = max_gap(ctx)
    segments = []
    for track, rows in by_track(persons):
        t = rows["t"].to_numpy()
        is_clear = rows["clear"].to_numpy()
        for a, b in runs(t, rows["on_road"].to_numpy(), gap):
            # qualifies if clearly on the road for min_sec; spans the whole stay on the road
            if any(t[d] - t[c] >= p["min_sec"] for c, d in runs(t[a : b + 1], is_clear[a : b + 1], gap)):
                segments.append(Segment(float(t[a]), float(t[b]), LABEL, 1.0, (track,)))
    return segments


def _riding(tt: pd.DataFrame, scene: Scene, p: dict[str, Any]) -> np.ndarray:
    """Per row: moving like a rider, fast along the lane or faster than anyone runs."""
    speed = np.nan_to_num(tt["speed"].to_numpy(dtype=np.float64), nan=0.0)
    vel = np.nan_to_num(tt[["vx", "vy"]].to_numpy(dtype=np.float64))
    lane_dir = lane_world_dirs(tt, scene) if scene.has("lanes") else np.full((len(tt), 2), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        along = np.abs((vel * lane_dir).sum(axis=1)) / speed > p["along_lane_cos"]
    return (speed > p["max_run_speed_mps"]) | ((speed > p["max_walk_speed_mps"]) & along)
