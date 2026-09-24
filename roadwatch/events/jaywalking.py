"""jaywalking: A pedestrian on the carriageway outside a crossing.

Trigger: Person footprint on the carriageway, outside crosswalks and sidewalks, for >= min_sec.
Riders are excluded (box centre inside a two-wheeler box expanded by rider_box_expand, or IoU >
rider_iou with one), and so are persons at least in_vehicle_frac inside a vehicle box (SPEC
§12.5-12.6).
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
from roadwatch.events.common import by_track, in_group, max_gap, runs
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


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    excluded = excluded_persons(tt, p)
    walker = in_group(tt, "persons") & ~excluded
    on_road = (tt["on_road"] & ~tt["in_crosswalk"] & ~tt["on_sidewalk"]).to_numpy()
    persons = tt[walker].assign(jay=on_road[walker])
    gap = max_gap(ctx)
    segments = []
    for tid, rows in by_track(persons):
        t = rows["t"].to_numpy()
        for a, b in runs(t, rows["jay"].to_numpy(), gap):
            if t[b] - t[a] >= p["min_sec"]:
                segments.append(Segment(float(t[a]), float(t[b]), LABEL, 1.0, (tid,)))
    return segments
