"""solid_line_crossing: A lane change or manoeuvre across a solid marking.

Trigger: the vehicle's ground rectangle goes from fully on one side of a solid polyline segment to fully
on the other, moving at least min_lateral_m across it, outside the intersection. A corner that touches
the line and returns does not count.
Start: the first corner (wheel) crosses the line.
End: the last corner has crossed (the vehicle is fully in the new lane).

Details:
- Corners come from the ground-plane vehicle model (features.vehicle_corners, SPEC §12.35), not from
  the image box: in this oblique view the box is much wider than the car's ground footprint.
- A corner counts only while its projection falls within the segment's extent (plus half a car
  length), so a vehicle passing the end of a solid line is not crossing it.
- Boundaries at sample transitions are timed at the midpoint of the two samples.
Score = 1.0 (all tests are geometric).

Thresholds: configs/thresholds.yaml -> classes.solid_line_crossing.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, in_group, max_gap, midpoint_before, midpoint_gap
from roadwatch.features import vehicle_corners
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "solid_line_crossing"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "solid_lines")


def _sides(seg_world: np.ndarray, corners_world: np.ndarray, margin: float) -> np.ndarray:
    """(N, 4) side of each corner (+1 / -1) relative to the segment's line; 0 where off the segment."""
    a, b = seg_world
    d = b - a
    length = float(np.hypot(*d))
    u = d / length
    rel = corners_world - a
    along = rel @ u
    cross = rel[..., 0] * u[1] - rel[..., 1] * u[0]
    on_segment = (along >= -margin) & (along <= length + margin)
    return np.where(on_segment, np.sign(cross), 0.0)


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    v = tt[in_group(tt, "vehicles", "two_wheelers") & tt["kin_valid"].to_numpy()]
    gap, mid = max_gap(ctx), midpoint_gap(ctx)
    segments = []
    for line in scene.layers.get("solid_lines") or []:
        poly = scene.to_world(np.asarray(line["polyline"], dtype=np.float64))
        for tid, rows in by_track(v):
            t = rows["t"].to_numpy()
            corners = scene.to_world(vehicle_corners(rows, scene).reshape(-1, 2)).reshape(len(rows), 4, 2)
            centre = rows[["cX", "cY"]].to_numpy(dtype=np.float64)
            inter = rows["in_intersection"].to_numpy()
            margin = float(np.nanmax(rows["veh_len"].to_numpy())) / 2
            for s in range(len(poly) - 1):
                seg = poly[s : s + 2]
                sides = _sides(seg, corners, margin)
                full = np.where((sides == 1).all(axis=1), 1, np.where((sides == -1).all(axis=1), -1, 0))
                event = _crossing(t, sides, full, centre, seg, inter, p, gap, mid)
                if event is not None:
                    segments.append(Segment(*event, LABEL, 1.0, (tid,), {"line": line["id"]}))
    return segments


def _crossing(
    t: np.ndarray,
    sides: np.ndarray,
    full: np.ndarray,
    centre: np.ndarray,
    seg: np.ndarray,
    inter: np.ndarray,
    p: dict[str, Any],
    gap: float,
    mid: float,
) -> tuple[float, float] | None:
    """(start, end) of the first full side change of one track across one segment, else None."""
    settled = np.flatnonzero(full != 0)
    for i, j in zip(settled, settled[1:], strict=False):
        if full[i] == full[j] or (t[i + 1 : j + 1] - t[i:j] > gap).any():
            continue  # same side (a touch) or a gap in the track
        if inter[i : j + 1].any():
            continue  # manoeuvres inside the intersection cross no lane marking
        d = seg[1] - seg[0]
        normal = np.array([-d[1], d[0]]) / float(np.hypot(*d))
        if abs(float((centre[j] - centre[i]) @ normal)) < p["min_lateral_m"]:
            continue
        # first sample after i where a corner left the starting side; the end is the settled sample j
        started = i + 1 + int(np.flatnonzero((sides[i + 1 : j + 1] != full[i]).any(axis=1))[0])
        return midpoint_before(t, started, mid), midpoint_before(t, j, mid)
    return None
