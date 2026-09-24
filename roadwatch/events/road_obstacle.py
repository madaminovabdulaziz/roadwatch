"""road_obstacle: Debris, an animal or a fallen object on the carriageway.

Trigger: (a) A COCO animal/bag class on the carriageway for >= coco_min_sec, or (b) a static
foreground blob (background = running median over background_window_sec) on the carriageway, area >=
min_area_m2, not overlapping any vehicle/person box expanded by box_expand, static for >=
static_sec.
Start: first appearance.
End: gone for >= gone_sec.

Details:
- (a) runs on the tracks of the `others` tracker group. A bag or animal box that overlaps a person,
  two-wheeler or vehicle box expanded by box_expand is carried by (or next to) someone: not an
  obstacle (COCO `backpack` fires on bags worn by pedestrians, SPEC §12.24).
- Sightings of one object are joined across gaps shorter than gone_sec, so a brief occlusion does not
  split it; the event ends at the last sighting.
- (b) needs the background model, which reads frames; it is not wired yet (ctx.background is None),
  so only (a) runs until then.
Score = 1.0 for an animal, 0.7 for a bag (bags are more often false detections).

Thresholds: configs/thresholds.yaml -> classes.road_obstacle.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, in_group, runs
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "road_obstacle"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "carriageway")
BAGS = {"backpack", "suitcase"}
_BOX = ["x1", "y1", "x2", "y2"]


def carried(tt: pd.DataFrame, expand: float) -> np.ndarray:
    """Per row: an `others` box overlapping a person/two-wheeler/vehicle box grown by `expand`."""
    out = np.zeros(len(tt), dtype=bool)
    obj = in_group(tt, "others")
    host = in_group(tt, "persons", "two_wheelers", "vehicles")
    boxes = tt[_BOX].to_numpy(dtype=np.float64)
    for _, idx in tt.groupby("frame", sort=False).indices.items():
        o, h = idx[obj[idx]], idx[host[idx]]
        if not len(o) or not len(h):
            continue
        hb = boxes[h]
        w, hh = hb[:, 2] - hb[:, 0], hb[:, 3] - hb[:, 1]
        grown = np.stack(
            [hb[:, 0] - expand * w, hb[:, 1] - expand * hh, hb[:, 2] + expand * w, hb[:, 3] + expand * hh],
            axis=1,
        )
        ob = boxes[o]
        overlap = (
            np.minimum(ob[:, None, 2], grown[None, :, 2]) > np.maximum(ob[:, None, 0], grown[None, :, 0])
        ) & (np.minimum(ob[:, None, 3], grown[None, :, 3]) > np.maximum(ob[:, None, 1], grown[None, :, 1]))
        out[o] = overlap.any(axis=1)
    return out


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    lone = in_group(tt, "others") & ~carried(tt, p["box_expand"]) & tt["on_road"].to_numpy()
    segments = []
    for tid, rows in by_track(tt[lone]):
        t = rows["t"].to_numpy()
        for a, b in runs(t, np.ones(len(t), dtype=bool), p["gone_sec"]):
            if t[b] - t[a] >= p["coco_min_sec"]:
                cls = str(rows["cls"].iloc[a])
                score = 0.7 if cls in BAGS else 1.0
                segments.append(Segment(float(t[a]), float(t[b]), LABEL, score, (tid,), {"cls": cls}))
    return segments
