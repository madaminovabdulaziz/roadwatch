"""solid_line_crossing: A lane change or manoeuvre across a solid marking.

Trigger: Both bottom box corners start on one side of a solid polyline and end on the other, the
lateral move is >= min_lateral_m, and it happens outside the intersection.
Start: the first bottom corner crosses.
End: the second bottom corner crosses.

Details:
- Crossings are found per polyline segment for each bottom corner, with interpolated times. A corner
  that crosses back before the other corner follows cancels its crossing (a wheel touching the line).
- The lateral move is the footprint's displacement across the crossed segment on the road plane, from
  the sample before the first crossing to the sample after the second.
Score = 1.0 (all tests are geometric).

Thresholds: configs/thresholds.yaml -> classes.solid_line_crossing.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, crossings, in_group, max_gap
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "solid_line_crossing"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "solid_lines")


def _side(seg: np.ndarray, pt: np.ndarray) -> float:
    (ax, ay), (bx, by) = seg
    return float(np.sign((bx - ax) * (pt[1] - ay) - (by - ay) * (pt[0] - ax)))


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    v = tt[in_group(tt, "vehicles", "two_wheelers")]
    gap = max_gap(ctx)
    segments = []
    for line in scene.layers.get("solid_lines") or []:
        poly = np.asarray(line["polyline"], dtype=np.float64)
        for tid, rows in by_track(v):
            t = rows["t"].to_numpy()
            corners = [
                rows[["x1", "y2"]].to_numpy(dtype=np.float64),
                rows[["x2", "y2"]].to_numpy(dtype=np.float64),
            ]
            foot = rows[["fx", "fy"]].to_numpy(dtype=np.float64)
            inter = rows["in_intersection"].to_numpy()
            for s in range(len(poly) - 1):
                seg = poly[s : s + 2]
                events = sorted(
                    (tc, c, k, _side(seg, corners[c][k - 1]))
                    for c in (0, 1)
                    for k, tc in crossings(t, corners[c], seg, gap)
                )
                pending: dict[int, tuple[float, int, float]] = {}  # corner -> (time, step, side it left)
                for tc, c, k, side in events:
                    other = 1 - c
                    if c in pending and pending[c][2] != side:
                        del pending[c]  # crossed back: it was only touching the line
                        continue
                    if other in pending and pending[other][2] == side:
                        t0, k0, _ = pending.pop(other)
                        before, after = foot[k0 - 1], foot[min(k, len(foot) - 1)]
                        a, b = scene.to_world(seg)
                        normal = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
                        moved = abs(
                            float((scene.to_world(after[None])[0] - scene.to_world(before[None])[0]) @ normal)
                        )
                        if moved >= p["min_lateral_m"] and not (inter[k0] or inter[k]):
                            segments.append(Segment(t0, tc, LABEL, 1.0, (tid,), {"line": line["id"]}))
                    else:
                        pending[c] = (tc, k, side)
    return segments
