"""illegal_u_turn: A U-turn where road markings or signs prohibit it.

Trigger: Cumulative heading change >= min_heading_change_deg within window_sec, speed >
min_start_speed_mps at the start, and the turn starts inside a no-U-turn zone (or the scene sets
u_turn_prohibited_everywhere). Either of those two layers is enough, so detect() checks them itself.
Start: yaw rate first > start_yaw_rate_dps.
End: heading stable within end_heading_tol_deg for end_stable_sec.

Details:
- Headings come from add_kinematics (held while the vehicle is slow), unwrapped, so a turn that
  pauses for oncoming traffic still accumulates.
- Candidate starts are the samples where |yaw rate| rises above start_yaw_rate_dps; the change is
  measured from the heading there. After an event the search resumes at its end.
Score = min(1, heading change / 180 deg).

Thresholds: configs/thresholds.yaml -> classes.illegal_u_turn.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, in_group, stable_from
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "illegal_u_turn"
REQUIRED_LAYERS: tuple[str, ...] = ("homography",)


def u_turns(rows: pd.DataFrame, p: dict[str, Any]) -> list[tuple[int, int, int, float]]:
    """(start, reached, end, change_deg) index triples of U-turns in one track (any location)."""
    t = rows["t"].to_numpy()
    heading = rows["heading_deg"].to_numpy(dtype=np.float64)
    yaw = np.abs(rows["yaw_rate"].to_numpy(dtype=np.float64))
    speed = rows["speed"].to_numpy(dtype=np.float64)
    known = np.isfinite(heading)
    unwrapped = np.full(len(t), np.nan)
    unwrapped[known] = np.degrees(np.unwrap(np.radians(heading[known])))
    turning = yaw > p["start_yaw_rate_dps"]
    rising = np.flatnonzero(turning & ~np.r_[False, turning[:-1]])

    out = []
    resume = 0
    for s in rising:
        if s < resume or not known[s] or speed[s] <= p["min_start_speed_mps"]:
            continue
        window = np.flatnonzero((t >= t[s]) & (t <= t[s] + p["window_sec"]) & known)
        change = np.abs(unwrapped[window] - unwrapped[s])
        hit = np.flatnonzero(change >= p["min_heading_change_deg"])
        if not len(hit):
            continue
        reached = int(window[hit[0]])
        end = stable_from(t, heading, reached, p["end_heading_tol_deg"], p["end_stable_sec"])
        end = len(t) - 1 if end is None else end
        out.append((int(s), reached, end, float(change.max())))
        resume = end + 1
    return out


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    everywhere = bool(scene.layers.get("u_turn_prohibited_everywhere"))
    if tt.empty or not (everywhere or scene.has("no_u_turn_zones")):
        return []
    p = cfg["params"]
    v = tt[in_group(tt, "vehicles", "two_wheelers") & tt["kin_valid"].to_numpy()]
    segments = []
    for tid, rows in by_track(v):
        zone = rows["in_no_uturn"].to_numpy()
        t = rows["t"].to_numpy()
        for s, _, e, change in u_turns(rows, p):
            if everywhere or zone[s]:
                segments.append(Segment(float(t[s]), float(t[e]), LABEL, min(1.0, change / 180.0), (tid,)))
    return segments
