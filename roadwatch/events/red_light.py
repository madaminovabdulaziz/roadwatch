"""red_light: A vehicle crosses its stop line while its signal is red.

Trigger: the signal of the vehicle's stop line is RED in a phase lasting >= red_stable_sec, the
vehicle's front bumper crosses the stop line in the lane direction at least min_after_red_onset_sec
after red onset, and then enters the intersection within max_entry_sec without stopping.
Start: the front bumper crosses the stop line (interpolated between processed frames).
End: the rear bumper leaves the intersection (the vehicle is clear of it), or the track ends (it left
the frame). Sample transitions are timed at the midpoint of the two samples.

Details:
- Front and rear bumpers come from the ground-plane vehicle model (features, SPEC §12.35).
- Only the lamp-based signal timeline counts; without it (state unknown) nothing fires.
- The continuity test separates red_light from stop_line: a vehicle that creeps past the line and stops
  (below stop_speed_mps for stop_hold_sec after crossing; maybe it goes on at green) is stop_line, not
  red_light (SPEC §12.36). Slow samples before the crossing do not count: pulling away from the head of
  the queue while the light is still red is the most common red-light violation.
- A red phase shorter than red_stable_sec is an occlusion of the lamps (a red bus passing), not a red
  light: real red phases last well over 10 s.
Score = min(1, 0.5 + (crossing - red onset) / 2 s): crossings right after the change are less certain.

Thresholds: configs/thresholds.yaml -> classes.red_light.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, crossings, in_group, max_gap, midpoint_after, runs
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "red_light"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "lanes", "stop_lines", "signals", "intersection")


def red_segment_at(segments: list[tuple[float, float, str]], t: float) -> tuple[float, float] | None:
    """(onset, end) of the red phase containing t, or None."""
    for t0, t1, state in segments:
        if state == "red" and t0 <= t <= t1:
            return t0, t1
    return None


def _stopped(t: np.ndarray, speed: np.ndarray, p: dict[str, Any], gap: float) -> bool:
    """Whether the vehicle stood still (below stop_speed_mps for stop_hold_sec) after crossing the line.

    Only samples after the crossing count: a car that waited at the line and pulls away on red crosses
    it slowly, and that is the most common red-light violation, not a stop past the line.
    """
    slow = speed < p["stop_speed_mps"]
    return any(t[b] - t[a] >= p["stop_hold_sec"] for a, b in runs(t, slow, gap))


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    dirs = scene.lane_directions()
    v = tt[in_group(tt, "vehicles", "two_wheelers")]
    gap = max_gap(ctx)
    segments = []
    for sl in scene.layers.get("stop_lines") or []:
        timeline = ctx.signal_timeline.get(str(sl.get("signal", "")), [])
        if not timeline:
            continue
        line = np.asarray(sl["line"], dtype=np.float64)
        lanes = set(sl.get("lanes", []))
        for tid, rows in by_track(v):
            t = rows["t"].to_numpy()
            front = rows[["front_x", "front_y"]].to_numpy(dtype=np.float64)
            rear = rows[["rear_x", "rear_y"]].to_numpy(dtype=np.float64)
            lane_ids = rows["lane_id"].to_numpy()
            speed = rows["speed"].to_numpy(dtype=np.float64)
            front_in = scene.point_in("intersection", front)
            occupied = front_in | scene.point_in("intersection", rear)
            for k, t_cross in crossings(t, front, line, gap):
                lane = lane_ids[k - 1]
                if lane not in lanes or lane not in dirs or (front[k] - front[k - 1]) @ dirs[lane] <= 0:
                    continue
                red = red_segment_at(timeline, t_cross)
                if red is None or red[1] - red[0] < p["red_stable_sec"]:
                    continue
                since = t_cross - red[0]
                if since < p["min_after_red_onset_sec"]:
                    continue
                entered = np.flatnonzero(front_in[k - 1 :] & (t[k - 1 :] <= t_cross + p["max_entry_sec"]))
                if not len(entered):
                    continue  # crept past the line without entering: stop_line's business
                entry = k - 1 + int(entered[0])
                if _stopped(t[k : entry + 1], speed[k : entry + 1], p, gap):
                    continue  # stopped between the line and the intersection: stop_line's business
                end = midpoint_after(t, occupied, entry, gap)
                score = float(min(1.0, 0.5 + since / 2.0))
                segments.append(Segment(t_cross, end, LABEL, score, (tid,), {"stop_line": sl["id"]}))
    return segments
