"""Helpers shared by the event rules: class groups, time runs, lane directions, signal lookups.

Rules receive the TrackTable after `features.add_kinematics` (one row per track per processed frame).
- `in_group(tt, *groups)`: rows whose class belongs to tracker groups from thresholds.yaml.
- `max_gap(ctx)`: the largest time step that still counts as continuous (1.5 processed-frame steps),
  so a single lost detection does not split a run.
- `runs(t, mask, gap)`: (first, last) index pairs of consecutive True samples, split where the mask is
  False or the time step exceeds `gap`.
- `by_track(tt, key)`: (id, rows sorted by t) per track (or per `obj_id` for persistence-merged objects).
- `lane_world_dirs(tt, scene)`: each row's lane direction as a unit vector on the road plane (metres),
  NaN outside lanes or for lanes without a direction.
- `lane_signal(scene)` / `signal_states(ctx, sid, t)`: the signal id of each lane and its state at times
  `t` ("unknown" without a timeline).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from roadwatch.config import load_thresholds
from roadwatch.events.base import VideoContext
from roadwatch.scene.light import UNKNOWN, state_at
from roadwatch.scene.scene import Scene

_DIR_PROBE_PX = 20.0  # image step along the lane direction used to measure it in metres


def group_members(*groups: str) -> set[str]:
    tracker_groups = load_thresholds()["perception"]["tracker_groups"]
    return {cls for g in groups for cls in tracker_groups[g]}


def in_group(tt: pd.DataFrame, *groups: str) -> np.ndarray:
    return tt["cls"].astype(str).isin(group_members(*groups)).to_numpy()


def max_gap(ctx: VideoContext) -> float:
    return 1.5 * ctx.stride / ctx.meta.fps


def runs(t: np.ndarray, mask: np.ndarray, gap: float) -> list[tuple[int, int]]:
    """Inclusive (first, last) indices of True runs; a False sample or a time step > gap ends a run."""
    out: list[tuple[int, int]] = []
    start = None
    for i in range(len(t)):
        if not mask[i]:
            if start is not None:
                out.append((start, i - 1))
                start = None
            continue
        if start is not None and t[i] - t[i - 1] > gap:
            out.append((start, i - 1))
            start = i
        elif start is None:
            start = i
    if start is not None:
        out.append((start, len(t) - 1))
    return out


def by_track(tt: pd.DataFrame, key: str = "track_id") -> Iterator[tuple[int, pd.DataFrame]]:
    for tid, rows in tt.sort_values([key, "t"], kind="stable").groupby(key, sort=True):
        yield int(tid), rows.reset_index(drop=True)


def lane_world_dirs(tt: pd.DataFrame, scene: Scene) -> np.ndarray:
    """(N, 2) unit lane directions on the road plane at each row's footprint (NaN where unknown)."""
    out = np.full((len(tt), 2), np.nan)
    if not scene.has("homography") or not len(tt):
        return out
    foot = tt[["fx", "fy"]].to_numpy(dtype=np.float64)
    lanes = tt["lane_id"].to_numpy()
    for lane_id, d_img in scene.lane_directions().items():
        rows = lanes == lane_id
        if not rows.any():
            continue
        a = scene.to_world(foot[rows])
        b = scene.to_world(foot[rows] + d_img * _DIR_PROBE_PX)
        d = b - a
        out[rows] = d / np.linalg.norm(d, axis=1, keepdims=True)
    return out


def lane_signal(scene: Scene) -> dict[str, str]:
    return {
        str(lane["id"]): str(lane["signal"]) for lane in scene.layers.get("lanes") or [] if lane.get("signal")
    }


def signal_states(ctx: VideoContext, sid: str | None, t: np.ndarray) -> np.ndarray:
    """State of signal `sid` at every time in `t` ("unknown" if no timeline or no signal)."""
    segments = ctx.signal_timeline.get(sid, []) if sid else []
    if not segments:
        return np.full(len(t), UNKNOWN, dtype=object)
    return np.array([state_at(segments, float(x)) for x in t], dtype=object)
