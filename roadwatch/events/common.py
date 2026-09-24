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
- `crossings(t, pts, line, gap)`: steps of one track crossing a line, with interpolated times.
- `past_line_m(scene, line, pts, lane_dir)`: signed metres beyond a line in the lane direction.
- `dist_to_polygon_m(scene, polygon, pts)`: metres to a polygon's boundary (0 inside).
- `stable_from(t, heading, start, tol, hold)`: first index whose heading then holds within `tol`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from roadwatch.config import load_thresholds
from roadwatch.events.base import VideoContext
from roadwatch.scene.light import UNKNOWN, state_at
from roadwatch.scene.scene import Scene, points_in_polygon, segments_cross

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


def crossings(t: np.ndarray, pts: np.ndarray, line: np.ndarray, gap: float) -> list[tuple[int, float]]:
    """Steps k-1 -> k of one track whose point crosses the 2-point `line`, with interpolated times.

    `pts` are image points (N, 2) of consecutive samples; steps longer than `gap` are ignored. The time
    is interpolated linearly by the signed distances of the two samples to the line.
    """
    if len(t) < 2:
        return []
    ok = np.diff(t) <= gap
    hit = np.flatnonzero(ok & segments_cross(line, pts[:-1], pts[1:])) + 1
    a, b = np.asarray(line, dtype=np.float64).reshape(2, 2)
    normal = np.array([-(b - a)[1], (b - a)[0]])
    out = []
    for k in hit:
        d0, d1 = float((pts[k - 1] - a) @ normal), float((pts[k] - a) @ normal)
        frac = d0 / (d0 - d1) if d0 != d1 else 1.0
        out.append((int(k), float(t[k - 1] + frac * (t[k] - t[k - 1]))))
    return out


def past_line_m(scene: Scene, line: np.ndarray, pts_img: np.ndarray, lane_dir_img: np.ndarray) -> np.ndarray:
    """Signed distance in metres of image points beyond `line`, positive in the lane direction."""
    a, b = scene.to_world(np.asarray(line, dtype=np.float64).reshape(2, 2))
    n = np.array([-(b - a)[1], (b - a)[0]])
    n /= np.linalg.norm(n)
    mid = (np.asarray(line, dtype=np.float64).reshape(2, 2)).mean(axis=0)
    ahead = scene.to_world(np.array([mid + lane_dir_img * _DIR_PROBE_PX]))[0] - scene.to_world(mid[None])[0]
    if n @ ahead < 0:
        n = -n
    return (scene.to_world(pts_img) - a) @ n


def dist_to_polygon_m(scene: Scene, polygon_img: np.ndarray, pts_img: np.ndarray) -> np.ndarray:
    """Distance in metres from image points to a polygon's boundary (0 inside), on the road plane."""
    poly = scene.to_world(np.asarray(polygon_img, dtype=np.float64))
    p = scene.to_world(pts_img)
    best = np.full(len(p), np.inf)
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        ab = b - a
        u = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p - (a + u[:, None] * ab), axis=1))
    best[points_in_polygon(p, poly)] = 0.0
    return best


def stable_from(t: np.ndarray, heading: np.ndarray, start: int, tol: float, hold: float) -> int | None:
    """First index >= start from which the heading stays within `tol` degrees for `hold` seconds."""
    for i in range(start, len(t)):
        if not np.isfinite(heading[i]):
            continue
        if t[-1] - t[i] < hold:
            return None  # not enough track left to confirm
        j = np.searchsorted(t, t[i] + hold, side="right")
        diff = (heading[i:j] - heading[i] + 180.0) % 360.0 - 180.0
        if np.all(np.abs(diff[np.isfinite(diff)]) <= tol):
            return i
    return None
