"""stopped_vehicle: A vehicle stationary on the carriageway for 10 s or more, not queued at a signal.

Trigger: Speed < stop_speed_mps continuously for >= min_stopped_sec, footprint on the carriageway,
not in a parking zone, and NOT queued: not within queue_upstream_m upstream of a stop line whose
signal is red/yellow, and no stopped vehicle ahead in the same lane within queue_ahead_m. Position
persistence bridges id switches of stationary objects (SPEC §3).
Start: first time the speed dropped below stop_speed_mps (backdated).
End: speed > resume_speed_mps for resume_hold_sec, or the track is gone; still stopped at the last
frame -> video duration.

Details:
- Stopping is a hysteresis: it starts below stop_speed_mps and ends only when the speed stays above
  resume_speed_mps for resume_hold_sec, so jitter around 0.5 m/s does not split a stop.
- Objects are followed by `obj_id` (persistence-merged), so runs may bridge gaps up to
  `kinematics.persistence_max_gap_sec`.
- A stop line whose signal state is unknown counts as possibly red: without the signal we cannot tell
  a broken-down car at the line from the first car of a red queue, and a false alarm costs more.
- The event needs one uninterrupted stretch of >= min_stopped_sec that is on the road, outside
  parking and not queued; it still starts when the vehicle first stopped.
Score = 0.5 + 0.5 * min(1, that stretch / (2 * min_stopped_sec)).

Thresholds: configs/thresholds.yaml -> classes.stopped_vehicle.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.config import load_thresholds
from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, in_group, lane_world_dirs, max_gap, runs, signal_states
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "stopped_vehicle"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "carriageway")
_QUEUE_STATES = ("red", "yellow", "unknown")


def stopped_hysteresis(t: np.ndarray, speed: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Per sample: is the vehicle in a stop (entered below stop_speed, not yet resumed)."""
    out = np.zeros(len(t), dtype=bool)
    stopped = False
    i = 0
    while i < len(t):
        if not stopped:
            stopped = bool(speed[i] < p["stop_speed_mps"])
            out[i] = stopped
            i += 1
            continue
        if speed[i] > p["resume_speed_mps"]:
            j = i
            while j < len(t) and speed[j] > p["resume_speed_mps"]:
                j += 1
            if t[j - 1] - t[i] >= p["resume_hold_sec"]:
                stopped = False  # resumed at i: samples i..j-1 are moving
                i = j
                continue
            out[i:j] = True  # a brief creep forward inside the stop
            i = j
            continue
        out[i] = True
        i += 1
    return out


def queued(v: pd.DataFrame, scene: Scene, ctx: VideoContext, p: dict[str, Any]) -> np.ndarray:
    """Per row of vehicle table `v`: waiting at a (possibly) red stop line, or behind a stopped vehicle."""
    out = np.zeros(len(v), dtype=bool)
    if v.empty:
        return out
    pos = v[["X", "Y"]].to_numpy(dtype=np.float64)
    lane_dir = lane_world_dirs(v, scene)
    lanes = v["lane_id"].to_numpy()
    t = v["t"].to_numpy()

    for sl in scene.layers.get("stop_lines") or []:
        if not sl.get("signal"):
            continue
        mid = scene.to_world(np.asarray(sl["line"], dtype=np.float64).mean(axis=0, keepdims=True))[0]
        along = ((mid - pos) * lane_dir).sum(axis=1)  # > 0: the line is still ahead
        near = np.isin(lanes, sl.get("lanes", [])) & (along >= 0) & (along <= p["queue_upstream_m"])
        if near.any():
            states = signal_states(ctx, str(sl["signal"]), t[near])
            out[np.flatnonzero(near)[np.isin(states, _QUEUE_STATES)]] = True

    slow = v["speed"].to_numpy() < p["stop_speed_mps"]
    for _, idx in v.groupby("frame", sort=False).indices.items():
        idx = idx[slow[idx] & (lanes[idx] != "")]
        if len(idx) < 2:
            continue
        rel = pos[idx][None, :, :] - pos[idx][:, None, :]  # [i, j] = j - i
        along = (rel * lane_dir[idx][:, None, :]).sum(axis=2)
        same_lane = lanes[idx][:, None] == lanes[idx][None, :]
        ahead = same_lane & (along > 0) & (along <= p["queue_ahead_m"])
        out[idx[ahead.any(axis=1)]] = True
    return out


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    v = tt[in_group(tt, "vehicles")].sort_values(["obj_id", "t"], kind="stable").reset_index(drop=True)
    if v.empty:
        return []
    v["ok"] = v["on_road"].to_numpy() & ~v["in_parking"].to_numpy() & ~queued(v, scene, ctx, p)
    gap = max(max_gap(ctx), load_thresholds()["kinematics"]["persistence_max_gap_sec"])

    segments = []
    for oid, rows in by_track(v, key="obj_id"):
        t = rows["t"].to_numpy()
        speed = rows["speed"].to_numpy(dtype=np.float64)
        ok = rows["ok"].to_numpy()
        for a, b in runs(t, stopped_hysteresis(t, speed, p), gap):
            if t[b] - t[a] < p["min_stopped_sec"]:
                continue
            clean = [t[e] - t[s] for s, e in runs(t[a : b + 1], ok[a : b + 1], gap)]
            best = max(clean, default=0.0)
            if best < p["min_stopped_sec"]:
                continue
            end = t[b + 1] if b + 1 < len(t) else t[b]  # the moment it moved off, or the last sighting
            score = 0.5 + 0.5 * min(1.0, best / (2 * p["min_stopped_sec"]))
            ids = tuple(sorted({int(x) for x in rows["track_id"].to_numpy()[a : b + 1]}))
            segments.append(Segment(float(t[a]), float(end), LABEL, score, ids, {"obj_id": oid}))
    return segments
