"""accident: Collision between road users, or between a road user and a fixed object.

Trigger: A pair (vehicle-vehicle, vehicle-person, vehicle-two-wheeler) with footprints closer than
contact_dist_m AND overlapping boxes, AND at least one kinematic shock within +/- shock_window_sec
(deceleration > shock_decel_mps2, heading change > shock_heading_deg in under
shock_heading_window_sec, or a person box flipping aspect as they fall); confirmed when the involved
objects stay stopped >= confirm_stopped_sec outside any queue, or leave. Single vehicle: speed >
single_speed_before_mps drops below single_speed_after_mps within single_window_sec, off-lane or
near the road edge. Box overlap alone is never enough (occlusion is the main false-positive source).
Start: first contact frame (refined at stride 1).
End: all involved speeds < end_speed_mps for end_hold_sec, or they leave the frame.

Details:
- "Stay stopped": one involved road vehicle slower than confirm_speed_mps for >= confirm_stopped_sec,
  starting within confirm_within_sec of the contact; "leave": an involved track ends within
  confirm_within_sec (a fallen rider or a car knocked out of view). A person falling needs no more.
- Queued pairs are excluded by the shock requirement: waiting cars do not decelerate hard.
- With a pedestrian in the pair the evidence has to come from the pedestrian: a fall, or the person lost
  mid-frame right after the contact. A car braking to a stop beside someone is a drop-off or a yield.
- Single vehicle "near the road edge" = footprint off every lane or off the carriageway.
- The end is where every involved object is slower than end_speed_mps for end_hold_sec; if that never
  happens, the last sample of the involved tracks.
Score = min(1, 0.6 + 0.2 * number of shock kinds seen) for pairs, 0.7 for single-vehicle crashes.

Thresholds: configs/thresholds.yaml -> classes.accident.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import by_track, group_members, in_group, midpoint_gap, runs
from roadwatch.scene.scene import Scene, ground_scale
from roadwatch.types import Segment

LABEL = "accident"
REQUIRED_LAYERS: tuple[str, ...] = ("homography",)
_BOX = ["x1", "y1", "x2", "y2"]


def contacts(tt: pd.DataFrame, p: dict[str, Any]) -> dict[tuple[int, int], float]:
    """First contact time of every eligible pair: footprints within contact_dist_m and boxes overlapping."""
    vehicles = group_members("vehicles")
    road = group_members("vehicles", "two_wheelers", "persons")
    rows = tt[tt["cls"].astype(str).isin(road)]
    first: dict[tuple[int, int], float] = {}
    for _, f in rows.groupby("frame", sort=True):
        if len(f) < 2:
            continue
        ids = f["track_id"].to_numpy()
        is_vehicle = f["cls"].astype(str).isin(vehicles).to_numpy()
        pos = f[["X", "Y"]].to_numpy(dtype=np.float64)
        box = f[_BOX].to_numpy(dtype=np.float64)
        i, j = np.triu_indices(len(f), k=1)
        close = np.hypot(*(pos[i] - pos[j]).T) < p["contact_dist_m"]
        overlap = (np.minimum(box[i, 2], box[j, 2]) > np.maximum(box[i, 0], box[j, 0])) & (
            np.minimum(box[i, 3], box[j, 3]) > np.maximum(box[i, 1], box[j, 1])
        )
        eligible = is_vehicle[i] | is_vehicle[j]  # every pair type in the rule involves a vehicle
        t = float(f["t"].iloc[0])
        for a, b in zip(i[close & overlap & eligible], j[close & overlap & eligible], strict=True):
            key = (int(min(ids[a], ids[b])), int(max(ids[a], ids[b])))
            first.setdefault(key, t)
    return first


def shocks(rows: pd.DataFrame, t0: float, p: dict[str, Any]) -> set[str]:
    """Kinds of kinematic shock of one track within +/- shock_window_sec of t0.

    Braking and heading shocks are not read in a track's first shock_min_track_age_sec: a new box (a
    vehicle entering or emerging from behind another) grows while it is revealed, so its first
    footprints race ahead and stop dead, which looks exactly like a hard stop (SPEC §12.48).
    """
    t = rows["t"].to_numpy()
    settled = t >= t.min() + p["shock_min_track_age_sec"]
    near = (t >= t0 - p["shock_window_sec"]) & (t <= t0 + p["shock_window_sec"]) & settled
    kinds = set()
    if (rows["accel"].to_numpy()[near] < -p["shock_decel_mps2"]).any():
        kinds.add("decel")
    heading = rows["heading_deg"].to_numpy(dtype=np.float64)
    # a sudden turn means something only for a moving vehicle: a stopped car's heading is jitter and
    # pedestrians turn all the time (their fall is the "fall" shock below) (SPEC §12.53)
    moving = rows["speed"].to_numpy(dtype=np.float64) >= p["shock_heading_min_speed_mps"]
    ok = np.isfinite(heading) & moving
    if ok.sum() >= 2 and str(rows["cls"].iloc[0]) in group_members("vehicles", "two_wheelers"):
        unwrapped = np.full(len(t), np.nan)
        unwrapped[ok] = np.degrees(np.unwrap(np.radians(heading[ok])))
        for k in np.flatnonzero(near & ok):
            later = ok & (t > t[k]) & (t <= t[k] + p["shock_heading_window_sec"])
            if later.any() and np.nanmax(np.abs(unwrapped[later] - unwrapped[k])) > p["shock_heading_deg"]:
                kinds.add("heading")
                break
    if str(rows["cls"].iloc[0]) in group_members("persons"):
        aspect = ((rows["y2"] - rows["y1"]) / (rows["x2"] - rows["x1"]).clip(lower=1e-6)).to_numpy()
        before = (t >= t0 - p["shock_window_sec"]) & (t <= t0)
        after = (t >= t0) & (t <= t0 + 2 * p["shock_window_sec"] + p["confirm_within_sec"])
        if (
            before.any()
            and after.any()
            and aspect[before].max() >= p["fall_aspect_upright"]
            and aspect[after].min() <= p["fall_aspect_down"]
        ):
            kinds.add("fall")
    return kinds


def person_hit(
    tracks: list[pd.DataFrame], t0: float, kinds: set[str], p: dict[str, Any], exits: set[int]
) -> bool:
    """For a pair with a pedestrian, whether the pedestrian shows it: a fall, or the person lost mid-frame
    right after the contact. Pairs without a pedestrian pass.

    A vehicle braking hard and then standing next to a person is what a drop-off, a pick-up or a driver
    giving way looks like, and in C3896 (3:15) it passed both the shock and the confirmation (SPEC §12.58).
    """
    persons = [rows for rows in tracks if str(rows["cls"].iloc[0]) in group_members("persons")]
    if not persons:
        return True
    if "fall" in kinds:
        return True
    return any(
        int(rows["track_id"].iloc[0]) not in exits
        and rows["t"].to_numpy()[-1] <= t0 + p["confirm_within_sec"]
        for rows in persons
    )


def confirmed(
    tracks: list[pd.DataFrame], t0: float, kinds: set[str], p: dict[str, Any], exits: set[int]
) -> bool:
    """Whether a contact is a crash: a fall, a vehicle that stays stopped, or a track that vanishes.

    A track that ends by reaching the frame edge, or that is still there when the video ends (`exits`),
    did not vanish, so it confirms nothing. Only a track lost mid-frame right after the contact (a rider
    thrown down, a car knocked out of view) counts.
    """
    if "fall" in kinds:
        return True
    vehicles = group_members("vehicles", "two_wheelers")
    for rows in tracks:
        t = rows["t"].to_numpy()
        vanished = int(rows["track_id"].iloc[0]) not in exits
        if vanished and t[-1] <= t0 + p["confirm_within_sec"]:
            return True  # lost mid-frame right after the contact
        if str(rows["cls"].iloc[0]) not in vehicles:
            continue
        slow = rows["speed"].to_numpy() < p["confirm_speed_mps"]
        for a, b in runs(t, slow, np.inf):
            if (
                t[a] <= t0 + p["confirm_within_sec"]
                and t[b] >= t0
                and t[b] - max(t[a], t0) >= p["confirm_stopped_sec"]
            ):
                return True
    return False


def end_time(tracks: list[pd.DataFrame], t0: float, p: dict[str, Any]) -> float:
    """Start of the first end_hold_sec during which every involved object is below end_speed_mps."""
    times = np.unique(np.concatenate([r["t"].to_numpy() for r in tracks]))
    times = times[times >= t0]
    still = np.ones(len(times), dtype=bool)
    for rows in tracks:
        speed = np.interp(times, rows["t"].to_numpy(), rows["speed"].to_numpy(dtype=np.float64))
        still &= speed < p["end_speed_mps"]
    for a, b in runs(times, still, np.inf):
        if times[b] - times[a] >= p["end_hold_sec"]:
            return float(times[a])
    return float(min(r["t"].iloc[-1] for r in tracks))


def single_vehicle(rows: pd.DataFrame, p: dict[str, Any]) -> list[tuple[float, float]]:
    """(start, stop) times of abrupt stops from speed off the lanes or off the road.

    Only from the track's settled part (not its first shock_min_track_age_sec, when a box being revealed
    races ahead and stops), and confirmed like a pair crash: the vehicle then stays below
    confirm_speed_mps for confirm_stopped_sec. An articulated bus whose box jumps while it turns, and
    which then drives on, did not crash (SPEC §12.53).
    """
    t = rows["t"].to_numpy()
    settled_from = t.min() + p["shock_min_track_age_sec"]
    speed = rows["speed"].to_numpy(dtype=np.float64)
    off = (rows["lane_id"].to_numpy() == "") | ~rows["on_road"].to_numpy()
    out = []
    k = 0
    while k < len(t):
        if speed[k] > p["single_speed_before_mps"]:
            j = np.flatnonzero(
                (t > t[k]) & (t <= t[k] + p["single_window_sec"]) & (speed < p["single_speed_after_mps"])
            )
            if len(j) and off[j[0]] and t[k] >= settled_from and _stays_stopped(t, speed, int(j[0]), p):
                start = int(np.argmax(speed[k : j[0] + 1])) + k  # the last fast moment before the drop
                out.append((float(t[start]), float(t[j[0]])))
                k = int(j[0]) + 1
                continue
        k += 1
    return out


def _stays_stopped(t: np.ndarray, speed: np.ndarray, i: int, p: dict[str, Any]) -> bool:
    """Whether the track stays under confirm_speed_mps for confirm_stopped_sec from sample i."""
    after = (t >= t[i]) & (t <= t[i] + p["confirm_stopped_sec"])
    return bool((speed[after] < p["confirm_speed_mps"]).all())


def continuation(a: pd.DataFrame, b: pd.DataFrame, p: dict[str, Any]) -> bool:
    """Whether one track is the other re-identified: it starts within reid_max_gap_sec of the other's
    end, where the other was last seen (within contact_dist_m). Such a pair is one object (SPEC §12.53)."""
    first, last = (a, b) if a["t"].min() <= b["t"].min() else (b, a)
    end, start = first.loc[first["t"].idxmax()], last.loc[last["t"].idxmin()]
    close = float(np.hypot(end["X"] - start["X"], end["Y"] - start["Y"])) <= p["contact_dist_m"]
    return bool(abs(float(start["t"]) - float(end["t"])) <= p["reid_max_gap_sec"] and close)


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    # only where the road is measured finely enough: far away a -60 m/s^2 "stop" is box jitter (§12.53)
    measurable = ground_scale(scene, tt[["fx", "fy"]].to_numpy(dtype=np.float64)) <= p["max_ground_m_per_px"]
    valid = tt[tt["kin_valid"].to_numpy() & measurable]
    tracks = dict(by_track(valid))
    last_rows = tt.sort_values(["track_id", "t"], kind="stable").groupby("track_id", sort=True).tail(1)
    exits = set(last_rows.loc[last_rows["at_edge"].to_numpy(dtype=bool), "track_id"].astype(int))
    # a track still there at the last processed frame did not vanish: the video ended (SPEC §12.48)
    ended = last_rows["t"].to_numpy() >= tt["t"].max() - midpoint_gap(ctx)
    exits |= set(last_rows.loc[ended, "track_id"].astype(int))
    segments = []
    for (a, b), t0 in sorted(contacts(valid, p).items(), key=lambda kv: kv[1]):
        pair = [tracks[a], tracks[b]]
        if continuation(pair[0], pair[1], p):
            continue  # one object re-identified by the tracker, not two colliding
        kinds = shocks(pair[0], t0, p) | shocks(pair[1], t0, p)
        if not kinds or not confirmed(pair, t0, kinds, p, exits):
            continue
        if not person_hit(pair, t0, kinds, p, exits):
            continue  # a car braking to a stop beside a pedestrian is a drop-off or a yield, not a crash
        score = min(1.0, 0.6 + 0.2 * len(kinds))
        segments.append(Segment(t0, end_time(pair, t0, p), LABEL, score, (a, b), {"shocks": sorted(kinds)}))

    for tid, rows in by_track(valid[in_group(valid, "vehicles", "two_wheelers")]):
        for t0, _ in single_vehicle(rows, p):
            segments.append(Segment(t0, end_time([rows], t0, p), LABEL, 0.7, (tid,), {"single": True}))
    return segments
