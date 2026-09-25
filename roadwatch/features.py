"""Per-track kinematics in metres and pairwise features (SPEC §3, RUNBOOK P1.3).

Contract:
- `add_kinematics(tt, scene, mode)` returns the TrackTable sorted by (t, track_id) with footprints
  mapped to metres (`X, Y`) and `vx, vy, speed, accel, heading_deg, yaw_rate, kin_valid`, the zone
  columns `lane_id, on_road, on_sidewalk, in_crosswalk, crosswalk_id, in_intersection, in_no_uturn,
  in_parking`, the vehicle's ground extent (`front_x, front_y`, `rear_x, rear_y` in image px, the box
  centre `cX, cY` in metres, `veh_len, veh_wid`; SPEC §12.35) and `obj_id`.
  mode="offline" smooths with Savitzky-Golay (Part A); mode="online" uses causal EMA (Part B), so
  every row depends only on earlier rows of its track.
  `accel` is signed along the heading (braking < 0). Below `heading_min_speed_mps` the heading is
  held from the last moving sample and `yaw_rate` is 0. Tracks shorter than `min_track_sec`, and every
  track when the scene has no homography, get `kin_valid = False` (metric columns are NaN then).
  `obj_id` joins stationary tracks broken by id switches (position persistence, SPEC §3); it equals
  `track_id` otherwise.
- `pair_features(frame_rows)` returns converging pairs within `risk.max_pair_dist_m` (closing speed
  above `risk.min_closing_speed_mps`) with distance, closing speed and time-to-collision.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from roadwatch.config import load_thresholds
from roadwatch.scene.scene import Scene

_DOWN_PROBE_PX = 20.0  # image step used to measure directions on the road plane

PAIR_COLUMNS = (
    "track_a",
    "track_b",
    "cls_a",
    "cls_b",
    "dist",
    "closing_speed",
    "ttc",
    "ttc_cv",
    "t_cpa",
    "d_cpa",
)


def add_kinematics(
    tt: pd.DataFrame, scene: Scene, mode: Literal["offline", "online"] = "offline"
) -> pd.DataFrame:
    """Append metric kinematics, zone flags, front point and persistent object ids (see module doc)."""
    if mode not in ("offline", "online"):
        raise ValueError(f"mode must be 'offline' or 'online', got {mode!r}")
    cfg = load_thresholds()["kinematics"]
    out = tt.sort_values(["track_id", "t"], kind="stable").reset_index(drop=True)
    foot = out[["fx", "fy"]].to_numpy(dtype=np.float64)

    _add_zone_columns(out, scene, foot)

    metric = scene.has("homography") and len(out) > 0
    xy = scene.to_world(foot) if metric else np.full((len(out), 2), np.nan)
    out["X"], out["Y"] = xy[:, 0], xy[:, 1]
    out["at_edge"] = _at_edge(out, scene, cfg)
    for col, values in _track_kinematics(out, xy, mode, cfg).items():
        out[col] = values
    _add_vehicle_extent(out, scene, foot, cfg)
    out["obj_id"] = _persistence_ids(out, cfg)

    return out.sort_values(["t", "track_id"], kind="stable").reset_index(drop=True)


def pair_features(frame_rows: pd.DataFrame) -> pd.DataFrame:
    """Pairs on a collision course among one frame's rows, sorted by time to collision (SPEC §12.39).

    Needs `X, Y, vx, vy` (metres, m/s); uses acceleration from `ax, ay`, or from `accel` along
    `heading_deg`, when present (else none). A pair is kept only if all of these hold:
    - it is within `max_pair_dist_m` and closing faster than `min_closing_speed_mps`;
    - its paths really meet: the closest point of approach at constant relative velocity (`t_cpa` from
      now, `d_cpa` apart) is under the collision radius, `conflict_radius_m` (vehicles) or
      `conflict_radius_person_m` (a pedestrian involved). A car turning past someone 3 m outside its
      path, or overtaking in the next lane, closes fast but is no conflict;
    Two times to collision are returned:
    - `ttc_cv` = d / c, at constant velocity: how imminent the collision would be without any reaction
      (near_miss asks this at the moment the evasive action starts);
    - `ttc`, braking-aware: the time for the distance to reach zero with the current relative
      acceleration along the line between them, the smallest positive t with c t + a t^2 / 2 = d,
      computed as 2d / (c + sqrt(c^2 + 2 a d)); inf when braking stops the approach short. A planned
      stop behind a queue never predicts contact (Part B's risk uses this one).
    """
    cfg = load_thresholds()["risk"]
    cols = ["X", "Y", "vx", "vy"]
    rows = frame_rows[np.isfinite(frame_rows[cols].to_numpy(dtype=np.float64)).all(axis=1)]
    if len(rows) < 2:
        return pd.DataFrame({c: [] for c in PAIR_COLUMNS})

    pos = rows[["X", "Y"]].to_numpy(dtype=np.float64)
    vel = rows[["vx", "vy"]].to_numpy(dtype=np.float64)
    acc = _acceleration_vectors(rows)
    i, j = np.triu_indices(len(rows), k=1)
    rel_pos, rel_vel, rel_acc = pos[j] - pos[i], vel[j] - vel[i], acc[j] - acc[i]
    dist = np.hypot(rel_pos[:, 0], rel_pos[:, 1])
    speed2 = (rel_vel**2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        closing = np.where(dist > 0, -(rel_pos * rel_vel).sum(axis=1) / dist, 0.0)
        closing_acc = np.where(dist > 0, -(rel_pos * rel_acc).sum(axis=1) / dist, 0.0)
        t_cpa = np.where(speed2 > 0, np.maximum(0.0, -(rel_pos * rel_vel).sum(axis=1) / speed2), 0.0)
        disc = closing**2 + 2 * closing_acc * dist
        ttc = np.where(disc >= 0, 2 * dist / (closing + np.sqrt(np.maximum(disc, 0.0))), np.inf)
        ttc_cv = dist / closing
    d_cpa = np.hypot(*(rel_pos + rel_vel * t_cpa[:, None]).T)

    track = rows["track_id"].to_numpy()
    cls = rows["cls"].astype(str).to_numpy()
    person = np.isin(cls, cfg["person_classes"])
    radius = np.where(person[i] | person[j], cfg["conflict_radius_person_m"], cfg["conflict_radius_m"])
    keep = (dist < cfg["max_pair_dist_m"]) & (closing > cfg["min_closing_speed_mps"]) & (d_cpa < radius)
    pairs = pd.DataFrame(
        {
            "track_a": track[i][keep],
            "track_b": track[j][keep],
            "cls_a": cls[i][keep],
            "cls_b": cls[j][keep],
            "dist": dist[keep],
            "closing_speed": closing[keep],
            "ttc": ttc[keep],
            "ttc_cv": ttc_cv[keep],
            "t_cpa": t_cpa[keep],
            "d_cpa": d_cpa[keep],
        }
    )
    return pairs.sort_values(["ttc", "track_a", "track_b"], kind="stable").reset_index(drop=True)


def _acceleration_vectors(rows: pd.DataFrame) -> np.ndarray:
    """(N, 2) acceleration in metres/s^2: `ax, ay` if given, else `accel` along the heading, else 0."""
    if {"ax", "ay"} <= set(rows.columns):
        acc = rows[["ax", "ay"]].to_numpy(dtype=np.float64)
    elif {"accel", "heading_deg"} <= set(rows.columns):
        rad = np.radians(rows["heading_deg"].to_numpy(dtype=np.float64))
        along = rows["accel"].to_numpy(dtype=np.float64)
        acc = along[:, None] * np.stack([np.cos(rad), np.sin(rad)], axis=1)
    else:
        acc = np.zeros((len(rows), 2))
    return np.nan_to_num(acc, nan=0.0)


def _add_zone_columns(out: pd.DataFrame, scene: Scene, foot: np.ndarray) -> None:
    out["lane_id"] = scene.lane_of(foot).astype(str)
    out["on_road"] = scene.point_in("carriageway", foot)
    # traffic islands (refuges, the median) are pavement for pedestrians (SPEC §12.6, §12.42)
    out["on_sidewalk"] = scene.point_in("sidewalks", foot) | scene.point_in("islands", foot)
    out["crosswalk_id"] = scene.region_of("crosswalks", foot).astype(str)
    out["in_crosswalk"] = out["crosswalk_id"] != ""
    out["in_intersection"] = scene.point_in("intersection", foot)
    out["in_no_uturn"] = scene.point_in("no_u_turn_zones", foot)
    out["in_parking"] = scene.point_in("parking_zones", foot)
    out["in_bus_stop"] = scene.point_in("bus_stops", foot)


def _add_vehicle_extent(out: pd.DataFrame, scene: Scene, foot: np.ndarray, cfg: dict[str, Any]) -> None:
    """Front and rear bumper points of every vehicle row from a ground-plane box model (SPEC §12.35).

    A vehicle is a `length x width` rectangle on the road, oriented by its heading (world frame; held
    while slow; the lane direction before it first moves). The box's bottom-centre (the footprint) is
    the point of that rectangle nearest the camera, so the rectangle's centre lies behind the footprint,
    away from the camera, by the rectangle's half-extent along the camera direction `u` (the road-plane
    direction of "down the image" at the footprint):
        centre = footprint - (L/2 |h.u| + W/2 |n.u|) u,   front = centre + L/2 h,   rear = centre - L/2 h.
    A car driving toward the camera then has its front at the footprint, one driving away its rear.
    Persons, rows without a heading or lane direction, and scenes without a homography fall back to the
    footprint for both points.
    """
    n = len(out)
    front, rear = foot.copy(), foot.copy()
    centre = np.full((n, 2), np.nan)
    dims = cfg["vehicle_dims_m"]
    cls = out["cls"].astype(str).to_numpy()
    length = np.array([dims.get(c, (0.0, 0.0))[0] for c in cls], dtype=np.float64)
    width = np.array([dims.get(c, (0.0, 0.0))[1] for c in cls], dtype=np.float64)
    if n and scene.has("homography"):
        rad = np.radians(out["heading_deg"].to_numpy(dtype=np.float64))
        h = np.stack([np.cos(rad), np.sin(rad)], axis=1)
        missing = ~np.isfinite(h).all(axis=1)
        if missing.any():
            h[missing] = _lane_world_dirs(out["lane_id"].to_numpy()[missing], foot[missing], scene)
        ok = (length > 0) & np.isfinite(h).all(axis=1)
        if ok.any():
            w0 = scene.to_world(foot[ok])
            u = scene.to_world(foot[ok] + np.array([0.0, _DOWN_PROBE_PX])) - w0
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            hk = h[ok]
            nk = np.stack([-hk[:, 1], hk[:, 0]], axis=1)
            half_l, half_w = length[ok] / 2, width[ok] / 2
            depth = half_l * np.abs((hk * u).sum(axis=1)) + half_w * np.abs((nk * u).sum(axis=1))
            c = w0 - depth[:, None] * u
            centre[ok] = c
            front[ok] = scene.to_image(c + half_l[:, None] * hk)
            rear[ok] = scene.to_image(c - half_l[:, None] * hk)
    out["front_x"], out["front_y"] = front[:, 0], front[:, 1]
    out["rear_x"], out["rear_y"] = rear[:, 0], rear[:, 1]
    out["cX"], out["cY"] = centre[:, 0], centre[:, 1]
    out["veh_len"], out["veh_wid"] = length, width


def _lane_world_dirs(lanes: np.ndarray, foot: np.ndarray, scene: Scene) -> np.ndarray:
    """Unit lane direction on the road plane at each footprint (NaN outside lanes with a direction)."""
    out = np.full((len(lanes), 2), np.nan)
    for lane_id, d_img in scene.lane_directions().items():
        rows = lanes == lane_id
        if rows.any():
            d = scene.to_world(foot[rows] + d_img * _DOWN_PROBE_PX) - scene.to_world(foot[rows])
            out[rows] = d / np.linalg.norm(d, axis=1, keepdims=True)
    return out


def vehicle_corners(rows: pd.DataFrame, scene: Scene) -> np.ndarray:
    """(N, 4, 2) image points of each row's ground rectangle: front-left, front-right, rear-right, rear-left.

    Rows without a vehicle model (persons, no heading) get their footprint four times.
    """
    n = len(rows)
    foot = rows[["fx", "fy"]].to_numpy(dtype=np.float64)
    out = np.repeat(foot[:, None, :], 4, axis=1)
    centre = rows[["cX", "cY"]].to_numpy(dtype=np.float64)
    ok = np.isfinite(centre).all(axis=1) & (rows["veh_len"].to_numpy() > 0)
    if not n or not ok.any():
        return out
    front = rows[["front_x", "front_y"]].to_numpy(dtype=np.float64)[ok]
    h = scene.to_world(front) - centre[ok]
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    nvec = np.stack([-h[:, 1], h[:, 0]], axis=1)
    half_l = rows["veh_len"].to_numpy(dtype=np.float64)[ok, None] / 2
    half_w = rows["veh_wid"].to_numpy(dtype=np.float64)[ok, None] / 2
    c = centre[ok]
    world = [
        c + half_l * h + half_w * nvec,
        c + half_l * h - half_w * nvec,
        c - half_l * h - half_w * nvec,
        c - half_l * h + half_w * nvec,
    ]
    out[ok] = np.stack([scene.to_image(w) for w in world], axis=1)
    return out


def _track_bounds(track_ids: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) row ranges of each track in a table sorted by track_id."""
    if len(track_ids) == 0:
        return []
    cuts = np.flatnonzero(np.diff(track_ids)) + 1
    return list(zip(np.r_[0, cuts].tolist(), np.r_[cuts, len(track_ids)].tolist(), strict=True))


def _at_edge(out: pd.DataFrame, scene: Scene, cfg: dict[str, Any]) -> np.ndarray:
    """Rows whose box touches the frame border (within edge_margin_frac of the frame size).

    The detector clips boxes to the frame, so an object half out of view has a footprint that no longer
    follows it: a car leaving through the bottom edge seems to stop dead. Without the frame size (no
    `image_size` in the scene) nothing is flagged.
    """
    size = scene.layers.get("image_size")
    if not size or out.empty:
        return np.zeros(len(out), dtype=bool)
    width, height = float(size[0]), float(size[1])
    mx, my = cfg["edge_margin_frac"] * width, cfg["edge_margin_frac"] * height
    return (
        (out["x1"].to_numpy(dtype=np.float64) <= mx)
        | (out["y1"].to_numpy(dtype=np.float64) <= my)
        | (out["x2"].to_numpy(dtype=np.float64) >= width - mx)
        | (out["y2"].to_numpy(dtype=np.float64) >= height - my)
    )


def _track_kinematics(
    out: pd.DataFrame, xy: np.ndarray, mode: str, cfg: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Per track, from its rows away from the frame edge only (edge rows keep NaN and kin_valid=False)."""
    n = len(out)
    t = out["t"].to_numpy(dtype=np.float64)
    edge = out["at_edge"].to_numpy(dtype=bool)
    res = {c: np.full(n, np.nan) for c in ("vx", "vy", "speed", "accel", "heading_deg", "yaw_rate")}
    res["kin_valid"] = np.zeros(n, dtype=bool)
    if not np.isfinite(xy).all():
        return res

    smooth = _savgol_track if mode == "offline" else _ema_track
    for s, e in _track_bounds(out["track_id"].to_numpy()):
        rows = np.arange(s, e)[~edge[s:e]]
        if len(rows) < 2:
            continue
        tt, pos = t[rows], xy[rows]
        vel, acc = smooth(tt, pos, cfg)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        heading, yaw_rate = _heading(tt, vel, speed, cfg["heading_min_speed_mps"], mode)
        unit = np.stack([np.cos(np.radians(heading)), np.sin(np.radians(heading))], axis=1)
        res["vx"][rows], res["vy"][rows], res["speed"][rows] = vel[:, 0], vel[:, 1], speed
        res["accel"][rows] = (acc * unit).sum(axis=1)  # NaN until the first moving sample
        res["heading_deg"][rows], res["yaw_rate"][rows] = heading, yaw_rate
        res["kin_valid"][rows] = tt[-1] - tt[0] >= cfg["min_track_sec"]
    return res


def _savgol_track(t: np.ndarray, xy: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Offline velocity and acceleration: Savitzky-Golay derivatives on a uniform time grid.

    Rows sit on multiples of one step (stride / fps); gaps (lost frames, paced detection) are filled
    by linear interpolation before filtering and the grid is sampled back at the original rows.
    """
    order = cfg["savgol_order"]
    steps = np.diff(t)
    if len(t) < order + 2 or not (steps > 0).any():
        return _gradient_track(t, xy)
    dt = steps[steps > 0].min()
    idx = np.rint((t - t[0]) / dt).astype(int)
    grid_t = t[0] + dt * np.arange(idx[-1] + 1)
    grid = np.stack([np.interp(grid_t, t, xy[:, k]) for k in range(2)], axis=1)

    win = int(round(cfg["savgol_window_sec"] / dt)) | 1  # odd
    win = min(max(win, order + 1 + (order % 2 == 0)), len(grid) - (len(grid) % 2 == 0))
    if win <= order:
        return _gradient_track(t, xy)
    vel = savgol_filter(grid, win, order, deriv=1, delta=dt, axis=0, mode="interp")
    acc = savgol_filter(grid, win, order, deriv=2, delta=dt, axis=0, mode="interp")
    return vel[idx], acc[idx]


def _gradient_track(t: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain finite differences for tracks too short to filter (they are kin_valid=False anyway)."""
    if len(t) < 2 or t[-1] <= t[0]:
        return np.zeros_like(xy), np.zeros_like(xy)
    vel = np.gradient(xy, t, axis=0)
    acc = np.gradient(vel, t, axis=0) if len(t) > 2 else np.zeros_like(xy)
    return vel, acc


def _ema_track(t: np.ndarray, xy: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Online velocity and acceleration: EMA on position, then on its finite differences (causal)."""
    a = cfg["ema_alpha"]
    pos = xy[0].copy()
    vel = np.zeros_like(xy)
    acc = np.zeros_like(xy)
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        if dt <= 0:
            vel[k], acc[k] = vel[k - 1], acc[k - 1]
            continue
        new_pos = a * xy[k] + (1 - a) * pos
        v_raw = (new_pos - pos) / dt
        vel[k] = v_raw if k == 1 else a * v_raw + (1 - a) * vel[k - 1]
        if k >= 2:
            a_raw = (vel[k] - vel[k - 1]) / dt
            acc[k] = a_raw if k == 2 else a * a_raw + (1 - a) * acc[k - 1]
        pos = new_pos
    return vel, acc


def _heading(
    t: np.ndarray, vel: np.ndarray, speed: np.ndarray, min_speed: float, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Heading in degrees (world frame, held while slow) and yaw rate in degrees/s (0 while held)."""
    moving = speed >= min_speed
    raw = np.degrees(np.arctan2(vel[:, 1], vel[:, 0]))
    heading = pd.Series(np.where(moving, raw, np.nan)).ffill().to_numpy()
    yaw_rate = np.zeros(len(t))
    known = np.flatnonzero(np.isfinite(heading))
    if len(known) >= 2:
        unwrapped = np.degrees(np.unwrap(np.radians(heading[known])))
        tk = t[known]
        if mode == "offline":
            rate = np.gradient(unwrapped, tk) if tk[-1] > tk[0] else np.zeros(len(tk))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.r_[0.0, np.diff(unwrapped) / np.diff(tk)]
            rate = np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
        yaw_rate[known] = rate
    yaw_rate[~moving] = 0.0
    return heading, yaw_rate


def _persistence_ids(out: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """obj_id per row: a track that starts stationary within `persistence_max_dist_m` of where a
    stationary track of the same tracker group vanished less than `persistence_max_gap_sec` earlier
    continues that object (nearest such track wins; each track continues at most one)."""
    track_ids = out["track_id"].to_numpy()
    obj = track_ids.astype(np.int64).copy()
    bounds = _track_bounds(track_ids)
    if len(bounds) < 2 or not np.isfinite(out["speed"].to_numpy()).any():
        return obj

    group_of = {
        cls: group
        for group, members in load_thresholds()["perception"]["tracker_groups"].items()
        for cls in members
    }
    t = out["t"].to_numpy(dtype=np.float64)
    xy = out[["X", "Y"]].to_numpy(dtype=np.float64)
    speed = out["speed"].to_numpy(dtype=np.float64)
    cls = out["cls"].astype(str).to_numpy()
    first = np.array([s for s, _ in bounds])
    last = np.array([e - 1 for _, e in bounds])
    group = np.array([group_of.get(c, c) for c in cls[first]], dtype=object)
    still_end = speed[last] <= cfg["persistence_max_speed_mps"]
    still_start = speed[first] <= cfg["persistence_max_speed_mps"]

    track_obj = track_ids[first].astype(np.int64)
    taken = np.zeros(len(bounds), dtype=bool)
    for b in np.lexsort((track_ids[first], t[first])):
        if not still_start[b]:
            continue
        gap = t[first[b]] - t[last]
        dist = np.hypot(*(xy[last] - xy[first[b]]).T)
        ok = (
            still_end
            & ~taken
            & (group == group[b])
            & (gap > 0)
            & (gap <= cfg["persistence_max_gap_sec"])
            & (dist <= cfg["persistence_max_dist_m"])
        )
        if ok.any():
            candidates = np.flatnonzero(ok)
            a = candidates[np.lexsort((track_ids[first][candidates], gap[candidates], dist[candidates]))[0]]
            taken[a] = True
            track_obj[b] = track_obj[a]
    for k, (s, e) in enumerate(bounds):
        obj[s:e] = track_obj[k]
    return obj
