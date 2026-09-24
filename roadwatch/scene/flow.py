"""Image-space traffic flow from tracks: lane directions and a grid flow field (RUNBOOK P1.2).

Contract:
- `step_velocities(tt)` gives one row per consecutive pair of rows of the same vehicle track: the step
  midpoint (px) and its image velocity (px/s). Only the `vehicles` and `two_wheelers` tracker groups.
- `lane_directions_from_tracks(tt, scene)` proposes each lane's direction as the mean unit vector of
  the moving steps whose midpoint lies in that lane, with `consistency` = length of that mean (1 = all
  steps agree). Lanes with too few steps or too little agreement get `ok = False`.
- `flow_field(tt, width, height)` averages step velocities over a `lane_flow.cell_px` grid (EDA).
Pixel thresholds are in native-resolution pixels (`configs/thresholds.yaml` -> `lane_flow`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.config import load_thresholds
from roadwatch.scene.scene import Scene


def step_velocities(tt: pd.DataFrame) -> pd.DataFrame:
    """Midpoint and image velocity of every step between consecutive rows of a road-vehicle track."""
    groups = load_thresholds()["perception"]["tracker_groups"]
    road_users = set(groups["vehicles"]) | set(groups["two_wheelers"])
    t = tt[tt["cls"].astype(str).isin(road_users)].sort_values(["track_id", "t"], kind="stable")
    tid, time = t["track_id"].to_numpy(), t["t"].to_numpy(dtype=np.float64)
    xy = t[["fx", "fy"]].to_numpy(dtype=np.float64)
    dt = np.diff(time)
    ok = (tid[1:] == tid[:-1]) & (dt > 0)
    vel = np.diff(xy, axis=0)[ok] / dt[ok, None]
    mid = (xy[1:] + xy[:-1])[ok] / 2
    return pd.DataFrame(
        {
            "track_id": tid[1:][ok],
            "t": time[1:][ok],
            "x": mid[:, 0],
            "y": mid[:, 1],
            "vx": vel[:, 0],
            "vy": vel[:, 1],
        }
    )


def _moving_units(steps: pd.DataFrame, min_speed: float) -> tuple[pd.DataFrame, np.ndarray]:
    speed = np.hypot(steps["vx"].to_numpy(), steps["vy"].to_numpy())
    moving = steps[speed >= min_speed]
    units = moving[["vx", "vy"]].to_numpy() / speed[speed >= min_speed, None]
    return moving, units


def lane_directions_from_tracks(tt: pd.DataFrame, scene: Scene) -> pd.DataFrame:
    """Per lane: steps used, proposed unit direction (dx, dy), consistency and whether it is usable."""
    cfg = load_thresholds()["lane_flow"]
    moving, units = _moving_units(step_velocities(tt), cfg["min_speed_px_s"])
    lanes = scene.lane_of(moving[["x", "y"]].to_numpy()) if len(moving) else np.array([], dtype=object)
    rows = []
    for lane_id, _ in scene.polygons("lanes"):
        u = units[lanes == lane_id]
        mean = u.mean(axis=0) if len(u) else np.array([np.nan, np.nan])
        r = float(np.hypot(*mean)) if len(u) else 0.0
        d = mean / r if r > 0 else np.array([np.nan, np.nan])
        rows.append(
            {
                "lane_id": lane_id,
                "n": len(u),
                "dx": float(d[0]),
                "dy": float(d[1]),
                "consistency": r,
                "ok": len(u) >= cfg["min_samples"] and r >= cfg["min_consistency"],
            }
        )
    return pd.DataFrame(rows, columns=["lane_id", "n", "dx", "dy", "consistency", "ok"])


def flow_field(tt: pd.DataFrame, width: int, height: int) -> dict[str, Any]:
    """Grid of mean image velocities: {"cell_px", "width", "height", "cells": [{x, y, vx, vy, n}]}."""
    cfg = load_thresholds()["lane_flow"]
    cell = cfg["cell_px"]
    moving, _ = _moving_units(step_velocities(tt), cfg["min_speed_px_s"])
    gx = np.clip((moving["x"].to_numpy() // cell).astype(int), 0, max(width // cell, 1) - 1)
    gy = np.clip((moving["y"].to_numpy() // cell).astype(int), 0, max(height // cell, 1) - 1)
    agg = (
        pd.DataFrame({"gx": gx, "gy": gy, "vx": moving["vx"].to_numpy(), "vy": moving["vy"].to_numpy()})
        .groupby(["gx", "gy"], sort=True)
        .agg(vx=("vx", "mean"), vy=("vy", "mean"), n=("vx", "size"))
        .reset_index()
    )
    agg = agg[agg["n"] >= cfg["min_cell_samples"]]
    cells = [
        {
            "x": int(r.gx * cell + cell // 2),
            "y": int(r.gy * cell + cell // 2),
            "vx": round(float(r.vx), 1),
            "vy": round(float(r.vy), 1),
            "n": int(r.n),
        }
        for r in agg.itertuples()
    ]
    return {"cell_px": cell, "width": int(width), "height": int(height), "cells": cells}
