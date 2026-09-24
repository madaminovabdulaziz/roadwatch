"""Per-track kinematics in metres and pairwise features (SPEC §3, RUNBOOK P1.3).

Contract:
- `add_kinematics(tt, scene, mode)` returns the TrackTable with footprints mapped to metres (`X, Y`)
  and `vx, vy, speed, accel, heading_deg, yaw_rate, kin_valid, lane_id, on_road, in_crosswalk,
  in_intersection, in_no_uturn, in_parking` plus the front point (SPEC §12.4).
  mode="offline" smooths with Savitzky-Golay (Part A); mode="online" uses causal EMA (Part B).
  Tracks shorter than `kinematics.min_track_sec` get `kin_valid = False`. Stationary tracks broken
  by id switches are merged by position persistence.
- `pair_features(frame_rows)` returns converging pairs within `risk.max_pair_dist_m` with distance,
  closing speed and time-to-collision.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from roadwatch.scene.scene import Scene


def add_kinematics(
    tt: pd.DataFrame, scene: Scene, mode: Literal["offline", "online"] = "offline"
) -> pd.DataFrame:
    raise NotImplementedError("Kinematics land in RUNBOOK P1.3")


def pair_features(frame_rows: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Pair features land in RUNBOOK P1.3")
