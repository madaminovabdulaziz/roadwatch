"""near_miss: Sharp braking or swerving to avoid a collision, with no contact.

Trigger: A pair with TTC < max_ttc_sec and closing speed > min_closing_speed_mps, followed by
evasive action (deceleration > evasive_decel_mps2 or yaw rate > evasive_yaw_rate_dps); separation
stays > min_separation_m and no accident follows within no_accident_within_sec.
Start: onset of evasive action (deceleration first > onset_decel_mps2).
End: separation increasing and TTC > end_ttc_sec.

Details:
- Pairs involve at least one vehicle or two-wheeler; TTC and closing speed come from
  features.pair_features on each processed frame.
- Evasive action is searched on either moving object within evasive_window_sec of the first trigger.
  Its onset walks back from the first evasive sample while the deceleration stays above
  onset_decel_mps2; a pure swerve starts at its first evasive sample.
- "No accident within": the pair's minimum separation over the episode and the following
  no_accident_within_sec must stay above min_separation_m.
Score = min(1, 0.5 + 0.5 * (max_ttc_sec - min TTC) / max_ttc_sec): the closer the call, the surer.

Thresholds: configs/thresholds.yaml -> classes.near_miss.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import group_members, max_gap
from roadwatch.features import pair_features
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "near_miss"
REQUIRED_LAYERS: tuple[str, ...] = ("homography",)


def triggers(tt: pd.DataFrame, p: dict[str, Any]) -> dict[tuple[int, int], tuple[float, float]]:
    """First trigger time and the minimum TTC of every pair that meets the TTC trigger."""
    movers = group_members("vehicles", "two_wheelers")
    road = movers | group_members("persons")
    rows = tt[tt["cls"].astype(str).isin(road)]
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for _, f in rows.groupby("frame", sort=True):
        if len(f) < 2:
            continue
        pairs = pair_features(f)
        hot = pairs[
            (pairs["ttc"] < p["max_ttc_sec"])
            & (pairs["closing_speed"] > p["min_closing_speed_mps"])
            & (pairs["cls_a"].isin(movers) | pairs["cls_b"].isin(movers))
        ]
        t = float(f["t"].iloc[0])
        for r in hot.itertuples():
            key = (int(min(r.track_a, r.track_b)), int(max(r.track_a, r.track_b)))
            first, best = out.get(key, (t, np.inf))
            out[key] = (first, min(best, float(r.ttc)))
    return out


def evasive_onset(rows: pd.DataFrame, t0: float, p: dict[str, Any], gap: float) -> float | None:
    """Onset of the first evasive action in [t0 - window, t0 + window], or None."""
    t = rows["t"].to_numpy()
    decel = -rows["accel"].to_numpy(dtype=np.float64)
    yaw = np.abs(rows["yaw_rate"].to_numpy(dtype=np.float64))
    window = (t >= t0 - p["evasive_window_sec"]) & (t <= t0 + p["evasive_window_sec"])
    hits = np.flatnonzero(window & ((decel > p["evasive_decel_mps2"]) | (yaw > p["evasive_yaw_rate_dps"])))
    if not len(hits):
        return None
    k = int(hits[0])
    while k > 0 and decel[k - 1] > p["onset_decel_mps2"] and t[k] - t[k - 1] <= gap:
        k -= 1
    return float(t[k])


def pair_series(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Joint samples of two tracks: t, distance, closing speed and TTC (inf when not closing)."""
    j = a[["frame", "t", "X", "Y", "vx", "vy"]].merge(
        b[["frame", "X", "Y", "vx", "vy"]], on="frame", suffixes=("_a", "_b")
    )
    rel = j[["X_b", "Y_b"]].to_numpy() - j[["X_a", "Y_a"]].to_numpy()
    vel = j[["vx_b", "vy_b"]].to_numpy() - j[["vx_a", "vy_a"]].to_numpy()
    dist = np.hypot(rel[:, 0], rel[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        closing = np.where(dist > 0, -(rel * vel).sum(axis=1) / dist, 0.0)
        ttc = np.where(closing > 0, dist / closing, np.inf)
    return pd.DataFrame({"t": j["t"].to_numpy(), "dist": dist, "closing": closing, "ttc": ttc})


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    valid = tt[tt["kin_valid"].to_numpy()].sort_values(["track_id", "t"], kind="stable")
    tracks = {int(k): g for k, g in valid.groupby("track_id", sort=True)}
    movers = group_members("vehicles", "two_wheelers")
    gap = max_gap(ctx)
    segments = []
    for (a, b), (t0, ttc_min) in sorted(triggers(valid, p).items(), key=lambda kv: kv[1][0]):
        onsets = [
            onset
            for tid in (a, b)
            if str(tracks[tid]["cls"].iloc[0]) in movers
            for onset in [evasive_onset(tracks[tid], t0, p, gap)]
            if onset is not None
        ]
        if not onsets:
            continue
        start = min(onsets)
        s = pair_series(tracks[a], tracks[b])
        after = s[s["t"] >= t0]
        closest = int(after["dist"].to_numpy().argmin())
        diverging = after.iloc[closest:]
        done = diverging[(diverging["closing"] <= 0) | (diverging["ttc"] > p["end_ttc_sec"])]
        end = float(done["t"].iloc[0]) if len(done) else float(after["t"].iloc[-1])
        check = s[(s["t"] >= start) & (s["t"] <= end + p["no_accident_within_sec"])]
        if check.empty or check["dist"].min() <= p["min_separation_m"]:
            continue  # too close: contact, possibly an accident, not a near miss
        score = float(min(1.0, 0.5 + 0.5 * (p["max_ttc_sec"] - ttc_min) / p["max_ttc_sec"]))
        segments.append(Segment(start, max(end, start), LABEL, score, (a, b)))
    return segments
