"""near_miss: Sharp braking or swerving to avoid a collision, with no contact.

Trigger: a conflict in Part B's sense (risk.conflicts, SPEC §12.47: a road user ahead of a mover's front
bumper and in its path, bumper-to-bumper TTC <= risk.max_conflict_ttc_sec and a deceleration needed to
stop short >= risk.drac_min_mps2, judged only where the camera resolves metres) that persists for
risk.conflict_persist_frames processed frames, and the mover reacting within evasive_window_sec with
evasive action: deceleration above evasive_decel_mps2 (read only from a track older than
risk.decel_min_age_sec, and at most risk.decel_cap_mps2: more is a tracker jump) or a yaw rate above
evasive_yaw_rate_dps. Separation stays > min_separation_m and no accident follows within
no_accident_within_sec (SPEC §12.39, §12.57).
Start: onset of evasive action (deceleration first > onset_decel_mps2).
End: separation increasing and TTC > end_ttc_sec.

Details:
- The old trigger, a collision course between box centres (features.pair_features), fired on queues
  closing up, side-by-side lanes and far-away box jitter: 23 false near misses in 7.4 min of the dev
  videos. risk.conflicts is the test Part B uses, which took Part B from 31 false alarms to 2; a near miss
  additionally needs the mover's evasive reaction, which neither of those 2 (a car passing alongside a
  long bus or truck) shows.
- Evasive action is searched on the mover within evasive_window_sec of the conflict. Its onset (the
  start) walks back from the first evasive sample while the deceleration stays above onset_decel_mps2;
  a pure swerve starts at its first evasive sample.
- "No accident within": the pair's minimum separation over the episode and the following
  no_accident_within_sec must stay above min_separation_m.
Score = min(1, 0.5 + 0.5 * (max_ttc_sec - min TTC) / max_ttc_sec): the closer the call, the surer.

Thresholds: configs/thresholds.yaml -> classes.near_miss.params (SPEC §5) and risk (the conflict test).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.config import load_thresholds
from roadwatch.events.base import VideoContext
from roadwatch.events.common import group_members, max_gap
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "near_miss"
REQUIRED_LAYERS: tuple[str, ...] = ("homography",)


def conflict_cfg() -> dict[str, Any]:
    """Part B's conflict settings (risk.*) with the class groups and vehicle sizes risk.conflicts needs."""
    th = load_thresholds()
    groups = th["perception"]["tracker_groups"]
    return dict(th["risk"]) | {
        "groups": {"movers": groups["vehicles"] + groups["two_wheelers"], "persons": groups["persons"]},
        "vehicle_dims": th["kinematics"]["vehicle_dims_m"],
    }


def triggers(
    tt: pd.DataFrame, scene: Scene, rcfg: dict[str, Any]
) -> dict[tuple[int, int], tuple[float, float]]:
    """(first time, minimum TTC) of every (mover, other) pair in a conflict that persists for
    conflict_persist_frames consecutive processed frames."""
    from roadwatch.risk import conflicts  # the Part B module; imported here to keep rule imports light

    road = group_members("vehicles", "two_wheelers", "persons")
    rows = tt[tt["cls"].astype(str).isin(road)]
    need = max(1, int(rcfg["conflict_persist_frames"]))
    streak: dict[
        tuple[int, int], tuple[int, float, float]
    ] = {}  # pair -> (frames in a row, first t, min ttc)
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for _, f in rows.groupby("frame", sort=True):
        t = float(f["t"].iloc[0])
        hot = (
            conflicts(f, scene, rcfg) if len(f) >= 2 else pd.DataFrame(columns=["track_a", "track_b", "ttc"])
        )
        now = {}
        for r in hot.itertuples():
            key = (int(r.track_a), int(r.track_b))
            count, first, best = streak.get(key, (0, t, np.inf))
            now[key] = (count + 1, first, min(best, float(r.ttc)))
            if now[key][0] >= need and key not in out:
                out[key] = (first, now[key][2])
            elif key in out:
                out[key] = (out[key][0], min(out[key][1], float(r.ttc)))
        streak = now
    return out


def evasive_action(
    rows: pd.DataFrame, t0: float, p: dict[str, Any], gap: float
) -> tuple[float, float] | None:
    """(onset, reaction) of the first evasive action in [t0 - window, t0 + window], or None.

    reaction = the first sample past the evasive threshold; onset = where it began (walked back while
    the deceleration stays above onset_decel_mps2).
    """
    t = rows["t"].to_numpy()
    decel = -rows["accel"].to_numpy(dtype=np.float64)
    yaw = np.abs(rows["yaw_rate"].to_numpy(dtype=np.float64))
    window = (t >= t0 - p["evasive_window_sec"]) & (t <= t0 + p["evasive_window_sec"])
    # braking is read only once the track has settled (a new box's first footprints race ahead and stop
    # dead), and never above the cap (a tracker jump between vehicles), as in Part B
    settled = t - t[0] >= p["decel_min_age_sec"]
    braking = settled & (decel > p["evasive_decel_mps2"]) & (decel <= p["decel_cap_mps2"])
    hits = np.flatnonzero(window & (braking | (yaw > p["evasive_yaw_rate_dps"])))
    if not len(hits):
        return None
    k = int(hits[0])
    while k > 0 and decel[k - 1] > p["onset_decel_mps2"] and t[k] - t[k - 1] <= gap:
        k -= 1
    return float(t[k]), float(t[hits[0]])


def pair_series(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Joint samples of two tracks: t, distance, closing speed and constant-velocity TTC (inf when
    not closing)."""
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
    rcfg = conflict_cfg()
    p = cfg["params"] | {k: rcfg[k] for k in ("decel_min_age_sec", "decel_cap_mps2")}
    valid = tt[tt["kin_valid"].to_numpy()].sort_values(["track_id", "t"], kind="stable")
    valid = valid.assign(age=valid["t"] - valid.groupby("track_id")["t"].transform("min"))
    tracks = {int(k): g for k, g in valid.groupby("track_id", sort=True)}
    gap = max_gap(ctx)
    segments = []
    done_pairs: set[tuple[int, int]] = set()
    for (a, b), (t0, ttc_min) in sorted(triggers(valid, scene, rcfg).items(), key=lambda kv: kv[1][0]):
        pair = (min(a, b), max(a, b))
        if pair in done_pairs:
            continue  # both directions of one pair can be in conflict; one near miss
        action = evasive_action(tracks[a], t0, p, gap)  # the mover, who has to avoid the other
        if action is None:
            continue
        s = pair_series(tracks[a], tracks[b])
        start = action[0]
        after = s[s["t"] >= t0]
        closest = int(after["dist"].to_numpy().argmin())
        diverging = after.iloc[closest:]
        done = diverging[(diverging["closing"] <= 0) | (diverging["ttc"] > p["end_ttc_sec"])]
        end = float(done["t"].iloc[0]) if len(done) else float(after["t"].iloc[-1])
        check = s[(s["t"] >= start) & (s["t"] <= end + p["no_accident_within_sec"])]
        if check.empty or check["dist"].min() <= p["min_separation_m"]:
            continue  # too close: contact, possibly an accident, not a near miss
        score = float(min(1.0, 0.5 + 0.5 * (p["max_ttc_sec"] - ttc_min) / p["max_ttc_sec"]))
        segments.append(Segment(start, max(end, start), LABEL, score, pair))
        done_pairs.add(pair)
    return segments
