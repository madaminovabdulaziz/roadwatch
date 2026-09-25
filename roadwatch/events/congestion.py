"""congestion: Traffic at a standstill or crawling across all lanes of one direction.

Trigger: per direction, a bin (step_sec) is jammed when the mean speed is < max_mean_speed_mps, every
lane holds >= min_vehicles_per_lane vehicles, and no vehicle moves faster than clear_speed_mps (a queue
with cars accelerating away through it is discharging, not jammed). A jam sustained >= min_sustain_sec
is congestion if it stays jammed for >= green_persist_sec of GREEN. Without a signal state (unknown
throughout), a jam and a long red look alike, so only a standstill of >= long_sec_unknown_signal counts
(SPEC §12.41). A normal red-light queue is never congestion (SPEC §12.1).
Start: the queue stopped moving (backdated to when the condition began).
End: mean speed > clear_speed_mps sustained for clear_hold_sec.

Details:
- A bin's vehicle count per lane is the number of distinct vehicles seen in that lane in the bin; the
  mean speed is over every vehicle sample of the direction in the bin.
- The direction's signal is the first `signal` among its lanes. Only green bins that are still jammed
  count: a normal discharge, whose waiting tail keeps the mean speed low for 10+ s of green at this
  junction (approach lanes end at the stop line, so departing cars leave them), must not qualify.
- After the trigger, the event lasts until the first clear_hold_sec stretch of bins whose mean speed is
  above clear_speed_mps (it ends where that stretch starts), or until the last bin with vehicles.
Score = 0.5 + 0.5 * min(1, duration / (2 * long_sec_unknown_signal)).

Thresholds: configs/thresholds.yaml -> classes.congestion.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import in_group, lane_signal, runs, signal_states
from roadwatch.scene.light import UNKNOWN
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "congestion"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "directions", "lanes")


def _bins(
    v: pd.DataFrame, lanes: list[str], step: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per bin: mean speed (NaN if empty), the fastest vehicle's speed, and the minimum over lanes of
    distinct vehicles in the lane."""
    b = np.minimum((v["t"].to_numpy() // step).astype(int), n_bins - 1)
    speeds = v["speed"].to_numpy(dtype=np.float64)
    speed = np.full(n_bins, np.nan)
    sums = np.bincount(b, weights=speeds, minlength=n_bins)
    counts = np.bincount(b, minlength=n_bins)
    speed[counts > 0] = sums[counts > 0] / counts[counts > 0]
    fastest = np.zeros(n_bins)
    np.maximum.at(fastest, b, speeds)
    per_lane = np.full((len(lanes), n_bins), 0)
    for k, lane in enumerate(lanes):
        rows = v["lane_id"].to_numpy() == lane
        uniq = pd.DataFrame({"b": b[rows], "id": v["track_id"].to_numpy()[rows]}).drop_duplicates()
        per_lane[k] = np.bincount(uniq["b"].to_numpy(), minlength=n_bins)
    return speed, fastest, per_lane.min(axis=0)


def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict[str, Any]) -> list[Segment]:
    if tt.empty:
        return []
    p = cfg["params"]
    step = p["step_sec"]
    n_bins = max(1, int(np.ceil(ctx.meta.duration / step)))
    bin_t = np.arange(n_bins) * step
    vehicles = tt[in_group(tt, "vehicles") & tt["kin_valid"].to_numpy()]
    signal_of = lane_signal(scene)

    segments = []
    for direction in scene.layers.get("directions") or []:
        lanes = [str(x) for x in direction.get("lanes", [])]
        v = vehicles[vehicles["lane_id"].isin(lanes)]
        if v.empty or not lanes:
            continue
        speed, fastest, min_count = _bins(v, lanes, step, n_bins)
        jammed = (
            (speed < p["max_mean_speed_mps"])
            & (min_count >= p["min_vehicles_per_lane"])
            & (fastest <= p["clear_speed_mps"])
        )
        clear = (speed > p["clear_speed_mps"]) | np.isnan(speed)  # an empty road has cleared too
        sid = next((signal_of[lane] for lane in lanes if lane in signal_of), None)
        states = signal_states(ctx, sid, bin_t + step / 2)
        green = states == "green"
        present = np.flatnonzero(~np.isnan(speed))

        for a, b in runs(bin_t, jammed, step * 1.5):
            if (b - a + 1) * step < p["min_sustain_sec"]:
                continue
            # the event continues until traffic clears for clear_hold_sec
            end_bin = present[-1]
            hold = int(np.ceil(p["clear_hold_sec"] / step))
            for k in range(b + 1, n_bins):
                if clear[k : k + hold].all() and k + hold <= n_bins:
                    end_bin = k - 1
                    break
            if any(s.meta.get("direction") == direction["id"] and s.end >= bin_t[a] for s in segments):
                continue  # already inside an event of this direction
            duration = (end_bin - a + 1) * step
            into_green = (green & jammed)[a : end_bin + 1].sum() * step
            signal_known = bool((states[a : end_bin + 1] != UNKNOWN).any())
            if signal_known and into_green < p["green_persist_sec"]:
                continue  # it only waited for red
            if not signal_known and duration < p["long_sec_unknown_signal"]:
                continue  # could be a long red phase
            score = 0.5 + 0.5 * min(1.0, duration / (2 * p["long_sec_unknown_signal"]))
            segments.append(
                Segment(
                    float(bin_t[a]),
                    float(bin_t[end_bin] + step),
                    LABEL,
                    score,
                    (),
                    {"direction": direction["id"]},
                )
            )
    return segments
