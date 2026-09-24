"""congestion: Traffic at a standstill or crawling across all lanes of one direction.

Trigger: Per direction, every step_sec: mean speed < max_mean_speed_mps AND >= min_vehicles_per_lane
vehicles in every lane, sustained >= min_sustain_sec, and either lasting >= long_sec or persisting
>= green_persist_sec into a GREEN phase. A normal red-light queue is not congestion (SPEC §12.1).
Start: the queue stopped moving (backdated to when the condition began).
End: mean speed > clear_speed_mps sustained for clear_hold_sec.

Details:
- A bin's vehicle count per lane is the number of distinct vehicles seen in that lane in the bin; the
  mean speed is over every vehicle sample of the direction in the bin.
- The direction's signal is the first `signal` among its lanes. Without a signal timeline only the
  long_sec criterion can confirm congestion.
- After the trigger, the event lasts until the first clear_hold_sec stretch of bins whose mean speed is
  above clear_speed_mps (it ends where that stretch starts), or until the last bin with vehicles.
Score = 0.5 + 0.5 * min(1, duration / (2 * long_sec)).

Thresholds: configs/thresholds.yaml -> classes.congestion.params (SPEC §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from roadwatch.events.base import VideoContext
from roadwatch.events.common import in_group, lane_signal, runs, signal_states
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment

LABEL = "congestion"
REQUIRED_LAYERS: tuple[str, ...] = ("homography", "directions", "lanes")


def _bins(v: pd.DataFrame, lanes: list[str], step: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Per bin: mean speed (NaN if empty) and the minimum over lanes of distinct vehicles in the lane."""
    b = np.minimum((v["t"].to_numpy() // step).astype(int), n_bins - 1)
    speed = np.full(n_bins, np.nan)
    sums = np.bincount(b, weights=v["speed"].to_numpy(), minlength=n_bins)
    counts = np.bincount(b, minlength=n_bins)
    speed[counts > 0] = sums[counts > 0] / counts[counts > 0]
    per_lane = np.full((len(lanes), n_bins), 0)
    for k, lane in enumerate(lanes):
        rows = v["lane_id"].to_numpy() == lane
        uniq = pd.DataFrame({"b": b[rows], "id": v["track_id"].to_numpy()[rows]}).drop_duplicates()
        per_lane[k] = np.bincount(uniq["b"].to_numpy(), minlength=n_bins)
    return speed, per_lane.min(axis=0)


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
        speed, min_count = _bins(v, lanes, step, n_bins)
        jammed = (speed < p["max_mean_speed_mps"]) & (min_count >= p["min_vehicles_per_lane"])
        clear = (speed > p["clear_speed_mps"]) | np.isnan(speed)  # an empty road has cleared too
        sid = next((signal_of[lane] for lane in lanes if lane in signal_of), None)
        green = signal_states(ctx, sid, bin_t + step / 2) == "green"
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
            into_green = green[a : end_bin + 1].sum() * step
            if duration < p["long_sec"] and into_green < p["green_persist_sec"]:
                continue
            score = 0.5 + 0.5 * min(1.0, duration / (2 * p["long_sec"]))
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
