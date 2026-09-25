"""Traffic-signal state from the lamp boxes in scene.json (RUNBOOK P1.2).

Contract:
- States are "red", "yellow", "green" or "unknown".
- Offline (Part A): `timeline(frames)` returns, per signal id, `[(t_start, t_end, state), ...]`
  covering the frames given, majority-filtered over `signal.median_sec`. The same in two steps:
  `observe(frame, t)` per frame, then `finish()` (Part A feeds frames it decodes anyway).
- Online (Part B): `update(frame, t_sec)` returns the current state per signal id using only frames
  seen so far (causal).
- Frames may be native or downscaled: lamp boxes are scaled from the scene's `image_size`.
- Without lamp boxes, `flow_timeline(kin, scene)` infers the state from traffic at the stop lines
  (vehicles crossing -> green, vehicles waiting while none cross -> red). It is low confidence:
  red_light must never rely on it.

How a lamp is judged lit (SPEC §12.36), in this order:
1. Colour excess (absolute, needs no history): the share of bright, saturated pixels of the lamp's
   colour in its own box, minus the largest share of that colour in its sibling boxes. A lit lamp shows
   its colour in its own box only; a red bus in front of the head shows red in every box (excess ~0),
   so it is not a red light. Red wins over green over yellow (the CIS red+yellow phase is still red).
2. Otherwise the relative brightness: mean V of the box plus mean V of lamp-coloured pixels, normalised
   by that lamp's own range (offline: percentiles over the video; online: running min/max); the best
   lamp past `lit_threshold` counts, but only if its colour excess is at least `fallback_min_excess`.
The absolute test recognises yellow (lit ~5 % of the time, so percentiles never see it "on") and clips
that stay in one state; the relative one covers washed-out colours.
"""

from __future__ import annotations

import bisect
import warnings
from collections import Counter, deque
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
import pandas as pd

from roadwatch.config import load_thresholds
from roadwatch.scene.scene import Scene

SignalTimeline = dict[str, list[tuple[float, float, str]]]
LAMPS = ("red", "yellow", "green")
UNKNOWN = "unknown"


def lamp_measures(
    frame: np.ndarray, scene: Scene, cfg: dict[str, Any] | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per signal id: (raw relative-brightness score, colour excess) of each lamp; NaN where no box."""
    cfg = cfg or load_thresholds()["signal"]
    width, height = scene.layers.get("image_size") or (frame.shape[1], frame.shape[0])
    sx, sy = frame.shape[1] / width, frame.shape[0] / height
    out = {}
    for sig in scene.layers.get("signals") or []:
        crops: list[np.ndarray | None] = []
        for lamp in LAMPS:
            crop = None
            if sig.get(lamp):
                x, y, w, h = sig[lamp]
                x0, y0 = int(np.floor(x * sx)), int(np.floor(y * sy))
                x1, y1 = max(x0 + 1, int(np.ceil((x + w) * sx))), max(y0 + 1, int(np.ceil((y + h) * sy)))
                crop = frame[max(y0, 0) : y1, max(x0, 0) : x1]
                crop = crop if crop.size else None
            crops.append(crop)
        hsv = [cv2.cvtColor(c, cv2.COLOR_BGR2HSV) if c is not None else None for c in crops]
        scores = np.array(
            [_score(h, lamp, cfg) if h is not None else np.nan for h, lamp in zip(hsv, LAMPS, strict=True)]
        )
        # share[c][k]: share of bright pixels of lamp colour c in lamp box k
        share = np.array([[_lit_share(h, c, cfg) if h is not None else np.nan for h in hsv] for c in LAMPS])
        excess = np.full(len(LAMPS), np.nan)
        for k in range(len(LAMPS)):
            if np.isfinite(share[k, k]):
                others = np.delete(share[k], k)
                excess[k] = share[k, k] - (np.nanmax(others) if np.isfinite(others).any() else 0.0)
        out[str(sig["id"])] = (scores, excess)
    return out


def lamp_scores(frame: np.ndarray, scene: Scene, cfg: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Per signal id, the raw relative-brightness score of each lamp (red, yellow, green); NaN if no box."""
    return {sid: scores for sid, (scores, _) in lamp_measures(frame, scene, cfg).items()}


def _colour_mask(hsv: np.ndarray, lamp: str, cfg: dict[str, Any]) -> np.ndarray:
    hue, sat = hsv[..., 0], hsv[..., 1]
    match = np.zeros(hue.shape, dtype=bool)
    for lo, hi in cfg["hue_ranges"][lamp]:
        match |= (hue >= lo) & (hue < hi)
    return match & (sat >= cfg["min_saturation"])


def _score(hsv: np.ndarray, lamp: str, cfg: dict[str, Any]) -> float:
    val = hsv[..., 2].astype(np.float64) / 255
    return float(val.mean() + (val * _colour_mask(hsv, lamp, cfg)).mean())


def _lit_share(hsv: np.ndarray, lamp: str, cfg: dict[str, Any]) -> float:
    bright = hsv[..., 2] >= 255 * cfg["lit_min_value"]
    return float((_colour_mask(hsv, lamp, cfg) & bright).mean())


_PRIORITY = (LAMPS.index("red"), LAMPS.index("green"), LAMPS.index("yellow"))


def _decide(norm: np.ndarray, contrast: np.ndarray, excess: np.ndarray, cfg: dict[str, Any]) -> str:
    """State from one signal's colour excess (absolute) or, failing that, normalised lamp scores."""
    for k in _PRIORITY:
        if np.isfinite(excess[k]) and excess[k] >= cfg["lit_fraction"]:
            return LAMPS[k]
    usable = np.isfinite(norm) & (contrast >= cfg["min_contrast"])
    if not usable.any():
        return UNKNOWN
    k = int(np.nanargmax(np.where(usable, norm, -np.inf)))
    if norm[k] < cfg["lit_threshold"] or not (excess[k] >= cfg["fallback_min_excess"]):
        return UNKNOWN
    return LAMPS[k]


def _mode(states: Iterable[str]) -> str:
    """Most common state; ties go to the earliest of LAMPS order, then unknown (deterministic)."""
    counts = Counter(states)
    return max(counts, key=lambda s: (counts[s], -(LAMPS + (UNKNOWN,)).index(s)))


def _segments(times: list[float], states: list[str]) -> list[tuple[float, float, str]]:
    """Runs of equal states; a run lasts from its first sample to the next run's first sample."""
    segs: list[tuple[float, float, str]] = []
    for t, s in zip(times, states, strict=True):
        if segs and segs[-1][2] == s:
            continue
        if segs:
            segs[-1] = (segs[-1][0], t, segs[-1][2])
        segs.append((t, t, s))
    if segs:
        segs[-1] = (segs[-1][0], times[-1], segs[-1][2])
    return segs


def state_at(segments: list[tuple[float, float, str]], t: float) -> str:
    """State of one signal at time t (the segment starting at or before t), or unknown outside."""
    starts = [s[0] for s in segments]
    i = bisect.bisect_right(starts, t) - 1
    if i < 0 or t > segments[-1][1]:
        return UNKNOWN
    return segments[i][2]


class SignalStateEstimator:
    """Per-signal lamp brightness -> state (see the module docstring)."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.cfg = load_thresholds()["signal"]
        self.ids = [str(sig["id"]) for sig in scene.layers.get("signals") or []]
        self._lo: dict[str, np.ndarray] = {}
        self._hi: dict[str, np.ndarray] = {}
        self._recent: dict[str, deque[tuple[float, str]]] = {sid: deque() for sid in self.ids}
        self._times: list[float] = []
        self._raw: dict[str, list[np.ndarray]] = {sid: [] for sid in self.ids}
        self._excess: dict[str, list[np.ndarray]] = {sid: [] for sid in self.ids}

    def timeline(self, frames: Iterable[tuple[int, float, np.ndarray]]) -> SignalTimeline:
        """Offline timeline from frames in time order (any stride), majority-filtered, per signal id."""
        for _, t, frame in frames:
            self.observe(frame, t)
        return self.finish()

    def observe(self, frame: np.ndarray, t_sec: float) -> None:
        """Record one frame's lamp scores for `finish()` (lets Part A feed frames it decodes anyway)."""
        self._times.append(float(t_sec))
        for sid, (scores, excess) in lamp_measures(frame, self.scene, self.cfg).items():
            self._raw[sid].append(scores)
            self._excess[sid].append(excess)

    def finish(self) -> SignalTimeline:
        """Timeline of every frame observed so far (offline: percentiles over the whole video)."""
        times, raw = self._times, self._raw
        if not times:
            return {sid: [] for sid in self.ids}

        p_lo, p_hi = self.cfg["baseline_percentiles"]
        half = self.cfg["median_sec"] / 2
        t_arr = np.asarray(times)
        out: SignalTimeline = {}
        for sid in self.ids:
            scores = np.asarray(raw[sid])  # (n_frames, 3)
            with warnings.catch_warnings():  # a lamp without a box is an all-NaN column
                warnings.simplefilter("ignore", RuntimeWarning)
                lo = np.nanpercentile(scores, p_lo, axis=0)
                hi = np.nanpercentile(scores, p_hi, axis=0)
            contrast = hi - lo
            norm = (scores - lo) / np.where(contrast > 0, contrast, np.inf)
            excess = np.asarray(self._excess[sid])
            states = [_decide(row, contrast, ex, self.cfg) for row, ex in zip(norm, excess, strict=True)]
            lo_idx = np.searchsorted(t_arr, t_arr - half, side="left")
            hi_idx = np.searchsorted(t_arr, t_arr + half, side="right")
            smooth = [_mode(states[a:b]) for a, b in zip(lo_idx, hi_idx, strict=True)]
            out[sid] = _segments(times, smooth)
        return out

    def update(self, frame: np.ndarray, t_sec: float) -> dict[str, str]:
        """Causal state per signal id: running per-lamp range, trailing majority filter."""
        states = {}
        for sid, (scores, excess) in lamp_measures(frame, self.scene, self.cfg).items():
            if sid not in self._lo:
                self._lo[sid], self._hi[sid] = scores.copy(), scores.copy()
            else:
                self._lo[sid] = np.fmin(self._lo[sid], scores)
                self._hi[sid] = np.fmax(self._hi[sid], scores)
            contrast = self._hi[sid] - self._lo[sid]
            norm = (scores - self._lo[sid]) / np.where(contrast > 0, contrast, np.inf)
            recent = self._recent[sid]
            recent.append((t_sec, _decide(norm, contrast, excess, self.cfg)))
            while recent and recent[0][0] < t_sec - self.cfg["median_sec"]:
                recent.popleft()
            states[sid] = _mode(s for _, s in recent)
        return states


def flow_timeline(kin: pd.DataFrame, scene: Scene) -> SignalTimeline:
    """Low-confidence signal timeline from traffic at each signal's stop lines (no lamp boxes needed).

    Per `signal.flow.bin_sec` bin: a vehicle's front point crossing a stop line of the signal -> green;
    otherwise a vehicle stopped within `queue_m` upstream in its lanes while another signal's lines
    are being crossed -> red; otherwise unknown. Needs `add_kinematics` output.
    """
    cfg = load_thresholds()["signal"]["flow"]
    vehicles = set(load_thresholds()["perception"]["tracker_groups"]["vehicles"])
    kin = kin[kin["cls"].astype(str).isin(vehicles)].sort_values(["track_id", "t"], kind="stable")
    lines_by_signal: dict[str, list[dict[str, Any]]] = {}
    for sl in scene.layers.get("stop_lines") or []:
        if sl.get("signal"):
            lines_by_signal.setdefault(str(sl["signal"]), []).append(sl)
    if kin.empty or not lines_by_signal:
        return {sid: [] for sid in lines_by_signal}

    t0, t1 = float(kin["t"].min()), float(kin["t"].max())
    edges = np.arange(t0, t1 + cfg["bin_sec"], cfg["bin_sec"])
    n_bins = max(len(edges) - 1, 1)
    front = kin[["front_x", "front_y"]].to_numpy(dtype=np.float64)
    same_track = kin["track_id"].to_numpy()[1:] == kin["track_id"].to_numpy()[:-1]
    t_step = kin["t"].to_numpy()[1:]
    bins_of_step = np.clip(np.searchsorted(edges, t_step, side="right") - 1, 0, n_bins - 1)

    crossing = {}
    waiting = {}
    bins_of_row = np.clip(np.searchsorted(edges, kin["t"].to_numpy(), side="right") - 1, 0, n_bins - 1)
    for sid, lines in lines_by_signal.items():
        crossed = np.zeros(n_bins, dtype=bool)
        queued = np.zeros(n_bins, dtype=bool)
        for sl in lines:
            hit = same_track & scene.crosses(np.asarray(sl["line"]), front[:-1], front[1:])
            crossed[np.unique(bins_of_step[hit])] = True
            if scene.has("homography"):
                mid = scene.to_world(np.asarray(sl["line"], dtype=np.float64).mean(axis=0, keepdims=True))[0]
                dist = np.hypot(kin["X"].to_numpy() - mid[0], kin["Y"].to_numpy() - mid[1])
                in_lanes = kin["lane_id"].isin(sl.get("lanes", [])).to_numpy()
                stopped = kin["speed"].to_numpy() <= cfg["stopped_speed_mps"]
                queued[np.unique(bins_of_row[in_lanes & stopped & (dist <= cfg["queue_m"])])] = True
        crossing[sid], waiting[sid] = crossed, queued

    out: SignalTimeline = {}
    bin_start = [float(e) for e in edges[:n_bins]]
    for sid in lines_by_signal:
        others = np.zeros(n_bins, dtype=bool)
        for other, crossed in crossing.items():
            if other != sid:
                others |= crossed
        states = np.where(crossing[sid], "green", np.where(waiting[sid] & others, "red", UNKNOWN))
        segs = _segments(bin_start + [t1], list(states) + [states[-1]])
        out[sid] = segs
    return out
