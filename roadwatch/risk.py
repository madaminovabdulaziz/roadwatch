"""Part B core: causal accident risk, used by `solution.RiskEstimator` (SPEC §8, RUNBOOK P2.2).

Contract:
- `RiskCore()` then `reset(meta)` once per video; `step(frame, t_sec)` is called for every frame in
  order and returns P(an accident starts within 5 s) in [0, 1].
- Causal by construction: it sees only frames passed to `step()`. It never opens the video file,
  never reads cache/ and never reuses Part A output (SPEC §12.3). Config, scene and weights are read in
  `reset()`; `step()` touches no file.
- Heavy work runs when at least `risk.stride` frames passed since the last processed one (frame index
  = round(t * fps), so the dev flag `--risk-stride` works too); other frames return the previous score.
- Time safety (SPEC §12.33): if the wall time spent inside step() exceeds
  `risk.budget_factor * t_sec + risk.budget_slack_sec`, processing slots are skipped (previous score)
  until it is back under. Over the 3x budget the whole video would score empty, Part A included; on a
  machine fast enough this never triggers, so normal runs stay deterministic.

The scene is aligned to the video on the first processed frames (OnlineRegistration; until then, and if
it never fits confidently, the scene is used as clicked).
Per processed frame: resize to the detector size -> detect -> own OnlineTracker -> causal kinematics
(EMA, the same update as features' online mode) -> features (`risk_features`) -> score (`risk_score`):
    z = b + w1 g(ttc_min) + w2 clip(closing/10) + w3 clip(decel_max/8) + w4 wrong_way + w5 red_light
        + w6 g(ped_ttc),  g(ttc) = exp(-ttc / ttc_scale_sec)  (0 when there is no converging pair)
    score = EMA(sigmoid(z), ema_alpha); once >= threshold it is held there for hold_sec.
Without a homography there are no metric features: the score is the constant sigmoid(b) and the detector
is never loaded (no GPU time, no model-load cost against the first video's budget).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd

from roadwatch import budget
from roadwatch.config import load_thresholds
from roadwatch.events.common import lane_world_dirs
from roadwatch.features import pair_features
from roadwatch.scene.light import SignalStateEstimator
from roadwatch.scene.registration import OnlineRegistration
from roadwatch.scene.scene import Scene
from roadwatch.types import FrameTracks

_FORGET_SEC = 2.0  # a track unseen this long is dropped from the online kinematics


@dataclass
class _TrackState:
    first_t: float
    samples: deque = field(default_factory=deque)  # (t, x, y) inside the fit window
    vels: deque = field(default_factory=deque)  # (t, vx, vy) inside the acceleration window
    last_lane: str = ""


class OnlineKinematics:
    """Causal per-track velocity and acceleration in metres, robust to detector jitter (SPEC §12.40).

    Velocity is the least-squares slope of the positions seen in the last `vel_window_sec`, and
    acceleration the least-squares slope of those velocities over `acc_window_sec`. Unlike stacked
    EMAs of finite differences, a line fit over ~8 samples averages jitter out instead of amplifying
    it (0.3 m of footprint jitter made a parked queue look like a collision course). A track needs
    `min_fit_samples` spanning `min_fit_span_sec` before it has a velocity (NaN until then: it is left
    out of pair features). Rows flagged at the frame edge are skipped: their clipped boxes freeze the
    footprint while the vehicle still moves.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.tracks: dict[int, _TrackState] = {}

    def update(
        self, t: float, ids: np.ndarray, world: np.ndarray, at_edge: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Velocities, accelerations (N, 2; NaN while unknown) and track ages (N,) for this frame."""
        c = self.cfg
        n = len(ids)
        vel, acc, age = np.full((n, 2), np.nan), np.full((n, 2), np.nan), np.zeros(n)
        edge = np.zeros(n, dtype=bool) if at_edge is None else at_edge
        for i, (tid, p) in enumerate(zip(ids.tolist(), world, strict=True)):
            s = self.tracks.get(tid)
            if s is None or (s.samples and t - s.samples[-1][0] > _FORGET_SEC):
                s = self.tracks[tid] = _TrackState(t)
            age[i] = t - s.first_t
            if edge[i] or not np.isfinite(p).all():
                continue
            s.samples.append((t, float(p[0]), float(p[1])))
            while s.samples and s.samples[0][0] < t - c["vel_window_sec"]:
                s.samples.popleft()
            v = _slope(s.samples, c)
            if v is None:
                continue
            vel[i] = v
            s.vels.append((t, float(v[0]), float(v[1])))
            while s.vels and s.vels[0][0] < t - c["acc_window_sec"]:
                s.vels.popleft()
            a = _slope(s.vels, c)
            acc[i] = a if a is not None else 0.0
        for tid in [k for k, s in self.tracks.items() if s.samples and t - s.samples[-1][0] > _FORGET_SEC]:
            del self.tracks[tid]
        return vel, acc, age

    def remember_lane(self, tid: int, lane: str) -> str:
        """Record the lane a track is in (if any) and return the last lane it was seen in."""
        s = self.tracks.get(tid)
        if s is None:
            return lane
        if lane:
            s.last_lane = lane
        return s.last_lane


def _slope(samples: deque, c: dict[str, Any]) -> np.ndarray | None:
    """Least-squares d/dt of (t, x, y) samples, or None with too few or too short a span."""
    if len(samples) < c["min_fit_samples"]:
        return None
    arr = np.asarray(samples, dtype=np.float64)
    t = arr[:, 0] - arr[:, 0].mean()
    span = arr[-1, 0] - arr[0, 0]
    if span < c["min_fit_span_sec"]:
        return None
    denom = float((t * t).sum())
    return (t[:, None] * (arr[:, 1:] - arr[:, 1:].mean(axis=0))).sum(axis=0) / denom


def risk_features(
    rows: pd.DataFrame, scene: Scene, red_lanes: set[str], cfg: dict[str, Any]
) -> dict[str, float]:
    """SPEC §8 features of one frame (with the §12.40 gates).

    `rows`: track_id, cls, X, Y, vx, vy, ax, ay (metres; NaN while a track has no velocity yet), fx, fy
    (px), and optionally `age` (seconds since the track appeared) and `last_lane` (the last lane it was
    seen in). Gates that keep normal traffic calm:
    - a pair counts only if its paths meet (features.pair_features), one of them moves at
      >= min_mover_speed_mps and both tracks are >= min_track_age_sec old (fresh tracks have no
      trustworthy velocity);
    - deceleration counts only above decel_floor_mps2: everyday braking is not a warning sign;
    - red_light = a moving vehicle past the stop line (in the intersection) that came from a lane whose
      signal is red. A car queueing or approaching inside its lane is legal.
    """
    groups = cfg["groups"]
    feats = {
        "ttc_min": math.inf,
        "closing_speed": 0.0,
        "decel_max": 0.0,
        "wrong_way": 0.0,
        "red_light": 0.0,
        "ped_ttc": math.inf,
    }
    if rows.empty:
        return feats
    cls = rows["cls"].astype(str).to_numpy()
    movers = np.isin(cls, groups["movers"])
    vel = rows[["vx", "vy"]].to_numpy(dtype=np.float64)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    known = np.isfinite(speed)
    acc = np.nan_to_num(rows[["ax", "ay"]].to_numpy(dtype=np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        along = np.where(known & (speed > 0.5), (acc * vel).sum(axis=1) / speed, 0.0)
    if (movers & known).any():
        feats["decel_max"] = float(max(0.0, -along[movers & known].min() - cfg["decel_floor_mps2"]))

    pairs = pair_features(rows)
    if len(pairs):
        tid = rows["track_id"].to_numpy()
        age = rows["age"].to_numpy(dtype=np.float64) if "age" in rows else np.full(len(rows), np.inf)
        mature = set(tid[age >= cfg["min_track_age_sec"]])
        moving = set(tid[movers & known & (speed >= cfg["min_mover_speed_mps"])])
        ok = (
            pairs["track_a"].isin(mature)
            & pairs["track_b"].isin(mature)
            & (pairs["track_a"].isin(moving) | pairs["track_b"].isin(moving))
        )
        pairs = pairs[ok]
        involve = pairs["cls_a"].isin(groups["movers"]) | pairs["cls_b"].isin(groups["movers"])
        person = pairs["cls_a"].isin(groups["persons"]) | pairs["cls_b"].isin(groups["persons"])
        road_ids = set(tid[scene.point_in("carriageway", rows[["fx", "fy"]].to_numpy())])
        on_road = pairs["track_a"].isin(road_ids) & pairs["track_b"].isin(road_ids)
        vv = pairs[involve & ~person]
        if len(vv):
            feats["ttc_min"] = float(vv["ttc"].iloc[0])
            feats["closing_speed"] = float(vv["closing_speed"].iloc[0])
        vp = pairs[involve & person & on_road]
        if len(vp):
            feats["ped_ttc"] = float(vp["ttc"].min())

    fast = movers & known & (speed > cfg["wrong_way_speed_mps"])
    if scene.has("lanes") and fast.any():
        foot = rows[["fx", "fy"]].to_numpy(dtype=np.float64)
        lanes = scene.lane_of(foot).astype(str)
        lane_dir = lane_world_dirs(rows.assign(lane_id=lanes), scene)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (vel * lane_dir).sum(axis=1) / speed
        feats["wrong_way"] = float((fast & (cos < cfg["wrong_way_max_cos"])).any())
        if red_lanes and scene.has("intersection"):
            came_from = rows["last_lane"].astype(str).to_numpy() if "last_lane" in rows else lanes
            past_line = scene.point_in("intersection", foot)
            feats["red_light"] = float((fast & past_line & np.isin(came_from, list(red_lanes))).any())
    return feats


def risk_score(feats: dict[str, float], cfg: dict[str, Any]) -> float:
    """sigmoid(z) of SPEC §8 (before smoothing)."""
    w = cfg["weights"]

    def g(ttc: float) -> float:
        return math.exp(-ttc / cfg["ttc_scale_sec"]) if math.isfinite(ttc) else 0.0

    z = (
        w["bias"]
        + w["ttc"] * g(feats["ttc_min"])
        + w["closing_speed"] * min(1.0, feats["closing_speed"] / cfg["closing_speed_norm_mps"])
        + w["decel"] * min(1.0, feats["decel_max"] / cfg["decel_norm_mps2"])
        + w["wrong_way"] * feats["wrong_way"]
        + w["red_light"] * feats["red_light"]
        + w["ped_ttc"] * g(feats["ped_ttc"])
    )
    return 1.0 / (1.0 + math.exp(-z))


class RiskCore:
    """Online perception + TTC features -> risk score (see the module docstring)."""

    def __init__(self, detector: Any = None, scene: Scene | None = None) -> None:
        """`detector`/`scene` default to the shared YOLO detector and configs/scene.json (tests inject)."""
        self._detector = detector
        self._scene = scene
        self.meta: dict[str, Any] = {}
        self.score = 0.0

    def reset(self, meta: dict[str, Any]) -> None:
        cfg = load_thresholds()
        self.cfg = dict(cfg["risk"])
        groups = cfg["perception"]["tracker_groups"]
        self.cfg["groups"] = {
            "movers": groups["vehicles"] + groups["two_wheelers"],
            "persons": groups["persons"],
        }
        self.cfg["wrong_way_speed_mps"] = cfg["classes"]["wrong_way"]["params"]["min_speed_mps"]
        self.cfg["wrong_way_max_cos"] = cfg["classes"]["wrong_way"]["params"]["max_cos_to_lane"]
        self.names = {int(k): v for k, v in cfg["perception"]["keep_classes"].items()}
        self.meta = dict(meta)
        self.fps = float(meta.get("fps") or 25.0)
        self.native = (int(meta["width"]), int(meta["height"]))
        self.scene = self._scene if self._scene is not None else Scene.load()
        self.metric = self.scene.has("homography")
        self.last_idx = -(10**9)
        self.ema: float | None = None
        self.hold_until = -math.inf
        self.score = 0.0
        self.spent = 0.0  # wall seconds inside step()'s processing
        self.skipped = 0
        # the video's real deadline: Part A marked its start (SPEC §12.40); else count from now
        self.reset_at = time.perf_counter()
        self.duration = float(meta.get("n_frames") or 0) / self.fps
        start = budget.started(str(meta.get("video_id", "")))
        self.deadline = (start if start is not None else self.reset_at) + (
            self.cfg["total_budget_factor"] * self.duration
        )
        if not self.metric:
            # no metric features are possible: the score is the constant bias, so skip perception
            self.score = risk_score(risk_features(pd.DataFrame(), self.scene, set(), self.cfg), self.cfg)
            return
        # torch loads only when perception can matter (its import alone costs seconds of the budget)
        from roadwatch.perception.detector import Detector
        from roadwatch.perception.tracker import OnlineTracker

        self.detector = self._detector or Detector.load()
        self.tracker = OnlineTracker(self.fps / self.cfg["stride"])
        self.kin = OnlineKinematics(self.cfg)
        self.edge_margin = cfg["kinematics"]["edge_margin_frac"]
        self.signals = SignalStateEstimator(self.scene) if self.scene.has("signals") else None
        self.signal_of = {
            str(lane["id"]): str(lane["signal"])
            for lane in self.scene.layers.get("lanes") or []
            if lane.get("signal")
        }
        self.det_size = self.detector.frame_size_for(*self.native)
        # the scene is aligned to this video from the frames step() receives (causal, SPEC §12.34)
        self.registration = OnlineRegistration(self.scene, self.native)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        idx = int(round(t_sec * self.fps))
        if not self.metric or idx - self.last_idx < self.cfg["stride"]:
            return self.score
        self.last_idx = idx
        started = time.perf_counter()
        if self._behind(started, t_sec):
            self.skipped += 1
            return self.score  # keep the budget for the harness's decoding; over 3x empties the video
        started = time.perf_counter()
        try:
            return self._process(frame, t_sec, idx)
        finally:
            self.spent += time.perf_counter() - started

    def _behind(self, now: float, t_sec: float) -> bool:
        """Whether to skip this slot's processing to finish inside the video's budget.

        Two checks, both idle on a fast enough machine (so normal runs stay deterministic):
        - projected finish: Part B's wall time per video second so far (the harness's decoding plus our
          processing) projected to the end of the video must land before the deadline, the Part A
          start + total_budget_factor x duration;
        - our processing alone stays under budget_factor x video time + slack.
        """
        if self.spent > self.cfg["budget_factor"] * t_sec + self.cfg["budget_slack_sec"]:
            return True
        if t_sec < self.cfg["pace_after_sec"]:
            return False
        rate = (now - self.reset_at) / t_sec
        return now + (self.duration - t_sec) * rate > self.deadline

    def _use_scene(self, scene: Scene) -> None:
        """Switch to the scene aligned to this video.

        Positions in metres change frame (a 100 px shift is 1-2 m), so the online kinematics restart
        instead of reading the jump as motion, and the lamp reader restarts on the moved boxes (its
        running per-lamp range was measured on the old ones).
        """
        self.scene = scene
        self.kin = OnlineKinematics(self.cfg)
        if self.signals is not None:
            self.signals = SignalStateEstimator(scene)

    def _process(self, frame: np.ndarray, t_sec: float, idx: int) -> float:
        if not self.registration.done:
            aligned = self.registration.offer(frame, t_sec)
            if aligned is not None:
                self._use_scene(aligned)
        small = cv2.resize(frame, self.det_size, interpolation=cv2.INTER_AREA)
        dets = self.detector.predict([(idx, t_sec, small)], native_size=self.native)[0]
        tracks: FrameTracks = self.tracker.update(dets)
        red_lanes: set[str] = set()
        if self.signals is not None:
            states = self.signals.update(frame, t_sec)
            red_lanes = {lane for lane, sid in self.signal_of.items() if states.get(sid) == "red"}
        raw = risk_score(self._features(tracks, t_sec, red_lanes), self.cfg)
        a = self.cfg["ema_alpha"]
        self.ema = raw if self.ema is None else a * raw + (1 - a) * self.ema
        out = self.ema
        if out >= self.cfg["threshold"]:
            self.hold_until = t_sec + self.cfg["hold_sec"]
        elif t_sec < self.hold_until:
            out = self.cfg["threshold"]
        self.score = float(min(1.0, max(0.0, out)))
        return self.score

    def _features(self, tracks: FrameTracks, t: float, red_lanes: set[str]) -> dict[str, float]:
        if not self.metric or not len(tracks.track_id):
            return risk_features(pd.DataFrame(), self.scene, red_lanes, self.cfg)
        foot = np.stack([(tracks.xyxy[:, 0] + tracks.xyxy[:, 2]) / 2, tracks.xyxy[:, 3]], axis=1).astype(
            np.float64
        )
        world = self.scene.to_world(foot)
        w, h = self.native
        mx, my = self.edge_margin * w, self.edge_margin * h
        box = tracks.xyxy.astype(np.float64)
        at_edge = (box[:, 0] <= mx) | (box[:, 1] <= my) | (box[:, 2] >= w - mx) | (box[:, 3] >= h - my)
        vel, acc, age = self.kin.update(t, tracks.track_id, world, at_edge)
        lanes = self.scene.lane_of(foot).astype(str) if self.scene.has("lanes") else [""] * len(foot)
        last_lane = [
            self.kin.remember_lane(int(i), str(lane)) for i, lane in zip(tracks.track_id, lanes, strict=True)
        ]
        rows = pd.DataFrame(
            {
                "track_id": tracks.track_id,
                "cls": [self.names.get(int(c), str(c)) for c in tracks.cls],
                "X": world[:, 0],
                "Y": world[:, 1],
                "vx": vel[:, 0],
                "vy": vel[:, 1],
                "ax": acc[:, 0],
                "ay": acc[:, 1],
                "fx": foot[:, 0],
                "fy": foot[:, 1],
                "age": age,
                "last_lane": last_lane,
            }
        )
        return risk_features(rows, self.scene, red_lanes, self.cfg)
