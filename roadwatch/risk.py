"""Part B core: causal accident risk, used by `solution.RiskEstimator` (SPEC §8, RUNBOOK P2.2).

Contract:
- `RiskCore()` then `reset(meta)` once per video; `step(frame, t_sec)` is called for every frame in
  order and returns P(an accident starts within 5 s) in [0, 1].
- Causal by construction: it sees only frames passed to `step()`. It never opens the video file,
  never reads cache/ and never reuses Part A output (SPEC §12.3). Config, scene and weights are read in
  `reset()`; `step()` touches no file.
- Heavy work runs when at least `risk.stride` frames passed since the last processed one (frame index
  = round(t * fps), so the dev flag `--risk-stride` works too); other frames return the previous score.

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
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd

from roadwatch.config import load_thresholds
from roadwatch.events.common import lane_world_dirs
from roadwatch.features import pair_features
from roadwatch.scene.light import SignalStateEstimator
from roadwatch.scene.scene import Scene
from roadwatch.types import FrameTracks

_FORGET_SEC = 2.0  # a track unseen this long is dropped from the online kinematics


@dataclass
class _TrackState:
    t: float
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(2))
    n: int = 1


class OnlineKinematics:
    """Causal per-track position/velocity/acceleration in metres (EMA, as features' online mode)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.tracks: dict[int, _TrackState] = {}

    def update(self, t: float, ids: np.ndarray, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Velocities and accelerations (N, 2) for this frame's tracks, in the order given."""
        a = self.alpha
        vel = np.zeros((len(ids), 2))
        acc = np.zeros((len(ids), 2))
        for i, (tid, p) in enumerate(zip(ids.tolist(), world, strict=True)):
            s = self.tracks.get(tid)
            if s is None or t - s.t > _FORGET_SEC:
                self.tracks[tid] = _TrackState(t, p.copy())
                continue
            dt = t - s.t
            if dt <= 0:
                vel[i], acc[i] = s.vel, s.acc
                continue
            pos = a * p + (1 - a) * s.pos
            v_raw = (pos - s.pos) / dt
            v = v_raw if s.n == 1 else a * v_raw + (1 - a) * s.vel
            if s.n >= 2:
                a_raw = (v - s.vel) / dt
                s.acc = a_raw if s.n == 2 else a * a_raw + (1 - a) * s.acc
            s.t, s.pos, s.vel, s.n = t, pos, v, s.n + 1
            vel[i], acc[i] = s.vel, s.acc
        for tid in [k for k, s in self.tracks.items() if t - s.t > _FORGET_SEC]:
            del self.tracks[tid]
        return vel, acc


def risk_features(
    rows: pd.DataFrame, scene: Scene, red_lanes: set[str], cfg: dict[str, Any]
) -> dict[str, float]:
    """SPEC §8 features of one frame. `rows`: track_id, cls, X, Y, vx, vy, ax, ay, fx, fy (metres/px)."""
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
    acc = rows[["ax", "ay"]].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        along = np.where(speed > 0.5, (acc * vel).sum(axis=1) / speed, 0.0)
    if movers.any():
        feats["decel_max"] = float(max(0.0, -along[movers].min()))

    pairs = pair_features(rows)
    if len(pairs):
        involve = pairs["cls_a"].isin(groups["movers"]) | pairs["cls_b"].isin(groups["movers"])
        person = pairs["cls_a"].isin(groups["persons"]) | pairs["cls_b"].isin(groups["persons"])
        road_ids = set(
            rows["track_id"].to_numpy()[scene.point_in("carriageway", rows[["fx", "fy"]].to_numpy())]
        )
        on_road = pairs["track_a"].isin(road_ids) & pairs["track_b"].isin(road_ids)
        vv = pairs[involve & ~person]
        if len(vv):
            feats["ttc_min"] = float(vv["ttc"].iloc[0])
            feats["closing_speed"] = float(vv["closing_speed"].iloc[0])
        vp = pairs[involve & person & on_road]
        if len(vp):
            feats["ped_ttc"] = float(vp["ttc"].min())

    if scene.has("lanes") and movers.any():
        lanes = scene.lane_of(rows[["fx", "fy"]].to_numpy(dtype=np.float64)).astype(str)
        lane_dir = lane_world_dirs(rows.assign(lane_id=lanes), scene)
        fast = movers & (speed > cfg["wrong_way_speed_mps"])
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (vel * lane_dir).sum(axis=1) / speed
        feats["wrong_way"] = float((fast & (cos < cfg["wrong_way_max_cos"])).any())
        feats["red_light"] = float((fast & np.isin(lanes, list(red_lanes))).any())
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
        from roadwatch.perception.detector import Detector
        from roadwatch.perception.tracker import OnlineTracker

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
        if not self.metric:
            # no metric features are possible: the score is the constant bias, so skip perception
            self.score = risk_score(risk_features(pd.DataFrame(), self.scene, set(), self.cfg), self.cfg)
            return
        self.detector = self._detector or Detector.load()
        self.tracker = OnlineTracker(self.fps / self.cfg["stride"])
        self.kin = OnlineKinematics(cfg["kinematics"]["ema_alpha"])
        self.signals = SignalStateEstimator(self.scene) if self.scene.has("signals") else None
        self.signal_of = {
            str(lane["id"]): str(lane["signal"])
            for lane in self.scene.layers.get("lanes") or []
            if lane.get("signal")
        }
        self.det_size = self.detector.frame_size_for(*self.native)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        idx = int(round(t_sec * self.fps))
        if not self.metric or idx - self.last_idx < self.cfg["stride"]:
            return self.score
        self.last_idx = idx
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
        vel, acc = self.kin.update(t, tracks.track_id, world)
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
            }
        )
        return risk_features(rows, self.scene, red_lanes, self.cfg)
