"""
solution.py: the interface the organizers' harness (run_submission.py) imports.

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

A thin, crash-proof wrapper: the logic lives in the `roadwatch` package (docs/SPEC.md). `roadwatch` is
imported lazily inside the functions, so even a broken dependency can never fail the harness's import
of this module (that would empty every video). Part A failures return [] for that video; Part B always
returns a finite float in [0, 1].
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

log = logging.getLogger("roadwatch.solution")

# Official class ids (14), unchanged from the starter kit. Which classes we actually emit is controlled
# by the `enabled` flags in configs/thresholds.yaml.
CLASSES: list[str] = [
    "accident",  # collision between road users / with a fixed object
    "near_miss",  # sharp braking or swerving to avoid a collision, no contact
    "red_light",  # crossing the stop line on red
    "wrong_way",  # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",  # U-turn where prohibited
    "stopped_vehicle",  # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",  # pedestrian on the carriageway outside a crossing
    "failure_to_yield",  # driving through a crossing while a pedestrian is on it
    "illegal_turn",  # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing",  # lane change / manoeuvre across a solid marking
    "stop_line",  # stopped past the stop line on red
    "congestion",  # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",  # debris, animal or fallen object on the carriageway
    "fire_smoke",  # visible fire or smoke from a vehicle or on the road
]


def detect_events(video_path: str) -> list[list]:
    """Part A: every traffic event in one .mp4 as [start_sec, end_sec, label]; [] on any failure."""
    try:
        from roadwatch.pipeline import detect_events as run_pipeline

        return run_pipeline(video_path)
    except Exception:
        log.exception("detect_events failed on %s; returning no events", video_path)
        return []


class RiskEstimator:
    """Part B: causal P(accident starts within 5 s), delegated to roadwatch.risk.RiskCore.

    The harness calls reset(meta) once per video, then step() for every frame in order. An exception
    escaping step() would cost the whole risk curve of the video, so every failure is contained here
    and the previous score is returned instead.
    """

    def __init__(self) -> None:
        self._core = None
        self._last = 0.0
        self._failures = 0

    def reset(self, meta: dict) -> None:
        self._last = 0.0
        self._failures = 0
        try:
            from roadwatch.risk import RiskCore

            self._core = RiskCore()
            self._core.reset(meta)
        except Exception:
            log.exception("RiskCore could not start for %s; risk stays at 0", meta.get("video_id"))
            self._core = None

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self._core is None:
            return self._last
        try:
            score = float(self._core.step(frame, t_sec))
        except Exception:
            self._failures += 1
            if self._failures == 1:  # log once per video; the harness calls step() for every frame
                log.exception("RiskCore.step failed at t=%.3f; keeping the previous score", t_sec)
            return self._last
        if math.isfinite(score):
            self._last = min(1.0, max(0.0, score))
        return self._last
