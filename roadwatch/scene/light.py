"""Traffic-signal state from the lamp boxes in scene.json (RUNBOOK P1.2).

Contract:
- States are "red", "yellow", "green" or "unknown".
- Offline (Part A): `timeline(frames)` returns, per signal id, `[(t_start, t_end, state), ...]`
  covering the video, median-filtered over `signal.median_sec`.
- Online (Part B): `update(frame, t_sec)` returns the current state per signal id using only frames
  seen so far (causal).
- Without signal boxes in the scene, a flow-based fallback may be used and is marked low confidence.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from roadwatch.scene.scene import Scene

SignalTimeline = dict[str, list[tuple[float, float, str]]]


class SignalStateEstimator:
    """Per-signal lamp brightness -> state."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene

    def timeline(self, frames: Iterable[tuple[int, float, np.ndarray]]) -> SignalTimeline:
        raise NotImplementedError("SignalStateEstimator lands in RUNBOOK P1.2")

    def update(self, frame: np.ndarray, t_sec: float) -> dict[str, str]:
        raise NotImplementedError("SignalStateEstimator lands in RUNBOOK P1.2")
