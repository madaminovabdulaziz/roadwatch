"""Part B core: causal accident risk, used by `solution.RiskEstimator` (SPEC §8, RUNBOOK P2.2).

Contract:
- `RiskCore()` then `reset(meta)` once per video; `step(frame, t_sec)` is called for every frame in
  order and returns P(an accident starts within 5 s) in [0, 1].
- Causal by construction: it sees only frames passed to `step()`. It never opens the video file,
  never reads cache/ and never reuses Part A output (SPEC §12.3).
- Heavy work runs on every `risk.stride`-th frame only; other frames return the previous score, so
  they cost almost nothing (the harness already spends its budget decoding every 4K frame).

Current state: no perception yet, so the score is always 0.0.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RiskCore:
    """Online perception + TTC features -> risk score."""

    def __init__(self) -> None:
        self.meta: dict[str, Any] = {}
        self.score = 0.0

    def reset(self, meta: dict[str, Any]) -> None:
        self.meta = dict(meta)
        self.score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self.score
