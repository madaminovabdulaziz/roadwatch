"""Hand-calibrated scene geometry (SPEC §4, RUNBOOK P1.1).

Contract:
- `Scene.load(path)` reads configs/scene.json; a missing file gives an empty scene, so every rule that
  needs geometry disables itself instead of guessing (SPEC §12.7).
- `Scene.has(layer)` tells whether a layer (e.g. "lanes", "stop_lines", "homography") is present.
- Geometry helpers work on pixel coordinates of the native resolution (3840x2160 for the samples):
  `point_in(layer, pts)`, `lane_of(pts)`, `crosses(line, p0, p1)`, and the homography pair
  `to_world(pts)` / `to_image(pts)` (metres on the road plane).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roadwatch.config import SCENE_PATH


@dataclass
class Scene:
    layers: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = SCENE_PATH) -> Scene:
        if not path.exists():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def has(self, layer: str) -> bool:
        return bool(self.layers.get(layer))

    def point_in(self, layer: str, pts: np.ndarray) -> np.ndarray:
        """Boolean mask: which (N, 2) points lie inside any polygon of `layer`."""
        raise NotImplementedError("Scene geometry lands in RUNBOOK P1.1")

    def lane_of(self, pts: np.ndarray) -> np.ndarray:
        """Lane id for each (N, 2) point, or "" outside every lane."""
        raise NotImplementedError("Scene geometry lands in RUNBOOK P1.1")

    def crosses(self, line: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Whether each step p0[i] -> p1[i] crosses the 2-point `line`."""
        raise NotImplementedError("Scene geometry lands in RUNBOOK P1.1")

    def to_world(self, pts: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) -> road-plane metres (N, 2) via the homography."""
        raise NotImplementedError("Scene geometry lands in RUNBOOK P1.1")

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        """Road-plane metres (N, 2) -> pixels (N, 2)."""
        raise NotImplementedError("Scene geometry lands in RUNBOOK P1.1")
