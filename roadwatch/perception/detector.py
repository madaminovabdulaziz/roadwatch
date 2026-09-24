"""Object detector: YOLO11m with COCO weights (SPEC §3, RUNBOOK P0.3).

Contract:
- `Detector.load()` returns the process-wide instance; weights are read once from weights/ by local
  path and never downloaded. One warm-up batch runs at load.
- `Detector.predict(batch)` takes `(frame_idx, t_sec, frame_bgr)` tuples and returns one
  `FrameDetections` per frame, boxes in native pixels, only classes in `perception.keep_classes`.
- FP16 on CUDA, FP32 on CPU; batch size, input size and thresholds come from thresholds.yaml.
- The `ultralytics` package is not imported at inference (SPEC §12): exported weights run on plain
  torch with our own letterbox and NMS.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from roadwatch.types import FrameDetections


class Detector:
    """Batched detector over BGR frames."""

    @classmethod
    def load(cls) -> Detector:
        raise NotImplementedError("Detector lands in RUNBOOK P0.3")

    def predict(self, batch: Sequence[tuple[int, float, np.ndarray]]) -> list[FrameDetections]:
        raise NotImplementedError("Detector lands in RUNBOOK P0.3")
