"""Online multi-object tracking (SPEC §3, §12.19, §12.21).

Contract:
- `OnlineTracker(frame_rate)` runs one ByteTrack per class group of `perception.tracker_groups`, so an
  id never jumps between object types. `frame_rate` is the rate of the frames it is fed (fps / stride).
- `update(dets)` consumes one frame's `FrameDetections` in time order (call it for every processed
  frame, even an empty one, so lost tracks age) and returns that frame's `FrameTracks`: the tracked
  detections with the detector's boxes, sorted by track id. Ids are unique across groups, assigned
  in order of first appearance, so identical input gives identical ids.
- Detections from `perception.conf_min` up are fed in: ByteTrack starts tracks from confident boxes
  (> track_activation_threshold + 0.1) and uses the weaker ones only to continue existing tracks
  through partial occlusion. Fixed camera, so no camera-motion compensation.

ByteTrack is supervision's implementation (MIT), imported from its module: the top-level
`sv.ByteTrack` alias is deprecated upstream, and requirements.txt pins supervision 0.30.5.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from supervision.detection.core import Detections
from supervision.tracker.byte_tracker.core import ByteTrack

from roadwatch.config import load_thresholds
from roadwatch.types import FrameDetections, FrameTracks

# supervision counts the lost-track buffer in frames at 30 fps: max_time_lost = frame_rate / 30 * buffer.
_BYTETRACK_BUFFER_FPS = 30.0


class OnlineTracker:
    """Per-class-group ByteTrack with ids unique across groups."""

    def __init__(self, frame_rate: float, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or load_thresholds()["perception"]
        bytetrack = cfg["bytetrack"]
        group_of_name = {
            name: g for g, members in enumerate(cfg["tracker_groups"].values()) for name in members
        }
        self._group_of_class = {int(cid): group_of_name[name] for cid, name in cfg["keep_classes"].items()}
        with warnings.catch_warnings():  # upstream deprecation of the class we pin on purpose
            warnings.filterwarnings(
                "ignore", message="The `ByteTrack` was deprecated", category=FutureWarning
            )
            self._trackers = [self._new_bytetrack(bytetrack, frame_rate) for _ in cfg["tracker_groups"]]
        self._ids: dict[tuple[int, int], int] = {}

    @staticmethod
    def _new_bytetrack(bytetrack: dict[str, Any], frame_rate: float) -> ByteTrack:
        return ByteTrack(
            track_activation_threshold=bytetrack["track_activation_threshold"],
            lost_track_buffer=round(bytetrack["lost_track_buffer_sec"] * _BYTETRACK_BUFFER_FPS),
            minimum_matching_threshold=bytetrack["minimum_matching_threshold"],
            frame_rate=max(1, round(frame_rate)),
            minimum_consecutive_frames=1,
        )

    def update(self, dets: FrameDetections) -> FrameTracks:
        groups = np.array([self._group_of_class.get(int(c), -1) for c in dets.cls], dtype=np.int64)
        ids, classes, confs, boxes = [], [], [], []
        for group, tracker in enumerate(self._trackers):
            mask = groups == group
            tracked = tracker.update_with_detections(
                Detections(
                    xyxy=dets.xyxy[mask].astype(np.float32).reshape(-1, 4),
                    confidence=dets.conf[mask].astype(np.float32),
                    class_id=dets.cls[mask].astype(np.int64),
                )
            )
            ids.extend(self._global_id(group, int(local)) for local in tracked.tracker_id)
            classes.append(tracked.class_id)
            confs.append(tracked.confidence)
            boxes.append(tracked.xyxy)
        track_id = np.asarray(ids, dtype=np.int64)
        order = np.argsort(track_id, kind="stable")
        return FrameTracks(
            frame_idx=dets.frame_idx,
            t=dets.t,
            track_id=track_id[order],
            cls=np.concatenate(classes).astype(np.int64)[order],
            conf=np.concatenate(confs).astype(np.float32)[order],
            xyxy=np.concatenate(boxes).astype(np.float32).reshape(-1, 4)[order],
        )

    def _global_id(self, group: int, local: int) -> int:
        key = (group, local)
        if key not in self._ids:
            self._ids[key] = len(self._ids) + 1
        return self._ids[key]
