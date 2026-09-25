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
- Each track ByteTrack updated this frame is output with the detection it was updated with: the one
  carrying the score the track copied from it (ties: best IoU with the track's box). supervision's
  own `update_with_detections` guessed that pairing by IoU > 0.5 with the Kalman box instead, which
  dropped tracked objects whose box changes shape faster than the filter follows: a bus entering at
  the frame edge vanished for 16 frames on C3902 (SPEC §12.45).

ByteTrack is supervision's implementation (MIT), imported from its module: the top-level
`sv.ByteTrack` alias is deprecated upstream, and requirements.txt pins supervision 0.30.5.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
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
        ids, rows = [], []
        for group, tracker in enumerate(self._trackers):
            members = np.flatnonzero(groups == group)
            boxes = dets.xyxy[members].astype(np.float32).reshape(-1, 4)
            scores = dets.conf[members].astype(np.float32)
            tracks = tracker.update_with_tensors(np.hstack([boxes, scores[:, None]]))
            for t, d in _pair(tracks, boxes, scores):
                ids.append(self._global_id(group, int(tracks[t].external_track_id)))
                rows.append(members[d])
        track_id = np.asarray(ids, dtype=np.int64)
        order = np.argsort(track_id, kind="stable")
        rows = np.asarray(rows, dtype=np.int64)[order]
        return FrameTracks(
            frame_idx=dets.frame_idx,
            t=dets.t,
            track_id=track_id[order],
            cls=dets.cls[rows].astype(np.int64),
            conf=dets.conf[rows].astype(np.float32),
            xyxy=dets.xyxy[rows].astype(np.float32).reshape(-1, 4),
        )

    def _global_id(self, group: int, local: int) -> int:
        key = (group, local)
        if key not in self._ids:
            self._ids[key] = len(self._ids) + 1
        return self._ids[key]


def _pair(tracks: list, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, int]]:
    """(track, detection) index pairs: each output track with the detection it was updated with.

    ByteTrack copies the score of the detection it updates (or starts) a track with, so that
    detection has exactly the track's score; equal scores are resolved by IoU with the track's box,
    greedily from the best pair (deterministic).
    """
    if not tracks or not len(boxes):
        return []
    track_boxes = np.array([t.tlbr for t in tracks], dtype=np.float64).reshape(-1, 4)
    same = np.abs(np.array([t.score for t in tracks], dtype=np.float64)[:, None] - scores[None, :]) < 1e-6
    overlap = _iou(track_boxes, boxes.astype(np.float64))
    pairs = sorted(zip(*np.nonzero(same), strict=True), key=lambda p: (-overlap[p], p[0], p[1]))
    used_t, used_d, out = set(), set(), []
    for t, d in pairs:
        if t not in used_t and d not in used_d:
            used_t.add(t)
            used_d.add(d)
            out.append((int(t), int(d)))
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, (len(a), len(b))."""
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(hi - lo, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)
