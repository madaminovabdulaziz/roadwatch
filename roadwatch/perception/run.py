"""Offline perception for Part A (SPEC §3, RUNBOOK P0.3).

Contract: `run_perception(video_path, stride)` decodes the video (PyAV, B-frames skipped, frames
converted straight to the detector's input size, decoded ahead on a background thread), detects in
batches with the shared `Detector`, tracks with a fresh `OnlineTracker`, and returns the TrackTable
(`types.TRACK_DTYPES`) sorted by `(t, track_id)`. Each track gets one class: the confidence-weighted
majority of its detections, so a vehicle does not flip between car and truck.

Time safety (SPEC §12.10): with `budget_sec`, perception paces itself (when wall time runs ahead of
`budget_sec * progress`, only every k-th decoded frame is detected, k <= `runtime.max_frame_skip`);
with `deadline` (a `clock()` value) it stops and returns what it has. Both only act when the machine
is too slow, so normal runs stay deterministic.

`scripts/cache_tracks.py` stores the table as cache/tracks/<video>.parquet for fast rule iteration.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from roadwatch.config import load_thresholds
from roadwatch.perception.detector import Detector
from roadwatch.perception.tracker import OnlineTracker
from roadwatch.types import TRACK_DTYPES, FrameTracks, empty_track_table
from roadwatch.video import FrameReader, prefetch, probe

log = logging.getLogger(__name__)

T = TypeVar("T")


def run_perception(
    video_path: str | Path,
    stride: int | None = None,
    *,
    detector: Detector | None = None,
    t_end: float = math.inf,
    budget_sec: float | None = None,
    deadline: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
    stats: dict[str, Any] | None = None,
    progress: Callable[[float], None] | None = None,
    on_frame: Callable[[np.ndarray, float], None] | None = None,
) -> pd.DataFrame:
    """TrackTable of one video (or of its first `t_end` seconds); fills `stats` with timings if given.

    `progress(fraction)` is called after every batch with the share of the video done (demo progress bar).
    `on_frame(frame, t)` sees every decoded frame (detector size), e.g. for the signal-lamp timeline.
    """
    cfg = load_thresholds()
    stride = stride or cfg["video"]["stride_part_a"]
    detector = detector or Detector.load()
    meta = probe(video_path)
    names = {int(k): v for k, v in cfg["perception"]["keep_classes"].items()}
    reader = FrameReader(
        video_path,
        stride,
        size=detector.frame_size_for(meta.width, meta.height),
        skip_nonref=cfg["video"]["skip_nonref"],
    )
    tracker = OnlineTracker(meta.fps / stride)
    horizon = min(t_end, meta.duration)
    pacer = _Pacer(budget_sec, horizon, cfg["runtime"]["max_frame_skip"], clock)

    timing = {"wait_frames_sec": 0.0, "detect_sec": 0.0, "track_sec": 0.0}
    counts = {"frames_decoded": 0, "frames_detected": 0, "detections": 0}
    stopped_early = False
    tracks: list[FrameTracks] = []
    start = clock()
    frames = prefetch(reader, cfg["perception"]["prefetch_frames"])
    try:
        batches = _batched(frames, detector.batch_size)
        while True:
            waited = clock()
            batch = next(batches, None)
            timing["wait_frames_sec"] += clock() - waited
            if not batch:
                break
            batch = [frame for frame in batch if frame[1] < t_end]
            counts["frames_decoded"] += len(batch)
            if deadline is not None and clock() > deadline:
                log.warning(
                    "%s: perception deadline reached at t=%.1f s; keeping partial tracks",
                    meta.video_id,
                    batch[0][1] if batch else horizon,
                )
                stopped_early = True
                break
            if on_frame is not None:
                for _, t, img in batch:
                    on_frame(img, t)
            selected = [frame for frame in batch if pacer.keep()]
            if selected:
                tick = clock()
                detections = detector.predict(selected, native_size=(meta.width, meta.height))
                timing["detect_sec"] += clock() - tick
                tick = clock()
                tracks.extend(tracker.update(d) for d in detections)
                timing["track_sec"] += clock() - tick
                counts["frames_detected"] += len(selected)
                counts["detections"] += sum(len(d.conf) for d in detections)
            if progress is not None and batch and horizon > 0:
                progress(min(1.0, batch[-1][1] / horizon))
            if len(batch) < detector.batch_size:  # t_end reached or end of video
                break
            pacer.update(batch[-1][1])
    finally:
        frames.close()

    table = tracks_to_table(tracks, names)
    if stats is not None:
        total = clock() - start
        stats.update(
            video_id=meta.video_id,
            video_sec=round(horizon, 3),
            total_sec=round(total, 3),
            x_duration=round(total / horizon, 3) if horizon else 0.0,
            **{k: round(v, 3) for k, v in timing.items()},
            **counts,
            tracks=int(table["track_id"].nunique()),
            max_frame_skip_used=pacer.max_used,
            stopped_early=stopped_early,
        )
    return table


def tracks_to_table(frames: list[FrameTracks], names: dict[int, str]) -> pd.DataFrame:
    """Stack per-frame tracks into the TrackTable, one class per track, sorted by (t, track_id)."""
    frames = [f for f in frames if len(f.track_id)]
    if not frames:
        return empty_track_table()
    sizes = [len(f.track_id) for f in frames]
    xyxy = np.concatenate([f.xyxy for f in frames])
    table = pd.DataFrame(
        {
            "frame": np.repeat([f.frame_idx for f in frames], sizes),
            "t": np.repeat([f.t for f in frames], sizes),
            "track_id": np.concatenate([f.track_id for f in frames]),
            "cls": [names[int(c)] for c in np.concatenate([f.cls for f in frames])],
            "conf": np.concatenate([f.conf for f in frames]),
            "x1": xyxy[:, 0],
            "y1": xyxy[:, 1],
            "x2": xyxy[:, 2],
            "y2": xyxy[:, 3],
            "fx": (xyxy[:, 0] + xyxy[:, 2]) / 2,
            "fy": xyxy[:, 3],
        }
    )
    votes = table.groupby(["track_id", "cls"], observed=True, sort=True)["conf"].sum().reset_index()
    majority = votes.sort_values(["track_id", "conf", "cls"], ascending=[True, False, True], kind="stable")
    table["cls"] = table["track_id"].map(majority.drop_duplicates("track_id").set_index("track_id")["cls"])
    table = table.astype(TRACK_DTYPES)
    return table.sort_values(["t", "track_id"], kind="stable").reset_index(drop=True)


class _Pacer:
    """Detect only every k-th frame while behind schedule; k moves one step per batch."""

    def __init__(self, budget_sec: float | None, horizon: float, max_skip: int, clock: Callable[[], float]):
        self.budget_sec, self.horizon, self.max_skip, self.clock = budget_sec, horizon, max_skip, clock
        self.start = clock()
        self.skip = 1
        self.max_used = 1
        self._count = 0

    def keep(self) -> bool:
        self._count += 1
        return self._count % self.skip == 0

    def update(self, t_sec: float) -> None:
        if self.budget_sec is None or self.horizon <= 0:
            return
        allowed = self.budget_sec * min(1.0, t_sec / self.horizon)
        elapsed = self.clock() - self.start
        if elapsed > allowed and self.skip < self.max_skip:
            self.skip += 1
        elif elapsed < 0.8 * allowed and self.skip > 1:
            self.skip -= 1
        self.max_used = max(self.max_used, self.skip)


def _batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
