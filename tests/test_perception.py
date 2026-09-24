"""run_perception and the TrackTable: schema, timestamps, one class per track, pacing and deadline.

A fake detector stands in for YOLO, so these tests run on the indexed test clip without weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from roadwatch.perception.run import run_perception, tracks_to_table
from roadwatch.types import TRACK_DTYPES, Frame, FrameDetections, FrameTracks
from roadwatch.video import probe
from tests.conftest import INDEXED_FRAMES, INDEXED_SIZE, read_index

CAR, TRUCK = 2, 7
NAMES = {0: "person", 2: "car", 7: "truck"}


class FakeDetector:
    """One vehicle per frame drifting right; class and confidence chosen per frame index."""

    batch_size = 4

    def __init__(self, cls_of: dict[int, int] | None = None) -> None:
        self.cls_of = cls_of or {}
        self.frames_seen: list[tuple[int, int, int]] = []

    def frame_size_for(self, width: int, height: int) -> tuple[int, int]:
        return width // 2, height // 2

    def predict(
        self, batch: Sequence[Frame], native_size: tuple[int, int] | None = None
    ) -> list[FrameDetections]:
        out = []
        for idx, t_sec, img in batch:
            assert read_index(img) == idx  # the frame really is the one its index says
            self.frames_seen.append((idx, img.shape[1], img.shape[0]))
            x = 10.0 + 2.0 * idx
            out.append(
                FrameDetections(
                    frame_idx=idx,
                    t=t_sec,
                    xyxy=np.array([[x, 20.0, x + 60.0, 80.0]], dtype=np.float32),
                    conf=np.array([0.9], dtype=np.float32),
                    cls=np.array([self.cls_of.get(idx, CAR)], dtype=np.int64),
                )
            )
        return out


class StepClock:
    """Deterministic clock: every call advances time by `step` seconds."""

    def __init__(self, step: float) -> None:
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def test_track_table_schema_timestamps_and_footprints(indexed_video: Path) -> None:
    detector = FakeDetector()
    stats: dict = {}
    table = run_perception(indexed_video, 3, detector=detector, stats=stats)
    fps = probe(indexed_video).fps

    assert dict(table.dtypes.astype(str)) == TRACK_DTYPES
    assert table["frame"].tolist() == list(range(0, INDEXED_FRAMES - 2, 3))  # reference frames only
    np.testing.assert_allclose(table["t"], table["frame"] / fps)
    assert table["track_id"].nunique() == 1 and set(table["cls"]) == {"car"}
    np.testing.assert_allclose(table["fx"], (table["x1"] + table["x2"]) / 2)
    np.testing.assert_allclose(table["fy"], table["y2"])
    assert {(w, h) for _, w, h in detector.frames_seen} == {(INDEXED_SIZE[0] // 2, INDEXED_SIZE[1] // 2)}
    assert stats["frames_detected"] == len(table) and stats["stopped_early"] is False


def test_t_end_limits_the_run(indexed_video: Path) -> None:
    fps = probe(indexed_video).fps
    table = run_perception(indexed_video, 3, detector=FakeDetector(), t_end=20 / fps)
    assert table["frame"].max() < 20


def test_each_track_gets_its_confidence_weighted_majority_class(indexed_video: Path) -> None:
    trucks = {idx: TRUCK for idx in (3, 9)}  # 2 of 16 detections say truck
    table = run_perception(indexed_video, 3, detector=FakeDetector(trucks))
    assert set(table["cls"]) == {"car"}


def test_tracks_to_table_breaks_class_ties_deterministically() -> None:
    frames = [
        FrameTracks(
            i,
            i / 10,
            np.array([1]),
            np.array([cls]),
            np.array([0.5], np.float32),
            np.ones((1, 4), np.float32),
        )
        for i, cls in enumerate([CAR, TRUCK])
    ]
    assert set(tracks_to_table(frames, NAMES)["cls"]) == {"car"}  # equal weight: alphabetical


def test_empty_video_gives_an_empty_table_with_the_schema() -> None:
    table = tracks_to_table([], NAMES)
    assert table.empty and dict(table.dtypes.astype(str)) == TRACK_DTYPES


def test_deadline_stops_early_and_keeps_partial_tracks(indexed_video: Path) -> None:
    stats: dict = {}
    clock = StepClock(1.0)
    table = run_perception(indexed_video, 3, detector=FakeDetector(), deadline=12.0, clock=clock, stats=stats)
    assert stats["stopped_early"] is True
    assert 0 < len(table) < INDEXED_FRAMES // 3


@pytest.mark.parametrize(("step", "skips"), [(0.0, False), (5.0, True)])
def test_pacing_thins_detection_only_when_behind_schedule(
    indexed_video: Path, step: float, skips: bool
) -> None:
    stats: dict = {}
    run_perception(
        indexed_video, 3, detector=FakeDetector(), budget_sec=1.0, clock=StepClock(step), stats=stats
    )
    assert (stats["frames_detected"] < stats["frames_decoded"]) is skips
    assert (stats["max_frame_skip_used"] > 1) is skips
