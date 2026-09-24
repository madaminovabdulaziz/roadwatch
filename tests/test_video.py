"""Our video metadata matches the harness's exactly (timestamps and duration drive the metric)."""

from __future__ import annotations

from pathlib import Path

import pytest

import run_submission
from roadwatch.video import probe
from tests.conftest import TINY_FPS, TINY_FRAMES, TINY_SIZE


def test_probe_matches_the_harness(tiny_video: Path) -> None:
    ours = probe(tiny_video)
    harness = run_submission.video_meta(tiny_video)

    assert ours.video_id == harness["video_id"]
    assert ours.fps == harness["fps"]
    assert ours.n_frames == harness["n_frames"]
    assert (ours.width, ours.height) == (harness["width"], harness["height"])
    assert ours.duration == harness["duration"]


def test_probe_reads_the_tiny_video(tiny_video: Path) -> None:
    meta = probe(tiny_video)
    assert meta.fps == pytest.approx(TINY_FPS)
    assert meta.n_frames == TINY_FRAMES
    assert (meta.width, meta.height) == TINY_SIZE
    assert meta.duration == pytest.approx(TINY_FRAMES / TINY_FPS)


def test_probe_raises_on_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        probe(tmp_path / "missing.mp4")
