"""Video reading: harness-identical metadata and timestamps, stride, B-frame skipping, windows, bad data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import run_submission
from roadwatch.video import FrameReader, probe, read_window
from tests.conftest import INDEXED_FRAMES, INDEXED_SIZE, TINY_FPS, TINY_FRAMES, TINY_SIZE, read_index

BACKENDS = ["pyav", "opencv"]


def indices(frames: list) -> list[int]:
    return [idx for idx, _, _ in frames]


def assert_content_matches_index(frames: list) -> None:
    assert frames, "no frames were read"
    for idx, _, img in frames:
        assert read_index(img) == idx, f"frame reported as {idx} shows {read_index(img)}"


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_frame_with_the_harness_timestamp(indexed_video: Path, backend: str) -> None:
    fps = run_submission.video_meta(indexed_video)["fps"]
    frames = list(FrameReader(indexed_video, backend=backend))

    assert indices(frames) == list(range(INDEXED_FRAMES))
    assert all(t_sec == idx / fps for idx, t_sec, _ in frames)
    assert all(img.shape == (INDEXED_SIZE[1], INDEXED_SIZE[0], 3) for _, _, img in frames)
    assert all(img.dtype == np.uint8 for _, _, img in frames)
    assert_content_matches_index(frames)


@pytest.mark.parametrize("backend", BACKENDS)
def test_stride_yields_exact_multiples(indexed_video: Path, backend: str) -> None:
    frames = list(FrameReader(indexed_video, stride=3, backend=backend))
    assert indices(frames) == list(range(0, INDEXED_FRAMES, 3))
    assert_content_matches_index(frames)


def test_skip_nonref_decodes_only_reference_frames_at_their_true_index(indexed_video: Path) -> None:
    every_ref = list(FrameReader(indexed_video, stride=1, skip_nonref=True))
    stride3 = list(FrameReader(indexed_video, stride=3, skip_nonref=True))

    # Every 3rd frame is a reference frame, plus the encoder's closing P-frame at the very end,
    # which stride 3 drops because it is only 2 frames after the previous one.
    assert indices(every_ref) == [*range(0, INDEXED_FRAMES - 2, 3), INDEXED_FRAMES - 1]
    assert indices(stride3) == list(range(0, INDEXED_FRAMES - 2, 3))
    assert_content_matches_index(every_ref)


def test_skip_nonref_stride_is_a_minimum_spacing(indexed_video: Path) -> None:
    idxs = indices(list(FrameReader(indexed_video, stride=6, skip_nonref=True)))
    assert idxs[:4] == [0, 6, 12, 18]
    assert all(b - a >= 6 for a, b in zip(idxs, idxs[1:], strict=False))


@pytest.mark.parametrize("backend", BACKENDS)
def test_size_converts_frames_directly(indexed_video: Path, backend: str) -> None:
    frames = list(FrameReader(indexed_video, stride=5, size=(128, 72), backend=backend))
    assert all(img.shape == (72, 128, 3) for _, _, img in frames)
    assert_content_matches_index(frames)


@pytest.mark.parametrize("backend", BACKENDS)
def test_read_window_seeks_to_exact_frames(indexed_video: Path, backend: str) -> None:
    fps = probe(indexed_video).fps
    frames = list(read_window(indexed_video, 17 / fps, 29 / fps, backend=backend))  # starts mid-GOP
    assert indices(frames) == list(range(17, 29))
    assert_content_matches_index(frames)


def test_read_window_with_stride_and_open_end(indexed_video: Path) -> None:
    fps = probe(indexed_video).fps
    frames = list(read_window(indexed_video, 40 / fps, 999.0, stride=2))
    assert indices(frames) == list(range(40, INDEXED_FRAMES, 2))
    assert_content_matches_index(frames)


def test_read_window_skip_nonref_yields_reference_frames_inside_the_window(indexed_video: Path) -> None:
    fps = probe(indexed_video).fps
    frames = list(read_window(indexed_video, 10 / fps, 40 / fps, stride=3, skip_nonref=True))
    assert indices(frames) == list(range(12, 40, 3))
    assert_content_matches_index(frames)


def test_invalid_stride_is_rejected(indexed_video: Path) -> None:
    with pytest.raises(ValueError):
        FrameReader(indexed_video, stride=0)
    with pytest.raises(ValueError):
        read_window(indexed_video, 0.0, 1.0, stride=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_corrupted_data_is_skipped_without_raising(indexed_video: Path, tmp_path: Path, backend: str) -> None:
    data = bytearray(indexed_video.read_bytes())
    mid = len(data) // 2
    data[mid : mid + 64] = bytes(range(64))  # garbage inside the frame data
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(data)

    frames = list(FrameReader(corrupt, backend=backend))

    assert len(frames) > INDEXED_FRAMES // 2
    assert indices(frames) == sorted(set(indices(frames)))
