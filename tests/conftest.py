"""Shared fixtures: small synthetic videos, so tests run in seconds without the samples."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

TINY_FPS = 10.0
TINY_FRAMES = 20
TINY_SIZE = (160, 96)  # (width, height)

INDEXED_FPS = Fraction(30000, 1001)  # the samples' frame rate
INDEXED_FRAMES = 48
INDEXED_SIZE = (256, 144)
_BARS = 8  # frame index written as 8 black/white vertical bars (LSB on the left)
# Like the samples: two non-reference B-frames between reference frames, open GOPs every 15 frames.
_X264_PARAMS = "bframes=2:b-adapt=0:b-pyramid=none:keyint=15:min-keyint=15:scenecut=0:open-gop=1"


def indexed_image(idx: int, size: tuple[int, int] = INDEXED_SIZE) -> np.ndarray:
    """BGR image whose vertical bars spell `idx` in binary."""
    width, height = size
    img = np.zeros((height, width, 3), np.uint8)
    bar = width // _BARS
    for bit in range(_BARS):
        if idx >> bit & 1:
            img[:, bit * bar : (bit + 1) * bar] = 255
    return img


def read_index(img: np.ndarray) -> int:
    """Inverse of `indexed_image`, robust to compression and resizing (samples each bar's centre)."""
    width = img.shape[1]
    gray = img.mean(axis=2)
    bar = width / _BARS
    bits = (gray[:, int((b + 0.3) * bar) : int((b + 0.7) * bar)].mean() > 128 for b in range(_BARS))
    return sum(1 << b for b, on in enumerate(bits) if on)


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2 s clip with a moving box, alone in its own folder (so it can be passed to the harness)."""
    path = tmp_path_factory.mktemp("videos") / "tiny.mp4"
    width, height = TINY_SIZE
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), TINY_FPS, TINY_SIZE)
    for i in range(TINY_FRAMES):
        frame = np.zeros((height, width, 3), np.uint8)
        cv2.rectangle(frame, (5 * i, 30), (5 * i + 20, 50), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0, "OpenCV could not write the test video"
    return path


@pytest.fixture(scope="session")
def indexed_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """H.264 at 29.97 fps with the samples' B-frame structure; every frame shows its own index."""
    path = tmp_path_factory.mktemp("indexed") / "indexed.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=INDEXED_FPS)
        stream.width, stream.height = INDEXED_SIZE
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "10", "x264-params": _X264_PARAMS}
        for idx in range(INDEXED_FRAMES):
            frame = av.VideoFrame.from_ndarray(indexed_image(idx), format="bgr24")
            frame.pts, frame.time_base = idx, 1 / INDEXED_FPS
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    return path
