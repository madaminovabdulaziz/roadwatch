"""Shared fixtures: a tiny synthetic video so interface tests run in seconds without the samples."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

TINY_FPS = 10.0
TINY_FRAMES = 20
TINY_SIZE = (160, 96)  # (width, height)


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
