"""The submission runs with the network blocked (CLAUDE.md rule 3; the official run has no internet).

solution.py swallows exceptions, so a blocked connection would silently turn into an empty result.
The guard therefore records every attempt and the test asserts there were none.
"""

from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path

import cv2
import pytest


@pytest.fixture
def network_attempts(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    attempts: list[tuple] = []

    def blocked(*args: object, **kwargs: object) -> None:
        attempts.append(args)
        raise OSError("network access is disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    return attempts


def test_solution_imports_and_runs_offline(
    network_attempts: list[tuple], tiny_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Re-import from scratch under the guard, as the harness would in a fresh process.
    for name in [m for m in sys.modules if m == "solution" or m.startswith("roadwatch")]:
        monkeypatch.delitem(sys.modules, name)
    solution = importlib.import_module("solution")

    events = solution.detect_events(str(tiny_video))

    estimator = solution.RiskEstimator()
    cap = cv2.VideoCapture(str(tiny_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    estimator.reset({"video_id": tiny_video.name, "fps": fps, "width": 0, "height": 0, "n_frames": 0})
    ok, frame = cap.read()
    cap.release()
    assert ok
    score = estimator.step(frame, 0.0)

    assert isinstance(events, list)
    assert 0.0 <= score <= 1.0
    assert network_attempts == []
