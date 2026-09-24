"""The submission runs with the network blocked (CLAUDE.md rule 3; the official run has no internet).

solution.py swallows exceptions, so a blocked connection would silently turn into an empty result.
The guard therefore records every attempt and each test asserts there were none.
"""

from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from roadwatch.config import WEIGHTS_DIR, load_thresholds

WEIGHTS = WEIGHTS_DIR / f"{load_thresholds()['perception']['model']}.torchscript"


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


def fresh_import(name: str, monkeypatch: pytest.MonkeyPatch):
    """Import `name` from scratch (as the harness would in a new process) under the network guard."""
    for module in [m for m in sys.modules if m in ("solution", name) or m.startswith("roadwatch")]:
        monkeypatch.delitem(sys.modules, module)
    return importlib.import_module(name)


def test_solution_imports_and_runs_offline(
    network_attempts: list[tuple], tiny_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solution = fresh_import("solution", monkeypatch)

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


@pytest.mark.skipif(not WEIGHTS.exists(), reason="weights not downloaded (bash weights/download.sh)")
def test_detector_loads_weights_and_detects_offline(
    network_attempts: list[tuple], monkeypatch: pytest.MonkeyPatch
) -> None:
    detector_module = fresh_import("roadwatch.perception.detector", monkeypatch)

    detector = detector_module.Detector.from_config()
    frame = np.full((1080, 1920, 3), 114, np.uint8)
    out = detector.predict([(0, 0.0, frame)], native_size=(3840, 2160))[0]

    assert out.xyxy.shape[1] == 4 and len(out.xyxy) == len(out.conf) == len(out.cls)
    assert network_attempts == []
