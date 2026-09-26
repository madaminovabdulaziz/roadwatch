"""solution.py honours the harness interface and never breaks it (SPEC §9, CLAUDE.md rules 2 and 8)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

import evaluate
import roadwatch.pipeline
import roadwatch.risk
import solution
from tests.conftest import TINY_FRAMES

REPO_ROOT = Path(__file__).resolve().parents[1]


def stream_scores(estimator: solution.RiskEstimator, video_path: Path) -> list[float]:
    """Feed every frame to the estimator exactly like run_submission.run_risk does."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    estimator.reset(
        {
            "video_id": video_path.name,
            "fps": fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    )
    scores, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scores.append(estimator.step(frame, idx / fps))
        idx += 1
    cap.release()
    return scores


def is_probability(x: object) -> bool:
    return isinstance(x, float) and math.isfinite(x) and 0.0 <= x <= 1.0


def test_solution_exposes_the_harness_interface() -> None:
    for name in ("CLASSES", "detect_events", "RiskEstimator"):
        assert hasattr(solution, name)
    assert solution.CLASSES == evaluate.OFFICIAL_CLASSES


def test_detect_events_output_passes_the_format_check(tiny_video: Path) -> None:
    events = solution.detect_events(str(tiny_video))
    assert isinstance(events, list)
    pred = {"team": "test", "videos": {tiny_video.name: {"events": events, "risk": []}}}
    errors, _ = evaluate.validate(pred)
    assert errors == []


def test_detect_events_returns_no_events_when_the_pipeline_crashes(
    tiny_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(_video_path: str) -> list[list]:
        raise RuntimeError("boom")

    monkeypatch.setattr(roadwatch.pipeline, "detect_events", crash)
    assert solution.detect_events(str(tiny_video)) == []


def test_risk_estimator_scores_every_frame(tiny_video: Path) -> None:
    scores = stream_scores(solution.RiskEstimator(), tiny_video)
    assert len(scores) == TINY_FRAMES
    assert all(is_probability(s) for s in scores)


@pytest.mark.parametrize("bad", [RuntimeError("boom"), float("nan"), float("inf"), 7.0, -1.0])
def test_risk_estimator_contains_bad_core_output(
    tiny_video: Path, monkeypatch: pytest.MonkeyPatch, bad: object
) -> None:
    def bad_step(self: roadwatch.risk.RiskCore, frame: object, t_sec: float) -> object:
        if isinstance(bad, Exception):
            raise bad
        return bad

    monkeypatch.setattr(roadwatch.risk.RiskCore, "step", bad_step)
    scores = stream_scores(solution.RiskEstimator(), tiny_video)
    assert len(scores) == TINY_FRAMES
    assert all(is_probability(s) for s in scores)


def test_official_harness_writes_valid_predictions(tiny_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "predictions.json"
    run = subprocess.run(
        [
            sys.executable,
            "run_submission.py",
            "--videos",
            str(tiny_video.parent),
            "--out",
            str(out),
            "--team",
            "roadwatch",
            # a 2 s clip cannot absorb the one-off model load that counts against the first video's
            # budget; this test checks plumbing and format, scripts/bench.py measures time on real clips
            "--time-factor",
            "100",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout + run.stderr

    pred = json.loads(out.read_text())
    assert pred["log"][tiny_video.name]["errors"] == []
    assert len(pred["videos"][tiny_video.name]["risk"]) == TINY_FRAMES

    check = subprocess.run(
        [sys.executable, "evaluate.py", "--pred", str(out), "--validate-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert check.returncode == 0, check.stdout


def test_the_unpaced_profile_only_switches_pacing_off() -> None:
    # make_predictions_samples.py --unpaced (SPEC §12.54): same pipeline, no wall-clock thinning
    import yaml

    from roadwatch.config import CONFIG_DIR, deep_merge, load_thresholds

    base = load_thresholds()
    extra = yaml.safe_load((CONFIG_DIR / "unpaced.yaml").read_text(encoding="utf-8"))
    merged = deep_merge(base, extra)
    assert set(extra) == {"runtime", "risk"}
    assert merged["runtime"]["perception_budget_factor"] >= 100
    assert merged["risk"]["total_budget_factor"] >= 100 and merged["risk"]["budget_factor"] >= 100
    assert merged["classes"] == base["classes"] and merged["perception"] == base["perception"]
