"""Part A orchestration on cached-style tracks: rule isolation, enable flags, events format."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from roadwatch import pipeline
from roadwatch.config import load_thresholds
from roadwatch.events.base import VideoContext
from roadwatch.scene.scene import Scene
from roadwatch.types import TRACK_DTYPES, Segment, VideoMeta

META = VideoMeta("v.mp4", 10.0, 100, 100, 100)  # 10 s
SCENE = Scene({"lanes": [{"id": "a", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}]})


def tracks() -> pd.DataFrame:
    t = np.arange(0, 10, 0.3)
    return pd.DataFrame(
        {
            "frame": np.round(t * 10).astype(int),
            "t": t,
            "track_id": 1,
            "cls": "car",
            "conf": 0.9,
            "x1": 10.0,
            "y1": 10.0,
            "x2": 20.0,
            "y2": 20.0,
            "fx": 15.0,
            "fy": 20.0,
        }
    ).astype(TRACK_DTYPES)


def fake_rule(label: str, layers: tuple = ("lanes",), segments=None, fail: bool = False):
    def detect(tt, scene, ctx, cfg):
        if fail:
            raise RuntimeError("boom")
        return segments if segments is not None else [Segment(2.0, 4.0, label, 0.9, (1,))]

    return SimpleNamespace(LABEL=label, REQUIRED_LAYERS=layers, detect=detect)


@pytest.fixture
def enabled() -> dict:
    th = copy.deepcopy(load_thresholds())
    for label in ("wrong_way", "jaywalking", "congestion"):
        th["classes"][label]["enabled"] = True
    return th


def test_failing_rule_does_not_stop_the_others(monkeypatch: pytest.MonkeyPatch, enabled: dict) -> None:
    monkeypatch.setattr(
        pipeline,
        "RULES",
        {"wrong_way": fake_rule("wrong_way", fail=True), "jaywalking": fake_rule("jaywalking")},
    )
    assert pipeline.events_from_tracks(tracks(), META, SCENE, thresholds=enabled) == [
        [2.0, 4.0, "jaywalking"]
    ]


def test_disabled_class_and_missing_layers_do_not_run(monkeypatch: pytest.MonkeyPatch, enabled: dict) -> None:
    enabled["classes"]["jaywalking"]["enabled"] = False
    monkeypatch.setattr(
        pipeline,
        "RULES",
        {
            "jaywalking": fake_rule("jaywalking"),
            "congestion": fake_rule("congestion", layers=("lanes", "stop_lines")),  # scene has no stop lines
            "wrong_way": fake_rule("wrong_way"),
        },
    )
    assert pipeline.events_from_tracks(tracks(), META, SCENE, thresholds=enabled) == [[2.0, 4.0, "wrong_way"]]


def test_force_runs_disabled_rules_for_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "RULES", {"jaywalking": fake_rule("jaywalking")})
    ctx = VideoContext(meta=META, stride=3)
    kin = pipeline.prepare(tracks(), SCENE)
    assert pipeline.run_rules(kin, SCENE, ctx) == []
    assert len(pipeline.run_rules(kin, SCENE, ctx, force=True)) == 1


def test_rule_cannot_emit_another_label(monkeypatch: pytest.MonkeyPatch, enabled: dict) -> None:
    rogue = fake_rule("wrong_way", segments=[Segment(1.0, 2.0, "jaywalking", 1.0)])
    monkeypatch.setattr(pipeline, "RULES", {"wrong_way": rogue})
    assert pipeline.events_from_tracks(tracks(), META, SCENE, thresholds=enabled) == []


def test_real_rules_on_real_config_give_no_events() -> None:
    assert pipeline.events_from_tracks(tracks(), META, SCENE) == []


def test_detect_events_skips_perception_when_nothing_is_enabled(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path
) -> None:
    def no_perception(*args, **kwargs):
        raise AssertionError("perception must not run when no class can be emitted")

    monkeypatch.setattr("roadwatch.perception.run.run_perception", no_perception)
    assert pipeline.detect_events(str(tiny_video)) == []
