"""configs/thresholds.yaml covers every official class and seeding is repeatable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import evaluate
from roadwatch.config import class_cfg, deep_merge, enabled_classes, load_thresholds, seed_everything

REQUIRED_KEYS = {"enabled", "min_score", "merge_gap", "min_dur", "params"}


def test_every_official_class_has_a_complete_config_block() -> None:
    classes = load_thresholds()["classes"]
    assert list(classes) == evaluate.OFFICIAL_CLASSES
    for label, cfg in classes.items():
        assert REQUIRED_KEYS <= set(cfg), label
        assert isinstance(cfg["enabled"], bool), label
        assert 0.0 <= cfg["min_score"] <= 1.0, label
        assert cfg["merge_gap"] >= 0 and cfg["min_dur"] > 0, label


def test_enabled_classes_are_official_and_fire_smoke_is_off() -> None:
    assert set(enabled_classes()) <= set(evaluate.OFFICIAL_CLASSES)
    assert class_cfg("fire_smoke")["enabled"] is False


def test_deep_merge() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": [1], "e": 5}
    assert deep_merge(base, {"a": {"c": 3}, "d": [2], "f": 6}) == {
        "a": {"b": 1, "c": 3},
        "d": [2],
        "e": 5,
        "f": 6,
    }
    assert base == {"a": {"b": 1, "c": 2}, "d": [1], "e": 5}  # not modified


def test_overrides_file_is_merged_only_when_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = tmp_path / "o.yaml"
    overrides.write_text("perception:\n  imgsz: 640\n", encoding="utf-8")
    monkeypatch.setenv("ROADWATCH_OVERRIDES", str(overrides))
    merged = load_thresholds.__wrapped__()  # bypass the per-process cache
    assert merged["perception"]["imgsz"] == 640
    assert merged["perception"]["conf_min"] == load_thresholds()["perception"]["conf_min"]
    monkeypatch.delenv("ROADWATCH_OVERRIDES")
    assert load_thresholds.__wrapped__()["perception"]["imgsz"] == load_thresholds()["perception"]["imgsz"]


def test_seed_everything_makes_numpy_repeatable() -> None:
    seed_everything(0)
    first = np.random.rand(5)
    seed_everything(0)
    assert np.array_equal(first, np.random.rand(5))
