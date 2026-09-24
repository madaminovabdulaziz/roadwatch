"""configs/thresholds.yaml covers every official class and seeding is repeatable."""

from __future__ import annotations

import numpy as np

import evaluate
from roadwatch.config import class_cfg, enabled_classes, load_thresholds, seed_everything

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


def test_seed_everything_makes_numpy_repeatable() -> None:
    seed_everything(0)
    first = np.random.rand(5)
    seed_everything(0)
    assert np.array_equal(first, np.random.rand(5))
