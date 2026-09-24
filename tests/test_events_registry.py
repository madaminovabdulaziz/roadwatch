"""Every official class has a rule module with the common contract (SPEC §5, §7)."""

from __future__ import annotations

import pytest

import evaluate
from roadwatch.config import class_cfg
from roadwatch.events import RULES
from roadwatch.events.base import VideoContext, is_runnable
from roadwatch.scene.scene import Scene
from roadwatch.types import VideoMeta, empty_track_table

CTX = VideoContext(
    meta=VideoMeta("clip.mp4", fps=30000 / 1001, width=3840, height=2160, n_frames=300), stride=3
)


def test_registry_lists_every_official_class_in_order() -> None:
    assert list(RULES) == evaluate.OFFICIAL_CLASSES


@pytest.mark.parametrize("label", evaluate.OFFICIAL_CLASSES)
def test_rule_follows_the_contract(label: str) -> None:
    rule = RULES[label]
    assert rule.LABEL == label
    assert isinstance(rule.REQUIRED_LAYERS, tuple)
    assert rule.detect(empty_track_table(), Scene(), CTX, class_cfg(label)) == []


def test_rule_is_not_runnable_when_disabled_or_geometry_is_missing() -> None:
    red_light = RULES["red_light"]
    enabled = {**class_cfg("red_light"), "enabled": True}
    full_scene = Scene({layer: [[0, 0], [1, 0], [1, 1]] for layer in red_light.REQUIRED_LAYERS})

    assert not is_runnable(red_light, Scene(), enabled)
    assert not is_runnable(red_light, full_scene, {**enabled, "enabled": False})
    assert is_runnable(red_light, full_scene, enabled)
