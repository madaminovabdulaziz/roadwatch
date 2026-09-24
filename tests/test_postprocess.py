"""Post-processing steps 1-3, 5, 6 (SPEC §6): score filter, merge, min-dur, union, clip, floats."""

from __future__ import annotations

import copy

import evaluate
from roadwatch.config import load_thresholds
from roadwatch.events.base import VideoContext
from roadwatch.postprocess import merge_close, postprocess, to_events
from roadwatch.types import Segment, VideoMeta

CTX = VideoContext(meta=VideoMeta("v.mp4", 30.0, 100, 100, 3000), stride=3)


def thresholds(**enabled: bool) -> dict:
    """The real thresholds with chosen classes switched on and round per-class values for the tests."""
    th = copy.deepcopy(load_thresholds())
    for label, on in enabled.items():
        th["classes"][label].update(enabled=on, min_score=0.5, merge_gap=1.0, min_dur=0.5)
    return th


def seg(start: float, end: float, label: str = "wrong_way", score: float = 1.0, ids: tuple = (1,)) -> Segment:
    return Segment(start, end, label, score, ids)


def test_score_filter_merge_and_min_duration() -> None:
    th = thresholds(wrong_way=True)
    out = postprocess(
        [seg(0, 2, score=0.4), seg(10, 11), seg(11.5, 13, ids=(2,)), seg(20, 20.3), seg(30, 32, score=0.9)],
        CTX,
        th,
    )
    assert [(s.start, s.end) for s in out] == [(10, 13), (30, 32)]
    assert out[0].track_ids == (1, 2) and out[0].score == 1.0


def test_gap_at_threshold_is_not_merged() -> None:
    assert len(merge_close([seg(0, 1), seg(2, 3)], 1.0)) == 2
    assert len(merge_close([seg(0, 1), seg(1.99, 3)], 1.0)) == 1


def test_classes_are_processed_separately_and_sorted() -> None:
    th = thresholds(wrong_way=True, jaywalking=True)
    out = postprocess([seg(5, 7, "jaywalking"), seg(1, 3), seg(5.5, 6.5)], CTX, th)
    assert [(s.label, s.start) for s in out] == [("wrong_way", 1), ("jaywalking", 5), ("wrong_way", 5.5)]


def test_to_events_clips_rounds_and_casts() -> None:
    th = thresholds(wrong_way=True)
    events = to_events(
        [seg(-1, 2.00049), seg(50, 99.97), seg(120, 130)], duration=100.0, ongoing_tol=0.1, thresholds=th
    )
    assert events == [[0.0, 2.0, "wrong_way"], [50.0, 100.0, "wrong_way"]]  # 99.97 was ongoing at the end
    assert all(type(x) is float for e in events for x in e[:2])


def test_to_events_never_emits_disabled_or_unknown_classes() -> None:
    th = thresholds(wrong_way=True)  # jaywalking stays disabled
    events = to_events([seg(1, 2, "jaywalking"), seg(1, 2, "not_a_class"), seg(3, 4)], 10.0, thresholds=th)
    assert events == [[3.0, 4.0, "wrong_way"]]


def test_to_events_output_passes_the_official_format_check() -> None:
    th = thresholds(wrong_way=True, jaywalking=True)
    raw = [seg(0, 5), seg(4, 6), seg(5.9996, 7), seg(1, 3, "jaywalking"), seg(9, 12, "jaywalking")]
    events = to_events(postprocess(raw, CTX, th), 10.0, thresholds=th)
    errors, _ = evaluate.validate(
        {"team": "t", "videos": {"v.mp4": {"events": events, "risk": [[0.0, 0.0]]}}}
    )
    assert errors == []
    assert events[-1] == [9.0, 10.0, "jaywalking"]


def test_empty() -> None:
    assert postprocess([], CTX, thresholds()) == []
    assert to_events([], 10.0) == []
