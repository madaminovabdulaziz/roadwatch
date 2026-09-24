"""Part A orchestration: `detect_events(video_path) -> [[start_sec, end_sec, label], ...]`.

Flow (SPEC §3-§7): probe -> perception (shared detector, stride `video.stride_part_a`) -> kinematics
-> every enabled, runnable rule in its own try/except -> postprocess -> to_events.

The same path runs from cached tracks (scripts/eval_dev.py, scripts/tune.py):
- `prepare(tt, scene)` adds kinematics once per video;
- `run_rules(kin, scene, ctx, thresholds, labels, force)` runs the chosen rules on it;
- `events_from_tracks(tt, meta, scene, ...)` = prepare + run_rules + postprocess + to_events.

Guarantees:
- one failing rule logs and contributes no events; it never takes down the others;
- with no class enabled, the video is not decoded at all (nothing could be emitted);
- perception paces itself to `runtime.perception_budget_factor` x duration and stops at
  `runtime.part_a_deadline_factor` x duration, because the harness decodes every frame for Part B
  inside the same 3x budget (SPEC §12.10);
- the output passes evaluate.py's format check (postprocess.to_events).

The signal-lamp timeline is built from the frames perception decodes anyway (detector size; lamp boxes
are scaled). Not wired yet: boundary refinement at stride 1 (postprocess step 4), which needs labelled
footage to show it helps (RUNBOOK P2.1).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd

from roadwatch.config import load_thresholds
from roadwatch.events import RULES
from roadwatch.events.base import VideoContext, is_runnable
from roadwatch.features import add_kinematics
from roadwatch.postprocess import postprocess, to_events
from roadwatch.scene.light import SignalStateEstimator, SignalTimeline
from roadwatch.scene.scene import Scene
from roadwatch.types import Segment, VideoMeta
from roadwatch.video import probe

log = logging.getLogger(__name__)


def prepare(tt: pd.DataFrame, scene: Scene) -> pd.DataFrame:
    """TrackTable with offline kinematics and zone columns (the input every rule expects)."""
    return add_kinematics(tt, scene, mode="offline")


def run_rules(
    kin: pd.DataFrame,
    scene: Scene,
    ctx: VideoContext,
    thresholds: dict[str, Any] | None = None,
    labels: Iterable[str] | None = None,
    force: bool = False,
) -> list[Segment]:
    """Raw segments of the chosen rules (default: all), each isolated in its own try/except.

    A rule runs if its class is enabled (or `force`, used when tuning a class that is still off) and
    the scene has every layer it needs.
    """
    thresholds = thresholds or load_thresholds()
    segments: list[Segment] = []
    for label in labels or RULES:
        rule, cfg = RULES[label], thresholds["classes"][label]
        if not is_runnable(rule, scene, {**cfg, "enabled": True} if force else cfg):
            continue
        try:
            found = rule.detect(kin, scene, ctx, cfg)
        except Exception:
            log.exception("%s: rule %s failed; it contributes no events", ctx.meta.video_id, label)
            continue
        segments.extend(s for s in found if s.label == label)
    return segments


def events_from_tracks(
    tt: pd.DataFrame,
    meta: VideoMeta,
    scene: Scene,
    stride: int | None = None,
    signal_timeline: SignalTimeline | None = None,
    thresholds: dict[str, Any] | None = None,
) -> list[list]:
    """Rules + post-processing on an existing TrackTable (cached or fresh): the Part A event list."""
    thresholds = thresholds or load_thresholds()
    stride = stride or thresholds["video"]["stride_part_a"]
    ctx = VideoContext(meta=meta, stride=stride, signal_timeline=signal_timeline or {})
    segments = run_rules(prepare(tt, scene), scene, ctx, thresholds)
    return to_events(postprocess(segments, ctx, thresholds), meta.duration, stride / meta.fps, thresholds)


def detect_events(video_path: str) -> list[list]:
    """Part A for one video; per-rule failures are contained, anything else propagates to solution.py."""
    start = time.perf_counter()
    thresholds = load_thresholds()
    meta = probe(video_path)
    scene = Scene.load()
    runnable = [
        label for label, rule in RULES.items() if is_runnable(rule, scene, thresholds["classes"][label])
    ]
    if not runnable:
        log.info("%s: no enabled class has the scene layers it needs; skipping perception", meta.video_id)
        return []

    from roadwatch.perception.run import run_perception  # torch loads only when something can be emitted

    runtime = thresholds["runtime"]
    signals = SignalStateEstimator(scene) if scene.has("signals") else None
    tt = run_perception(
        video_path,
        thresholds["video"]["stride_part_a"],
        budget_sec=runtime["perception_budget_factor"] * meta.duration,
        deadline=start + runtime["part_a_deadline_factor"] * meta.duration,
        on_frame=signals.observe if signals else None,
    )
    timeline = signals.finish() if signals else {}
    events = events_from_tracks(tt, meta, scene, signal_timeline=timeline, thresholds=thresholds)
    log.info(
        "%s: %d events from %d track rows in %.1f s (%.2fx duration)",
        meta.video_id,
        len(events),
        len(tt),
        time.perf_counter() - start,
        (time.perf_counter() - start) / max(meta.duration, 1e-9),
    )
    return events
