"""Segment post-processing (SPEC §6, RUNBOOK P2.1), applied in this order:

1. drop segments with score < the class's `min_score`;
2. per class, merge segments separated by less than `merge_gap`;
3. drop segments shorter than `min_dur`;
4. boundary refinement: re-read +/- `refine_window_sec` at stride 1 around starts/ends of
   `refine_classes`, within the `runtime.refine_budget_frac` frame budget;
5. same-class union (overlaps become one segment);
6. clip to [0, duration], drop start >= end, plain Python floats rounded to `round_decimals`.

Contract: `postprocess(segments, ctx, thresholds)` runs steps 1-5; `to_events(segments, duration,
ongoing_tol, thresholds)` runs step 6 and returns `[[start, end, label], ...]` sorted by (start, label)
that passes evaluate.py's format check. A segment ending within `ongoing_tol` of the video end was
still ongoing at the last processed frame, so it ends at `duration`. As a last guard, `to_events`
drops every label that is not an official class or not `enabled` (CLAUDE.md rule 9).

Step 4 needs video reads and lands with the rest of RUNBOOK P2.1; until then it is skipped.
Class values (`min_score`, `merge_gap`, `min_dur`) fall back to the `postprocess` defaults.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from roadwatch.config import load_thresholds
from roadwatch.events.base import VideoContext
from roadwatch.types import Segment


def class_value(thresholds: dict[str, Any], label: str, key: str) -> Any:
    """A post-processing value for one class, falling back to the `postprocess` defaults."""
    cls = thresholds["classes"].get(label, {})
    return cls[key] if key in cls else thresholds["postprocess"][key]


def _join(a: Segment, b: Segment) -> Segment:
    """One segment covering both (same label): max score, union of track ids, a's meta first."""
    return Segment(
        start=min(a.start, b.start),
        end=max(a.end, b.end),
        label=a.label,
        score=max(a.score, b.score),
        track_ids=tuple(sorted(set(a.track_ids) | set(b.track_ids))),
        meta={**b.meta, **a.meta},
    )


def merge_close(segments: list[Segment], gap: float) -> list[Segment]:
    """Merge same-label segments whose gap is < `gap` (overlaps have a negative gap, so they merge too)."""
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        if out and seg.start - out[-1].end < gap:
            out[-1] = _join(out[-1], seg)
        else:
            out.append(seg)
    return out


def postprocess(
    segments: list[Segment], ctx: VideoContext, thresholds: dict[str, Any] | None = None
) -> list[Segment]:
    """Steps 1-5 per class (see the module docstring); the result is sorted by (start, label)."""
    thresholds = thresholds or load_thresholds()
    by_label: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        by_label[seg.label].append(seg)

    out: list[Segment] = []
    for label in sorted(by_label):
        kept = [s for s in by_label[label] if s.score >= class_value(thresholds, label, "min_score")]
        kept = merge_close(kept, class_value(thresholds, label, "merge_gap"))
        kept = [s for s in kept if s.end - s.start >= class_value(thresholds, label, "min_dur")]
        out.extend(merge_close(kept, 0.0))  # step 5: union of what still overlaps after step 4
    return sorted(out, key=lambda s: (s.start, s.label))


def to_events(
    segments: list[Segment],
    duration: float,
    step: float = 0.0,
    thresholds: dict[str, Any] | None = None,
) -> list[list]:
    """Step 6: clipped, rounded, non-overlapping `[start, end, label]` lists of enabled classes only.

    `step` is the sample step (stride / fps). A segment still running at the last processed frame,
    which can sit up to one step before the end, ends at the video's duration. The tolerance is 1.5
    steps plus an epsilon, because (n - 3) / fps and n / fps - 3 / fps can differ in the last bit.
    """
    ongoing_tol = 1.5 * step + 1e-6 if step > 0 else 0.0
    thresholds = thresholds or load_thresholds()
    # thresholds["classes"] lists exactly the official ids (tests/test_config.py guards that)
    enabled = {label for label, cfg in thresholds["classes"].items() if cfg.get("enabled", False)}
    decimals = thresholds["postprocess"]["round_decimals"]
    clipped: list[Segment] = []
    for seg in segments:
        if seg.label not in enabled:
            continue
        start = max(0.0, float(seg.start))
        end = float(duration) if seg.end >= duration - ongoing_tol else min(float(seg.end), float(duration))
        if start < end:
            clipped.append(Segment(start, end, seg.label, seg.score, seg.track_ids, seg.meta))

    events: list[list] = []
    by_label: dict[str, list[Segment]] = defaultdict(list)
    for seg in clipped:
        by_label[seg.label].append(seg)
    for label, segs in by_label.items():
        for seg in merge_close(segs, 0.0):  # clipping can make segments touch; keep them disjoint
            start, end = round(seg.start, decimals), round(seg.end, decimals)
            if start < end:
                events.append([float(start), float(end), label])
    return sorted(events, key=lambda e: (e[0], e[2]))
