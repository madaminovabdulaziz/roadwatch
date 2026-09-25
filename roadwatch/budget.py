"""Per-video wall-clock budget shared by Part A and Part B (SPEC §12.40).

The harness gives each video 3 x its duration. The clock runs from just before `detect_events` to the
end of the Part B loop, and it includes the harness decoding every frame for Part B. Going over
empties the whole video, Part A's events included. So Part A records when a video started
(`mark_start`), and Part B paces itself against that real deadline (`started`) instead of guessing
how long Part A took.

Only a clock reading crosses from Part A to Part B, never anything about the video's content, so
Part B stays causal.
"""

from __future__ import annotations

import time

_STARTS: dict[str, float] = {}


def mark_start(video_id: str) -> None:
    """Record now as the start of this video's budget (called first thing in detect_events)."""
    _STARTS[video_id] = time.perf_counter()


def started(video_id: str) -> float | None:
    """When this video's budget started, or None if Part A never marked it."""
    return _STARTS.get(video_id)
