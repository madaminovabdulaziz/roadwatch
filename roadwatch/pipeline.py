"""Part A orchestration: `detect_events(video_path) -> [[start_sec, end_sec, label], ...]`.

Flow (SPEC §3-§7): probe -> perception (shared detector, stride `video.stride_part_a`) -> kinematics
-> signal timeline -> every enabled, runnable rule in its own try/except -> postprocess -> to_events.

Guarantees once implemented (RUNBOOK P0.3, P2.1):
- one failing rule logs and contributes no events; it never takes down the others;
- Part A stops refining and returns what it has after `runtime.part_a_deadline_factor` x duration,
  because the harness decodes every frame for Part B inside the same 3x budget (SPEC §12);
- the output passes evaluate.py's format check.

Current state: perception is not implemented, so this returns [] without decoding the video.
"""

from __future__ import annotations

import logging

from roadwatch.video import probe

log = logging.getLogger(__name__)


def detect_events(video_path: str) -> list[list]:
    meta = probe(video_path)
    log.info(
        "%s: %.1f s @ %.3f fps; pipeline not implemented yet, no events",
        meta.video_id,
        meta.duration,
        meta.fps,
    )
    return []
