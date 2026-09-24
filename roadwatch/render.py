"""Annotated video for the website and demo (WEBSITE_SPEC "Visual rules", RUNBOOK P2.3).

Contract: `render_video(video_path, tt, events, risk, scene, out_path)` writes an H.264 MP4
(libx264, yuv420p, +faststart, via ffmpeg) with class-coloured boxes, track id + speed, 2 s footprint
trails, semi-transparent scene layers, active-event banners, a risk bar and the signal state.
It never uses Ultralytics `.plot()` or any helper that fetches fonts or assets (CLAUDE.md rule 3).
`tt` is in the pixels of `video_path`; scene layers are in the scene's `image_size` pixels and must be
scaled to the video size (the demo renders on a 720p re-encode).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from roadwatch.scene.scene import Scene


def render_video(
    video_path: str | Path,
    tt: pd.DataFrame,
    events: list[list],
    risk: list[list[float]],
    scene: Scene,
    out_path: str | Path,
) -> Path:
    raise NotImplementedError("Rendering lands in RUNBOOK P2.3")
