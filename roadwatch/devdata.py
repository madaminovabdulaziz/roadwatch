"""Development data: our dev labels and cached tracks (RUNBOOK P1.4). Never used at inference.

Contract:
- `parse_time("75.5" | "1:15.5" | "0:01:15.5")` -> seconds.
- `read_label_rows(csv_paths)` reads `video,start,end,label,notes` rows (labels/raw/*.csv). A row with
  label `none` marks a video as fully watched with no events, so it enters the ground truth with [].
- `build_gt(rows, meta_of)` validates and returns the organizers' GT format
  `{video: {"duration", "fps", "events": [[start, end, label], ...]}}`, same-class overlaps (and
  touching segments) unioned; problems come back as messages with file:line.
- `cached_meta(stem, cache_dir)` / `load_cached(stem, cache_dir)` read cache/tracks/<stem>.json and
  .parquet written by scripts/cache_tracks.py.
- `video_meta(name, videos_dir, cache_dir)` probes the video if it is here, else uses the cache sidecar
  (the samples may only live on Kaggle).
- `cached_signals(stem, cache_dir, scene)` returns the cached signal timeline, or {} when it was measured
  with other lamp boxes than the current scene's.
- `diff_predictions(a, b)` lists the differences between two harness outputs (determinism and
  reproducibility checks, RUNBOOK P3.1/P3.3).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from roadwatch.scene.light import SignalTimeline
from roadwatch.scene.scene import Scene
from roadwatch.types import VideoMeta
from roadwatch.video import probe

NO_EVENTS = "none"
DURATION_SLACK = 0.5  # evaluate.py accepts end <= duration + 0.5 s


@dataclass(frozen=True)
class LabelRow:
    source: str  # file:line, for error messages
    video: str
    start: float
    end: float
    label: str
    notes: str = ""


def parse_time(text: str) -> float:
    """Seconds from "75.5", "1:15.5" (m:s) or "0:01:15.5" (h:m:s)."""
    parts = [float(p) for p in str(text).strip().split(":")]
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise ValueError(f"not a time: {text!r}")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def read_label_rows(paths: Iterable[Path]) -> tuple[list[LabelRow], list[str]]:
    """All rows of the label CSVs, plus parse problems (file:line: message)."""
    rows: list[LabelRow] = []
    problems: list[str] = []
    for path in sorted(paths):
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            missing = {"video", "start", "end", "label"} - set(reader.fieldnames or [])
            if missing:
                problems.append(f"{path.name}: missing column(s) {', '.join(sorted(missing))}")
                continue
            for line, rec in enumerate(reader, start=2):
                where = f"{path.name}:{line}"
                label = (rec.get("label") or "").strip()
                video = (rec.get("video") or "").strip()
                if not video and not label:
                    continue  # blank line
                try:
                    start = parse_time(rec["start"]) if label != NO_EVENTS else 0.0
                    end = parse_time(rec["end"]) if label != NO_EVENTS else 0.0
                except (ValueError, TypeError) as exc:
                    problems.append(f"{where}: {exc}")
                    continue
                rows.append(LabelRow(where, video, start, end, label, (rec.get("notes") or "").strip()))
    return rows, problems


def _union(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(segments):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def build_gt(
    rows: list[LabelRow], meta_of: Callable[[str], VideoMeta | None], classes: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Organizer-format ground truth from label rows, plus validation problems."""
    problems: list[str] = []
    per_video: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    metas: dict[str, VideoMeta] = {}
    labelled: set[str] = set()
    for r in rows:
        if r.video not in metas:
            meta = meta_of(r.video)
            if meta is None:
                problems.append(f"{r.source}: video {r.video!r} not found (no file, no cache sidecar)")
                continue
            metas[r.video] = meta
        meta = metas[r.video]
        labelled.add(r.video)  # a "none" row still puts the video in the ground truth
        if r.label == NO_EVENTS:
            continue
        if r.label not in classes:
            problems.append(f"{r.source}: label {r.label!r} is not an official class")
        elif not 0 <= r.start < r.end:
            problems.append(f"{r.source}: need 0 <= start < end, got {r.start} .. {r.end}")
        elif r.end > meta.duration + DURATION_SLACK:
            problems.append(f"{r.source}: end {r.end} is past the video end {meta.duration:.3f}")
        else:
            per_video[r.video][r.label].append((r.start, min(r.end, meta.duration)))

    gt: dict[str, Any] = {}
    for video in sorted(labelled):
        events = [
            [round(s, 3), round(e, 3), label]
            for label, segs in per_video[video].items()
            for s, e in _union(segs)
        ]
        gt[video] = {
            "duration": round(metas[video].duration, 3),
            "fps": metas[video].fps,
            "events": sorted(events, key=lambda ev: (ev[0], ev[2])),
        }
    return gt, problems


def cached_meta(stem: str, cache_dir: Path) -> VideoMeta | None:
    """VideoMeta from cache/tracks/<stem>.json (written by scripts/cache_tracks.py), or None."""
    sidecar = cache_dir / f"{stem}.json"
    if not sidecar.exists():
        return None
    v = json.loads(sidecar.read_text(encoding="utf-8"))["video"]
    return VideoMeta(v["name"], float(v["fps"]), int(v["width"]), int(v["height"]), int(v["n_frames"]))


def load_cached(stem: str, cache_dir: Path) -> tuple[pd.DataFrame, VideoMeta]:
    """Cached TrackTable and its video metadata; raises FileNotFoundError with the fix if missing."""
    parquet = cache_dir / f"{stem}.parquet"
    meta = cached_meta(stem, cache_dir)
    if not parquet.exists() or meta is None:
        raise FileNotFoundError(f"no track cache for {stem} in {cache_dir} (run scripts/cache_tracks.py)")
    sidecar = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
    if sidecar.get("seconds"):
        raise ValueError(f"{parquet} covers only the first {sidecar['seconds']} s; recache the full video")
    return pd.read_parquet(parquet), meta


def cached_signals(stem: str, cache_dir: Path, scene: Scene) -> SignalTimeline:
    """The lamp timeline stored by scripts/cache_tracks.py, if it matches the scene's signal boxes."""
    sidecar = cache_dir / f"{stem}.json"
    if not sidecar.exists():
        return {}
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if data.get("signals") != (scene.layers.get("signals") or []):
        return {}
    return {sid: [tuple(seg) for seg in segs] for sid, segs in data.get("signal_timeline", {}).items()}


def video_meta(name: str, videos_dir: Path, cache_dir: Path) -> VideoMeta | None:
    """Probe `videos_dir/name` if present, else fall back to the track-cache sidecar."""
    path = videos_dir / name
    if path.exists():
        return probe(path)
    return cached_meta(Path(name).stem, cache_dir)


def diff_predictions(a: dict[str, Any], b: dict[str, Any], risk_tol: float = 1e-6) -> list[str]:
    """Differences between two harness outputs: events must match exactly, risk within `risk_tol`."""
    out: list[str] = []
    va, vb = a.get("videos", {}), b.get("videos", {})
    for video in sorted(set(va) | set(vb)):
        if video not in va or video not in vb:
            out.append(f"{video}: only in {'the first' if video in va else 'the second'} run")
            continue
        ea, eb = va[video].get("events", []), vb[video].get("events", [])
        if ea != eb:
            out.append(f"{video}: events differ ({len(ea)} vs {len(eb)})")
        ra, rb = va[video].get("risk", []), vb[video].get("risk", [])
        if len(ra) != len(rb):
            out.append(f"{video}: risk lengths differ ({len(ra)} vs {len(rb)})")
        elif ra:
            worst = max(max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(ra, rb, strict=True))
            if worst > risk_tol:
                out.append(f"{video}: risk differs by up to {worst:.2g}")
    return out
