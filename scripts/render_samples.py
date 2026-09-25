"""Annotated sample videos and results data for the website (RUNBOOK P2.3, web/README.md "Data").

For every sample with a track cache, writes web/public/data/results/<id>/: annotated.mp4 (720p H.264),
poster.jpg, events.json ([[start, end, label], ...]) and risk.json ([[t, score], ...]); one short clip
per detected class (2 s before -> 2 s after its first event) for the gallery (results/gallery.json);
and results/index.json.

Events, in order of preference:
- `--predictions predictions_samples.json`: exactly what the official harness produced (also gives the
  risk curves and the runtime x duration shown on the home page);
- otherwise Part A from the track cache with the current thresholds, i.e. what the submission emits;
- `--preview-all`: every runnable rule, enabled or not, for reviewing classes before switching them on
  (RUNBOOK P2.5). Such output is marked "preview" in index.json and must not be published as results.

Usage: python scripts/render_samples.py samples/ [--predictions predictions_samples.json] [--preview-all]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch import pipeline  # noqa: E402
from roadwatch.config import CACHE_DIR, REPO_ROOT, enabled_classes, load_thresholds  # noqa: E402
from roadwatch.devdata import cached_scene, cached_signals, load_cached  # noqa: E402
from roadwatch.events import RULES  # noqa: E402
from roadwatch.events.base import VideoContext  # noqa: E402
from roadwatch.postprocess import postprocess, to_events  # noqa: E402
from roadwatch.render import render_video  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402
from roadwatch.video import read_window  # noqa: E402

CLIP_PAD_SEC = 2.0
POSTER_HEIGHT = 720


def preview_events(tt, meta, scene: Scene, timeline) -> list[list]:
    """Every runnable rule as if its class were enabled (for review only)."""
    th = copy.deepcopy(load_thresholds())
    for cfg in th["classes"].values():
        cfg["enabled"] = True
    stride = th["video"]["stride_part_a"]
    ctx = VideoContext(meta, stride, timeline)
    segs = pipeline.run_rules(pipeline.prepare(tt, scene), scene, ctx, th, labels=list(RULES))
    return to_events(postprocess(segs, ctx, th), meta.duration, stride / meta.fps, th)


def save_poster(video: Path, t: float, out: Path) -> None:
    frame = next((img for _, _, img in read_window(video, t, t + 1.0)), None)
    if frame is not None:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (round(w * POSTER_HEIGHT / h), POSTER_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, help="folder with the sample .mp4 files")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--predictions", type=Path, help="harness output to take events and risk from")
    ap.add_argument("--preview-all", action="store_true", help="run every rule (review only)")
    ap.add_argument("--no-blur", action="store_true", help="do not blur faces")
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "web" / "public" / "data" / "results")
    args = ap.parse_args()

    pred = json.loads(args.predictions.read_text(encoding="utf-8")) if args.predictions else None
    scene = Scene.load()
    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")
    index: dict[str, Any] = {"videos": [], "enabled_classes": enabled_classes()}
    if args.preview_all:
        index["preview"] = True
    gallery: list[dict[str, Any]] = []
    ratios = []

    for video in videos:
        stem = video.stem
        try:
            tt, meta = load_cached(stem, args.cache)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{video.name}: skipped ({exc})")
            continue
        timeline = cached_signals(stem, args.cache, scene)
        video_scene = cached_scene(stem, args.cache, scene)
        risk: list[list[float]] = []
        if pred and video.name in pred.get("videos", {}):
            events = pred["videos"][video.name]["events"]
            risk = pred["videos"][video.name].get("risk", [])
            log = pred.get("log", {}).get(video.name)
            if log and log.get("duration"):
                ratios.append(log["total_sec"] / log["duration"])
        elif args.preview_all:
            events = preview_events(tt, meta, video_scene, timeline)
        else:
            events = pipeline.events_from_tracks(tt, meta, video_scene, signal_timeline=timeline)

        folder = args.out / stem
        folder.mkdir(parents=True, exist_ok=True)
        render_video(
            video,
            tt,
            events,
            risk,
            video_scene,
            folder / "annotated.mp4",
            blur_faces=not args.no_blur,
            signal_timeline=timeline,
        )
        save_poster(video, events[0][0] if events else 0.0, folder / "poster.jpg")
        (folder / "events.json").write_text(json.dumps(events) + "\n", encoding="utf-8")
        (folder / "risk.json").write_text(json.dumps(risk) + "\n", encoding="utf-8")
        index["videos"].append(
            {
                "id": stem,
                "name": video.name,
                "duration": round(meta.duration, 3),
                "video": f"results/{stem}/annotated.mp4",
                "poster": f"results/{stem}/poster.jpg",
                "events": f"results/{stem}/events.json",
                "risk": f"results/{stem}/risk.json",
            }
        )
        if not args.no_clips:
            seen = {g["label"] for g in gallery}
            for s, e, label in events:
                if label in seen:
                    continue
                seen.add(label)
                clip = folder / f"clip_{label}.mp4"
                t0, t1 = max(0.0, s - CLIP_PAD_SEC), min(meta.duration, e + CLIP_PAD_SEC)
                render_video(
                    video,
                    tt,
                    events,
                    risk,
                    video_scene,
                    clip,
                    t0=t0,
                    t1=t1,
                    blur_faces=not args.no_blur,
                    signal_timeline=timeline,
                )
                save_poster(video, s, folder / f"clip_{label}.jpg")
                gallery.append(
                    {
                        "label": label,
                        "clip": f"results/{stem}/{clip.name}",
                        "poster": f"results/{stem}/clip_{label}.jpg",
                        "video": video.name,
                        "start": s,
                    }
                )
        print(f"{video.name}: {len(events)} events, {len(risk)} risk samples -> {folder}", flush=True)

    if ratios:
        index["runtime_x_duration"] = round(max(ratios), 2)  # the worst video is what the budget sees
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    (args.out / "gallery.json").write_text(json.dumps(gallery, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'index.json'} ({len(index['videos'])} videos, {len(gallery)} gallery clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
