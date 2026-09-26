"""Contact sheets of what each rule emits on one video, for review by eye (SPEC §7's final check).

Runs the chosen rules on the video's track cache (every rule forced on, as `eval_dev.py --all-classes`
does), and for every raw segment writes one JPEG: six frames from 1 s before its start to its end, with
the boxes of the tracks behind it drawn and the time on each frame. Needs the video itself, so it can run
where the video is (e.g. Kaggle) and only the small sheets travel.

Usage: python scripts/event_sheets.py VIDEO.mp4 [--cache cache/tracks] [--labels accident red_light ...]
       [--out outputs/sheets]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch import pipeline  # noqa: E402
from roadwatch.config import CACHE_DIR, REPO_ROOT, load_thresholds  # noqa: E402
from roadwatch.devdata import cached_scene, cached_signals, load_cached  # noqa: E402
from roadwatch.events.base import VideoContext  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402
from roadwatch.video import read_window  # noqa: E402

FRAMES = 6
TILE_W = 960
COLOURS = [(0, 0, 255), (0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 255, 0), (255, 255, 255)]


def frame_at(video: Path, t: float) -> np.ndarray | None:
    return next((img for _, _, img in read_window(video, t, t + 0.5)), None)


def sheet(video: Path, prep, label: str, start: float, end: float, tids: tuple[int, ...]) -> np.ndarray:
    tiles = []
    for t in np.linspace(max(0.0, start - 1.0), end, FRAMES):
        img = frame_at(video, float(t))
        if img is None:
            continue
        k = TILE_W / img.shape[1]
        img = cv2.resize(img, (TILE_W, round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        for n, tid in enumerate(tids):
            rows = prep[prep["track_id"] == tid]
            if not len(rows):
                continue
            row = rows.iloc[(rows["t"] - t).abs().argsort()[:1]]
            if abs(float(row["t"].iloc[0]) - t) > 0.6:
                continue
            x1, y1, x2, y2 = (row[["x1", "y1", "x2", "y2"]].to_numpy()[0] * k).astype(int)
            colour = COLOURS[n % len(COLOURS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(img, str(tid), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        cv2.putText(img, f"{label} t={t:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(img)
    rows_of_two = [cv2.hconcat(tiles[i : i + 2]) for i in range(0, len(tiles) - 1, 2)]
    return cv2.vconcat(rows_of_two) if rows_of_two else np.zeros((10, 10, 3), np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--labels", nargs="*", help="rules to run (default: every rule but fire_smoke)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "sheets")
    args = ap.parse_args()
    th = copy.deepcopy(load_thresholds())
    labels = args.labels or [c for c in th["classes"] if c != "fire_smoke"]
    for label in labels:
        th["classes"][label]["enabled"] = True
    stem = args.video.stem
    tt, meta = load_cached(stem, args.cache)
    scene = cached_scene(stem, args.cache, Scene.load())
    ctx = VideoContext(meta, th["video"]["stride_part_a"], cached_signals(stem, args.cache, Scene.load()))
    prep = pipeline.prepare(tt, scene)
    segments = pipeline.run_rules(prep, scene, ctx, th, labels=labels)
    args.out.mkdir(parents=True, exist_ok=True)
    for s in sorted(segments, key=lambda s: (s.label, s.start)):
        name = f"{stem}_{s.label}_{s.start:06.1f}.jpg"
        img = sheet(args.video, prep, s.label, s.start, s.end, tuple(int(t) for t in s.track_ids))
        cv2.imwrite(str(args.out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{s.label:18s} {s.start:7.1f}-{s.end:7.1f} tracks {s.track_ids} -> {name}")
    print(f"{len(segments)} sheets in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
