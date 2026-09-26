"""Where events happen, for the website's Dashboard (web/public/data/dashboard/event_heat.jpg).

Published events carry no position ([start, end, label]), so each one is traced back to the road users
behind it: the rules are run on the video's track cache (the same code the submission runs) and every
raw segment of the event's class that overlaps it in time names its tracks. Their footprints while that
segment holds (a jaywalker only while on the road) are drawn on configs/reference.jpg in the class
colour (configs/palette.json), mapped into the reference frame by each video's registration so that
videos framed differently line up.

Usage: python scripts/event_heat.py [--predictions predictions_samples.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch import pipeline  # noqa: E402
from roadwatch.config import CACHE_DIR, CONFIG_DIR, REPO_ROOT, load_thresholds  # noqa: E402
from roadwatch.devdata import cached_scene, cached_signals, load_cached, tracks_in_reference  # noqa: E402
from roadwatch.events.base import VideoContext  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402

OUT_WIDTH = 1920
JPEG_QUALITY = 85


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def event_paths(stem: str, events: list[list], cache: Path, scene: Scene) -> list[tuple[str, np.ndarray]]:
    """(label, reference-frame footprints) of every track behind each published event of one video."""
    tt, meta = load_cached(stem, cache)
    th = load_thresholds()
    video_scene = cached_scene(stem, cache, scene)
    ctx = VideoContext(meta, th["video"]["stride_part_a"], cached_signals(stem, cache, scene))
    labels = sorted({e[2] for e in events})
    segments = pipeline.run_rules(pipeline.prepare(tt, video_scene), video_scene, ctx, th, labels=labels)
    in_ref = tracks_in_reference(tt, stem, cache)
    paths = []
    for start, end, label in events:
        for s in segments:
            if s.label != label or s.start > end or s.end < start:
                continue
            # only while the rule held for this track (a jaywalker on the road, not on the pavement)
            a, b = max(start, s.start), min(end, s.end)
            for tid in s.track_ids:
                rows = in_ref[(in_ref["track_id"] == tid) & in_ref["t"].between(a, b)].sort_values("t")
                if len(rows):
                    paths.append((label, rows[["fx", "fy"]].to_numpy(np.float64)))
    return paths


def draw(paths: list[tuple[str, np.ndarray]], bg: np.ndarray, palette: dict[str, str]) -> np.ndarray:
    k = OUT_WIDTH / bg.shape[1]
    img = cv2.resize(bg, (OUT_WIDTH, round(bg.shape[0] * k)), interpolation=cv2.INTER_AREA)
    img = (img * 0.45).astype(np.uint8)
    layer = img.copy()
    for label, pts in paths:
        colour = _bgr(palette.get(label, "#ffffff"))
        p = np.round(pts * k).astype(np.int32)
        if len(p) > 1:
            cv2.polylines(layer, [p.reshape(-1, 1, 2)], False, colour, 3, cv2.LINE_AA)
        cv2.circle(layer, tuple(int(v) for v in p[-1]), 6, colour, -1, cv2.LINE_AA)
    img = cv2.addWeighted(layer, 0.85, img, 0.15, 0)
    shown = sorted({label for label, _ in paths})
    for i, label in enumerate(shown):  # legend, top left
        y = 40 + 34 * i
        cv2.rectangle(img, (24, y - 18), (48, y + 4), _bgr(palette.get(label, "#ffffff")), -1)
        cv2.putText(img, label, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=REPO_ROOT / "predictions_samples.json")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "web" / "public" / "data" / "dashboard" / "event_heat.jpg"
    )
    args = ap.parse_args()
    pred = json.loads(args.predictions.read_text(encoding="utf-8"))["videos"]
    palette = json.loads((CONFIG_DIR / "palette.json").read_text(encoding="utf-8"))["classes"]
    scene = Scene.load()
    paths = []
    for name in sorted(pred):
        stem = Path(name).stem
        if not (args.cache / f"{stem}.parquet").exists():
            print(f"{name}: skipped (no track cache)")
            continue
        found = event_paths(stem, pred[name]["events"], args.cache, scene)
        print(f"{name}: {len(pred[name]['events'])} events, {len(found)} tracks behind them")
        paths += found
    bg = cv2.imread(str(CONFIG_DIR / "reference.jpg"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), draw(paths, bg, palette), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
