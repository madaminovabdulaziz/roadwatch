"""Run detection + tracking once per video and save cache/tracks/<video>.parquet (RUNBOOK P0.3).

Each parquet gets a JSON sidecar with the video metadata, the exact perception settings and weights
hash that produced it (a cache made with other settings is recomputed, never silently reused), per-class
track counts and timings. Rule development then reads tracks in seconds instead of re-running YOLO.

Usage: python scripts/cache_tracks.py samples/ [--out cache/tracks] [--seconds N] [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.config import CACHE_DIR, WEIGHTS_DIR, load_thresholds  # noqa: E402
from roadwatch.perception.detector import Detector, detector_info  # noqa: E402
from roadwatch.perception.run import run_perception  # noqa: E402
from roadwatch.video import probe  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}  # same as run_submission.py


def perception_settings() -> dict[str, Any]:
    """Everything that changes the tracks: a cache is valid only if this matches.

    Normalised through JSON (integer class ids become strings) so it compares equal to a sidecar.
    """
    cfg = load_thresholds()
    perception = cfg["perception"]
    weights_meta = json.loads((WEIGHTS_DIR / f"{perception['model']}.json").read_text(encoding="utf-8"))
    settings = {
        "weights_sha256": weights_meta["sha256"],
        "stride": cfg["video"]["stride_part_a"],
        "skip_nonref": cfg["video"]["skip_nonref"],
        **{k: perception[k] for k in ("model", "imgsz", "conf_min", "nms_iou", "max_det", "keep_classes")},
        "tracker_groups": perception["tracker_groups"],
        "bytetrack": perception["bytetrack"],
    }
    return json.loads(json.dumps(settings))


def summarize(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {"rows": 0, "tracks": 0, "tracks_per_class": {}, "median_track_sec": 0.0}
    per_track = table.groupby("track_id").agg(cls=("cls", "first"), start=("t", "min"), end=("t", "max"))
    return {
        "rows": len(table),
        "tracks": len(per_track),
        "tracks_per_class": per_track["cls"].astype(str).value_counts().sort_index().to_dict(),
        "median_track_sec": round(float((per_track["end"] - per_track["start"]).median()), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, help="folder of .mp4 files, or one .mp4")
    ap.add_argument("--out", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--seconds", type=float, help="only the first N seconds of each video (quick checks)")
    ap.add_argument("--force", action="store_true", help="recompute even if a matching cache exists")
    args = ap.parse_args()

    videos = (
        [args.videos]
        if args.videos.is_file()
        else sorted(p for p in args.videos.iterdir() if p.suffix in VIDEO_EXTS)
    )
    if not videos:
        ap.error(f"no .mp4 files in {args.videos}")
    args.out.mkdir(parents=True, exist_ok=True)
    settings = perception_settings()
    detector = Detector.load()
    print("detector:", detector_info(detector))

    for path in videos:
        parquet = args.out / f"{path.stem}.parquet"
        sidecar = parquet.with_suffix(".json")
        if parquet.exists() and sidecar.exists() and not args.force:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            if cached.get("settings") == settings and cached.get("seconds") == args.seconds:
                print(f"{path.name}: cache is current, skipped ({parquet})")
                continue
        stats: dict[str, Any] = {}
        table = run_perception(path, detector=detector, t_end=args.seconds or math.inf, stats=stats)
        table.to_parquet(parquet, index=False)
        meta = probe(path)
        summary = summarize(table)
        sidecar.write_text(
            json.dumps(
                {
                    "video": {
                        "name": path.name,
                        "fps": meta.fps,
                        "width": meta.width,
                        "height": meta.height,
                        "n_frames": meta.n_frames,
                        "duration": round(meta.duration, 3),
                    },
                    "seconds": args.seconds,
                    "settings": settings,
                    "detector": detector_info(detector),
                    "stats": stats,
                    "summary": summary,
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        timing = ", ".join(
            f"{k.removesuffix('_sec')} {stats[k]:.0f} s" for k in ("wait_frames_sec", "detect_sec")
        )
        print(
            f"{path.name}: {stats['video_sec']:.0f} s of video in {stats['total_sec']:.0f} s "
            f"({stats['x_duration']:.2f}x; {timing}) | {summary['tracks']} tracks "
            f"{summary['tracks_per_class']}, median {summary['median_track_sec']} s -> {parquet}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
