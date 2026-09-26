"""Replay cached tracks through Part B's scoring: risk curve, alarms and their causes per video.

Part B normally sees frames. Here its perception half (detect, track, read the lamps) is replaced by the
Part A track cache (same detector, tracker and 10 Hz stride) and the cached lamp timeline, while its
scoring half, `RiskCore.score_tracks` (online kinematics -> features -> score -> EMA -> hold), runs
unchanged. A video takes seconds instead of a detector pass, so a feature change can be measured on
real traffic before it ships (SPEC §12.47).

Alarms are counted as evaluate.py counts them: runs of score >= risk.threshold, merged when less than
2 s apart. The samples contain no accident, so every alarm is a false alarm; each is printed with the
feature that drove it and the pair or track behind it, so it can be checked in the video.

Usage: python scripts/risk_replay.py [--videos C3902 C3905] [--cache cache/tracks] [--quiet]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import roadwatch.risk as risk  # noqa: E402
from roadwatch.config import CACHE_DIR, load_thresholds  # noqa: E402
from roadwatch.devdata import cached_meta, cached_scene, cached_signals  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402
from roadwatch.types import FrameTracks  # noqa: E402

ALARM_MERGE_SEC = 2.0  # evaluate.py merges alarm runs closer than this


class _NoDetector:
    """RiskCore wants a detector at reset(); the replay never calls it."""

    def frame_size_for(self, width: int, height: int) -> tuple[int, int]:
        return width, height


def frames_from_cache(tt: pd.DataFrame, names_to_id: dict[str, int]) -> list[FrameTracks]:
    """The cached TrackTable back as the per-frame FrameTracks the tracker produced."""
    out = []
    for frame, g in tt.sort_values(["frame", "track_id"]).groupby("frame", sort=True):
        out.append(
            FrameTracks(
                frame_idx=int(frame),
                t=float(g["t"].iloc[0]),
                track_id=g["track_id"].to_numpy(np.int64),
                cls=np.array([names_to_id.get(c, -1) for c in g["cls"]], np.int64),
                conf=g["conf"].to_numpy(np.float32),
                xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
            )
        )
    return out


def terms(feats: dict[str, float], cfg: dict[str, Any]) -> dict[str, float]:
    """Each feature's contribution to z (the logit), as risk_score adds them."""
    w = cfg["weights"]

    def g(ttc: float) -> float:
        return math.exp(-ttc / cfg["ttc_scale_sec"]) if math.isfinite(ttc) else 0.0

    return {
        "ttc": w["ttc"] * g(feats["ttc_min"]),
        "closing": w["closing_speed"] * min(1.0, feats["closing_speed"] / cfg["closing_speed_norm_mps"]),
        "decel": w["decel"] * min(1.0, feats["decel_max"] / cfg["decel_norm_mps2"]),
        "wrong_way": w["wrong_way"] * feats["wrong_way"],
        "red_light": w["red_light"] * feats["red_light"],
        "ped_ttc": w["ped_ttc"] * g(feats["ped_ttc"]),
    }


def replay(stem: str, cache: Path) -> dict[str, Any]:
    """Risk curve of one cached video with, per processed frame, its features and inputs."""
    clicked = Scene.load()
    meta = cached_meta(stem, cache)
    scene = cached_scene(stem, cache, clicked)
    timeline = cached_signals(stem, cache, clicked)
    names_to_id = {v: int(k) for k, v in load_thresholds()["perception"]["keep_classes"].items()}
    tt = pd.read_parquet(cache / f"{stem}.parquet")

    core = risk.RiskCore(detector=_NoDetector(), scene=scene)
    core.reset(
        {
            "video_id": meta.video_id,
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "n_frames": meta.n_frames,
        }
    )
    signal_of = core.signal_of
    captured: dict[str, Any] = {}
    original = risk.risk_features

    def spy(rows: pd.DataFrame, *args: Any, **kwargs: Any) -> dict[str, float]:
        feats = original(rows, *args, **kwargs)
        captured["rows"], captured["feats"] = rows, feats
        return feats

    risk.risk_features = spy
    curve = []
    try:
        for ft in frames_from_cache(tt, names_to_id):
            red = set()
            for lane, sid in signal_of.items():
                state = next((s for a, b, s in timeline.get(sid, []) if a <= ft.t < b), "unknown")
                if state == "red":
                    red.add(lane)
            captured.clear()
            score = core.score_tracks(ft, ft.t, red)
            curve.append((ft.t, score, captured.get("feats"), captured.get("rows")))
    finally:
        risk.risk_features = original
    return {"meta": meta, "curve": curve, "cfg": core.cfg, "scene": core.scene}


def alarms(curve: list[tuple], threshold: float) -> list[tuple[float, float]]:
    """(start, end) of score >= threshold runs, merged when closer than ALARM_MERGE_SEC."""
    runs: list[list[float]] = []
    for t, s, *_ in curve:
        if s >= threshold:
            if runs and t - runs[-1][1] < ALARM_MERGE_SEC:
                runs[-1][1] = t
            else:
                runs.append([t, t])
    return [(a, b) for a, b in runs]


def _who(cls: str, tid: float, r: pd.Series) -> str:
    return f"{cls} {int(tid)} ({math.hypot(r.vx, r.vy):.1f} m/s, px {r.fx / 3:.0f},{r.fy / 3:.0f})"


def describe(curve: list[tuple], start: float, end: float, cfg: dict[str, Any], scene: Scene) -> str:
    """The driving feature at the alarm's peak and the pair or track behind it."""
    window = [c for c in curve if start - 1.0 <= c[0] <= end and c[2] is not None]
    if not window:
        return "no features"
    t, _, feats, rows = max(window, key=lambda c: risk.risk_score(c[2], cfg))
    contrib = terms(feats, cfg)
    top = max(contrib, key=contrib.get)
    what = f"peak {t:.1f}s {top} (+{contrib[top]:.1f})"
    if top in ("ttc", "closing", "ped_ttc") and rows is not None and len(rows) > 1:
        pairs = risk.conflicts(rows, scene, cfg)
        person = pairs["cls_b"].isin(cfg["groups"]["persons"]) if len(pairs) else pairs
        pairs = pairs[person] if top == "ped_ttc" else pairs[~person] if len(pairs) else pairs
        if len(pairs):
            p = pairs.iloc[0]
            ra = rows[rows["track_id"] == p.track_a].iloc[0]
            rb = rows[rows["track_id"] == p.track_b].iloc[0]
            what += f" | {_who(p.cls_a, p.track_a, ra)} x {_who(p.cls_b, p.track_b, rb)}"
            what += f" gap {p.gap:.1f} m closing {p.closing:.1f} ttc {p.ttc:.2f} drac {p.drac:.1f}"
    elif top == "decel" and rows is not None and len(rows):
        speed = np.hypot(rows["vx"], rows["vy"])
        along = (rows["ax"] * rows["vx"] + rows["ay"] * rows["vy"]) / speed.replace(0, np.nan)
        k = along.idxmin()
        r = rows.loc[k]
        what += (
            f" | {r.cls} {int(r.track_id)} decel {-along[k]:.1f} m/s^2 at {speed[k]:.1f} m/s,"
            f" age {r.age:.1f}s, px {r.fx / 3:.0f},{r.fy / 3:.0f}"
        )
    return what


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", help="cache stems (default: every cached video)")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--quiet", action="store_true", help="only the per-video summary")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    stems = args.videos or sorted(p.stem for p in args.cache.glob("*.parquet"))
    total_min = total_alarms = 0.0
    for stem in stems:
        result = replay(stem, args.cache)
        cfg, curve, meta = result["cfg"], result["curve"], result["meta"]
        found = alarms(curve, cfg["threshold"])
        scores = np.array([s for _, s, *_ in curve])
        minutes = meta.duration / 60
        total_min += minutes
        total_alarms += len(found)
        print(
            f"{stem}: {len(found)} alarms in {minutes:.1f} min ({len(found) / minutes:.2f}/min), "
            f"score median {np.median(scores):.3f}, p99 {np.percentile(scores, 99):.3f}, "
            f"time above threshold {(scores >= cfg['threshold']).mean():.1%}"
        )
        if not args.quiet:
            for a, b in found:
                span = f"{int(a // 60)}:{a % 60:04.1f}-{int(b // 60)}:{b % 60:04.1f}"
                print(f"  {span}  {describe(curve, a, b, cfg, result['scene'])}")
    if total_min:
        rate = total_alarms / total_min
        print(f"TOTAL: {int(total_alarms)} alarms in {total_min:.1f} min = {rate:.2f} per minute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
