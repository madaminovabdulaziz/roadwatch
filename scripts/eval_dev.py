"""Score the pipeline on the dev labels with the official evaluate.py (RUNBOOK P1.4).

Runs Part A from cached tracks (cache/tracks/<video>.parquet; --no-cache runs the full pipeline on
samples/), writes predictions_dev.json, prints evaluate.py's per-class table, then every false
positive and false negative at --tiou with timestamps so they can be looked at in the video. Also
writes the dev metrics for the website (web/public/data/metrics.json).

Usage: python scripts/eval_dev.py [--no-cache] [--tiou 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
from roadwatch import pipeline  # noqa: E402
from roadwatch.config import CACHE_DIR, REPO_ROOT  # noqa: E402
from roadwatch.devdata import cached_signals, load_cached  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402


def unmatched(
    gt: list[tuple[float, float]], pred: list[tuple[float, float]], thr: float
) -> tuple[list[int], list[int]]:
    """Indices of unmatched GT (FN) and predictions (FP): evaluate.match_segments' greedy matching."""
    pairs = sorted(
        ((evaluate.tiou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)),
        reverse=True,
    )
    used_g, used_p = set(), set()
    for iou, i, j in pairs:
        if iou >= thr and i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
    return [i for i in range(len(gt)) if i not in used_g], [j for j in range(len(pred)) if j not in used_p]


def error_list(gt: dict[str, Any], pred: dict[str, Any], thr: float) -> list[str]:
    """One line per FP/FN: video, class, kind, [start, end], best tIoU with the other side."""
    lines = []
    for video in sorted(gt):
        g_events, p_events = gt[video]["events"], pred["videos"][video]["events"]
        for label in sorted({e[2] for e in g_events} | {e[2] for e in p_events}):
            gs, ps = evaluate.segs(g_events, label), evaluate.segs(p_events, label)
            fn, fp = unmatched(gs, ps, thr)
            for kind, idx, own, other in (("FN", fn, gs, ps), ("FP", fp, ps, gs)):
                for k in idx:
                    best = max((evaluate.tiou(own[k], o) for o in other), default=0.0)
                    span = f"[{own[k][0]:8.2f}, {own[k][1]:8.2f}]"
                    lines.append(f"{video:<16} {label:<20} {kind} {span} best tIoU {best:.2f}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=REPO_ROOT / "labels" / "dev_gt.json")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--videos", type=Path, default=REPO_ROOT / "samples")
    ap.add_argument("--no-cache", action="store_true", help="run perception too (slow)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "predictions_dev.json")
    ap.add_argument("--metrics", type=Path, default=REPO_ROOT / "web" / "public" / "data" / "metrics.json")
    ap.add_argument("--tiou", type=float, default=0.5, help="tIoU for the FP/FN list")
    args = ap.parse_args()

    if not args.gt.exists():
        ap.error(f"{args.gt} does not exist (write labels/raw/*.csv, then run scripts/labels_to_gt.py)")
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    scene = Scene.load()
    pred: dict[str, Any] = {"team": "roadwatch", "videos": {}}
    for video in sorted(gt):
        if args.no_cache:
            events = pipeline.detect_events(str(args.videos / video))
        else:
            tt, meta = load_cached(Path(video).stem, args.cache)
            timeline = cached_signals(Path(video).stem, args.cache, scene)
            events = pipeline.events_from_tracks(tt, meta, scene, signal_timeline=timeline)
        pred["videos"][video] = {"events": events, "risk": []}
    args.out.write_text(json.dumps(pred, indent=1) + "\n", encoding="utf-8")

    errors, _ = evaluate.validate(pred, gt)
    if errors:
        for e in errors:
            print("ERROR:", e)
        return 1
    report = evaluate.evaluate(gt, pred, per_video=True)
    evaluate.print_report(report)

    lines = error_list(gt, pred, args.tiou)
    print(f"\nFalse negatives / positives at tIoU {args.tiou} ({len(lines)}):")
    for line in lines:
        print("  " + line)

    a = report["part_a"]
    metrics = {
        "source": "scripts/eval_dev.py on labels/dev_gt.json",
        "videos": len(gt),
        "gt_events": sum(len(v["events"]) for v in gt.values()),
        "pred_events": sum(len(v["events"]) for v in pred["videos"].values()),
        "score_a": round(a["score_a"], 4),
        "per_class": {
            c: {
                **{f"f1@{t}": round(a["per_class"][c][str(t)]["f1"], 4) for t in evaluate.TIOU_THRESHOLDS},
                "f1_mean": round(a["per_class"][c]["f1_mean"], 4),
                "tp_fp_fn@0.5": [a["per_class"][c]["0.5"][k] for k in ("tp", "fp", "fn")],
            }
            for c in a["classes"]
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out} and {args.metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
