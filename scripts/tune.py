"""Small grid search over per-class thresholds on cached tracks (RUNBOOK P1.4 / P2.5).

For each class in configs/tune_grid.yaml (or --classes): every combination of its candidate values is
applied to a copy of configs/thresholds.yaml, only that class's rule runs (even if still disabled) on
the cached tracks of every labelled video, and its mean F1 over tIoU {0.3, 0.5, 0.7} is scored with
evaluate.py's own matching. Kinematics are computed once per video. The table shows the best
combinations and the current values; nothing is written: copy round, defensible values into
thresholds.yaml by hand (small dev set, avoid overfitting).

Usage: python scripts/tune.py [--classes wrong_way congestion] [--top 5]
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
from roadwatch import pipeline  # noqa: E402
from roadwatch.config import CACHE_DIR, CONFIG_DIR, REPO_ROOT, load_thresholds  # noqa: E402
from roadwatch.devdata import load_cached  # noqa: E402
from roadwatch.events.base import VideoContext  # noqa: E402
from roadwatch.postprocess import postprocess, to_events  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402


def apply(thresholds: dict[str, Any], label: str, values: dict[str, Any]) -> dict[str, Any]:
    """Copy of the thresholds with `values` set for one class, and that class enabled."""
    th = copy.deepcopy(thresholds)
    cls = th["classes"][label]
    cls["enabled"] = True
    for key, value in values.items():
        target = cls["params"] if key in cls.get("params", {}) else cls
        if key not in target:
            raise KeyError(f"{label}: unknown parameter {key!r} (not in params nor in the class block)")
        target[key] = value
    return th


def score_class(
    label: str, th: dict[str, Any], videos: list[dict[str, Any]], gt: dict[str, Any]
) -> dict[str, Any]:
    """evaluate.py's per-class result for one class under thresholds `th`."""
    pred = {}
    for v in videos:
        segs = pipeline.run_rules(v["kin"], v["scene"], v["ctx"], th, labels=[label], force=True)
        events = to_events(postprocess(segs, v["ctx"], th), v["ctx"].meta.duration, v["ongoing_tol"], th)
        pred[v["name"]] = {"events": events}
    gt_one = {name: {**g, "events": [e for e in g["events"] if e[2] == label]} for name, g in gt.items()}
    a = evaluate.evaluate_part_a(gt_one, pred)
    return a["per_class"].get(label) or {"f1_mean": 0.0, "0.5": evaluate.prf(0, 0, 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=REPO_ROOT / "labels" / "dev_gt.json")
    ap.add_argument("--grid", type=Path, default=CONFIG_DIR / "tune_grid.yaml")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--classes", nargs="*", help="only these classes (default: every class in the grid)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max-combos", type=int, default=500, help="refuse bigger grids per class")
    args = ap.parse_args()

    if not args.gt.exists():
        ap.error(f"{args.gt} does not exist (run scripts/labels_to_gt.py first)")
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    grid = yaml.safe_load(args.grid.read_text(encoding="utf-8")) or {}
    base = load_thresholds()
    scene = Scene.load()
    stride = base["video"]["stride_part_a"]
    videos = []
    for name in sorted(gt):
        tt, meta = load_cached(Path(name).stem, args.cache)
        videos.append(
            {
                "name": name,
                "kin": pipeline.prepare(tt, scene),
                "scene": scene,
                "ctx": VideoContext(meta=meta, stride=stride),
                "ongoing_tol": stride / meta.fps,
            }
        )

    for label in args.classes or list(grid):
        if label not in grid:
            print(f"{label}: not in {args.grid.name}, skipped")
            continue
        keys = list(grid[label])
        combos = list(itertools.product(*(grid[label][k] for k in keys)))
        if len(combos) > args.max_combos:
            ap.error(f"{label}: {len(combos)} combinations > --max-combos {args.max_combos}")
        current = {
            k: base["classes"][label].get("params", {}).get(k, base["classes"][label].get(k)) for k in keys
        }
        n_gt = sum(1 for g in gt.values() for e in g["events"] if e[2] == label)
        results = []
        for combo in [tuple(current[k] for k in keys), *combos]:
            res = score_class(label, apply(base, label, dict(zip(keys, combo, strict=True))), videos, gt)
            results.append((res["f1_mean"], combo, res["0.5"]))
        now = results.pop(0)
        results.sort(key=lambda r: (-r[0], str(r[1])))  # ties: deterministic order

        print(f"\n{label}: {len(combos)} combination(s), {n_gt} labelled event(s) in {len(videos)} video(s)")
        print(f"  {'':<8}{'mean F1':>8}  {'TP/FP/FN@0.5':>13}  " + "  ".join(keys))
        for tag, (f1, combo, m) in [("current", now), *(("", r) for r in results[: args.top])]:
            counts = "{}/{}/{}".format(m["tp"], m["fp"], m["fn"])
            print(f"  {tag:<8}{f1:8.3f}  {counts:>13}  " + "  ".join(str(c) for c in combo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
