"""Ablations and error analysis for the website (web/public/data/ablations.json).

Every row says where its number comes from. "measured" rows are computed here, now: evaluate.py on the
committed runs against labels/dev_gt.json, or Part A re-run on the track caches with one setting
switched off. "recorded" rows are before/after numbers of code that no longer exists (the old Part B
risk, the old near_miss trigger), quoted from the SPEC decision that measured them.

Error analysis, from predictions_samples.json against the dev labels:
- a temporal confusion matrix: every predicted and labelled event matched class-agnostically at
  tIoU >= 0.3 (greedy, like evaluate.py), so a row shows what each labelled class was predicted as
  ("none" = missed) and the "none" row what fired on unlabelled time;
- per class, the median offset of predicted starts and ends against the labels (matched same-class
  pairs), the bias a boundary correction would remove.

Usage: python scripts/write_ablations.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
from roadwatch import pipeline  # noqa: E402
from roadwatch.config import CACHE_DIR, REPO_ROOT, load_thresholds  # noqa: E402
from roadwatch.devdata import cached_scene, cached_signals, load_cached  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402

MATCH_TIOU = 0.3


def score(gt: dict[str, Any], videos: dict[str, Any]) -> dict[str, Any]:
    """evaluate.py's Part A on the videos the dev labels cover."""
    pred = {"videos": {v: {"events": videos[v]["events"], "risk": []} for v in gt}}
    return evaluate.evaluate(gt, pred)["part_a"]


def cache_events(gt: dict[str, Any], cache: Path, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Part A from the track caches with the given thresholds, for the videos the labels cover."""
    scene = Scene.load()
    out = {}
    for video in sorted(gt):
        stem = Path(video).stem
        tt, meta = load_cached(stem, cache)
        events = pipeline.events_from_tracks(
            tt,
            meta,
            cached_scene(stem, cache, scene),
            signal_timeline=cached_signals(stem, cache, scene),
            thresholds=thresholds,
        )
        out[video] = {"events": events}
    return out


def class_f1(part_a: dict[str, Any], label: str) -> float:
    return round(part_a["per_class"].get(label, {}).get("f1_mean", 0.0), 3)


def ablations(gt: dict[str, Any], cache: Path) -> list[dict[str, Any]]:
    unpaced = json.loads((REPO_ROOT / "predictions_samples.json").read_text(encoding="utf-8"))["videos"]
    official_path = REPO_ROOT / "runs" / "kaggle_t4_official.json"
    rows: list[dict[str, Any]] = []

    frame_rate = [
        {"variant": "every 3rd frame (10 Hz), as submitted", "value": round(score(gt, unpaced)["score_a"], 3)}
    ]
    if official_path.exists():
        paced = json.loads(official_path.read_text(encoding="utf-8"))["videos"]
        frame_rate.append(
            {
                "variant": "frames thinned by the old pacer (4-core Kaggle T4)",
                "value": round(score(gt, paced)["score_a"], 3),
            }
        )
    rows.append(
        {
            "title": "Frame rate",
            "metric": "dev Score A",
            "rows": frame_rate,
            "takeaway": "Tracking at 10 Hz matters: when the old pacer dropped frames to save time, tracks "
            "broke and a fifth of the score went. The pacer now thins only when a run would overrun.",
            "source": "measured: evaluate.py on predictions_samples.json and runs/kaggle_t4_official.json "
            "(SPEC §12.56)",
        }
    )

    base = load_thresholds()
    no_guard = copy.deepcopy(base)
    no_guard["classes"]["solid_line_crossing"]["params"]["occluded_cover"] = 2.0  # never occluded
    with_guard = score(gt, cache_events(gt, cache, base))
    without_guard = score(gt, cache_events(gt, cache, no_guard))
    rows.append(
        {
            "title": "Occlusion guard for line crossings",
            "metric": "solid_line_crossing mean F1 (dev)",
            "rows": [
                {"variant": "with the guard", "value": class_f1(with_guard, "solid_line_crossing")},
                {"variant": "without it", "value": class_f1(without_guard, "solid_line_crossing")},
            ],
            "takeaway": "A bus half hidden behind another has a box that ends at the other bus's roof; its "
            "footprint slid across a solid line. Footprints a nearer vehicle covers no longer count.",
            "source": "measured: Part A on the track caches, with and without the guard (SPEC §12.55)",
        }
    )

    rows += [
        {
            "title": "Part B: what counts as a conflict",
            "metric": "false alarms in 7.4 min of accident-free traffic",
            "rows": [
                {"variant": "collision course between box centres", "value": 31},
                {"variant": "bumper gaps, lane-aware path, emergency braking needed", "value": 2},
            ],
            "takeaway": "Queues closing up, side-by-side lanes and far-away jitter all looked like collision "
            "courses. A conflict now has to need emergency braking, measured where the camera resolves it.",
            "source": "recorded: SPEC §12.47 (the old risk features no longer exist)",
        },
        {
            "title": "failure_to_yield: what 'on the crossing' means",
            "metric": "false positives on the dev videos",
            "rows": [
                {"variant": "anyone inside the zebra polygon", "value": 51},
                {"variant": "on its roadway part, in the vehicle's path, cars/buses/trucks only", "value": 8},
            ],
            "takeaway": "The zebras' ends lie on the kerb and the island, where people wait all day.",
            "source": "recorded: SPEC §12.49",
        },
        {
            "title": "near_miss: trigger",
            "metric": "false near misses on the dev videos (the samples contain none)",
            "rows": [
                {"variant": "collision course + hard braking", "value": 23},
                {"variant": "Part B's conflict test + the mover's own braking", "value": 4},
            ],
            "takeaway": "Better, but 4 false events in 7.4 min with no real near miss to measure recall on: "
            "the class stays off rather than risk adding a zero class to the score.",
            "source": "recorded: SPEC §12.52 and §12.57",
        },
        {
            "title": "Rejected: jaywalking only in the near field",
            "metric": "jaywalking mean F1 (dev)",
            "rows": [
                {"variant": "whole carriageway (kept)", "value": 0.444},
                {"variant": "only where 1 px <= 8 cm of road", "value": 0.412},
            ],
            "takeaway": "It split a long merged event but lost a real one; people waiting in the kerb lane "
            "are on the road by the definition, so the rule was kept as it is.",
            "source": "recorded: SPEC §12.55",
        },
    ]
    return rows


def _greedy(pairs: list[tuple[float, int, int]]) -> list[tuple[int, int]]:
    used_p, used_g, out = set(), set(), []
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j))
    return out


def error_analysis(gt: dict[str, Any], videos: dict[str, Any]) -> dict[str, Any]:
    counts: dict[tuple[str, str], int] = {}
    offsets: dict[str, list[tuple[float, float]]] = {}
    for video, truth in gt.items():
        pred, labels = videos[video]["events"], truth["events"]
        pairs = [
            (iou, i, j)
            for i, p in enumerate(pred)
            for j, g in enumerate(labels)
            if (iou := evaluate.tiou((p[0], p[1]), (g[0], g[1]))) >= MATCH_TIOU
        ]
        matched = _greedy(pairs)
        for i, j in matched:
            key = (labels[j][2], pred[i][2])
            counts[key] = counts.get(key, 0) + 1
        for i in set(range(len(pred))) - {i for i, _ in matched}:
            counts[("none", pred[i][2])] = counts.get(("none", pred[i][2]), 0) + 1
        for j in set(range(len(labels))) - {j for _, j in matched}:
            counts[(labels[j][2], "none")] = counts.get((labels[j][2], "none"), 0) + 1
        same = [(iou, i, j) for iou, i, j in pairs if pred[i][2] == labels[j][2]]
        for i, j in _greedy(same):
            offsets.setdefault(pred[i][2], []).append((pred[i][0] - labels[j][0], pred[i][1] - labels[j][1]))
    truth_classes = sorted({g for g, _ in counts if g != "none"}) + ["none"]
    pred_classes = sorted({p for _, p in counts if p != "none"}) + ["none"]
    return {
        "match_tiou": MATCH_TIOU,
        "confusion": {
            "labelled": truth_classes,
            "predicted": pred_classes,
            "counts": [[counts.get((g, p), 0) for p in pred_classes] for g in truth_classes],
        },
        "boundaries": {
            c: {
                "n": len(v),
                "start_median_sec": round(float(np.median([s for s, _ in v])), 2) + 0.0,  # no "-0.0"
                "end_median_sec": round(float(np.median([e for _, e in v])), 2) + 0.0,
            }
            for c, v in sorted(offsets.items())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=REPO_ROOT / "labels" / "dev_gt.json")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "web" / "public" / "data" / "ablations.json")
    args = ap.parse_args()
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    videos = json.loads((REPO_ROOT / "predictions_samples.json").read_text(encoding="utf-8"))["videos"]
    data = {"ablations": ablations(gt, args.cache), "errors": error_analysis(gt, videos)}
    args.out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
