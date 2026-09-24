"""Merge labels/raw/*.csv into labels/dev_gt.json in the organizers' ground-truth format (RUNBOOK P1.4).

CSV columns: video,start,end,label,notes. Times are seconds ("75.5") or m:s / h:m:s ("1:15.5"). A
row with label `none` marks a video as watched with no events. Labels are validated (official class,
0 <= start < end, end within the video), same-class overlaps are unioned, and duration/fps come from
each video (samples/ if present, else the cache/tracks sidecar). Nothing is written if a row is wrong.

Usage: python scripts/labels_to_gt.py [--raw labels/raw] [--videos samples] [--out labels/dev_gt.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
from roadwatch.config import CACHE_DIR, REPO_ROOT  # noqa: E402
from roadwatch.devdata import build_gt, read_label_rows, video_meta  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=REPO_ROOT / "labels" / "raw")
    ap.add_argument("--videos", type=Path, default=REPO_ROOT / "samples")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "labels" / "dev_gt.json")
    args = ap.parse_args()

    csvs = sorted(args.raw.glob("*.csv"))
    if not csvs:
        ap.error(f"no label CSVs in {args.raw}")
    rows, problems = read_label_rows(csvs)
    gt, more = build_gt(
        rows, lambda name: video_meta(name, args.videos, args.cache), evaluate.OFFICIAL_CLASSES
    )
    problems += more
    for p in problems:
        print("ERROR:", p)
    if problems:
        print(f"{len(problems)} problem(s); {args.out} not written")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gt, indent=1) + "\n", encoding="utf-8")
    counts = Counter(label for v in gt.values() for _, _, label in v["events"])
    print(
        f"{len(rows)} row(s) from {len(csvs)} file(s) -> {len(gt)} video(s), {sum(counts.values())} event(s)"
    )
    for label in evaluate.OFFICIAL_CLASSES:
        if counts[label]:
            print(f"  {label:<20} {counts[label]}")
    for r in rows:
        if r.notes:
            print(f"  note {r.source} {r.video} {r.start:.1f}-{r.end:.1f} {r.label}: {r.notes}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
