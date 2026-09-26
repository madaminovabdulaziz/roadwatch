"""Regenerate predictions_samples.json with the official harness on samples/ (RUNBOOK P3.3).

Writes predictions_samples.json at the repository root (a task deliverable), validates it with
evaluate.py, and copies it to web/public/data/ for the Links page. With --check it instead compares a
fresh run against the committed file (events exact, risk within 1e-6): the reproducibility claim.

--unpaced runs the harness with configs/unpaced.yaml (pacing off). On a machine too slow for the 3x
budget (Kaggle's 4 cores: 2.7x with pacing) the pipeline thins its work by wall-clock time, so its
output depends on the machine. Unpaced, the output is the deterministic full-quality one that any
machine fast enough to finish inside the budget reproduces exactly (SPEC §12.54).

Usage: python scripts/make_predictions_samples.py [samples/] [--check] [--unpaced]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
from roadwatch.config import REPO_ROOT  # noqa: E402
from roadwatch.devdata import diff_predictions  # noqa: E402
from scripts.check_determinism import run_harness  # noqa: E402

TARGET = REPO_ROOT / "predictions_samples.json"
WEB_COPY = REPO_ROOT / "web" / "public" / "data" / "predictions_samples.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, nargs="?", default=REPO_ROOT / "samples")
    ap.add_argument("--check", action="store_true", help="compare a fresh run with the committed file")
    ap.add_argument("--out", type=Path, default=TARGET)
    ap.add_argument("--unpaced", action="store_true", help="pacing off (configs/unpaced.yaml)")
    args = ap.parse_args()
    if args.unpaced:
        # the harness subprocess inherits the environment
        os.environ["ROADWATCH_OVERRIDES"] = str(REPO_ROOT / "configs" / "unpaced.yaml")

    if args.check:
        if not args.out.exists():
            ap.error(f"{args.out} does not exist yet")
        fresh = run_harness(args.videos.resolve(), Path(tempfile.mkdtemp()) / "fresh.json")
        problems = diff_predictions(json.loads(args.out.read_text(encoding="utf-8")), fresh)
        for p in problems:
            print("DIFF:", p)
        print("REPRODUCED" if not problems else f"{len(problems)} difference(s)")
        return 0 if not problems else 1

    pred = run_harness(args.videos.resolve(), args.out)
    errors, warnings = evaluate.validate(pred)
    for w in warnings:
        print("warning:", w)
    if errors:
        for e in errors:
            print("ERROR:", e)
        return 1
    if args.out == TARGET:
        WEB_COPY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.out, WEB_COPY)
    n_events = sum(len(v["events"]) for v in pred["videos"].values())
    print(f"wrote {args.out}: {len(pred['videos'])} video(s), {n_events} event(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
