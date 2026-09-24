"""Run the official harness twice on the samples and diff the two predictions.json (RUNBOOK P3.1).

Events must match exactly, risk within 1e-6. Also prints each run's wall time per video as a multiple
of the video length (the harness's own log), so the same command doubles as a budget check.

Usage: python scripts/check_determinism.py samples/ [--keep out_dir]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.config import REPO_ROOT  # noqa: E402
from roadwatch.devdata import diff_predictions  # noqa: E402


def run_harness(videos: Path, out: Path) -> dict:
    """One official run; raises with the harness output if it fails."""
    cmd = [
        sys.executable,
        "run_submission.py",
        "--videos",
        str(videos),
        "--out",
        str(out),
        "--team",
        "roadwatch",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"harness failed:\n{res.stdout}\n{res.stderr}")
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path)
    ap.add_argument("--keep", type=Path, help="keep both outputs in this folder")
    args = ap.parse_args()

    folder = args.keep or Path(tempfile.mkdtemp(prefix="roadwatch-det-"))
    folder.mkdir(parents=True, exist_ok=True)
    runs = [run_harness(args.videos.resolve(), folder / f"run{i}.json") for i in (1, 2)]
    for i, pred in enumerate(runs, 1):
        for video, log in sorted(pred.get("log", {}).items()):
            ratio = log["total_sec"] / log["duration"] if log.get("duration") else float("nan")
            errors = log.get("errors") or "none"
            print(f"run {i}: {video}: {ratio:.2f}x duration (budget 3.00x), errors: {errors}")
    problems = diff_predictions(*runs)
    for p in problems:
        print("DIFF:", p)
    print("IDENTICAL" if not problems else f"{len(problems)} difference(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
