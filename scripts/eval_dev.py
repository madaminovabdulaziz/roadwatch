"""Score the pipeline on the dev labels with the official evaluate.py.

Runs from cached tracks, writes predictions_dev.json, prints the per-class table and every FP/FN
with timestamps.

Usage: python scripts/eval_dev.py [--no-cache]
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("scripts/eval_dev.py lands in RUNBOOK P1.4")


if __name__ == "__main__":
    sys.exit(main())
