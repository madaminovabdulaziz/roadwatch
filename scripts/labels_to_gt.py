"""Merge labels/raw/*.csv into labels/dev_gt.json in the organizers' ground-truth format.

Validates labels, unions same-class overlaps, reads duration/fps from each video.

Usage: python scripts/labels_to_gt.py
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("scripts/labels_to_gt.py lands in RUNBOOK P1.4")


if __name__ == "__main__":
    sys.exit(main())
