"""Runtime against the 3x budget: decode, detect, track, rules, refine, Part B.

Reports seconds per video-second per stage; target total <= 1.0x.

Usage: python scripts/bench.py samples/
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("scripts/bench.py lands in RUNBOOK P0.2")


if __name__ == "__main__":
    sys.exit(main())
