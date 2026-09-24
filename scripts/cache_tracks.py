"""Run detection + tracking once per video and save cache/tracks/<video>.parquet.

Rule iteration then reads the cache in seconds.

Usage: python scripts/cache_tracks.py samples/
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("scripts/cache_tracks.py lands in RUNBOOK P0.3")


if __name__ == "__main__":
    sys.exit(main())
