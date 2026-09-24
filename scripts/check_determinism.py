"""Run the official harness twice on the samples and diff the two predictions.json.

Events must match exactly, risk within 1e-6.

Usage: python scripts/check_determinism.py samples/
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("scripts/check_determinism.py lands in RUNBOOK P3.1")


if __name__ == "__main__":
    sys.exit(main())
