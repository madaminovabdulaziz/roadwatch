"""Propose lane directions from cached tracks; write them to scene.json after confirmation (RUNBOOK P1.2).

For every lane polygon in configs/scene.json it prints the proposed image-space direction (mean unit
vector of moving vehicles' steps), how many steps support it, how consistent they are, and the angle
to the direction already in scene.json. Nothing is written unless you answer "y" (or pass --yes);
the previous file is kept as scene.json.bak. Lanes that fail `lane_flow.min_samples` or
`lane_flow.min_consistency` are never written.

Also writes the grid flow field used by the EDA page (--field).

Usage: python scripts/learn_lane_flow.py [--tracks cache/tracks] [--field web/public/data/eda/flow_field.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.config import CACHE_DIR, SCENE_PATH  # noqa: E402
from roadwatch.scene.flow import flow_field, lane_directions_from_tracks  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between two 2-D vectors in degrees."""
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, default=CACHE_DIR / "tracks", help="folder of <video>.parquet")
    ap.add_argument("--scene", type=Path, default=SCENE_PATH)
    ap.add_argument("--field", type=Path, help="also write the flow-field JSON here")
    ap.add_argument("--yes", action="store_true", help="write without asking")
    args = ap.parse_args()

    parquets = sorted(args.tracks.glob("*.parquet"))
    if not parquets:
        ap.error(f"no track caches in {args.tracks} (run scripts/cache_tracks.py first)")
    if not args.scene.exists():
        ap.error(f"{args.scene} does not exist (run scripts/calibrate_scene.py first)")
    # track ids restart per video: offset them so steps never join two videos
    tables, offset = [], 0
    for p in parquets:
        t = pd.read_parquet(p)
        t["track_id"] = t["track_id"].astype(np.int64) + offset
        offset = int(t["track_id"].max()) + 1 if len(t) else offset
        tables.append(t)
    tt = pd.concat(tables, ignore_index=True)
    scene_dict = json.loads(args.scene.read_text(encoding="utf-8"))
    scene = Scene(scene_dict)

    proposals = lane_directions_from_tracks(tt, scene)
    current = scene.lane_directions()
    print(f"{len(parquets)} video(s), {len(tt)} track rows")
    print(f"{'lane':<10}{'steps':>7}{'consistency':>13}{'proposed (dx, dy)':>22}{'vs current':>12}  use")
    for r in proposals.itertuples():
        diff = angle_deg(np.array([r.dx, r.dy]), current[r.lane_id]) if r.lane_id in current and r.n else None
        vec = f"({r.dx:+.3f}, {r.dy:+.3f})" if r.n else "-"
        print(
            f"{r.lane_id:<10}{r.n:>7}{r.consistency:>13.2f}{vec:>22}"
            f"{(f'{diff:.0f} deg' if diff is not None else '-'):>12}  {'yes' if r.ok else 'no'}"
        )

    if args.field:
        width, height = scene_dict.get("image_size", (3840, 2160))
        args.field.parent.mkdir(parents=True, exist_ok=True)
        args.field.write_text(json.dumps(flow_field(tt, width, height)) + "\n", encoding="utf-8")
        print(f"wrote {args.field}")

    usable = proposals[proposals["ok"]]
    if usable.empty:
        print("no lane has a usable proposal; scene.json unchanged")
        return 0
    if (
        not args.yes
        and input(f"write {len(usable)} lane direction(s) to {args.scene}? [y/N] ").strip().lower() != "y"
    ):
        print("scene.json unchanged")
        return 0
    new_dirs = {r.lane_id: [round(r.dx, 4), round(r.dy, 4)] for r in usable.itertuples()}
    for lane in scene_dict.get("lanes", []):
        if lane["id"] in new_dirs:
            lane["direction"] = new_dirs[lane["id"]]
    shutil.copyfile(args.scene, args.scene.with_suffix(".json.bak"))
    args.scene.write_text(json.dumps(scene_dict, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(new_dirs)} lane direction(s) to {args.scene} (previous kept as .bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
