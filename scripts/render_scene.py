"""Save an overlay PNG of configs/scene.json drawn on the reference frame (RUNBOOK P1.1).

Also prints the scene's consistency problems (roadwatch.scene.calibration.validate_scene).

Usage: python scripts/render_scene.py [--scene configs/scene.json] [--image configs/reference.jpg]
                                       [--out outputs/scene_overlay.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.config import CONFIG_DIR, REPO_ROOT, SCENE_PATH  # noqa: E402
from roadwatch.scene.calibration import validate_scene  # noqa: E402
from roadwatch.scene.overlay import draw_scene  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=SCENE_PATH)
    ap.add_argument("--image", type=Path, default=CONFIG_DIR / "reference.jpg")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "scene_overlay.png")
    args = ap.parse_args()

    if not args.scene.exists():
        ap.error(f"{args.scene} does not exist (run scripts/calibrate_scene.py first)")
    img = cv2.imread(str(args.image))
    if img is None:
        ap.error(f"cannot read {args.image}")
    scene = Scene.load(args.scene)
    problems = validate_scene(scene.layers)
    for problem in problems:
        print("problem:", problem)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), draw_scene(img, scene))
    print(f"wrote {args.out} ({len(problems)} problem(s))")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
