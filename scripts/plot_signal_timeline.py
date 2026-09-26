"""Plot each signal's lamp scores and estimated state for a quick visual check (RUNBOOK P1.2).

For every sample video: align the scene to the video (the tripod moves between recordings and a lamp
box is only ~20 px wide: on C3902 the unaligned boxes miss the lamps by ~140 px; SPEC §12.34), decode
reference frames every `--every` frames at native resolution, run SignalStateEstimator.timeline, print
the state segments (compare a few with the video by eye), and save outputs/signals/<video>.png (raw
lamp scores + state band) and <video>.json (the timeline).
matplotlib comes with the runtime lock (a supervision dependency); it is used only here.

Usage: python scripts/plot_signal_timeline.py samples/ [--every 6] [--seconds 120]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.config import REPO_ROOT, SCENE_PATH  # noqa: E402
from roadwatch.scene.light import LAMPS, SignalStateEstimator, lamp_scores  # noqa: E402
from roadwatch.scene.registration import scene_for_video  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402
from roadwatch.types import Frame  # noqa: E402
from roadwatch.video import read_window  # noqa: E402

STATE_COLOURS = {"red": "#d62728", "yellow": "#e6b800", "green": "#2ca02c", "unknown": "#bbbbbb"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, help="folder of .mp4 files, or one .mp4")
    ap.add_argument("--every", type=int, default=6, help="use every N-th frame (default 6, ~0.2 s)")
    ap.add_argument("--seconds", type=float, help="only the first N seconds")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "signals")
    ap.add_argument("--scene", type=Path, default=SCENE_PATH)
    args = ap.parse_args()

    scene = Scene.load(args.scene)
    if not scene.has("signals"):
        ap.error(f"{args.scene} has no signal lamp boxes (calibrate them first)")
    videos = (
        [args.videos]
        if args.videos.is_file()
        else sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")
    )
    args.out.mkdir(parents=True, exist_ok=True)

    for path in videos:
        video_scene, reg = scene_for_video(path, scene)
        aligned = "aligned" if reg is not None and reg.confident else "NOT aligned (lamp boxes as clicked)"
        times: list[float] = []
        scores: dict[str, list[np.ndarray]] = {}

        def frames(
            video: Path = path,
            times: list[float] = times,
            scores: dict[str, list[np.ndarray]] = scores,
            scene: Scene = video_scene,
        ) -> Iterator[Frame]:
            end = args.seconds if args.seconds else math.inf
            for idx, t, img in read_window(video, 0.0, end, stride=args.every, skip_nonref=True):
                times.append(t)
                for sid, s in lamp_scores(img, scene).items():
                    scores.setdefault(sid, []).append(s)
                yield idx, t, img

        timeline = SignalStateEstimator(video_scene).timeline(frames())
        (args.out / f"{path.stem}.json").write_text(json.dumps(timeline, indent=1) + "\n", encoding="utf-8")

        fig, axes = plt.subplots(len(timeline), 1, figsize=(14, 2.6 * len(timeline)), squeeze=False)
        for ax, (sid, segs) in zip(axes[:, 0], timeline.items(), strict=True):
            s = np.asarray(scores[sid])
            for k, lamp in enumerate(LAMPS):
                ax.plot(times, s[:, k], color=STATE_COLOURS[lamp], lw=1, label=f"{lamp} lamp score")
            top = np.nanmax(s) * 1.2 if np.isfinite(s).any() else 1.0  # headroom for the state band
            for t0, t1, state in segs:
                ax.axvspan(t0, t1, ymin=0.92, ymax=1.0, color=STATE_COLOURS[state], lw=0)
            ax.set_ylim(0, top)
            ax.set_title(f"{path.name} signal {sid} (band on top = estimated state)")
            ax.set_xlabel("t (s)")
            ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / f"{path.stem}.png", dpi=110)
        plt.close(fig)

        print(f"{path.name}: scene {aligned}, {len(times)} frames -> {args.out / (path.stem + '.png')}")
        for sid, segs in timeline.items():
            print(f"  {sid}: " + ", ".join(f"{t0:.1f}-{t1:.1f} {state}" for t0, t1, state in segs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
