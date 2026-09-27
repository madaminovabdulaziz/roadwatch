"""Honest failure cases for the website's Results page (WEBSITE_SPEC "Results": failure cases with short
clips and why).

Each case below was found by reviewing the submission's errors against the dev labels (SPEC §12.55) and
checked on the frames; the text says what happened and what we did about it. Writes one annotated clip
per case (roadwatch/render.py, the same overlay as the samples) and web/public/data/results/failures.json
([{title, why, clip}]). Events and risk for the overlay come from --predictions (the harness output).

Usage: python scripts/render_failures.py samples/ [--predictions predictions_samples.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.config import CACHE_DIR, REPO_ROOT  # noqa: E402
from roadwatch.devdata import cached_scene, cached_signals, load_cached  # noqa: E402
from roadwatch.render import render_video  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402
from scripts.render_samples import WEB_CRF, WEB_PRESET  # noqa: E402

CASES: list[dict[str, Any]] = [
    {
        "slug": "jaywalking_kerb_waiters",
        "video": "C3902",
        "t0": 114.0,
        "t1": 124.0,
        "title": "People waiting in the kerb lane join crossings into one long jaywalking event",
        "why": (
            "At the far bus stop people wait in the kerb lane for 10-30 s. By the letter of the definition "
            "(a pedestrian on the carriageway outside a crossing) they count, but our labels did not mark "
            "them. While one of them is always there, the union across people turns four separate "
            "crossings (1:19-3:19) into one 120 s event, which matches none of them. Limiting the rule to "
            "the near field split it but lost a real event elsewhere, so we kept the rule as it is."
        ),
    },
    {
        "slug": "solid_line_occlusion",
        "video": "C3902",
        "t0": 226.0,
        "t1": 232.0,
        "title": "Fixed: a bus hidden behind another bus 'crossed' a solid line",
        "why": (
            "The green bus's box ended at the blue bus's roof, not on the road, so its footprint slid "
            "across the lane line while it stayed in its lane. A footprint whose bottom edge a nearer "
            "vehicle covers no longer counts for line crossings; this false event is gone, the real "
            "crossing on the same video is kept."
        ),
    },
    {
        "slug": "accident_car_beside_pedestrian",
        "video": "C3896",
        "t0": 193.0,
        "t1": 199.0,
        "title": "Fixed: a car pulling up beside a pedestrian read as an accident",
        "why": (
            "Found by reviewing the unlabelled third sample event by event. The car braked hard and then "
            "stood right next to the person, which passed both the shock and the confirmation. With a "
            "pedestrian involved, the evidence now has to come from the pedestrian: a fall, or vanishing "
            "right after the contact. A car braking beside someone is what an avoided crash looks like."
        ),
    },
    {
        "slug": "stopped_bus_far_stop",
        "video": "C3902",
        "t0": 18.0,
        "t1": 28.0,
        "title": "A bus dwelling at the far bus stop is too far away to judge",
        "why": (
            "The definition of a stopped vehicle (10 s or more on the carriageway, not queued at a signal) "
            "arguably includes a bus loading at the stop. There one pixel of the 4K frame is 9-16 cm of "
            "road, too coarse to tell standing from creeping reliably, so stopped_vehicle is switched off "
            "rather than guessed."
        ),
    },
    {
        "slug": "u_turn_with_pause",
        "video": "C3902",
        "t0": 72.0,
        "t1": 84.0,
        "title": "A U-turn with a 26 s pause, and no sign that forbids it",
        "why": (
            "The white SUV pulled into the corner at 0:48, waited 26 s for a gap and turned back at 1:17. "
            "The class is a U-turn that markings or signs prohibit; we found no such sign in this view, so "
            "no zone is drawn and illegal_u_turn is off. Our test also expects the turn within 12 s, and "
            "this one took 34 s."
        ),
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, help="folder with the sample .mp4 files")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--predictions", type=Path, help="harness output to take events and risk from")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "web" / "public" / "data" / "results")
    args = ap.parse_args()

    pred = json.loads(args.predictions.read_text(encoding="utf-8"))["videos"] if args.predictions else {}
    scene = Scene.load()
    folder = args.out / "failures"
    folder.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in CASES:
        stem = case["video"]
        matches = [p for p in sorted(args.videos.iterdir()) if p.stem == stem and p.suffix.lower() == ".mp4"]
        video = matches[0] if matches else None
        if video is None:
            print(f"{case['slug']}: skipped (no {stem} video in {args.videos})")
            continue
        tt, _ = load_cached(stem, args.cache)
        ran = pred.get(video.name, {})
        clip = folder / f"{case['slug']}.mp4"
        render_video(
            video,
            tt,
            ran.get("events", []),
            ran.get("risk", []),
            cached_scene(stem, args.cache, scene),
            clip,
            t0=case["t0"],
            t1=case["t1"],
            signal_timeline=cached_signals(stem, args.cache, scene),
            crf=WEB_CRF,
            preset=WEB_PRESET,
        )
        cases.append({"title": case["title"], "why": case["why"], "clip": f"results/failures/{clip.name}"})
        print(f"{case['slug']}: {clip.stat().st_size / 1e6:.1f} MB")
    (args.out / "failures.json").write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'failures.json'} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
