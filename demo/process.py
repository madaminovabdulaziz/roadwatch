"""Demo processing of one upload (WEBSITE_SPEC "Demo API"): the same roadwatch pipeline on CPU.

Contract:
- `probe_upload(path)` -> duration in seconds; raises `BadInput` with a readable message if the file
  cannot be opened or decoded.
- `reencode(src, dst, height, max_sec)` writes the first `max_sec` seconds as H.264 (libx264, yuv420p,
  +faststart) at `height` px with PyAV, so any input becomes something our decoder and every browser
  play cheaply. Raw camera files are 4K 10-bit at ~140 Mbps, so it decodes only reference frames
  (`skip_nonref`) and holds the last one over the skipped B-frames: the output keeps the source frame
  rate and frame count (every later stage times things exactly as for the original) at a third of the
  decode cost. Frame rates above `work_fps` are decimated to about it (a 60 fps phone clip -> 30). The
  re-encode has no B-frames, so later stages reading it with `skip_nonref` get every stride-th frame.
- `match_camera(video, scene)` -> (matched, scene in the video's pixels). "Our camera" means the scene
  registration to configs/reference.jpg is confident (SIFT on the median background, CLAHE against
  lighting, a plausible camera move; SPEC §12.34). An ORB check at 960 px, used before, found 8 of the
  40 inliers it needed on C3902 in late-afternoon shade, where the registration finds 99.
- `process(src, workdir, progress)` runs: re-encode -> camera match + scene alignment -> perception
  (with the signal-lamp timeline) -> rules -> Part B risk curve -> render. Another camera gets an empty
  scene, so the scene rules disable themselves and Part B gives its bias only. Part B reuses Part
  A's detections (`SharedDetections`), so the risk curve costs no second detector pass. A failing risk
  curve or renderer degrades the result (no curve / no annotations), never the job.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import yaml

from roadwatch.config import load_thresholds
from roadwatch.pipeline import events_from_tracks
from roadwatch.scene.light import SignalStateEstimator
from roadwatch.scene.registration import align_scene, can_register, register_video
from roadwatch.scene.scene import Scene
from roadwatch.types import FrameDetections
from roadwatch.video import FrameReader, probe, read_window

log = logging.getLogger(__name__)

DEMO_CONFIG = Path(__file__).with_name("config.yaml")
Progress = Callable[[str, float], None]  # (stage, percent 0-100)


class BadInput(ValueError):
    """The upload cannot be processed; the message is shown to the user."""


class JobTimeout(RuntimeError):
    """The job ran past `max_job_sec` (raised from its progress callback); the message is shown."""


@functools.cache
def demo_cfg() -> dict[str, Any]:
    """The `demo` block of demo/config.yaml (read directly: the roadwatch config may be cached already)."""
    return yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))["demo"]


def probe_upload(path: Path) -> float:
    """Duration of a decodable video, else BadInput."""
    try:
        meta = probe(path)
        with contextlib.closing(iter(FrameReader(path, 1))) as frames:
            first = next(frames, None)
    except Exception as exc:
        raise BadInput("this file could not be read as a video") from exc
    if first is None or meta.fps <= 0 or meta.n_frames <= 0:
        raise BadInput("no video frames could be decoded from this file")
    return meta.duration


def reencode(
    src: Path,
    dst: Path,
    height: int,
    max_sec: float = math.inf,
    progress: Callable[[float], None] | None = None,
) -> float:
    """H.264 yuv420p copy of the first `max_sec` s of `src` at `height` px; returns the seconds written."""
    meta = probe(src)
    k = max(1, round(meta.fps / demo_cfg()["work_fps"]))  # keep every k-th source frame
    rate = Fraction(meta.fps).limit_denominator(1001) / k
    h = min(height, meta.height) // 2 * 2
    w = max(2, round(meta.width * h / meta.height / 2) * 2)
    end = min(max_sec, meta.duration)
    n_slots = math.ceil(end * float(rate) - 1e-6)  # output frame s shows source frame s * k
    slot, prev, last_idx, gap = 0, None, -1, 1
    with av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
        stream = out.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        # no B-frames: every stage after this reads it with skip_nonref, which would skip them
        stream.options = {"preset": "veryfast", "crf": "23", "x264-params": "bframes=0"}

        def emit(frame: av.VideoFrame) -> None:
            nonlocal slot
            frame.pts, frame.time_base = slot, 1 / rate
            out.mux(stream.encode(frame))
            slot += 1

        with contextlib.closing(read_window(src, 0.0, end, size=(w, h), skip_nonref=True)) as frames:
            for n, (idx, _, img) in enumerate(frames):
                cur = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
                # skipped frames show the last decoded one (the first one, before any was decoded)
                while slot < n_slots and slot * k <= idx:
                    emit(cur if prev is None or slot * k == idx else prev)
                if prev is not None:
                    gap = max(gap, idx - last_idx)
                prev, last_idx = cur, idx
                if progress and n % 10 == 0:
                    progress(min(1.0, idx / max(end * meta.fps, 1.0)))
        if prev is None:
            raise BadInput("no video frames could be decoded from this file")
        # the non-reference frames after the last decoded one (not a frozen tail of a truncated file)
        while slot < min(n_slots, (last_idx + gap - 1) // k + 1):
            emit(prev)
        out.mux(stream.encode())
    return slot / float(rate)


class SharedDetections:
    """A detector that remembers its detections by frame index.

    Part B's pass over the frames Part A already detected then costs no second inference: on a CPU
    Space the detector is nearly all of the demo's time. Everything else is delegated to the wrapped
    detector.
    """

    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self.seen: dict[int, FrameDetections] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.detector, name)

    def predict(
        self, batch: Sequence[Any], native_size: tuple[int, int] | None = None
    ) -> list[FrameDetections]:
        todo = [item for item in batch if item[0] not in self.seen]
        if todo:
            for item, dets in zip(todo, self.detector.predict(todo, native_size=native_size), strict=True):
                self.seen[item[0]] = dets
        return [self.seen[item[0]] for item in batch]


def match_camera(video: Path, scene: Scene) -> tuple[bool, Scene]:
    """Whether `video` shows our camera's view, and the scene aligned to it (empty if not)."""
    if not can_register(scene):
        return False, Scene()
    meta = probe(video)
    try:
        reg = register_video(video, scene)
    except Exception:
        log.warning("registration failed; treating the upload as another camera", exc_info=True)
        return False, Scene()
    if reg is None or not reg.confident:
        return False, Scene()
    return True, align_scene(scene, reg, (meta.width, meta.height))


def risk_curve(
    video: Path, scene: Scene, detector: Any, progress: Callable[[float], None] | None = None
) -> list[list[float]]:
    """Part B's score at each frame Part A detected, as [t, score] pairs.

    RiskCore runs as in the submission (causal, aligning `scene` - as clicked - from the frames it
    receives) but is fed only Part A's frames, so with a `SharedDetections` detector it reuses their
    detections.
    """
    from roadwatch.risk import RiskCore

    meta = probe(video)
    core = RiskCore(detector=detector, scene=scene)
    core.reset(
        {
            "video_id": meta.video_id,
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "n_frames": meta.n_frames,
        }
    )
    cfg = load_thresholds()["video"]
    curve = []
    with contextlib.closing(
        iter(FrameReader(video, cfg["stride_part_a"], skip_nonref=cfg["skip_nonref"]))
    ) as frames:
        for n, (_, t, frame) in enumerate(frames):
            curve.append([round(t, 3), round(min(1.0, max(0.0, float(core.step(frame, t)))), 4)])
            if progress and n % 10 == 0:
                progress(min(1.0, t / max(meta.duration, 1e-9)))
    return curve


def process(src: Path, workdir: Path, progress: Progress) -> dict[str, Any]:
    """Full demo run on one upload; returns the result dict of the Demo API (minus video_url)."""
    from roadwatch.perception.detector import Detector  # torch loads in the worker, not at import
    from roadwatch.perception.run import run_perception

    cfg = demo_cfg()
    source = probe(src)
    work = workdir / "work.mp4"
    progress("decoding", 0)
    reencode(src, work, cfg["work_height"], cfg["max_process_sec"], lambda f: progress("decoding", 25 * f))
    meta = probe(work)
    matched, scene = match_camera(work, Scene.load())
    clicked = Scene.load() if matched else Scene()  # Part B aligns the scene itself, from its frames
    signals = SignalStateEstimator(scene) if scene.has("signals") else None

    progress("detecting", 25)
    detector = SharedDetections(Detector.load())
    tt = run_perception(
        work,
        detector=detector,
        progress=lambda f: progress("detecting", 25 + 60 * f),
        on_frame=signals.observe if signals else None,
    )

    progress("rules", 85)
    timeline = signals.finish() if signals else {}
    events = events_from_tracks(tt, meta, scene, signal_timeline=timeline)
    try:
        risk = risk_curve(work, clicked, detector, lambda f: progress("rules", 85 + 5 * f))
    except JobTimeout:
        raise
    except Exception:
        log.warning("the risk curve failed; returning the events without it", exc_info=True)
        risk = []

    progress("rendering", 90)
    video = work
    try:
        from roadwatch.render import render_video

        video = render_video(
            work, tt, events, risk, scene, workdir / "annotated.mp4", signal_timeline=timeline
        )
    except Exception:
        log.warning("rendering failed; returning the video without annotations", exc_info=True)
    progress("rendering", 100)
    return {
        "events": events,
        "risk": risk,
        "video_path": str(video),
        "camera_match": matched,
        "duration": round(meta.duration, 3),
        "source_duration": round(source.duration, 3),
    }
