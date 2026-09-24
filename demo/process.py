"""Demo processing of one upload (WEBSITE_SPEC "Demo API"): the same roadwatch pipeline on CPU.

Contract:
- `probe_upload(path)` -> duration in seconds; raises `BadInput` with a readable message if the file
  cannot be opened or decoded.
- `reencode(src, dst, height)` writes H.264 (libx264, yuv420p, +faststart) at `height` px with PyAV,
  so odd inputs become something both our decoder and every browser play.
- `camera_match(video, reference)` compares the median of a few upload frames with
  configs/reference.jpg (ORB features + RANSAC homography inliers).
- `process(src, workdir, progress)` runs: re-encode -> camera match -> perception -> rules (scene
  rules only when the camera matches; tracks are scaled to the scene's pixel size) -> render. If the
  renderer fails, the re-encoded video is returned without annotations (WEBSITE_SPEC fallback).
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import yaml

from roadwatch.config import CONFIG_DIR
from roadwatch.pipeline import events_from_tracks
from roadwatch.scene.scene import Scene
from roadwatch.video import FrameReader, probe, read_window

log = logging.getLogger(__name__)

DEMO_CONFIG = Path(__file__).with_name("config.yaml")
Progress = Callable[[str, float], None]  # (stage, percent 0-100)


class BadInput(ValueError):
    """The upload cannot be processed; the message is shown to the user."""


@functools.cache
def demo_cfg() -> dict[str, Any]:
    """The `demo` block of demo/config.yaml (read directly: the roadwatch config may be cached already)."""
    return yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))["demo"]


def probe_upload(path: Path) -> float:
    """Duration of a decodable video, else BadInput."""
    try:
        meta = probe(path)
        first = next(iter(FrameReader(path, 1)), None)
    except Exception as exc:
        raise BadInput("this file could not be read as a video") from exc
    if first is None or meta.fps <= 0 or meta.n_frames <= 0:
        raise BadInput("no video frames could be decoded from this file")
    return meta.duration


def reencode(src: Path, dst: Path, height: int, progress: Callable[[float], None] | None = None) -> None:
    """H.264 yuv420p copy of `src` at `height` px (even width, same frame rate, no audio)."""
    with av.open(str(src)) as inp:
        s_in = inp.streams.video[0]
        s_in.thread_type = "AUTO"
        rate = s_in.average_rate or Fraction(30, 1)
        total = s_in.frames or 0
        with av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
            s_out = out.add_stream("libx264", rate=rate)
            src_w, src_h = s_in.codec_context.width, s_in.codec_context.height
            h = min(height, src_h) // 2 * 2
            s_out.height, s_out.width = h, max(2, round(src_w * h / src_h / 2) * 2)
            s_out.pix_fmt = "yuv420p"
            s_out.options = {"preset": "veryfast", "crf": "23"}
            for i, frame in enumerate(inp.decode(s_in)):
                img = frame.reformat(width=s_out.width, height=s_out.height, format="yuv420p")
                img.pts, img.time_base = i, 1 / rate
                out.mux(s_out.encode(img))
                if progress and total and i % 30 == 0:
                    progress(min(1.0, i / total))
            out.mux(s_out.encode())


def _median_frame(video: Path, n: int, width: int) -> np.ndarray:
    meta = probe(video)
    frames = []
    for idx in np.unique(np.linspace(0, max(meta.n_frames - 1, 0), n).round().astype(int)):
        img = next((f for _, _, f in read_window(video, idx / meta.fps, (idx + 1) / meta.fps)), None)
        if img is not None:
            frames.append(cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1]))))
    if not frames:
        raise BadInput("no frames could be decoded for the camera check")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def match_images(a: np.ndarray, b: np.ndarray, cfg: dict[str, Any]) -> int:
    """RANSAC homography inliers between two BGR images (0 when they do not match)."""
    orb = cv2.ORB_create(nfeatures=cfg["orb_features"])
    ka, da = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg["ratio"] * p[1].distance]
    if len(good) < 4:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def camera_match(video: Path, reference: Path = CONFIG_DIR / "reference.jpg") -> bool:
    """Whether the upload shows the same camera view as our reference frame."""
    cfg = demo_cfg()["camera_match"]
    ref = cv2.imread(str(reference)) if reference.exists() else None
    if ref is None:
        return False
    ref = cv2.resize(ref, (cfg["width"], round(ref.shape[0] * cfg["width"] / ref.shape[1])))
    return match_images(_median_frame(video, cfg["frames"], cfg["width"]), ref, cfg) >= cfg["min_inliers"]


def _to_scene_pixels(tt: Any, width: int, height: int, scene: Scene) -> Any:
    """Tracks measured on the re-encoded video, rescaled to the scene's native pixel size."""
    size = scene.layers.get("image_size")
    if not size:
        return tt
    sx, sy = size[0] / width, size[1] / height
    out = tt.copy()
    for col in ("x1", "x2", "fx"):
        out[col] = out[col] * sx
    for col in ("y1", "y2", "fy"):
        out[col] = out[col] * sy
    return out


def process(src: Path, workdir: Path, progress: Progress) -> dict[str, Any]:
    """Full demo run on one upload; returns the result dict of the Demo API (minus video_url)."""
    from roadwatch.perception.run import run_perception  # torch loads in the worker, not at import

    work = workdir / "work.mp4"
    progress("decoding", 0)
    reencode(src, work, demo_cfg()["work_height"], lambda f: progress("decoding", 15 * f))
    meta = probe(work)
    matched = camera_match(work)
    progress("detecting", 15)
    tt = run_perception(work, progress=lambda f: progress("detecting", 15 + 70 * f))

    progress("rules", 85)
    scene = Scene.load() if matched else Scene()
    events = events_from_tracks(_to_scene_pixels(tt, meta.width, meta.height, scene), meta, scene)

    progress("rendering", 90)
    video = work
    try:
        from roadwatch.render import render_video

        video = render_video(work, tt, events, [], scene, workdir / "annotated.mp4")
    except Exception:
        log.warning("rendering failed; returning the video without annotations", exc_info=True)
    progress("rendering", 100)
    return {
        "events": events,
        "risk": [],
        "video_path": str(video),
        "camera_match": matched,
        "duration": round(meta.duration, 3),
    }
