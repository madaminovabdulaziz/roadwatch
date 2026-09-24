"""Video decoding.

Contract:
- `probe(path)` returns `VideoMeta` read exactly like the harness (OpenCV CAP_PROP_FPS and
  CAP_PROP_FRAME_COUNT). Every timestamp we produce is `t_sec = frame_idx / meta.fps`, with frame_idx
  counted in display order from the first frame, so it equals the harness's `idx / fps` for that frame.
- `FrameReader(path, stride)` yields `(frame_idx, t_sec, frame_bgr)` for frames whose index is a
  multiple of `stride`. `read_window(path, t0, t1, stride)` does the same for `t0 <= t_sec < t1`,
  seeking to the window first (boundary refinement, and chunks for parallel decoding).
- Frames are BGR uint8 at native size, or converted straight to `size=(width, height)` in the same
  colour-conversion pass (much cheaper than converting 4K and resizing afterwards).
- Undecodable data is skipped; a reader gives up only after `video.max_consecutive_errors` in a row.
- `prefetch(frames, depth)` decodes ahead on a background thread, so decoding overlaps the detector.

Backends:
- "pyav" (default) decodes with FFmpeg frame threading and derives frame_idx from each frame's pts
  (checked against OpenCV's sequential index on the samples: identical). With `skip_nonref=True`,
  non-reference frames are never decoded: the samples' two B-frames between reference frames are
  skipped, so only every 3rd frame (idx 2, 5, 8, ...) is decoded. `stride` is then the minimum
  spacing between yielded frames instead of an exact multiple.
- "opencv" decodes exactly like the harness (cv2.VideoCapture) and uses grab() for frames it does not
  return. It is also the fallback when PyAV cannot open a file (skip_nonref is then ignored).

The samples are 4K H.264 4:2:2 10-bit at 29.97 fps and the T4 cannot decode them in hardware
(SPEC §12.15), so decoding is CPU-bound: never convert frames that are not used.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal, TypeVar

import av
import cv2
import numpy as np

from roadwatch.config import load_thresholds
from roadwatch.types import Frame, Size, VideoMeta

log = logging.getLogger(__name__)

Backend = Literal["pyav", "opencv"]
T = TypeVar("T")

# Tolerance when turning a time bound into a frame index, so t0 = k / fps maps back to exactly k.
_INDEX_EPS = 1e-6


def probe(video_path: str | Path) -> VideoMeta:
    """Read fps, size and frame count without decoding; raises if the file cannot be opened."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        return VideoMeta(
            video_id=path.name,
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


class FrameReader:
    """Iterate `(frame_idx, t_sec, frame_bgr)` over a whole video (see the module docstring)."""

    def __init__(
        self,
        video_path: str | Path,
        stride: int = 1,
        *,
        size: Size | None = None,
        skip_nonref: bool = False,
        backend: Backend = "pyav",
    ) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.video_path = Path(video_path)
        self.meta = probe(self.video_path)
        self.stride = stride
        self.size = size
        self.skip_nonref = skip_nonref
        self.backend = backend

    def __iter__(self) -> Iterator[Frame]:
        return _read(
            self.video_path, self.meta, 0.0, math.inf, self.stride, self.size, self.skip_nonref, self.backend
        )


def read_window(
    video_path: str | Path,
    t0: float,
    t1: float,
    stride: int = 1,
    *,
    size: Size | None = None,
    skip_nonref: bool = False,
    backend: Backend = "pyav",
) -> Iterator[Frame]:
    """Frames with `t0 <= t_sec < t1` whose index is a multiple of `stride` (or, with `skip_nonref`,
    reference frames at least `stride` apart).

    Returns a generator rather than a list: a few seconds of native 4K frames take gigabytes.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    path = Path(video_path)
    return _read(path, probe(path), max(0.0, t0), t1, stride, size, skip_nonref, backend)


def _read(
    path: Path,
    meta: VideoMeta,
    t0: float,
    t1: float,
    stride: int,
    size: Size | None,
    skip_nonref: bool,
    backend: Backend,
) -> Iterator[Frame]:
    if backend == "pyav":
        try:
            container = av.open(str(path))
        except av.FFmpegError as exc:
            log.warning("PyAV cannot open %s (%s); falling back to OpenCV", path.name, exc)
        else:
            with container:
                yield from _read_pyav(container, meta, t0, t1, stride, size, skip_nonref)
            return
    yield from _read_opencv(path, meta, t0, t1, stride, size)


def _read_pyav(
    container: av.container.InputContainer,
    meta: VideoMeta,
    t0: float,
    t1: float,
    stride: int,
    size: Size | None,
    skip_nonref: bool,
) -> Iterator[Frame]:
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    if skip_nonref:
        stream.codec_context.skip_frame = "NONREF"
    time_base = stream.time_base
    origin = stream.start_time or 0
    next_idx = math.ceil(t0 * meta.fps - _INDEX_EPS)
    if next_idx > 0:
        container.seek(origin + int(t0 / time_base), stream=stream, backward=True, any_frame=False)

    max_errors = load_thresholds()["video"]["max_consecutive_errors"]
    errors = 0
    try:
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.FFmpegError as exc:
                errors += 1
                if errors > max_errors:
                    log.error(
                        "%s: %d undecodable packets in a row (%s); stopping", meta.video_id, errors, exc
                    )
                    return
                continue
            errors = 0
            for frame in frames:
                if frame.pts is None:
                    continue
                idx = round(float((frame.pts - origin) * time_base) * meta.fps)
                if idx < next_idx or (not skip_nonref and idx % stride):
                    continue
                t_sec = idx / meta.fps
                if t_sec >= t1:
                    return
                yield idx, t_sec, _to_bgr(frame, size)
                next_idx = idx + (stride if skip_nonref else 1)
    finally:
        _drain(stream)


def _drain(stream: av.video.stream.VideoStream) -> None:
    """Flush the decoder so no frame is in flight when the container closes.

    Closing a frame-threaded FFmpeg decoder that still holds frames can deadlock in its thread
    teardown (seen when a read stops early, e.g. at a window's end or a deadline), so every PyAV
    read ends by sending end-of-stream and collecting the remaining frames.
    """
    try:
        for _ in stream.codec_context.decode(None):
            pass
    except av.FFmpegError:
        pass


def _to_bgr(frame: av.VideoFrame, size: Size | None) -> np.ndarray:
    if size is None:
        return frame.to_ndarray(format="bgr24")
    return frame.to_ndarray(format="bgr24", width=size[0], height=size[1])


def _read_opencv(
    path: Path, meta: VideoMeta, t0: float, t1: float, stride: int, size: Size | None
) -> Iterator[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    max_errors = load_thresholds()["video"]["max_consecutive_errors"]
    try:
        idx = math.ceil(t0 * meta.fps - _INDEX_EPS)
        if idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        errors = 0
        while (t_sec := idx / meta.fps) < t1:
            # Like the harness, count only frames that decode; retry a bad frame a few times.
            if not cap.grab():
                if idx >= meta.n_frames:
                    return
                errors += 1
                if errors > max_errors:
                    log.error("%s: %d failed frames in a row at idx %d; stopping", meta.video_id, errors, idx)
                    return
                continue
            errors = 0
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    if size is not None:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    yield idx, t_sec, frame
            idx += 1
    finally:
        cap.release()


class _Failure:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


_DONE = object()


def prefetch(items: Iterable[T], depth: int) -> Iterator[T]:
    """Iterate `items` on a background thread, at most `depth` ahead of the consumer.

    Order is preserved and an exception in the producer is re-raised in the consumer. Closing the
    returned generator early (break, return, error) stops the producer and closes `items`. PyAV and
    CUDA both release the GIL, so decoding really overlaps the detector.
    """
    buffer: queue.Queue = queue.Queue(maxsize=max(1, depth))
    stop = threading.Event()
    source = iter(items)

    def produce() -> None:
        try:
            for item in source:
                while not stop.is_set():
                    try:
                        buffer.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as exc:  # handed to the consumer, which re-raises it
            buffer.put(_Failure(exc))
        finally:
            buffer.put(_DONE)

    worker = threading.Thread(target=produce, name="frame-prefetch", daemon=True)
    worker.start()
    try:
        while (item := buffer.get()) is not _DONE:
            if isinstance(item, _Failure):
                raise item.exc
            yield item
    finally:
        stop.set()
        while worker.is_alive():
            try:
                buffer.get(timeout=0.1)
            except queue.Empty:
                continue
        close = getattr(source, "close", None)
        if close:
            close()
