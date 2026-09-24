"""Runtime against the 3x budget (RUNBOOK P0.2, P0.3): decode strategies and perception, per video.

For every video and mode, processes the first `--seconds` of the video and prints wall time per
video-second ("x", the unit of the harness's 3x budget). Decode modes:

  harness          cv2 read() of every native frame: exactly what run_submission.py does for Part B
  opencv_s3        FrameReader(backend="opencv", stride=3): grab() every frame, convert every 3rd
  pyav_s1          FrameReader(stride=1): PyAV, every frame converted at native size
  pyav_s3          FrameReader(stride=3): PyAV decodes every frame, converts every 3rd
  nonref_s3        FrameReader(stride=3, skip_nonref=True): B-frames are never decoded
  nonref_s3_small  as nonref_s3, converted straight to --size (the planned Part A input)
  parN             nonref_s3_small split into N time chunks decoded by N processes in parallel
                   (--workers; frames stay in the workers, so transfer to the GPU is not included)

Perception modes (--perception; detector loaded once, --imgsz to override the input size):

  part_a           run_perception: decode thread + batched detection + tracking (Part A's path)
  part_b           Part B's path: the harness decodes every native frame and the real RiskCore steps
                   through them (resize, detect + track every risk.stride-th frame, kinematics, score);
                   without a calibrated homography a stand-in metric scene keeps it doing full work
  --check-fp16     on CUDA, compares FP16 detections with FP32 on the first seconds of each video

--verify checks, per video, that the fast path's frames are the harness's frames at the same index
(grayscale thumbnails, best match among neighbouring indices).

Usage: python scripts/bench.py samples/ [--seconds 30] [--modes harness,nonref_s3_small] [--workers 2,4]
                               [--perception] [--imgsz 1280] [--check-fp16] [--verify] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.types import Frame, Size, VideoMeta  # noqa: E402
from roadwatch.video import FrameReader, probe, read_window  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}  # same as run_submission.py
BUDGET_X = 3.0  # run_submission.py TIME_FACTOR_DEFAULT
THUMB = (160, 90)
VERIFY_RADIUS = 3  # neighbouring indices searched for the best-matching harness frame


def harness_frames(path: Path, meta: VideoMeta) -> Iterator[Frame]:
    """Decode exactly like run_submission.run_risk: cv2 read() of every frame, t = idx / fps."""
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield idx, idx / meta.fps, frame
            idx += 1
    finally:
        cap.release()


MODES: dict[str, Callable[[Path, VideoMeta, Size], Iterator[Frame]]] = {
    "harness": lambda path, meta, size: harness_frames(path, meta),
    "opencv_s3": lambda path, meta, size: iter(FrameReader(path, 3, backend="opencv")),
    "pyav_s1": lambda path, meta, size: iter(FrameReader(path, 1)),
    "pyav_s3": lambda path, meta, size: iter(FrameReader(path, 3)),
    "nonref_s3": lambda path, meta, size: iter(FrameReader(path, 3, skip_nonref=True)),
    "nonref_s3_small": lambda path, meta, size: iter(FrameReader(path, 3, size=size, skip_nonref=True)),
}


def time_mode(frames: Iterator[Frame], limit_sec: float) -> tuple[int, float]:
    """Consume frames up to `limit_sec`; return (frames yielded, wall seconds)."""
    start = time.perf_counter()
    count = 0
    try:
        for _, t_sec, _ in frames:
            if t_sec >= limit_sec:
                break
            count += 1
    finally:
        close = getattr(frames, "close", None)
        if close:
            close()
    return count, time.perf_counter() - start


def count_chunk(path: Path, t0: float, t1: float, size: Size) -> int:
    """Worker: decode one time chunk the nonref_s3_small way and count its frames."""
    return sum(1 for _ in read_window(path, t0, t1, 3, size=size, skip_nonref=True))


def warm_up(_: int) -> None:
    """Worker start-up (imports) happens before timing; a real pipeline keeps its workers alive."""


def time_parallel(
    pool: ProcessPoolExecutor, workers: int, path: Path, limit_sec: float, size: Size
) -> tuple[int, float]:
    edges = [limit_sec * i / workers for i in range(workers + 1)]
    start = time.perf_counter()
    futures = [pool.submit(count_chunk, path, t0, t1, size) for t0, t1 in zip(edges, edges[1:], strict=False)]
    count = sum(f.result() for f in futures)
    return count, time.perf_counter() - start


def time_part_a(path: Path, limit_sec: float, detector) -> tuple[int, float, dict]:
    from roadwatch.perception.run import run_perception  # torch stays out of the decode-only workers

    stats: dict = {}
    start = time.perf_counter()
    run_perception(path, detector=detector, t_end=limit_sec, stats=stats)
    wall = time.perf_counter() - start
    notes = {k: stats[k] for k in ("wait_frames_sec", "detect_sec", "track_sec", "tracks")}
    return stats["frames_detected"], wall, notes


def time_part_b(path: Path, meta: VideoMeta, limit_sec: float, detector) -> tuple[int, float, dict]:
    """The real RiskCore on the harness's frames. Without a calibrated homography RiskCore would skip
    perception entirely, so a stand-in metric scene makes it do its full per-frame work."""
    from roadwatch.config import load_thresholds
    from roadwatch.risk import RiskCore
    from roadwatch.scene.scene import Scene

    scene = Scene.load()
    if not scene.has("homography"):
        unit = [[0, 0], [100, 0], [100, 100], [0, 100]]
        scene = Scene({"homography": {"image_pts": unit, "world_pts": [[0, 0], [1, 0], [1, 1], [0, 1]]}})
    core = RiskCore(detector=detector, scene=scene)
    core.reset(
        {
            "video_id": path.name,
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "n_frames": meta.n_frames,
        }
    )
    stride = load_thresholds()["risk"]["stride"]
    count, step_sec = 0, 0.0
    start = time.perf_counter()
    frames = harness_frames(path, meta)
    try:
        for idx, t_sec, frame in frames:
            if t_sec >= limit_sec:
                break
            tick = time.perf_counter()
            core.step(frame, t_sec)
            step_sec += time.perf_counter() - tick
            count += idx % stride == 0
    finally:
        frames.close()
    return count, time.perf_counter() - start, {"step_sec": round(step_sec, 2)}


def check_fp16(path: Path, meta: VideoMeta, fp16, fp32, seconds: float = 3.0) -> dict:
    """Share of confident FP32 boxes (conf >= 0.3) that FP16 finds with IoU >= 0.9, same class."""
    total = matched = 0
    size = fp32.frame_size_for(meta.width, meta.height)
    for idx, t_sec, img in FrameReader(path, 3, size=size, skip_nonref=True):
        if t_sec >= seconds:
            break
        a = fp32.predict([(idx, t_sec, img)])[0]
        b = fp16.predict([(idx, t_sec, img)])[0]
        keep = a.conf >= 0.3
        for box, cls in zip(a.xyxy[keep], a.cls[keep], strict=True):
            total += 1
            same = b.cls == cls
            if same.any():
                lt = np.maximum(box[:2], b.xyxy[same, :2])
                rb = np.minimum(box[2:], b.xyxy[same, 2:])
                inter = np.clip(rb - lt, 0, None).prod(1)
                union = np.prod(box[2:] - box[:2]) + np.prod(b.xyxy[same, 2:] - b.xyxy[same, :2], 1) - inter
                matched += bool((inter / union).max() >= 0.9)
    return {
        "fp32_boxes": total,
        "fp16_matched": matched,
        "ratio": round(matched / total, 4) if total else 1.0,
    }


def thumbnail(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def verify_alignment(path: Path, meta: VideoMeta, limit_sec: float) -> Counter[int]:
    """Histogram of (best-matching harness index - reported index) for the skip_nonref fast path."""
    reference: dict[int, np.ndarray] = {}
    for idx, t_sec, img in harness_frames(path, meta):
        if t_sec >= limit_sec:
            break
        reference[idx] = thumbnail(img)
    offsets: Counter[int] = Counter()
    for idx, t_sec, img in FrameReader(path, 3, skip_nonref=True):
        if t_sec >= limit_sec:
            break
        thumb = thumbnail(img)
        candidates = [i for i in range(idx - VERIFY_RADIUS, idx + VERIFY_RADIUS + 1) if i in reference]
        best = min(candidates, key=lambda i: float(np.abs(reference[i] - thumb).mean()))
        offsets[best - idx] += 1
    return offsets


def torch_float16():
    import torch

    return torch.float16


def list_videos(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return sorted(p for p in src.iterdir() if p.suffix in VIDEO_EXTS)


def parse_size(text: str) -> Size:
    width, height = text.lower().split("x")
    return int(width), int(height)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, help="folder of .mp4 files, or one .mp4")
    ap.add_argument("--seconds", type=float, default=30.0, help="decode this much of each video per mode")
    ap.add_argument("--modes", default=",".join(MODES), help=f"comma-separated subset of {list(MODES)}")
    ap.add_argument("--size", type=parse_size, default=(1920, 1080), help="output size for *_small modes")
    ap.add_argument("--workers", default="2,4", help="process counts for the parallel modes ('' = none)")
    ap.add_argument("--perception", action="store_true", help="also time part_a and part_b (needs weights)")
    ap.add_argument("--imgsz", type=int, help="detector input size for the perception modes")
    ap.add_argument("--check-fp16", action="store_true", help="compare FP16 and FP32 detections (CUDA)")
    ap.add_argument("--verify", action="store_true", help="check fast-path frame alignment (first 10 s)")
    ap.add_argument("--json", type=Path, help="also write the results here")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        ap.error(f"unknown modes {unknown}; choose from {list(MODES)}")
    videos = list_videos(args.videos)
    if not videos:
        ap.error(f"no .mp4 files in {args.videos}")

    env = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "pyav": av.__version__,
    }
    print("env:", ", ".join(f"{k}={v}" for k, v in env.items()))
    print(f"{'video':<22}{'mode':<17}{'secs':>6}{'frames':>8}{'wall s':>9}{'x dur':>8}{'fps':>8}")

    workers = [int(n) for n in args.workers.split(",") if n.strip()]
    pools = {}
    for n in workers:
        pools[n] = ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"))
        list(pools[n].map(warm_up, range(n)))

    detector = fp32 = None
    if args.perception or args.check_fp16:
        from roadwatch.perception.detector import Detector, detector_info

        detector = Detector.from_config(imgsz=args.imgsz)
        env["detector"] = detector_info(detector)
        print("detector:", env["detector"])
        if args.check_fp16 and detector.dtype == torch_float16():
            fp32 = Detector.from_config(imgsz=args.imgsz, half=False)
    extra_modes = [f"par{n}" for n in workers] + (["part_a", "part_b"] if args.perception else [])

    rows = []
    for path in videos:
        meta = probe(path)
        limit = min(args.seconds, meta.duration)
        for mode in modes + extra_modes:
            notes: dict = {}
            if mode in MODES:
                count, wall = time_mode(MODES[mode](path, meta, args.size), limit)
            elif mode == "part_a":
                count, wall, notes = time_part_a(path, limit, detector)
            elif mode == "part_b":
                count, wall, notes = time_part_b(path, meta, limit, detector)
            else:
                n = int(mode[3:])
                count, wall = time_parallel(pools[n], n, path, limit, args.size)
            row = {
                "video": path.name,
                "mode": mode,
                "seconds": round(limit, 2),
                "frames": count,
                "wall_sec": round(wall, 2),
                "x_duration": round(wall / limit, 3),
                "fps": round(count / wall, 1) if wall else 0.0,
                **notes,
            }
            rows.append(row)
            print(
                f"{path.name:<22}{mode:<17}{limit:>6.1f}{count:>8}{wall:>9.1f}"
                f"{row['x_duration']:>8.2f}{row['fps']:>8.1f}  {notes or ''}",
                flush=True,
            )
        if fp32 is not None:
            result = check_fp16(path, meta, detector, fp32)
            print(f"{path.name:<22}fp16 vs fp32: {result}", flush=True)
            rows.append({"video": path.name, "mode": "check_fp16", **result})
        if args.verify:
            offsets = verify_alignment(path, meta, min(10.0, meta.duration))
            total = sum(offsets.values())
            status = "OK" if set(offsets) == {0} else "MISALIGNED"
            print(
                f"{path.name:<22}verify nonref vs harness: {offsets[0]}/{total} frames at offset 0 "
                f"{dict(offsets)} -> {status}",
                flush=True,
            )
            rows.append({"video": path.name, "mode": "verify", "offsets": dict(offsets), "status": status})

    for pool in pools.values():
        pool.shutdown()

    harness = [r["x_duration"] for r in rows if r.get("mode") == "harness"]
    if harness:
        worst = max(harness)
        print(
            f"\nharness decode (Part B frames): worst {worst:.2f}x of the {BUDGET_X:.0f}x budget "
            f"-> {BUDGET_X - worst:.2f}x left for Part A and our Part B compute"
        )
    by_video: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.get("mode") in ("part_a", "part_b"):
            by_video.setdefault(r["video"], {})[r["mode"]] = r["x_duration"]
    for video, parts in by_video.items():
        if len(parts) == 2:
            total = parts["part_a"] + parts["part_b"]
            print(
                f"{video}: perception A {parts['part_a']:.2f}x + B {parts['part_b']:.2f}x = {total:.2f}x "
                f"of {BUDGET_X:.0f}x -> {BUDGET_X - total:.2f}x left for rules, refinement and risk features"
            )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"env": env, "rows": rows}, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
