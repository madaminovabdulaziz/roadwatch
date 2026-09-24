"""EDA data for the website from the sample videos and cache/tracks (RUNBOOK P1.5, WEBSITE_SPEC "EDA").

Writes web/public/data/eda/ (JSON for interactive charts, images only for overlays on the frame):
- summary.json: per video resolution, fps, duration, codec, pixel format, mean brightness over time
  (lighting proxy; needs the video file, else stats come from the track-cache sidecar);
- <video>/counts.json: mean visible objects per class in 1 s bins;
- <video>/density.json: new vehicle / person tracks per minute;
- <video>/speeds.json: speed histogram (km/h) of moving vehicles per lane (needs the homography);
- <video>/signals.json: signal timeline (only with --signals: decodes frames at native size);
- flow_field.json + lane_flow.jpg, heatmap.jpg, trajectories.jpg over configs/reference.jpg, pooled
  over all videos (1920 px wide; JPEG, since PNGs of a photo are ~3 MB each on a phone).
Everything is deterministic (sorted keys, rounded numbers), so re-running gives identical files.

Usage: python scripts/eda.py samples/ [--cache cache/tracks] [--signals]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.config import CACHE_DIR, CONFIG_DIR, REPO_ROOT, load_thresholds  # noqa: E402
from roadwatch.devdata import cached_meta  # noqa: E402
from roadwatch.features import add_kinematics  # noqa: E402
from roadwatch.scene.flow import flow_field, step_velocities  # noqa: E402
from roadwatch.scene.light import SignalStateEstimator  # noqa: E402
from roadwatch.scene.overlay import draw_scene  # noqa: E402
from roadwatch.scene.scene import Scene  # noqa: E402
from roadwatch.video import probe, read_window  # noqa: E402

OUT_WIDTH = 1920  # overlay image width for the web
BRIGHTNESS_SAMPLES = 30
SPEED_BIN_KMH = 5
SPEED_MAX_KMH = 100
MOVING_MPS = 1.0  # speeds below this are waiting, not driving, and stay out of the histogram
JPEG_QUALITY = 85


def _dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def video_stats(path: Path) -> dict[str, Any]:
    """Container/stream facts and mean luma at evenly spaced times (0-255)."""
    meta = probe(path)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        codec, pix_fmt = s.codec_context.name, s.codec_context.pix_fmt
        bit_rate = c.bit_rate
    times, luma = [], []
    for k in range(BRIGHTNESS_SAMPLES):
        t = meta.duration * k / BRIGHTNESS_SAMPLES
        frame = next((img for _, _, img in read_window(path, t, t + 1.0, size=(320, 180))), None)
        if frame is not None:
            times.append(round(t, 2))
            luma.append(round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()), 1))
    return {
        "name": path.name,
        "width": meta.width,
        "height": meta.height,
        "fps": round(meta.fps, 3),
        "n_frames": meta.n_frames,
        "duration": round(meta.duration, 2),
        "codec": codec,
        "pix_fmt": pix_fmt,
        "bit_rate_mbps": round(bit_rate / 1e6, 1) if bit_rate else None,
        "brightness": {"t": times, "mean_luma": luma},
    }


def counts_per_second(tt: pd.DataFrame, duration: float) -> dict[str, Any]:
    """Mean number of visible objects per class in each 1 s bin (averaged over processed frames)."""
    n_bins = max(1, math.ceil(duration))
    frames_per_bin = tt.groupby(tt["t"].astype(int).clip(0, n_bins - 1))["frame"].nunique()
    per = tt.assign(sec=tt["t"].astype(int).clip(0, n_bins - 1), cls=tt["cls"].astype(str))
    counts = per.groupby(["cls", "sec"]).size().unstack(fill_value=0)
    series = {}
    for cls in sorted(counts.index):
        row = counts.loc[cls].reindex(range(n_bins), fill_value=0)
        denom = frames_per_bin.reindex(range(n_bins)).fillna(0).to_numpy()
        series[cls] = [
            round(float(c / d), 2) if d else 0.0 for c, d in zip(row.to_numpy(), denom, strict=True)
        ]
    return {"t": list(range(n_bins)), "series": series}


def density_per_minute(tt: pd.DataFrame, duration: float) -> dict[str, Any]:
    """Tracks that appear in each minute, for vehicles and persons."""
    groups = load_thresholds()["perception"]["tracker_groups"]
    n_min = max(1, math.ceil(duration / 60))
    first = tt.groupby("track_id").agg(t=("t", "min"), cls=("cls", "first"))
    minute = (first["t"] // 60).astype(int).clip(0, n_min - 1)
    out: dict[str, Any] = {"minute": list(range(n_min))}
    for name, members in (
        ("vehicles", groups["vehicles"] + groups["two_wheelers"]),
        ("persons", groups["persons"]),
    ):
        hits = minute[first["cls"].astype(str).isin(members)]
        out[name] = [int(v) for v in np.bincount(hits, minlength=n_min)]
    return out


def speed_histograms(kin: pd.DataFrame) -> dict[str, Any]:
    """Speed histogram (km/h) of moving, kinematically valid vehicles per lane ("" = outside lanes)."""
    groups = load_thresholds()["perception"]["tracker_groups"]
    v = kin[kin["kin_valid"] & kin["cls"].astype(str).isin(groups["vehicles"]) & (kin["speed"] >= MOVING_MPS)]
    edges = list(range(0, SPEED_MAX_KMH + SPEED_BIN_KMH, SPEED_BIN_KMH))
    lanes = {}
    for lane, rows in v.groupby("lane_id", sort=True):
        counts, _ = np.histogram(np.clip(rows["speed"] * 3.6, 0, SPEED_MAX_KMH - 1e-6), bins=edges)
        lanes[str(lane)] = {
            "counts": [int(c) for c in counts],
            "median_kmh": round(float(rows["speed"].median() * 3.6), 1),
        }
    return {"bin_edges_kmh": edges, "lanes": lanes}


def _scaled(bg: np.ndarray) -> tuple[np.ndarray, float]:
    k = OUT_WIDTH / bg.shape[1]
    return cv2.resize(bg, (OUT_WIDTH, round(bg.shape[0] * k)), interpolation=cv2.INTER_AREA), k


def heatmap_image(tt: pd.DataFrame, bg: np.ndarray) -> np.ndarray:
    """Footprint density (log scale, blurred) as a colour overlay on the background."""
    img, k = _scaled(bg)
    h, w = img.shape[:2]
    xs = np.clip((tt["fx"].to_numpy() * k).astype(int), 0, w - 1)
    ys = np.clip((tt["fy"].to_numpy() * k).astype(int), 0, h - 1)
    density = np.zeros((h, w), np.float32)
    np.add.at(density, (ys, xs), 1.0)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=6)
    if density.max() > 0:
        density = np.log1p(density) / np.log1p(density.max())
    colour = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(density * 1.5, 0, 0.85)[..., None]
    return (img * (1 - alpha) + colour * alpha).astype(np.uint8)


def trajectories_image(tt: pd.DataFrame, bg: np.ndarray) -> np.ndarray:
    """Every road-user track as a thin line, hue = its overall image-space direction."""
    img, k = _scaled(bg)
    img = (img * 0.5).astype(np.uint8)
    for _, rows in tt.sort_values(["track_id", "t"], kind="stable").groupby("track_id", sort=True):
        pts = (rows[["fx", "fy"]].to_numpy() * k).round().astype(np.int32)
        if len(pts) < 2:
            continue
        dx, dy = (pts[-1] - pts[0]).astype(float)
        if math.hypot(dx, dy) < 5:
            continue
        hue = int((math.degrees(math.atan2(dy, dx)) % 360) / 2)  # OpenCV hue is 0-179
        bgr = cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, bgr, 1, cv2.LINE_AA)
    return img


def lane_flow_image(field: dict[str, Any], bg: np.ndarray, scene: Scene) -> np.ndarray:
    """Flow-field arrows (white) over the scene layers."""
    img, k = _scaled(draw_scene(bg, scene) if scene.layers else bg)
    cell = field["cell_px"] * k
    for c in field["cells"]:
        v = np.array([c["vx"], c["vy"]], dtype=float)
        norm = float(np.hypot(*v))
        if norm == 0:
            continue
        tail = np.array([c["x"], c["y"]], dtype=float) * k
        head = tail + v / norm * cell * 0.8
        p0 = (int(round(tail[0])), int(round(tail[1])))
        p1 = (int(round(head[0])), int(round(head[1])))
        cv2.arrowedLine(img, p0, p1, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "videos", type=Path, nargs="?", default=REPO_ROOT / "samples", help="folder of sample .mp4"
    )
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "tracks")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "web" / "public" / "data" / "eda")
    ap.add_argument(
        "--signals", action="store_true", help="also estimate the signal timeline (decodes 4K frames)"
    )
    args = ap.parse_args()

    scene = Scene.load()
    stems = sorted({p.stem for p in args.cache.glob("*.parquet")})
    if not stems:
        ap.error(f"no track caches in {args.cache} (run scripts/cache_tracks.py first)")
    video_files = (
        {p.stem: p for p in args.videos.glob("*") if p.suffix.lower() == ".mp4"}
        if args.videos.exists()
        else {}
    )

    summary, pooled, id_offset = [], [], 0
    for stem in stems:
        meta = cached_meta(stem, args.cache)
        tt = pd.read_parquet(args.cache / f"{stem}.parquet")
        path = video_files.get(stem)
        stats = (
            video_stats(path)
            if path
            else {
                "name": meta.video_id,
                "width": meta.width,
                "height": meta.height,
                "fps": round(meta.fps, 3),
                "n_frames": meta.n_frames,
                "duration": round(meta.duration, 2),
                "brightness": None,
            }
        )
        summary.append(stats)
        _dump(args.out / stem / "counts.json", counts_per_second(tt, meta.duration))
        _dump(args.out / stem / "density.json", density_per_minute(tt, meta.duration))
        if scene.has("homography"):
            _dump(args.out / stem / "speeds.json", speed_histograms(add_kinematics(tt, scene)))
        if args.signals and path and scene.has("signals"):
            frames = read_window(path, 0.0, math.inf, stride=6, skip_nonref=True)
            _dump(args.out / stem / "signals.json", SignalStateEstimator(scene).timeline(frames))
        # track ids restart per video: offset them so pooled tracks never join across videos
        pooled.append(tt.assign(track_id=tt["track_id"].astype(np.int64) + id_offset))
        id_offset += int(tt["track_id"].max()) + 1 if len(tt) else 0
        print(f"{stem}: {len(tt)} track rows, {tt['track_id'].nunique()} tracks", flush=True)
    _dump(args.out / "summary.json", {"videos": summary})

    all_tt = pd.concat(pooled, ignore_index=True)
    width, height = summary[0]["width"], summary[0]["height"]
    ref_path = CONFIG_DIR / "reference.jpg"
    reference = cv2.imread(str(ref_path)) if ref_path.exists() else None
    bg = reference if reference is not None else np.zeros((height, width, 3), np.uint8)
    field = flow_field(all_tt, width, height)
    _dump(args.out / "flow_field.json", field)
    for name, img in (
        ("heatmap.jpg", heatmap_image(all_tt, bg)),
        ("trajectories.jpg", trajectories_image(all_tt, bg)),
        ("lane_flow.jpg", lane_flow_image(field, bg, scene)),
    ):
        cv2.imwrite(str(args.out / name), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    print(f"{len(step_velocities(all_tt))} moving steps pooled; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
