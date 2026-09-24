"""EDA data generation end to end on a synthetic cache and clip; output is deterministic (P1.5)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from roadwatch.config import REPO_ROOT
from roadwatch.types import TRACK_DTYPES


def make_cache(cache: Path, stem: str, width: int, height: int, fps: float, n_frames: int) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for k, (cls, y, v) in enumerate([("car", 30, 40), ("car", 60, -40), ("person", 80, 10)]):
        t = np.arange(0, n_frames / fps, 0.3)
        fx = (10 + v * t) % width if v > 0 else (width - 10 + v * t) % width
        parts.append(
            pd.DataFrame(
                {
                    "frame": np.round(t * fps).astype(int),
                    "t": t,
                    "track_id": k,
                    "cls": cls,
                    "conf": 0.9,
                    "x1": fx - 5,
                    "y1": y - 10.0,
                    "x2": fx + 5,
                    "y2": float(y),
                    "fx": fx,
                    "fy": float(y),
                }
            )
        )
    pd.concat(parts, ignore_index=True).astype(TRACK_DTYPES).to_parquet(
        cache / f"{stem}.parquet", index=False
    )
    video = {
        "name": f"{stem}.mp4",
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "duration": n_frames / fps,
    }
    (cache / f"{stem}.json").write_text(json.dumps({"video": video, "seconds": None}), encoding="utf-8")


def test_eda_writes_all_outputs_deterministically(tmp_path: Path, tiny_video: Path) -> None:
    videos, cache = tmp_path / "videos", tmp_path / "tracks"
    videos.mkdir()
    shutil.copy(tiny_video, videos / "tiny.mp4")
    make_cache(cache, "tiny", 160, 96, 10.0, 20)
    make_cache(cache, "other", 160, 96, 10.0, 700)  # 70 s, no video file: stats from the sidecar

    outs = []
    for run in ("a", "b"):
        out = tmp_path / run
        r = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "eda.py"),
                str(videos),
                "--cache",
                str(cache),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stdout + r.stderr
        outs.append(out)

    files = sorted(p.relative_to(outs[0]).as_posix() for p in outs[0].rglob("*") if p.is_file())
    for expected in (
        "summary.json",
        "flow_field.json",
        "heatmap.jpg",
        "trajectories.jpg",
        "lane_flow.jpg",
        "tiny/counts.json",
        "tiny/density.json",
        "other/counts.json",
    ):
        assert expected in files
    for f in files:
        assert (outs[0] / f).read_bytes() == (outs[1] / f).read_bytes(), f"{f} differs between runs"

    summary = {v["name"]: v for v in json.loads((outs[0] / "summary.json").read_text())["videos"]}
    assert summary["tiny.mp4"]["codec"] and summary["tiny.mp4"]["brightness"]["mean_luma"]
    assert summary["other.mp4"]["brightness"] is None and summary["other.mp4"]["duration"] == 70.0
    counts = json.loads((outs[0] / "other" / "counts.json").read_text())
    assert len(counts["t"]) == 70 and set(counts["series"]) == {"car", "person"}
    assert max(counts["series"]["car"]) == 2.0  # two cars visible in every frame
    density = json.loads((outs[0] / "other" / "density.json").read_text())
    assert density["vehicles"] == [2, 0] and density["persons"] == [1, 0]
