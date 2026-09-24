"""Annotated rendering (RUNBOOK P2.3): browser-friendly H.264, windows, privacy blur, scene scaling."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

from roadwatch.config import REPO_ROOT
from roadwatch.render import _scaled_scene, render_video
from roadwatch.scene.scene import Scene
from roadwatch.types import TRACK_DTYPES
from roadwatch.video import probe
from tests.conftest import TINY_FPS, TINY_FRAMES, TINY_SIZE
from tests.test_eda import make_cache

SCENE = Scene(
    {
        "image_size": [320, 192],  # twice the tiny video: layers must be scaled down
        "homography": {
            "image_pts": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "world_pts": [[0, 0], [10, 0], [10, 10], [0, 10]],
        },
        "lanes": [{"id": "e1", "polygon": [[0, 100], [320, 100], [320, 160], [0, 160]], "direction": [1, 0]}],
        "signals": [{"id": "L1", "red": [300, 4, 8, 8], "yellow": [300, 14, 8, 8], "green": [300, 24, 8, 8]}],
    }
)


def tracks() -> pd.DataFrame:
    f = np.arange(0, TINY_FRAMES, 3)
    t = f / TINY_FPS
    car = pd.DataFrame(
        {
            "frame": f,
            "t": t,
            "track_id": 1,
            "cls": "car",
            "conf": 0.9,
            "x1": 5 * f,
            "y1": 30.0,
            "x2": 5 * f + 20,
            "y2": 50.0,
        }
    )
    person = pd.DataFrame(
        {
            "frame": f,
            "t": t,
            "track_id": 2,
            "cls": "person",
            "conf": 0.9,
            "x1": 60.0,  # the red square of the tiny clip passes behind the face at frame 12
            "y1": 20.0,
            "x2": 80.0,
            "y2": 80.0,
        }
    )
    tt = pd.concat([car, person], ignore_index=True)
    tt["fx"], tt["fy"] = (tt["x1"] + tt["x2"]) / 2, tt["y2"]
    return tt.astype(TRACK_DTYPES)


def frames_of(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        assert s.codec_context.name == "h264" and s.codec_context.pix_fmt == "yuv420p"
        return [f.to_ndarray(format="bgr24") for f in c.decode(s)]


def test_render_full_video(tmp_path: Path, tiny_video: Path) -> None:
    out = render_video(
        tiny_video,
        tracks(),
        [[0.2, 1.5, "jaywalking"]],
        [[0.0, 0.1], [2.0, 0.9]],
        SCENE,
        tmp_path / "a.mp4",
        signal_timeline={"L1": [(0.0, 2.0, "red")]},
    )
    frames = frames_of(out)
    assert len(frames) == TINY_FRAMES and frames[0].shape[:2] == (TINY_SIZE[1], TINY_SIZE[0])
    assert probe(out).fps == TINY_FPS


def test_window_clip_and_blur(tmp_path: Path, tiny_video: Path) -> None:
    clip = render_video(tiny_video, tracks(), [], [], Scene(), tmp_path / "clip.mp4", t0=0.5, t1=1.0)
    assert len(frames_of(clip)) == 5
    sharp = frames_of(
        render_video(tiny_video, tracks(), [], [], Scene(), tmp_path / "s.mp4", blur_faces=False)
    )
    blurred = frames_of(
        render_video(tiny_video, tracks(), [], [], Scene(), tmp_path / "b.mp4", blur_faces=True)
    )
    face = (slice(21, 32), slice(61, 79))  # top fifth of the person box, inside its outline
    assert not np.array_equal(sharp[12][face], blurred[12][face])
    elsewhere = (slice(85, 95), slice(100, 160))
    assert (
        np.abs(sharp[12][elsewhere].astype(int) - blurred[12][elsewhere].astype(int)).max() <= 12
    )  # codec noise only


def test_scene_scaling_keeps_metres_and_ids() -> None:
    small = _scaled_scene(SCENE, 160, 96)
    assert small.layers["lanes"][0]["polygon"][1] == [160, 50]
    assert small.layers["lanes"][0]["direction"] == [1, 0] and small.layers["lanes"][0]["id"] == "e1"
    assert small.layers["homography"]["world_pts"] == SCENE.layers["homography"]["world_pts"]
    assert small.layers["signals"][0]["red"] == [150, 2, 4, 4]


def test_render_samples_end_to_end(tmp_path: Path, tiny_video: Path) -> None:
    videos, cache, out = tmp_path / "videos", tmp_path / "tracks", tmp_path / "results"
    videos.mkdir()
    shutil.copy(tiny_video, videos / "tiny.mp4")
    make_cache(cache, "tiny", TINY_SIZE[0], TINY_SIZE[1], TINY_FPS, TINY_FRAMES)
    r = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_samples.py"),
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
    index = json.loads((out / "index.json").read_text())
    assert [v["id"] for v in index["videos"]] == ["tiny"]
    for key in ("video", "poster", "events", "risk"):  # paths are relative to public/data/
        assert (out.parent / index["videos"][0][key]).exists(), key
    assert json.loads((out / "tiny" / "events.json").read_text()) == []  # every class is still disabled
    assert json.loads((out / "gallery.json").read_text()) == []
