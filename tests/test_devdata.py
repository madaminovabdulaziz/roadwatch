"""Dev labels -> GT, track-cache loading, and the eval/tune scripts end to end (RUNBOOK P1.4)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate
from roadwatch.config import REPO_ROOT
from roadwatch.devdata import build_gt, cached_meta, load_cached, parse_time, read_label_rows
from roadwatch.types import TRACK_DTYPES, VideoMeta

META = VideoMeta("a.mp4", 30.0, 3840, 2160, 3000)  # 100 s


def test_parse_time() -> None:
    assert parse_time("75.5") == 75.5
    assert parse_time("1:15.5") == 75.5
    assert parse_time("0:01:15.5") == 75.5
    for bad in ("", "a", "1:2:3:4", "-1"):
        with pytest.raises(ValueError):
            parse_time(bad)


def write_csv(path: Path, lines: list[str]) -> Path:
    path.write_text("video,start,end,label,notes\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_build_gt_unions_and_keeps_empty_videos(tmp_path: Path) -> None:
    rows, problems = read_label_rows(
        [
            write_csv(
                tmp_path / "p1.csv", ["a.mp4,10,15,jaywalking,", "a.mp4,0:14,0:20,jaywalking,same person"]
            ),
            write_csv(tmp_path / "p2.csv", ["a.mp4,20,25,jaywalking,", "b.mp4,,,none,", ""]),
        ]
    )
    assert problems == []
    gt, problems = build_gt(rows, lambda name: VideoMeta(name, 30.0, 1, 1, 3000), evaluate.OFFICIAL_CLASSES)
    assert problems == []
    assert gt["a.mp4"]["events"] == [[10.0, 25.0, "jaywalking"]]  # overlap and touching segment unioned
    assert gt["b.mp4"] == {"duration": 100.0, "fps": 30.0, "events": []}
    assert evaluate.validate({"videos": {v: {"events": g["events"]} for v, g in gt.items()}}, gt)[0] == []


def test_build_gt_reports_every_bad_row(tmp_path: Path) -> None:
    rows, problems = read_label_rows(
        [
            write_csv(
                tmp_path / "p.csv",
                [
                    "a.mp4,5,4,jaywalking,",
                    "a.mp4,1,2,speeding,",
                    "a.mp4,90,101,congestion,",
                    "c.mp4,1,2,congestion,",
                    "a.mp4,x,2,congestion,",
                ],
            )
        ]
    )
    assert len(problems) == 1 and "p.csv:6" in problems[0]
    known = {"a.mp4": VideoMeta("a.mp4", 30.0, 1, 1, 3000)}
    _, problems = build_gt(rows, known.get, evaluate.OFFICIAL_CLASSES)
    text = " | ".join(problems)
    for expected in ("p.csv:2", "p.csv:3", "not an official class", "p.csv:4", "past the video end", "c.mp4"):
        assert expected in text


def make_cache(cache: Path, stem: str = "a", seconds: float | None = None) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    t = np.arange(0, 10, 0.1)
    pd.DataFrame(
        {
            "frame": np.arange(len(t)) * 3,
            "t": t,
            "track_id": 1,
            "cls": "car",
            "conf": 0.9,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 10.0,
            "fx": 5.0,
            "fy": 10.0,
        }
    ).astype(TRACK_DTYPES).to_parquet(cache / f"{stem}.parquet", index=False)
    video = {
        "name": f"{stem}.mp4",
        "fps": 30.0,
        "width": 3840,
        "height": 2160,
        "n_frames": 3000,
        "duration": 100.0,
    }
    (cache / f"{stem}.json").write_text(json.dumps({"video": video, "seconds": seconds}), encoding="utf-8")


def test_cache_loading(tmp_path: Path) -> None:
    make_cache(tmp_path)
    tt, meta = load_cached("a", tmp_path)
    assert meta == META and len(tt) == 100
    assert cached_meta("missing", tmp_path) is None
    with pytest.raises(FileNotFoundError):
        load_cached("missing", tmp_path)
    make_cache(tmp_path, "partial", seconds=30)
    with pytest.raises(ValueError):
        load_cached("partial", tmp_path)


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_label_eval_and_tune_scripts_end_to_end(tmp_path: Path) -> None:
    cache, raw = tmp_path / "tracks", tmp_path / "raw"
    make_cache(cache)
    raw.mkdir()
    write_csv(raw / "me.csv", ["a.mp4,10,15,jaywalking,"])
    gt = tmp_path / "gt.json"

    r = run(
        "labels_to_gt.py",
        "--raw",
        str(raw),
        "--videos",
        str(tmp_path),
        "--cache",
        str(cache),
        "--out",
        str(gt),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(gt.read_text())["a.mp4"]["events"] == [[10.0, 15.0, "jaywalking"]]

    pred, metrics = tmp_path / "pred.json", tmp_path / "metrics.json"
    r = run(
        "eval_dev.py", "--gt", str(gt), "--cache", str(cache), "--out", str(pred), "--metrics", str(metrics)
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Score A = 0.0000" in r.stdout and "FN [   10.00,    15.00]" in r.stdout
    assert json.loads(metrics.read_text())["per_class"]["jaywalking"]["f1_mean"] == 0.0

    # --pred scores a finished harness run instead, and leaves the Part A output file alone
    harness = tmp_path / "harness.json"
    events = [[10.2, 15.0, "jaywalking"]]
    harness.write_text(json.dumps({"videos": {"a.mp4": {"events": events, "risk": []}}}))
    pred.unlink()
    r = run(
        "eval_dev.py", "--gt", str(gt), "--pred", str(harness), "--out", str(pred), "--metrics", str(metrics)
    )
    assert r.returncode == 0, r.stdout + r.stderr
    written = json.loads(metrics.read_text())
    assert written["per_class"]["jaywalking"]["f1_mean"] == 1.0 and "harness.json" in written["source"]
    assert not pred.exists()

    r = run("tune.py", "--gt", str(gt), "--cache", str(cache), "--classes", "jaywalking", "--top", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "jaywalking: 3 combination(s), 1 labelled event(s)" in r.stdout and "current" in r.stdout

    write_csv(raw / "bad.csv", ["a.mp4,10,5,jaywalking,"])
    r = run(
        "labels_to_gt.py",
        "--raw",
        str(raw),
        "--videos",
        str(tmp_path),
        "--cache",
        str(cache),
        "--out",
        str(gt),
    )
    assert r.returncode == 1 and "bad.csv:2" in r.stdout


def test_cached_signals_only_for_the_same_lamp_boxes(tmp_path: Path) -> None:
    from roadwatch.devdata import cached_signals
    from roadwatch.scene.scene import Scene

    boxes = [{"id": "L1", "red": [0, 0, 5, 5], "yellow": [0, 6, 5, 5], "green": [0, 12, 5, 5]}]
    sidecar = {"video": {}, "signals": boxes, "signal_timeline": {"L1": [[0.0, 5.0, "red"]]}}
    (tmp_path / "a.json").write_text(json.dumps(sidecar), encoding="utf-8")
    assert cached_signals("a", tmp_path, Scene({"signals": boxes})) == {"L1": [(0.0, 5.0, "red")]}
    moved = [{**boxes[0], "red": [1, 0, 5, 5]}]
    assert cached_signals("a", tmp_path, Scene({"signals": moved})) == {}
    assert cached_signals("missing", tmp_path, Scene()) == {}


def test_diff_predictions() -> None:
    from roadwatch.devdata import diff_predictions

    a = {"videos": {"v.mp4": {"events": [[1.0, 2.0, "jaywalking"]], "risk": [[0.0, 0.1], [0.1, 0.2]]}}}
    assert diff_predictions(a, json.loads(json.dumps(a))) == []
    b = {"videos": {"v.mp4": {"events": [], "risk": [[0.0, 0.1], [0.1, 0.25]]}, "w.mp4": {"events": []}}}
    problems = " | ".join(diff_predictions(a, b))
    assert (
        "events differ" in problems and "risk differs" in problems and "w.mp4: only in the second" in problems
    )


def test_determinism_and_reproduction_scripts(tmp_path: Path, tiny_video: Path) -> None:
    r = run("check_determinism.py", str(tiny_video.parent), "--keep", str(tmp_path / "det"))
    assert r.returncode == 0 and "IDENTICAL" in r.stdout, r.stdout + r.stderr
    out = tmp_path / "pred.json"
    r = run("make_predictions_samples.py", str(tiny_video.parent), "--out", str(out))
    assert r.returncode == 0 and out.exists(), r.stdout + r.stderr
    r = run("make_predictions_samples.py", str(tiny_video.parent), "--out", str(out), "--check")
    assert r.returncode == 0 and "REPRODUCED" in r.stdout, r.stdout + r.stderr
