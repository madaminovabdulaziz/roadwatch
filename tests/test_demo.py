"""Demo backend: API contract, validation that never 500s, the job lifecycle, camera matching (P1.7)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from demo.app import create_app
from demo.process import BadInput, demo_cfg, match_images, probe_upload, process, reencode
from roadwatch.config import WEIGHTS_DIR
from roadwatch.video import probe


def fake_processor(src: Path, workdir: Path, progress) -> dict:
    progress("detecting", 50)
    return {
        "events": [[1.0, 2.0, "jaywalking"]],
        "risk": [[0.0, 0.1]],
        "video_path": str(src),
        "camera_match": True,
        "duration": 2.0,
    }


def failing_processor(src: Path, workdir: Path, progress) -> dict:
    raise RuntimeError("boom")


def wait_done(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.05)
    raise AssertionError("job never finished")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(fake_processor, tmp_path / "jobs"))


def test_upload_poll_and_download(client: TestClient, tiny_video: Path) -> None:
    res = client.post("/api/jobs", files={"file": ("clip.mp4", tiny_video.read_bytes(), "video/mp4")})
    assert res.status_code == 200, res.text
    status = wait_done(client, res.json()["job_id"])
    assert status["status"] == "done" and status["progress"] == 100
    result = status["result"]
    assert result["events"] == [[1.0, 2.0, "jaywalking"]] and result["camera_match"] is True
    assert "video_path" not in result
    video = client.get(result["video_url"])
    assert video.status_code == 200 and video.content == tiny_video.read_bytes()
    assert client.get("/api/health").json()["status"] == "ok"


@pytest.mark.parametrize(
    ("name", "body", "status", "text"),
    [
        ("notes.txt", b"hello", 400, ".mp4"),
        ("broken.mp4", b"not a video at all", 400, "could not be read"),
    ],
)
def test_bad_uploads_get_readable_4xx(
    client: TestClient, name: str, body: bytes, status: int, text: str
) -> None:
    res = client.post("/api/jobs", files={"file": (name, body, "video/mp4")})
    assert res.status_code == status and text in res.json()["message"]


def test_too_large_and_too_long(tmp_path: Path, tiny_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(demo_cfg(), "max_upload_mb", 0)
    small = TestClient(create_app(fake_processor, tmp_path / "a"))
    res = small.post("/api/jobs", files={"file": ("clip.mp4", tiny_video.read_bytes(), "video/mp4")})
    assert res.status_code == 413
    monkeypatch.setitem(demo_cfg(), "max_upload_mb", 100)
    monkeypatch.setitem(demo_cfg(), "max_duration_sec", 1)
    short = TestClient(create_app(fake_processor, tmp_path / "b"))
    res = short.post("/api/jobs", files={"file": ("clip.mp4", tiny_video.read_bytes(), "video/mp4")})
    assert res.status_code == 400 and "limit is 1 s" in res.json()["message"]


def test_processing_failure_becomes_a_message(tmp_path: Path, tiny_video: Path) -> None:
    client = TestClient(create_app(failing_processor, tmp_path / "jobs"))
    job_id = client.post(
        "/api/jobs", files={"file": ("clip.mp4", tiny_video.read_bytes(), "video/mp4")}
    ).json()["job_id"]
    status = wait_done(client, job_id)
    assert status["status"] == "error" and "try another clip" in status["message"]


def test_unknown_job_and_file(client: TestClient) -> None:
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/files/nope/annotated.mp4").status_code == 404


def test_reencode_gives_browser_friendly_h264(tmp_path: Path, tiny_video: Path) -> None:
    out = tmp_path / "work.mp4"
    reencode(tiny_video, out, height=48)
    meta = probe(out)
    assert (meta.width, meta.height) == (80, 48) and meta.n_frames == probe(tiny_video).n_frames
    import av

    with av.open(str(out)) as c:
        assert c.streams.video[0].codec_context.name == "h264"
        assert c.streams.video[0].codec_context.pix_fmt == "yuv420p"
    assert probe_upload(out) == pytest.approx(meta.duration)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(BadInput):
        probe_upload(bad)


@pytest.mark.skipif(not (WEIGHTS_DIR / "yolo11m.torchscript").exists(), reason="weights not downloaded")
def test_real_processing_end_to_end(tmp_path: Path, tiny_video: Path) -> None:
    stages = []
    result = process(tiny_video, tmp_path, lambda stage, pct: stages.append((stage, pct)))
    assert result["events"] == [] and result["risk"] == []  # every class is still disabled
    assert result["camera_match"] is False  # a synthetic clip is not our camera
    assert Path(result["video_path"]).exists() and result["duration"] == pytest.approx(2.0)
    assert [s for s, _ in stages][0] == "decoding" and stages[-1] == ("rendering", 100)
    assert all(b >= a for (_, a), (_, b) in zip(stages, stages[1:], strict=False))


def test_camera_match_same_view_vs_other_view() -> None:
    rng = np.random.default_rng(0)
    scene = (rng.random((540, 960)) * 255).astype(np.uint8)
    scene = cv2.cvtColor(cv2.GaussianBlur(scene, (0, 0), 2), cv2.COLOR_GRAY2BGR)
    cfg = demo_cfg()["camera_match"]
    shifted = np.roll(scene, 7, axis=1)  # same camera, tiny shake
    other = cv2.cvtColor(
        cv2.GaussianBlur((rng.random((540, 960)) * 255).astype(np.uint8), (0, 0), 2), cv2.COLOR_GRAY2BGR
    )
    assert match_images(shifted, scene, cfg) >= cfg["min_inliers"]
    assert match_images(other, scene, cfg) < cfg["min_inliers"]
