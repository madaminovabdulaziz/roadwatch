"""Demo backend: API contract, validation that never 500s, the job lifecycle, camera matching (P1.7)."""

from __future__ import annotations

import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from demo.app import create_app
from demo.jobs import JobQueue
from demo.process import (
    BadInput,
    SharedDetections,
    demo_cfg,
    match_camera,
    probe_upload,
    process,
    reencode,
)
from roadwatch.config import CONFIG_DIR, WEIGHTS_DIR
from roadwatch.types import FrameDetections
from roadwatch.video import FrameReader, probe
from tests.conftest import INDEXED_FRAMES, indexed_image, read_index
from tests.test_registration import CAMERA_MOVE, moved, scene_with_reference


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


def test_too_large(tmp_path: Path, tiny_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(demo_cfg(), "max_upload_mb", 0)
    small = TestClient(create_app(fake_processor, tmp_path / "a"))
    res = small.post("/api/jobs", files={"file": ("clip.mp4", tiny_video.read_bytes(), "video/mp4")})
    assert res.status_code == 413 and "larger than" in res.json()["message"]
    # a declared size far over the limit is refused before the body is read
    res = small.post("/api/jobs", files={"file": ("clip.mp4", b"\0" * (3 << 20), "video/mp4")})
    assert res.status_code == 413 and "larger than" in res.json()["message"]


def test_long_videos_are_accepted(tmp_path: Path, tiny_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # raw camera files run for minutes: they are accepted and only the first max_process_sec analysed
    monkeypatch.setitem(demo_cfg(), "max_process_sec", 0.5)
    client = TestClient(create_app(fake_processor, tmp_path / "jobs"))
    res = client.post("/api/jobs", files={"file": ("clip.mp4", tiny_video.read_bytes(), "video/mp4")})
    assert res.status_code == 200, res.text


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
    with av.open(str(out)) as c:
        assert c.streams.video[0].codec_context.name == "h264"
        assert c.streams.video[0].codec_context.pix_fmt == "yuv420p"
    assert probe_upload(out) == pytest.approx(meta.duration)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(BadInput):
        probe_upload(bad)


def shown(video: Path) -> list[int]:
    """The source frame index each frame of a re-encoded indexed video shows."""
    return [read_index(img) for _, _, img in FrameReader(video, 1)]


def test_reencode_skips_b_frames_and_holds_the_last_reference_frame(
    tmp_path: Path, indexed_video: Path
) -> None:
    out = tmp_path / "work.mp4"
    reencode(indexed_video, out, height=144)
    src, meta = probe(indexed_video), probe(out)
    assert meta.n_frames == INDEXED_FRAMES and meta.fps == pytest.approx(src.fps)
    idx = shown(out)
    decoded = sorted(set(idx))
    assert len(decoded) <= INDEXED_FRAMES // 3 + 1  # the two B-frames between references were never decoded
    for slot, i in enumerate(idx):  # each frame shows the latest decoded one at or before it
        assert i == max([d for d in decoded if d <= slot] or [decoded[0]])
        assert abs(i - slot) <= 2
    # the re-encode itself has no B-frames: later stages reading it with skip_nonref get every stride-th frame
    assert [i for i, _, _ in FrameReader(out, 3, skip_nonref=True)] == list(range(0, INDEXED_FRAMES, 3))


def test_reencode_stops_at_max_sec(tmp_path: Path, indexed_video: Path) -> None:
    out = tmp_path / "work.mp4"
    assert reencode(indexed_video, out, height=144, max_sec=1.0) == pytest.approx(
        30 / probe(indexed_video).fps
    )
    assert probe(out).n_frames == 30  # t < 1 s at 29.97 fps


def test_reencode_brings_high_frame_rates_down_to_about_30(tmp_path: Path) -> None:
    src = tmp_path / "fast.mp4"
    with av.open(str(src), "w") as c:
        stream = c.add_stream("libx264", rate=60)
        stream.width, stream.height, stream.pix_fmt = 256, 144, "yuv420p"
        stream.options = {"x264-params": "bframes=0"}
        for i in range(60):
            frame = av.VideoFrame.from_ndarray(indexed_image(i), format="bgr24")
            frame.pts, frame.time_base = i, Fraction(1, 60)
            c.mux(stream.encode(frame))
        c.mux(stream.encode())
    out = tmp_path / "work.mp4"
    reencode(src, out, height=144)
    meta = probe(out)
    assert meta.fps == pytest.approx(30) and meta.n_frames == 30
    assert shown(out) == list(range(0, 60, 2))


class CountingDetector:
    batch_size = 2

    def __init__(self) -> None:
        self.calls: list[int] = []

    def frame_size_for(self, width: int, height: int) -> tuple[int, int]:
        return width // 2, height // 2

    def predict(self, batch, native_size=None):
        self.calls.extend(idx for idx, _, _ in batch)
        empty = np.zeros((0, 4), np.float32)
        return [
            FrameDetections(idx, t, empty, np.zeros(0, np.float32), np.zeros(0, np.int64))
            for idx, t, _ in batch
        ]


def test_shared_detections_run_the_detector_once_per_frame() -> None:
    inner = CountingDetector()
    shared = SharedDetections(inner)
    img = np.zeros((4, 4, 3), np.uint8)
    first = shared.predict([(0, 0.0, img), (6, 0.2, img)], native_size=(8, 8))
    again = shared.predict([(6, 0.2, img), (12, 0.4, img)], native_size=(8, 8))
    assert inner.calls == [0, 6, 12]
    assert again[0] is first[1] and [d.frame_idx for d in again] == [6, 12]
    assert shared.batch_size == 2 and shared.frame_size_for(8, 8) == (4, 4)


# ------------------------------------------------------------------ job lifecycle: timeout, ETA, queue
def blocking_processor(release: threading.Event):
    def run(src: Path, workdir: Path, progress) -> dict:
        progress("detecting", 50)
        release.wait(5)
        return fake_processor(src, workdir, progress)

    return run


def test_a_job_that_runs_too_long_ends_with_a_message(tmp_path: Path, tiny_video: Path) -> None:
    def slow(src: Path, workdir: Path, progress) -> dict:
        for pct in range(100):
            progress("detecting", pct)
            time.sleep(0.02)
        return fake_processor(src, workdir, progress)

    jobs = JobQueue(tmp_path, slow, max_queue=2, expiry_sec=60, max_job_sec=0.2)
    job = jobs.new_job()
    (job.workdir / "input.mp4").write_bytes(tiny_video.read_bytes())
    jobs.submit(job)
    for _ in range(100):
        if job.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert job.status == "error" and "took longer than" in job.message
    assert not (job.workdir / "input.mp4").exists()  # the upload is deleted once the job ends


def test_status_shows_eta_and_queue_position(tmp_path: Path, tiny_video: Path) -> None:
    release = threading.Event()
    jobs = JobQueue(tmp_path, blocking_processor(release), max_queue=3, expiry_sec=60, max_job_sec=60)
    first, second = jobs.new_job(), jobs.new_job()
    for job in (first, second):
        (job.workdir / "input.mp4").write_bytes(tiny_video.read_bytes())
        jobs.submit(job)
    for _ in range(100):
        if first.progress == 50:
            break
        time.sleep(0.02)
    time.sleep(0.1)
    running, queued = jobs.public(first), jobs.public(second)
    release.set()
    assert running["status"] == "running" and running["eta_sec"] >= 0
    assert queued["status"] == "queued" and queued["ahead"] == 1


@pytest.mark.skipif(not (WEIGHTS_DIR / "yolo11m.torchscript").exists(), reason="weights not downloaded")
def test_real_processing_end_to_end(tmp_path: Path, tiny_video: Path) -> None:
    stages = []
    result = process(tiny_video, tmp_path, lambda stage, pct: stages.append((stage, pct)))
    assert result["events"] == []  # every class is still disabled
    assert result["camera_match"] is False  # a synthetic clip is not our camera
    assert Path(result["video_path"]).exists() and result["duration"] == pytest.approx(2.0)
    assert result["source_duration"] == pytest.approx(2.0)
    times = [t for t, _ in result["risk"]]
    assert times and times == sorted(set(times)) and all(0.0 <= r <= 1.0 for _, r in result["risk"])
    assert [s for s, _ in stages][0] == "decoding" and stages[-1] == ("rendering", 100)
    assert all(b >= a for (_, a), (_, b) in zip(stages, stages[1:], strict=False))


def write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> Path:
    with av.open(str(path), "w") as c:
        stream = c.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for i, img in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            c.mux(stream.encode(frame))
        c.mux(stream.encode())
    return path


def test_our_camera_in_other_light_matches_and_the_scene_follows_it(tmp_path: Path) -> None:
    # C3902 at 720p against the C3896 reference: moved ~100 px and in late-afternoon shade. The ORB
    # check this replaced found 8 of the 40 inliers it needed there; the SIFT registration found 99.
    ref = cv2.imread(str(CONFIG_DIR / "reference.jpg"))
    view = cv2.warpPerspective(ref, CAMERA_MOVE, (ref.shape[1], ref.shape[0]))
    shade = (255 * (cv2.resize(view, (1280, 720), interpolation=cv2.INTER_AREA) / 255.0) ** 1.8).astype(
        np.uint8
    )
    video = write_video(tmp_path / "ours.mp4", [shade] * 10)
    scene = scene_with_reference()
    matched, aligned = match_camera(video, scene)
    assert matched
    np.testing.assert_allclose(
        aligned.layers["stop_lines"][0]["line"],
        moved(CAMERA_MOVE, scene.layers["stop_lines"][0]["line"]) / 3,
        atol=1.5,
    )


def test_another_camera_does_not_match(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    noise = [
        cv2.GaussianBlur((rng.random((720, 1280, 3)) * 255).astype(np.uint8), (0, 0), 2) for _ in range(5)
    ]
    matched, scene = match_camera(write_video(tmp_path / "other.mp4", noise), scene_with_reference())
    assert not matched and scene.layers == {}
