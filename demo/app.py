"""FastAPI app of the live demo (WEBSITE_SPEC "Demo API").

    POST /api/jobs              multipart file=video.mp4 -> {"job_id"}
    GET  /api/jobs/{id}         -> {"status", "progress", "stage", "message"?, "result"?}
    GET  /api/files/{id}/annotated.mp4
    GET  /api/health

Bad input gets a 4xx JSON `{"message": ...}` the website shows as is; nothing a user uploads can cause a
500. Run from the repository root with the CPU profile (demo/Dockerfile sets the same variable):

    ROADWATCH_OVERRIDES=demo/config.yaml uvicorn demo.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from demo.jobs import JobQueue, QueueFull
from demo.process import BadInput, demo_cfg, probe_upload
from demo.process import process as run_process

CHUNK = 1 << 20


def create_app(processor=run_process, root: Path | None = None) -> FastAPI:
    """The app with its own job queue; tests pass a fake `processor` and a temporary `root`."""
    cfg = demo_cfg()
    jobs = JobQueue(
        root or Path(tempfile.gettempdir()) / "roadwatch-demo", processor, cfg["max_queue"], cfg["expiry_sec"]
    )
    app = FastAPI(title="RoadWatch demo", docs_url=None, redoc_url=None)
    # A public, read-only demo: the static website on another origin calls it from the browser.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    def reject(status: int, message: str) -> JSONResponse:
        return JSONResponse({"message": message}, status_code=status)

    @app.post("/api/jobs")
    async def create_job(file: Annotated[UploadFile, File()]) -> JSONResponse:
        if not (file.filename or "").lower().endswith(".mp4"):
            return reject(400, "please upload an .mp4 file")
        try:
            job = jobs.new_job()
        except QueueFull as exc:
            return reject(503, str(exc))
        limit = cfg["max_upload_mb"] * 1024 * 1024
        size = 0
        with open(job.workdir / "input.mp4", "wb") as f:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    jobs.discard(job)
                    return reject(413, f"the file is larger than {cfg['max_upload_mb']} MB")
                f.write(chunk)
        try:
            duration = probe_upload(job.workdir / "input.mp4")
        except BadInput as exc:
            jobs.discard(job)
            return reject(400, str(exc))
        if duration > cfg["max_duration_sec"]:
            jobs.discard(job)
            return reject(
                400, f"the video is {round(duration)} s long; the limit is {cfg['max_duration_sec']} s"
            )
        jobs.submit(job)
        return JSONResponse({"job_id": job.id})

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse({"status": "error", "message": "unknown or expired job"}, status_code=404)
        return JSONResponse(job.public())

    @app.get("/api/files/{job_id}/annotated.mp4", response_model=None)
    def job_video(job_id: str) -> FileResponse | JSONResponse:
        job = jobs.get(job_id)
        if job is None or job.result is None:
            return reject(404, "no video for this job")
        return FileResponse(job.result["video_path"], media_type="video/mp4")

    @app.get("/api/health")
    def health() -> dict:
        jobs.expire()
        return {"status": "ok", "waiting": jobs.waiting()}

    return app


app = create_app()
