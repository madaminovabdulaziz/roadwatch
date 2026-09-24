"""In-memory job queue with one worker thread and hourly expiry (WEBSITE_SPEC "Demo API").

Contract: `JobQueue.submit(path)` -> job id or `QueueFull`; `get(id)` -> Job or None; one daemon worker
runs `process(src, workdir, progress)` per job and records "done" with its result or "error" with a
readable message (never an exception to the caller); jobs and their files expire after `expiry_sec`.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from demo.process import BadInput

log = logging.getLogger(__name__)

Processor = Callable[[Path, Path, Callable[[str, float], None]], dict[str, Any]]


class QueueFull(RuntimeError):
    """Too many jobs waiting; the user should retry later."""


@dataclass
class Job:
    id: str
    workdir: Path
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | error
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        """The JSON the API returns for this job."""
        out: dict[str, Any] = {"status": self.status, "progress": round(self.progress), "stage": self.stage}
        if self.status == "error":
            out["message"] = self.message
        if self.status == "done" and self.result is not None:
            result = {k: v for k, v in self.result.items() if k != "video_path"}
            out["result"] = {**result, "video_url": f"/api/files/{self.id}/annotated.mp4"}
        return out


class JobQueue:
    def __init__(self, root: Path, processor: Processor, max_queue: int, expiry_sec: float) -> None:
        self.root = root
        self.processor = processor
        self.max_queue = max_queue
        self.expiry_sec = expiry_sec
        self.jobs: dict[str, Job] = {}
        self._pending: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._work, name="demo-worker", daemon=True).start()

    def new_job(self) -> Job:
        """A job with its own folder; the caller writes the upload to `job.workdir / "input.mp4"`."""
        self.expire()
        with self._lock:
            waiting = sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))
            if waiting >= self.max_queue:
                raise QueueFull("the demo is busy; please try again in a few minutes")
            job_id = uuid.uuid4().hex[:12]
            job = Job(job_id, self.root / job_id)
            job.workdir.mkdir(parents=True)
            self.jobs[job.id] = job
        return job

    def submit(self, job: Job) -> None:
        self._pending.put(job.id)

    def discard(self, job: Job) -> None:
        with self._lock:
            self.jobs.pop(job.id, None)
        shutil.rmtree(job.workdir, ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def waiting(self) -> int:
        return self._pending.qsize()

    def expire(self) -> None:
        now = time.time()
        with self._lock:
            old = [
                j
                for j in self.jobs.values()
                if now - j.created > self.expiry_sec and j.status in ("done", "error")
            ]
            for j in old:
                del self.jobs[j.id]
        for j in old:
            shutil.rmtree(j.workdir, ignore_errors=True)

    def _work(self) -> None:
        while True:
            job = self.jobs.get(self._pending.get())
            if job is None:
                continue

            def progress(stage: str, pct: float, job: Job = job) -> None:
                job.stage, job.progress = stage, max(job.progress, min(100.0, pct))

            job.status = "running"
            try:
                job.result = self.processor(job.workdir / "input.mp4", job.workdir, progress)
                job.status, job.progress = "done", 100.0
            except BadInput as exc:
                job.status, job.message = "error", str(exc)
            except Exception:
                log.exception("job %s failed", job.id)
                job.status, job.message = "error", "processing failed on our side; please try another clip"
