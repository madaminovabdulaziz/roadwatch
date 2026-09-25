# Demo backend

FastAPI service for the website's live demo (contract: `docs/WEBSITE_SPEC.md`, "Demo API"; decisions:
`docs/SPEC.md` §12.30). One worker thread processes uploads one at a time with the same `roadwatch`
pipeline as the submission, on CPU: `demo/config.yaml` is deep-merged over `configs/thresholds.yaml`
through `ROADWATCH_OVERRIDES` (imgsz 640, 10 Hz like the submission). The submission never sets that
variable.

Per upload: validate (.mp4, <= 2 GB, decodable; otherwise a 4xx with a readable message) -> re-encode
the first 20 s to 720p H.264 yuv420p +faststart from reference frames only (PyAV) -> camera check =
scene registration to `configs/reference.jpg` (a different camera runs without scene rules) ->
perception + signal lamps -> rules -> Part B risk curve (reusing Part A's detections) -> annotated
video (`roadwatch/render.py`; if it fails, the re-encoded video is returned). A job has 15 minutes;
running jobs report an ETA, queued ones their place in the queue. Uploads are deleted when their job
ends; jobs and results expire after an hour.

A 20 s window should take 3-5 minutes on a 2-vCPU Space (an estimate), mostly detection and 4K decoding; `max_process_sec`
in `demo/config.yaml` trades the window against the wait. Check the time on the real Space after
deploying (measured here: 54 s end-to-end for an 8 s 4K clip of C3902 through the live server on an M1).

## Run locally

```bash
ROADWATCH_OVERRIDES=demo/config.yaml ROADWATCH_DEVICE=cpu uvicorn demo.app:app --port 7860
```

## Deploy to a Hugging Face Space (human step)

1. `bash weights/download.sh`, then `docker build -f demo/Dockerfile -t roadwatch-demo .` and
   `docker run --rm -p 7860:7860 roadwatch-demo` to check it locally.
2. Create a Space (SDK: Docker, CPU basic). Push `roadwatch/`, `configs/`, `demo/`, `weights/`,
   `solution.py`, `evaluate.py`, plus `demo/Dockerfile` as the Space's root `Dockerfile` and a root
   `README.md` whose front matter says `sdk: docker` and `app_port: 7860`.
3. Set the website's `NEXT_PUBLIC_DEMO_API` to the Space URL, and point an uptime monitor at
   `/api/health` every 10 minutes so the Space does not sleep during judging.
