# Demo backend

FastAPI service for the website's live demo (contract: `docs/WEBSITE_SPEC.md`, "Demo API"). One worker
thread processes uploads one at a time with the same `roadwatch` pipeline as the submission, on CPU:
`demo/config.yaml` is deep-merged over `configs/thresholds.yaml` through `ROADWATCH_OVERRIDES`
(imgsz 640, one detection per 6 frames). The submission never sets that variable.

Per upload: validate (.mp4, <= 100 MB, <= 120 s, decodable; otherwise a 4xx with a readable message)
-> re-encode to 720p H.264 yuv420p +faststart (PyAV) -> camera check against `configs/reference.jpg`
(ORB + RANSAC; a different camera runs without scene rules) -> perception -> rules -> annotated video
(`roadwatch/render.py`; until it exists, or if it fails, the re-encoded video is returned). Jobs and
files expire after an hour.

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
