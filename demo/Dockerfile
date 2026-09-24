# Demo backend image for a Hugging Face Docker Space (CPU). Build context = repository root:
#   docker build -f demo/Dockerfile -t roadwatch-demo .
#   docker run --rm -p 7860:7860 roadwatch-demo
# Weights must be in weights/ before the build (bash weights/download.sh), like the submission image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ROADWATCH_OVERRIDES=/app/demo/config.yaml \
    ROADWATCH_DEVICE=cpu

# Hugging Face runs Spaces as uid 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY demo/requirements.txt demo/requirements.txt
RUN pip install -r demo/requirements.txt

COPY --chown=app roadwatch roadwatch
COPY --chown=app configs configs
COPY --chown=app demo demo
COPY --chown=app weights weights
COPY --chown=app solution.py evaluate.py ./
RUN test -f weights/yolo11m.torchscript || (echo "weights/ is empty: run bash weights/download.sh first" && exit 1)

USER app
EXPOSE 7860
CMD ["uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "7860"]
