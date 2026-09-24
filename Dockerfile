# RoadWatch submission image: the organizers' two commands inside a container (task FAQ, SPEC §11).
#
#   bash weights/download.sh                      # once, on the host (skip if weights/ is already filled)
#   docker build -t roadwatch .
#   docker run --rm --gpus all --network none \
#     -v /data/test:/data/videos:ro -v "$PWD/out":/data/out roadwatch
#
# The run writes /data/out/predictions.json and then checks it with evaluate.py --validate-only.
# Nothing is downloaded at run time: dependencies and weights are baked in at build time.

# CUDA "base" is enough: the cu126 torch wheels bundle cuBLAS/cuDNN themselves, and the NVIDIA
# container runtime injects the driver. Ubuntu 22.04 ships Python 3.10, which requirements.txt supports.
FROM nvidia/cuda:12.6.3-base-ubuntu22.04 AS app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Dependencies first, so code changes do not reinstall torch.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV VIDEOS=/data/videos \
    OUT=/data/out/predictions.json \
    TEAM=roadwatch

CMD ["bash", "-c", "mkdir -p \"$(dirname \"$OUT\")\" && python run_submission.py --videos \"$VIDEOS\" --out \"$OUT\" --team \"$TEAM\" && python evaluate.py --pred \"$OUT\" --validate-only"]

# Final stage (the default target): weights. Files already in weights/ with the right checksum are
# kept; anything missing is fetched from the release, so the build fails loudly without weights.
# While the repository is private, pass a token as a BuildKit secret:
#   docker build --secret id=github_token,env=GITHUB_TOKEN -t roadwatch .
# `--target app` skips this stage (a weightless image, for testing the packaging only).
FROM app AS final
RUN --mount=type=secret,id=github_token,required=false \
    GITHUB_TOKEN="$(cat /run/secrets/github_token 2>/dev/null || true)" bash weights/download.sh
